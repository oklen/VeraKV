"""kv_keepsel.py -- KEEP-style fresh-budget selector, same-harness (review-proposed arm).

KEEP (2602.23592) selects which memory GROUPS to recompute by query-conditioned attention
propagation (pure attention, no causal structure). Our b_hot fixes the fresh budget on the
RECENT events instead. This experiment swaps ONLY the fresh-recompute selector inside the
b_hot serving base (header + rblk + anchored stores; identical evidence set, identical
2-event fresh budget) -- the 7th entry of the "smart <= random" ledger, now aimed at the
fresh slot rather than routing/filling/conditioning:

  bh      fresh = last-2 events (the .2209 gate, expected to replicate byte-identically)
  kp_att  fresh = top-2 kept events by KEEP-style score: s = s0 + 0.5 * P^T s0, where
          s0[e] = question's eager attention mass onto event e's rows (query conditioning)
          and P = row-normalized event->event attention matrix (one propagation step over
          the trajectory's own attention graph; layer/head-mean signal, aggregation-robust
          per the orphan campaign)
  kp_rnd  fresh = seeded-random 2 kept events (selection control)

Non-selected events (including the hot tail, when unselected) serve from their anchored
stores; selected events are recomputed fresh IN POSITION ORDER over the partially
assembled cache (piecewise, causal -- serve_rep discipline from kv_orphan).

Pre-registered readings: kp_att > bh would say query-conditioned attention finds a better
home for the fresh budget than recency; kp_att ~ kp_rnd <= bh completes the ledger (no
content signal beats the structural default, fresh edition).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_keepsel --shard 0 --queue_dir ./out/kp_q
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_orphan import build_attn_matrix

ARMS = ["bh", "kp_att", "kp_rnd"]


@torch.no_grad()
def q_attn_scores(llm, cache, total, q_ids, spans, band):
    """Eager forward of the question over the full-context cache; returns per-event
    attention mass (mean over question tokens of summed span attention). Cache is
    cropped back to `total` afterwards."""
    core = getattr(llm.model, "model", llm.model)
    old = core.config._attn_implementation
    core.config._attn_implementation = "eager"
    try:
        L = q_ids.shape[1]
        dev = llm.device
        out = core(input_ids=q_ids.to(dev), past_key_values=cache, use_cache=True,
                   output_attentions=True,
                   position_ids=torch.arange(total, total + L, device=dev).unsqueeze(0),
                   cache_position=torch.arange(total, total + L, device=dev),
                   attention_mask=torch.ones(1, total + L, dtype=torch.long, device=dev))
        acc = None
        for li in band:
            m = out.attentions[li][0].float().mean(0)  # (L, total+L)
            acc = m if acc is None else acc + m
        acc = acc / len(band)
        s = torch.zeros(len(spans))
        for i, (a, b) in enumerate(spans):
            s[i] = acc[:, a:b].sum(1).mean().item()
        del out, acc
    finally:
        core.config._attn_implementation = old
        cache.crop(total)
    return s


def event_attn_matrix(A, spans):
    """Row-normalized event->event attention (P[i,j] = event i's mean attention to j)."""
    E = len(spans)
    P = torch.zeros(E, E)
    for i, (a, b) in enumerate(spans):
        rm = A[a:b, :].float().mean(0)
        for j, (c0, d0) in enumerate(spans):
            P[i, j] = rm[c0:d0].sum()
    return P / P.sum(1, keepdim=True).clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--sel_events", type=int, default=2)
    ap.add_argument("--prop_w", type=float, default=0.5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/kp.json")
    ap.add_argument("--ans_out", default="./out/kp_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sig", default="mean", choices=["mean", "late_vnorm"])
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    core = getattr(llm.model, "model", llm.model)
    nl = core.config.num_hidden_layers
    band = list(range(nl)) if args.sig == "mean" else list(range(nl // 2, nl))

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    target = args.ep_offset + args.max_ep
    eps = []
    while len(eps) < target and any(queues):
        for qd in queues:
            if qd and len(eps) < target:
                eps.append(qd.popleft())
    eps = eps[args.ep_offset:]

    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
    fresh_tok = {a: 0 for a in RUN}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        all_seg_ids, spans, cur = [], [], H
        for txt in segment_texts:
            sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
            all_seg_ids.append(sids)
            spans.append((cur, cur + len(sids)))
            cur += len(sids)
        total = cur
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k)
                      if p in id2idx}
            k5 = sorted(hot_idx | picked)
            routed.append(k5)
            need |= set(k5)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st_anch = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                      pos, keep_a=H + alen)
        full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts)
        assert ftot == total
        A = build_attn_matrix(llm, header_ids + traj_flat, sig=args.sig)
        P = event_attn_matrix(A, spans)
        del A
        torch.cuda.empty_cache()
        print(f"[w{wid}] ep {ep.episode_id} total={total} segs={n_seg} need={len(need)} "
              f"qa={len(qas)}", flush=True)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                s0, e0 = spans[i]
                for pth in range(max(H, s0), min(H + w_anch, e0)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def serve_bh(kept, qids_):
            keep = [i for i in kept if i not in hot_idx]
            rbl = rblk_for(kept)
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            p = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        def append_kv(c, kv):
            for li, (K, V) in enumerate(kv):
                c.update(K.to(llm.device, non_blocking=True),
                         V.to(llm.device, non_blocking=True), li)

        def serve_sel(kept, sel, qids_):
            """b_hot serving base with the fresh budget on `sel` (position-ordered,
            piecewise causal: each fresh event sees exactly the cache rows before it)."""
            sel_sorted = sorted(sel)
            sel_starts = [spans[i][0] for i in sel_sorted]
            keep = [i for i in kept if i not in sel]
            items = [(0, "kv", (header_kv, list(range(H))))]
            rbl = rblk_for(kept)
            if rbl:
                kvl, pos = rbl
                groups = {}
                for j, p_ in enumerate(pos):
                    g = bisect.bisect_right(sel_starts, p_)
                    groups.setdefault(g, []).append(j)
                for g in sorted(groups):
                    idx = torch.tensor(groups[g], dtype=torch.long)
                    piece = [(K.index_select(2, idx), V.index_select(2, idx)) for K, V in kvl]
                    items.append((pos[groups[g][0]], "kv",
                                  (piece, [pos[j] for j in groups[g]])))
            for i in keep:
                items.append((spans[i][0], "kv",
                              (st_anch[i], list(range(spans[i][0], spans[i][1])))))
            for i in sel_sorted:
                items.append((spans[i][0], "fresh", i))
            items.sort(key=lambda t: t[0])
            c = DynamicCache()
            pos_list = []
            for _, kind, payload in items:
                if kind == "kv":
                    kv, pos = payload
                    append_kv(c, kv)
                    pos_list += pos
                else:
                    i = payload
                    s0, e0 = spans[i]
                    prefill_fresh(llm, c, len(pos_list), all_seg_ids[i],
                                  list(range(s0, e0)))
                    pos_list += list(range(s0, e0))
            out, _, _ = llm._greedy_pos(
                c, torch.tensor(pos_list, dtype=torch.long, device=llm.device),
                qids_, args.ans_tokens)
            del c
            return out

        for qi, (qa, k5) in enumerate(zip(qas, routed)):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans, meta = {}, {}

            s0 = q_attn_scores(llm, full_cache, total, qids, spans, band)
            s0n = s0 / s0.sum().clamp_min(1e-9)
            score = s0n + args.prop_w * (P.T @ s0n)
            pool = sorted(k5)
            sel_att = sorted(pool, key=lambda i: -float(score[i]))[: args.sel_events]
            rng = random.Random(int(hashlib.md5(
                f"{ep.episode_id}|{qi}".encode()).hexdigest()[:8], 16))
            sel_rnd = sorted(rng.sample(pool, min(args.sel_events, len(pool))))
            meta["sel_att"] = sel_att
            meta["sel_rnd"] = sel_rnd
            meta["att_hot_overlap"] = len(set(sel_att) & hot_idx)

            if "bh" in RUN:
                ans["bh"] = serve_bh(k5, qids)
                fresh_tok["bh"] += len(hot_ids) + qids.shape[1]
            if "kp_att" in RUN:
                ans["kp_att"] = serve_sel(k5, sel_att, qids)
                fresh_tok["kp_att"] += sum(spans[i][1] - spans[i][0] for i in sel_att) \
                    + qids.shape[1]
            if "kp_rnd" in RUN:
                ans["kp_rnd"] = serve_sel(k5, sel_rnd, qids)
                fresh_tok["kp_rnd"] += sum(spans[i][1] - spans[i][0] for i in sel_rnd) \
                    + qids.shape[1]

            row = {"episode_id": ep.episode_id, "qi": qi, "q": q, "gold": gold,
                   "domain": ep.domain, **meta}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc, "fresh_tok": fresh_tok},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} done | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] KEEPSEL QUEUE over {len(eps)} episodes", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM ep idx {i} skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 60)
    print(f"KEEPSEL w{wid}: {n} QA")
    for a in RUN:
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%  fresh/QA {fresh_tok[a]//nn}")
    print(f"KP_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
