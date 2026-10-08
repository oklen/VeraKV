"""kv_orphan.py -- the orphan-attention hypothesis (user, 2026-07-14), tested on the newly
established zero-write harvest path (glob_R), where the signal is for the first time
actually computable: the full-context cache exists, so we can measure, for every trajectory
token, how much of its ORIGINAL attention fell on rows that serving drops.

Two instantiations of the signal, each with its own control (all arms share glob_R's serving
base: header + opening rows + routed-span rows gathered from the full cache, hot frozen):

  (A) REPAIR: recompute the top-orphan served events over the pruned view (their write-time
      attention leaned most on dropped rows -> dangling references; recomputing re-forms them
      consistent with what is actually served, at the cost of DESTROYING the absorbed copies
      of dropped content they carry).
  (B) FILL: gather the dropped events that receive the most attention FROM served rows
      (attention-guided hole-filling round two; gh showed hole-filling is what works here).

Arms (n=824 paired; deterministic — pairs row-for-row with gh/da/dl runs):
  bh        gate (.2209, 13th replication expected)
  glob_R    gate/base (expect 173/824, replicating gh)
  or_rep    glob_R + recompute top-2 orphan-mass served events (position-ordered rebuild)
  or_reprnd glob_R + recompute 2 seeded-random served events        (selection control)
  or_fill   glob_R + gather top orphan-target dropped events (<=2048 tok)
  or_fillrnd glob_R + gather random dropped events (<=2048 tok)     (selection control)

Signal: one extra eager chunked prefill per episode accumulates A = layer/head-mean
attention (total x total, fp16, GPU). Per QA: dropped = trajectory rows not served;
orphan(event i) = mean_t A[t in span_i, dropped].sum(); target(dropped event j) =
A[served rows, span_j].sum().

Pre-registered readings:
  A-branch: or_rep > glob_R and > or_reprnd  -> dangling references are real damage
            or_rep ~ glob_R                  -> carrier view holds (absorbed info harmless,
                                                recompute neither fixes nor breaks)
            or_rep < glob_R                  -> absorbed-info destruction dominates (rows
                                                are NOTES; erasing notes hurts)
  B-branch: or_fill > glob_R and > or_fillrnd -> attention-guided hole-filling pays; the
            harvest path is not evidence-saturated (unlike the anchored path, evext).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_orphan --shard 0 --queue_dir ./out/op_q
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend, _iter_cache_kv
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_globhot import gather_rows

ARMS = ["bh", "glob_R", "or_rep", "or_reprnd", "or_fill", "or_fillrnd", "or_topm", "or_anti"]


@torch.no_grad()
def build_attn_matrix(llm, ids, chunk=96, sig="mean"):
    """Attention matrix over one contiguous prefill (fp16, GPU).

    sig="mean": uniform layer/head mean (v1 — conservative, matches frontier's f_attn).
    sig="late_vnorm": literature-guided v2 — later-half layers only (retrieval heads live
    almost exclusively in the latter half: 2601.11020, 2606.21249) and each attention edge
    weighted by the target's value norm ||V_j|| (weight != information flow; sinks carry
    near-zero value norm: Kobayashi 2004.10102) — suppresses sink/local-head dilution and
    the left-boundary-exposure artifact of the uniform mean."""
    dev = llm.device
    core = getattr(llm.model, "model", llm.model)
    total = len(ids)
    nl = core.config.num_hidden_layers
    band = list(range(nl)) if sig == "mean" else list(range(nl // 2, nl))
    rep = None
    A = torch.zeros(total, total, dtype=torch.half, device=dev)
    ecache = DynamicCache()
    old_impl = core.config._attn_implementation
    core.config._attn_implementation = "eager"
    try:
        for s in range(0, total, chunk):
            seg = torch.tensor([ids[s: s + chunk]], dtype=torch.long, device=dev)
            L = seg.shape[1]
            out = core(input_ids=seg, past_key_values=ecache, use_cache=True,
                       output_attentions=True,
                       position_ids=torch.arange(s, s + L, device=dev).unsqueeze(0),
                       cache_position=torch.arange(s, s + L, device=dev),
                       attention_mask=torch.ones(1, s + L, dtype=torch.long, device=dev))
            acc = None
            for li in band:
                m = out.attentions[li][0].float()  # (Hq, L, S)
                if sig == "late_vnorm":
                    V = ecache[li][1][0]  # (Hkv, S, dim)
                    if rep is None:
                        rep = m.shape[0] // V.shape[0]
                    vn = V.norm(dim=-1).repeat_interleave(rep, 0)  # (Hq, S)
                    m = m * vn.unsqueeze(1)
                m = m.mean(0)  # (L, S)
                acc = m if acc is None else acc + m
            A[s:s + L, : s + L] = (acc / len(band)).half()
            del out, acc
    finally:
        core.config._attn_implementation = old_impl
    del ecache
    torch.cuda.empty_cache()
    return A


def append_gather(llm, cache, full_cache, rows):
    t = torch.tensor(rows, dtype=torch.long, device=llm.device)
    for li, K0, V0 in _iter_cache_kv(full_cache):
        cache.update(K0.index_select(2, t).contiguous(),
                     V0.index_select(2, t).contiguous(), li)


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
    ap.add_argument("--dep_budget", type=int, default=2048)  # fill volume budget
    ap.add_argument("--rep_events", type=int, default=2)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/op.json")
    ap.add_argument("--ans_out", default="./out/op_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sig", default="mean", choices=["mean", "late_vnorm"])
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

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

    NEED_A = any(a.startswith("or_") for a in RUN)
    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
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
        seg_lens = [len(s) for s in all_seg_ids]
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
        R_kv, st_anch = None, {}
        if "bh" in RUN:
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
            for i in sorted(need):
                s0, e0 = spans[i]
                alen = min(w_anch, s0 - H)
                pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
                st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                          pos, keep_a=H + alen)
        full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts)
        assert ftot == total
        A = build_attn_matrix(llm, header_ids + traj_flat, sig=args.sig) if NEED_A else None

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

        def opening_rows(kept):
            rows = []
            for p in range(H, H + w_anch):
                cov = any(spans[i][0] <= p < spans[i][1] for i in kept)
                if not cov:
                    rows.append(p)
            return rows

        def gr_rows(kept, extra_events=()):
            idx = list(range(H)) + opening_rows(kept)
            for i in sorted(set(kept) | set(extra_events)):
                idx += list(range(spans[i][0], spans[i][1]))
            return sorted(set(idx))

        def decode(c, pos_list, qids_):
            p = torch.tensor(pos_list, dtype=torch.long, device=llm.device)
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        def serve_gr(kept, extra_events, qids_):
            idx = gr_rows(kept, extra_events)
            return decode(gather_rows(llm, full_cache, idx), idx, qids_)

        def serve_rep(kept, rep_set, qids_):
            idx = gr_rows(kept)
            rep_spans = sorted((spans[i][0], spans[i][1], i) for i in rep_set)
            c = DynamicCache()
            pos_list = []
            cur_rows = []
            ri = 0
            for p in idx:
                while ri < len(rep_spans) and p >= rep_spans[ri][1]:
                    ri += 1
                in_rep = ri < len(rep_spans) and rep_spans[ri][0] <= p < rep_spans[ri][1]
                if in_rep:
                    continue
                cur_rows.append(p)
            # piecewise: gather rows before each rep span, prefill the span fresh, continue
            done_rows = 0
            for s0, e0, i in rep_spans:
                pre = [p for p in cur_rows[done_rows:] if p < s0]
                if pre:
                    append_gather(llm, c, full_cache, pre)
                    pos_list += pre
                    done_rows += len(pre)
                prefill_fresh(llm, c, len(pos_list), all_seg_ids[i],
                              list(range(s0, e0)))
                pos_list += list(range(s0, e0))
            rest = cur_rows[done_rows:]
            if rest:
                append_gather(llm, c, full_cache, rest)
                pos_list += rest
            return decode(c, pos_list, qids_)

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

        for qi, (qa, k5) in enumerate(zip(qas, routed)):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans, meta = {}, {}
            try:
                served = set(gr_rows(k5))
                dropped = [p for p in range(H, total) if p not in served]
                dt = torch.tensor(dropped, dtype=torch.long, device=llm.device) \
                    if dropped else None
                orph, tgt = {}, {}
                if A is not None and dt is not None:
                    for i in k5:
                        s0, e0 = spans[i]
                        orph[i] = float(A[s0:e0].index_select(1, dt).sum(1).mean())
                    sr = torch.tensor(sorted(served), dtype=torch.long, device=llm.device)
                    Asr = A.index_select(0, sr)
                    dropped_ev = [j for j in range(n_seg) if j not in k5
                                  and spans[j][0] >= H + w_anch]
                    for j in dropped_ev:
                        s0, e0 = spans[j]
                        tgt[j] = float(Asr[:, s0:e0].sum())
                    del Asr
                rep2 = sorted(sorted(orph, key=lambda i: -orph[i])[: args.rep_events])
                rng = random.Random(4242 * ei + 17 * qi)
                reprnd = sorted(rng.sample(k5, min(args.rep_events, len(k5))))
                fill, tot = [], 0
                for j in sorted(tgt, key=lambda j: -tgt[j]):
                    if tot + seg_lens[j] <= args.dep_budget:
                        fill.append(j)
                        tot += seg_lens[j]
                cand = [j for j in tgt]
                rng2 = random.Random(9999 * ei + 13 * qi)
                rng2.shuffle(cand)
                fillrnd, tot2 = [], 0
                for j in cand:
                    if tot2 + seg_lens[j] <= args.dep_budget:
                        fillrnd.append(j)
                        tot2 += seg_lens[j]
                # per-token-mean dial (length-confound-free): top vs anti on the SAME metric
                tgtm = {j: tgt[j] / max(1, seg_lens[j]) for j in tgt}

                def budget_walk(order):
                    out, t = [], 0
                    for j in order:
                        if t + seg_lens[j] <= args.dep_budget:
                            out.append(j)
                            t += seg_lens[j]
                    return out, t

                topm, tot3 = budget_walk(sorted(tgtm, key=lambda j: -tgtm[j]))
                anti, tot4 = budget_walk(sorted(tgtm, key=lambda j: tgtm[j]))
                meta = {"rep2": rep2, "rep_recency_overlap":
                        len(set(rep2) & hot_idx),
                        "fill_n": len(fill), "fill_tok": tot,
                        "fillrnd_tok": tot2, "fill_ids": fill,
                        "topm_ids": topm, "topm_tok": tot3,
                        "anti_ids": anti, "anti_tok": tot4,
                        "fillrnd_ids": fillrnd}
                if "bh" in RUN:
                    ans["bh"] = serve_bh(k5, qids)
                if "glob_R" in RUN:
                    ans["glob_R"] = serve_gr(k5, (), qids)
                if "or_rep" in RUN:
                    ans["or_rep"] = serve_rep(k5, rep2, qids)
                if "or_reprnd" in RUN:
                    ans["or_reprnd"] = serve_rep(k5, reprnd, qids)
                if "or_fill" in RUN:
                    ans["or_fill"] = serve_gr(k5, fill, qids)
                if "or_fillrnd" in RUN:
                    ans["or_fillrnd"] = serve_gr(k5, fillrnd, qids)
                if "or_topm" in RUN:
                    ans["or_topm"] = serve_gr(k5, topm, qids)
                if "or_anti" in RUN:
                    ans["or_anti"] = serve_gr(k5, anti, qids)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold}
            row.update(meta)
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, header_kv, R_kv, full_cache, A
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] ORPHAN QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
        for i in range(len(eps)):
            cdir = os.path.join(args.queue_dir, f"c{i}")
            try:
                os.mkdir(cdir)
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM on episode idx {i} -- skipped", flush=True)
                torch.cuda.empty_cache()
            open(os.path.join(cdir, "done"), "w").close()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()
    print(f"ORPHAN_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
