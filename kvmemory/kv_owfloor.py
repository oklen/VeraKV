"""kv_owfloor.py -- b_hot on the OVER-WINDOW home turf (30k < total <= 100k, YaRN 4x for all arms).

Within-window verdict: b_hot (anchored per-event stores + fresh hot-tail integration, hot=2)
beats text re-prefill at 8B (+5.6pp p=3e-5) and 32B (hot=4: +2.9pp p=.053). Over-window is where
the cost story is sharpest: tx must re-prefill the routed view per query, `full` must chunk-prefill
the WHOLE 30-100k trajectory (YaRN brute force), while b_hot serves cached span KV and freshly
computes only the last `hot` segments + question.

Arms:
  tx      header + routed spans, compact re-prefill              (per-query text path)
  full    chunked YaRN prefill of the whole trajectory, amortized across the episode's QAs
  b_anch  anchored spans + served anchor, question only          (fully cached)
  b_hot   b_anch stores; hot tail (last `hot` segs) recomputed fresh over the view

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa SPRAG_ROPE_FACTOR=4.0 SPRAG_MAX_CTX=131072 \
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_owfloor \
        --shard 0 --queue_dir ./out/owf_q --out ./out/owfo/owf_s0.json \
        --ans_out ./out/owfo/owf_ans_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked

ARMS = ["tx", "full", "b_anch", "b_hot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--min_tokens", type=int, default=30000)
    ap.add_argument("--ow_max", type=int, default=100000)
    ap.add_argument("--max_ep", type=int, default=66)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/owfo/owf.json")
    ap.add_argument("--ans_out", default="./out/owfo/owf_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--pos", default="native", choices=["native", "packed"],
                    help="span position scheme: original trajectory positions vs packed-union (bridge mode)")
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    assert os.environ.get("SPRAG_ROPE_FACTOR"), "over-window run requires SPRAG_ROPE_FACTOR (YaRN)"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    alleps = [e for e in load_episodes(args.data)
              if args.min_tokens < e.total_tokens <= args.ow_max]
    bydom = defaultdict(list)
    for e in alleps:
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())

    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
    fresh_tok = {a: 0 for a in RUN}
    t_arm = {a: 0.0 for a in RUN}
    sel_tok_total = traj_tok_total = 0
    byhop, bydom_t = {}, {}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n, sel_tok_total, traj_tok_total
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
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx}
            kept = sorted(hot_idx | picked)
            routed.append(kept)
            need |= set(kept)

        if args.pos == "packed":
            # packed-union layout (the >125k bridge fallback, validated here on the 30-100k band
            # where native positions are also available): union spans laid out after the anchor
            cur2 = H + w_anch
            packed = {}
            for i in sorted(need):
                packed[i] = (cur2, cur2 + len(all_seg_ids[i]))
                cur2 += len(all_seg_ids[i])
            spans = [packed.get(i, spans[i]) for i in range(n_seg)]
        # per-event stores (write-time, cacheable): span conditioned on [header; first-w anchor]
        torch.cuda.synchronize(); t0 = time.time()
        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st_anch = {}
        for i in sorted(need):
            st, en = spans[i]
            alen = min(w_anch, st - H)
            st_anch[i] = encode_block(
                llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                list(range(H)) + list(range(H, H + alen)) + list(range(st, en)),
                keep_a=H + alen)
        torch.cuda.synchronize()
        t_write = time.time() - t0

        # brute-force competitor: whole-trajectory chunked YaRN prefill, once per episode
        full_cache, total_len, t_pf = None, total, 0.0
        if "full" in RUN:
            full_cache, total_len, t_pf = prefill_chunked(llm, header, segment_texts)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} {total_len} tok | full prefill {t_pf:.1f}s "
              f"write-stores {t_write:.1f}s ({n_seg} seg, {len(qas)} qa)", flush=True)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                st, en = spans[i]
                for pth in range(max(H, st), min(H + w_anch, en)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
            return assemble(llm, bs)

        for qa, kept in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            sel_tok = sum(spans[i][1] - spans[i][0] for i in kept)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            qlen = qids.shape[1]
            ans = {}

            if "tx" in RUN:
                text = header + "".join(segment_texts[i] for i in kept) + qtext
                ans["tx"], ti, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
                fresh_tok["tx"] += H + sel_tok + qlen
                t_arm["tx"] += ti

            if "full" in RUN:
                ans["full"], ti, _ = llm._greedy(full_cache, total_len, qids, args.ans_tokens)
                full_cache.crop(total_len)
                fresh_tok["full"] += qlen + total_len // max(1, len(qas))
                t_arm["full"] += ti + t_pf / max(1, len(qas))

            r5 = rblk_for(kept)
            ab = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept]
            torch.cuda.synchronize(); t0 = time.time()
            c, p = build([hkb] + ([r5] if r5 else []) + ab)
            torch.cuda.synchronize(); tb = time.time() - t0
            a, ti, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
            ans["b_anch"] = a
            t_arm["b_anch"] += tb + ti
            fresh_tok["b_anch"] += qlen
            del c

            kept_old = [i for i in kept if i not in hot_idx]
            abo = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept_old]
            torch.cuda.synchronize(); t0 = time.time()
            c, p = build([hkb] + ([r5] if r5 else []) + abo)
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            torch.cuda.synchronize(); tb = time.time() - t0
            a, ti, _ = llm._greedy_pos(c, p2, qids, args.ans_tokens)
            ans["b_hot"] = a
            t_arm["b_hot"] += tb + ti
            fresh_tok["b_hot"] += len(hot_ids) + qlen
            del c

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold,
                   "sel_tok": sel_tok, "total_tok": total_len, "hot_tok": len(hot_ids)}
            for a_ in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a_]))
                acc[a_] += ok
                row[a_] = ok
                row["ans_" + a_] = ans[a_]
            n += 1
            sel_tok_total += sel_tok
            traj_tok_total += total_len
            for tbl, key in ((byhop, row["hop"]), (bydom_t, ep.domain)):
                d = tbl.setdefault(key, {a_: 0 for a_ in RUN})
                d.setdefault("_n", 0)
                d["_n"] += 1
                for a_ in RUN:
                    d[a_] += row[a_]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del full_cache, st_anch, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc, "fresh_tok": fresh_tok,
                   "t_arm": t_arm, "sel_tok_total": sel_tok_total,
                   "traj_tok_total": traj_tok_total, "byhop": byhop, "bydom": bydom_t},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} done | n={n} | " +
              " ".join(f"{a_} {acc[a_]}" for a_ in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] OWF QUEUE over {len(eps)} episodes ({args.min_tokens}<tok<={args.ow_max}, "
              f"hot={args.hot}, domains {sorted(bydom)})", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(eps[i])
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM on episode idx {i} -- skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for ep in eps[args.shard::args.nshards]:
            run_episode(ep)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 76)
    print(f"OWF w{wid}: {n} QA | mean traj {traj_tok_total//nn} tok, view {sel_tok_total//nn} tok")
    for a in RUN:
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%  fresh-tok/QA {fresh_tok[a]//nn:6d}  "
              f"t/QA {t_arm[a]/nn:5.2f}s")
    print(f"OWF_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
