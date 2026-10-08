"""kv_fixanchor.py -- C3's engineering dividend: can the write-time conditioner be a FIXED,
episode-independent prefix (encode once globally, prefix-cache across all events/episodes)?

Deployment arithmetic: per-event store encode currently pays (alen + |event|) tokens; with a
fixed shared conditioner the [header; FIX] prefix is encoded ONCE and cached, so each event
pays only |event| — write cost collapses from ~5x one episode prefill to ~1x (the per-episode
anchor-row encode for the served rblk stays, one 4k prefill). Accuracy question, arms:

  bh_true  [header; rblk(true anchor rows); stores(true-cond); fresh last2]   deployed ref
  bf_hot   [header; rblk(true anchor rows); stores(FIXED-cond); fresh last2]  <- money arm
  bn_true  same minus rblk (ties to E1 b_noR)
  bn_fix   same minus rblk, FIXED-cond (maps onto E1 c_other)

Primary: bf_hot vs bh_true (same visible evidence, only write-time conditioning differs).
FIX text = first-4k tokens of the first loaded episode (episode 0 uses episode 1's), so it is
real trajectory-register text, deterministic, and identical for (almost) all episodes.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_fixanchor --shard 0 --queue_dir ./out/fx_q
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh

ARMS = ["bh_true", "bf_hot", "bn_true", "bn_fix"]


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
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/fx.json")
    ap.add_argument("--ans_out", default="./out/fx_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
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

    def flat_tokens(ep):
        out = []
        for s in ep.segments:
            out += list(llm.tok(f"<step {s.turn}>\n{s.text}\n",
                                add_special_tokens=False).input_ids)
        return out

    FIX_MAIN = flat_tokens(eps[0])[: args.w]
    FIX_ALT = flat_tokens(eps[1 % len(eps)])[: args.w]
    while len(FIX_MAIN) < args.w:
        FIX_MAIN = FIX_MAIN + FIX_MAIN
        FIX_MAIN = FIX_MAIN[: args.w]
    while len(FIX_ALT) < args.w:
        FIX_ALT = FIX_ALT + FIX_ALT
        FIX_ALT = FIX_ALT[: args.w]

    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        FIX = FIX_ALT if ei == 0 else FIX_MAIN
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
        R_kv = None
        if any(a in RUN for a in ("bh_true", "bf_hot")):
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
        st_true, st_fix = {}, {}
        for i in sorted(need):
            s0, e0 = spans[i]
            span_ids = all_seg_ids[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            if any(a in RUN for a in ("bh_true", "bn_true")):
                st_true[i] = encode_block(llm, header_ids + traj_flat[:alen] + span_ids,
                                          pos, keep_a=H + alen)
            if any(a in RUN for a in ("bf_hot", "bn_fix")):
                st_fix[i] = encode_block(llm, header_ids + FIX[:alen] + span_ids,
                                         pos, keep_a=H + alen)
        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            if R_kv is None:
                return None
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

        def serve(st, kept, rbl, qids_):
            keep = [i for i in kept if i not in hot_idx]
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            p = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            rbl = rblk_for(k5)
            ans = {}
            try:
                if "bh_true" in RUN:
                    ans["bh_true"] = serve(st_true, k5, rbl, qids)
                if "bf_hot" in RUN:
                    ans["bf_hot"] = serve(st_fix, k5, rbl, qids)
                if "bn_true" in RUN:
                    ans["bn_true"] = serve(st_true, k5, None, qids)
                if "bn_fix" in RUN:
                    ans["bn_fix"] = serve(st_fix, k5, None, qids)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_true, st_fix, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] FIXANCHOR QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"FIXANCHOR_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
