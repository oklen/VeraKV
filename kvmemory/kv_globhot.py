"""kv_globhot.py -- the free-rider cell: serve the agent's OWN full-context rollout KV
(gathered rows) wearing b_hot's full dressing (opening rows + fresh last-2 tail).

Deployment question (user, 2026-07-14): in a real agent run the trajectory KV already exists
(sunk cost). b_hot instead re-encodes every event offline at ~5x one episode prefill. T7's
bare glob (.175 << b_hot .221) is NOT a fair test of harvesting -- it had neither the opening
rows nor the fresh tail, both of which are FREE for the harvest path (gather more rows from
the same cache; recompute two events). This run completes the 2x2:

  bh         anchored stores + rblk + fresh tail          deployed ref (gate .2209, 12th rep)
  glob       gather header + routed spans (frozen, incl. last-2)   bare harvest = T7 replicate
  glob_R     glob + opening-region rows [H, H+4096) gathered exact  (+evidence, still frozen)
  glob_hot   gather header + cold spans, fresh-recompute last-2     (+integration, no opening)
  glob_hotR  gather header + opening rows + cold spans + fresh tail <- MONEY: zero-write b_hot

Decision: glob_hotR >= bh  -> deployed recipe flips to zero-write harvesting (rollout cache
+ tail); write-cost story collapses in harvest's favor (rollout KV is sunk). glob_hotR < bh
-> the 5x write is doing real work beyond row quality (cross-span interference story gets a
controlled confirmation); anchored re-encoding vindicated against its strongest alternative.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_globhot --shard 0 --queue_dir ./out/gh_q
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.llm_hf import _iter_cache_kv
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked

ARMS = ["bh", "glob", "glob_R", "glob_hot", "glob_hotR"]


def gather_rows(llm, cache, indices):
    """New DynamicCache holding the given absolute row indices of a contiguous cache."""
    t = torch.tensor(indices, dtype=torch.long, device=llm.device)
    new = DynamicCache()
    for i, K0, V0 in _iter_cache_kv(cache):
        new.update(K0.index_select(2, t).contiguous(), V0.index_select(2, t).contiguous(), i)
    return new


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
    ap.add_argument("--dep_budget", type=int, default=4096)  # unused; runner flag-sanity
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/gh.json")
    ap.add_argument("--ans_out", default="./out/gh_ans.jsonl")
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

    NEED_BH = "bh" in RUN
    NEED_GLOB = any(a.startswith("glob") for a in RUN)

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
        if NEED_BH:
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
            for i in sorted(need):
                s0, e0 = spans[i]
                alen = min(w_anch, s0 - H)
                pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
                st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                          pos, keep_a=H + alen)
        full_cache = None
        if NEED_GLOB:
            full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts)
            assert ftot == total, f"token mismatch {ftot} vs {total}"

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

        def opening_rows(kept, exclude_hot):
            rows = []
            for p in range(H, H + w_anch):
                cov = False
                for i in kept:
                    if spans[i][0] <= p < spans[i][1]:
                        cov = True
                        break
                if not cov and exclude_hot:
                    for i in hot_idx:
                        if spans[i][0] <= p < spans[i][1]:
                            cov = True
                            break
                if not cov:
                    rows.append(p)
            return rows

        def serve_gather(kept, with_opening, fresh_hot, qids_):
            idx = list(range(H))
            if with_opening:
                idx += opening_rows(kept, exclude_hot=fresh_hot)
            for i in sorted(kept):
                if fresh_hot and i in hot_idx:
                    continue
                idx += list(range(spans[i][0], spans[i][1]))
            idx = sorted(set(idx))
            c = gather_rows(llm, full_cache, idx)
            p = torch.tensor(idx, dtype=torch.long, device=llm.device)
            if fresh_hot:
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                p = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

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

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans = {}
            try:
                if "bh" in RUN:
                    ans["bh"] = serve_bh(k5, qids)
                if "glob" in RUN:
                    ans["glob"] = serve_gather(k5, False, False, qids)
                if "glob_R" in RUN:
                    ans["glob_R"] = serve_gather(k5, True, False, qids)
                if "glob_hot" in RUN:
                    ans["glob_hot"] = serve_gather(k5, False, True, qids)
                if "glob_hotR" in RUN:
                    ans["glob_hotR"] = serve_gather(k5, True, True, qids)
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
        del st_anch, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] GLOBHOT QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"GLOBHOT_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
