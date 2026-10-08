"""kv_hxfer.py -- the deployment blocker test for the zero-write harvest recipe (user
direction, 2026-07-14: push the glob_R line): in a REAL run the rollout cache is built under
the AGENT's system prompt, while memory QA happens under a review prompt. All harvest results
so far used the QA header for the cache (clean replay proxy). How much does prompt mismatch
cost, and does appending the review instruction at the END (the only place deployment can
write) rescue it?

2x2 (cache header: review / agent-style) x (QA side: bare question / +appended review
instruction) + bh gate. All harvest arms serve [header rows; opening rows; routed spans]
gathered from their respective cache; no fresh tail (glob_R form).

  bh        gate (.2209)
  glob_R    review-header cache, bare question       = gate (expect 173/824)
  g_rev_i   review-header cache, +instruction        isolates the instruction-append effect
  g_agt     AGENT-header cache, bare question        pure mismatch exposure
  g_agt_i   AGENT-header cache, +instruction         <- THE deployment configuration

Write-serve-consistency prediction: the mismatch penalty should ride on the header/opening
rows (content mismatch); event rows are relatively immune (their virtual context unchanged).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_hxfer --shard 0 --queue_dir ./out/hx_q
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
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_globhot import gather_rows

ARMS = ["bh", "glob_R", "g_rev_i", "g_agt", "g_agt_i", "full"]

INSTR = ("\n\nYou are now reviewing the completed agent trajectory above. "
         "Use it to answer the question precisely.")


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
    ap.add_argument("--dep_budget", type=int, default=4096)  # runner flag-sanity
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/hx.json")
    ap.add_argument("--ans_out", default="./out/hx_ans.jsonl")
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

    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header_rev = (head + "You are reviewing a completed agent trajectory. Use it to "
                      f"answer the question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_agt = (head + "You are an autonomous agent. Execute the task step by step, "
                      "issuing actions and reading observations until the task is complete."
                      f"\n\nYour task: {ep.task}\n\nExecution log:\n")
        qas = ep.qa[: args.max_qa]

        def layout(header):
            hids = list(llm.tok(header, add_special_tokens=False).input_ids)
            Hh = len(hids)
            seg_ids, spans, cur = [], [], Hh
            for txt in segment_texts:
                sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
                seg_ids.append(sids)
                spans.append((cur, cur + len(sids)))
                cur += len(sids)
            return hids, Hh, seg_ids, spans, cur

        hids_r, Hr, seg_r, spans_r, total_r = layout(header_rev)
        hids_a, Ha, seg_a, spans_a, total_a = layout(header_agt)
        traj_flat_r = [t for s in seg_r for t in s]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        w_r = min(args.w, total_r - Hr)
        w_a = min(args.w, total_a - Ha)

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k)
                      if p in id2idx}
            k5 = sorted(hot_idx | picked)
            routed.append(k5)
            need |= set(k5)

        header_kv = encode_block(llm, hids_r, list(range(Hr)))
        R_kv, st_anch = None, {}
        if "bh" in RUN:
            R_kv = encode_block(llm, hids_r + traj_flat_r[:w_r], list(range(Hr + w_r)),
                                keep_a=Hr, keep_b=Hr + w_r)
            for i in sorted(need):
                s0, e0 = spans_r[i]
                alen = min(w_r, s0 - Hr)
                st_anch[i] = encode_block(
                    llm, hids_r + traj_flat_r[:alen] + seg_r[i],
                    list(range(Hr)) + list(range(Hr, Hr + alen)) + list(range(s0, e0)),
                    keep_a=Hr + alen)
        cache_r, tr, _ = prefill_chunked(llm, header_rev, segment_texts, chunk=4096)
        assert tr == total_r
        cache_a = None
        if any(a.startswith("g_agt") for a in RUN):
            cache_a, ta, _ = prefill_chunked(llm, header_agt, segment_texts, chunk=4096)
            assert ta == total_a

        hkb = (header_kv, list(range(Hr)))
        hot_sorted = sorted(hot_idx)
        hot_ids_r = [t for i in hot_sorted for t in seg_r[i]]
        hot_pos_r = [p for i in hot_sorted for p in range(spans_r[i][0], spans_r[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_r
            for i in kept:
                s0, e0 = spans_r[i]
                for pth in range(max(Hr, s0), min(Hr + w_r, e0)):
                    keep_mask[pth - Hr] = False
            ridx = [j for j in range(w_r) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [Hr + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def gr_rows(kept, Hh, spans, wl):
            idx = list(range(Hh))
            for p in range(Hh, Hh + wl):
                if not any(spans[i][0] <= p < spans[i][1] for i in kept):
                    idx.append(p)
            for i in sorted(kept):
                idx += list(range(spans[i][0], spans[i][1]))
            return sorted(set(idx))

        def serve_gather(cache, kept, Hh, spans, wl, qids_):
            idx = gr_rows(kept, Hh, spans, wl)
            c = gather_rows(llm, cache, idx)
            p = torch.tensor(idx, dtype=torch.long, device=llm.device)
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        def serve_bh(kept, qids_):
            keep = [i for i in kept if i not in hot_idx]
            rbl = rblk_for(kept)
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st_anch[i], list(range(spans_r[i][0], spans_r[i][1])))
                            for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids_r, hot_pos_r)
            p = torch.cat([p, torch.tensor(hot_pos_r, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            q_bare = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            q_instr = INSTR + f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            ans = {}
            try:
                if "bh" in RUN:
                    ans["bh"] = serve_bh(k5, llm._ids(q_bare))
                if "glob_R" in RUN:
                    ans["glob_R"] = serve_gather(cache_r, k5, Hr, spans_r, w_r,
                                                 llm._ids(q_bare))
                if "g_rev_i" in RUN:
                    ans["g_rev_i"] = serve_gather(cache_r, k5, Hr, spans_r, w_r,
                                                  llm._ids(q_instr))
                if "g_agt" in RUN:
                    ans["g_agt"] = serve_gather(cache_a, k5, Ha, spans_a, w_a,
                                                llm._ids(q_bare))
                if "g_agt_i" in RUN:
                    ans["g_agt_i"] = serve_gather(cache_a, k5, Ha, spans_a, w_a,
                                                  llm._ids(q_instr))
                if "full" in RUN:
                    fp = torch.arange(total_r, device=llm.device)
                    out, _, _ = llm._greedy_pos(cache_r, fp, llm._ids(q_bare),
                                                args.ans_tokens)
                    cache_r.crop(total_r)
                    ans["full"] = out
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
        del st_anch, header_kv, R_kv, cache_r, cache_a
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] HXFER QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"HXFER_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
