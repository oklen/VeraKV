"""kv_ow.py -- Phase C probe: the OVER-WINDOW home turf of base-KV + recency-tail replay.

Episodes with 30k < total_tokens <= 100k (SOFTWARE-dominant) do not fit the native 32k window, so the
within-window champion (global gather from a full prefill) does not exist as a deployed option. The
run compares, under ONE model (Qwen3-8B + YaRN 4x so 131k positions are valid for every arm):

  tx      : header + routed spans, compact re-prefill      (deployed text path; per-query prefill)
  iso     : per-event base KV at ORIGINAL trajectory positions, gathered; question only
            (0% query-time replay -- the fully-cached floor)
  sp_hot  : iso + fully replay the hot tail over the view   (the design validated within-window)
  rep100  : full view replay                                (upper anchor of the replay dial)
  full    : brute-force window extension -- chunked YaRN prefill of the WHOLE trajectory, reused
            across the episode's QAs (the "just extend the context" competitor; pays 30-100k prefill)

Efficiency is part of the result: per-QA fresh-prefill tokens and wall time are recorded per arm.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa SPRAG_ROPE_FACTOR=4.0 SPRAG_MAX_CTX=131072 \
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_ow \
        --shard 0 --queue_dir ./out/ow_q --out ./out/ow_s0.json --ans_out ./out/ow_ans_s0.jsonl
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
from kvmemory.kv_equiv import judge, norm
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_replay import iso_prefill, replayed_cache, build_alloc

ARMS = ["tx", "iso", "sp_hot", "rep100", "full"]
REPLAY_ARMS = ["sp_hot", "rep100"]


@torch.no_grad()
def prefill_chunked(llm, header_text, segment_texts, chunk=8192):
    """Prefill header + ALL segments into one cache in chunks (a single 100k forward would
    materialize a ~60GB fp32 logits tensor). Contiguous positions, plain causal. Returns
    (cache, total_len, t_prefill)."""
    ids = list(llm.tok(header_text, add_special_tokens=False).input_ids)
    for t in segment_texts:
        ids += llm.tok(t, add_special_tokens=False).input_ids
    total = len(ids)
    cache = DynamicCache()
    dev = llm.device
    torch.cuda.synchronize(); t0 = time.time()
    for s in range(0, total, chunk):
        seg = torch.tensor([ids[s: s + chunk]], dtype=torch.long, device=dev)
        L = seg.shape[1]
        pos = torch.arange(s, s + L, device=dev)
        attn = torch.ones(1, s + L, dtype=torch.long, device=dev)
        llm.model(input_ids=seg, past_key_values=cache, use_cache=True,
                  cache_position=pos, attention_mask=attn)
    torch.cuda.synchronize()
    return cache, total, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--min_tokens", type=int, default=30000)
    ap.add_argument("--ow_max", type=int, default=100000, help="YaRN-4x validity ceiling")
    ap.add_argument("--max_ep", type=int, default=66)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="./out/ow.json")
    ap.add_argument("--ans_out", default="./out/ow_ans.jsonl")
    args = ap.parse_args()

    assert os.environ.get("SPRAG_ROPE_FACTOR"), "over-window run requires SPRAG_ROPE_FACTOR (YaRN)"
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
    acc = {a: 0 for a in ARMS}
    n = 0
    sel_tok_total = traj_tok_total = 0
    fresh_tok = {a: 0 for a in ARMS}   # fresh prefill/replay tokens per arm (query-time compute)
    t_arm = {a: 0.0 for a in ARMS}
    byhop, bydom_t = {}, {}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n, sel_tok_total, traj_tok_total
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_len = len(llm.tok(header, add_special_tokens=False).input_ids)
        spans, cur = [], header_len
        for txt in segment_texts:
            tl = len(llm.tok(txt, add_special_tokens=False).input_ids)
            spans.append((cur, cur + tl))
            cur += tl
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]

        # brute-force arm: chunked YaRN prefill of the WHOLE trajectory, once per episode
        full_cache, total_len, t_pf = prefill_chunked(llm, header, segment_texts)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} {total_len} tok prefill {t_pf:.1f}s "
              f"({n_seg} seg, {len(qas)} qa)", flush=True)

        for qa in qas:
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            hb = hop_bucket(q)
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            sel_tok = sum(spans[i][1] - spans[i][0] for i in kept)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            qlen = qids.shape[1]
            ans, arep = {}, {}

            # tx: compact re-prefill of the routed view
            text = header + "".join(segment_texts[i] for i in kept) + qtext
            ans["tx"], ti, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            fresh_tok["tx"] += header_len + sel_tok + qlen
            t_arm["tx"] += ti
            # full: reuse the episode-level whole-trajectory cache
            ans["full"], ti, _ = llm._greedy(full_cache, total_len, qids, args.ans_tokens)
            full_cache.crop(total_len)
            fresh_tok["full"] += qlen + total_len // max(1, len(qas))  # amortized share
            t_arm["full"] += ti + t_pf / max(1, len(qas))
            # base-KV arms
            iso_c, ipos, iids, blocks = iso_prefill(llm, header, segment_texts, kept, spans, header_len)
            for arm in REPLAY_ARMS:
                alloc = build_alloc(arm, sorted(kept), spans, segment_texts, hot_idx, q)
                rc, rpos, lr, tr = replayed_cache(llm, iso_c, ipos, iids, blocks, header_len, alloc)
                a, ti, _ = llm._greedy_pos(rc, rpos, qids, args.ans_tokens)
                ans[arm] = a
                arep[arm] = lr
                fresh_tok[arm] += lr + qlen
                t_arm[arm] += tr + ti
            ans["iso"], ti, _ = llm._greedy_pos(iso_c, ipos, qids, args.ans_tokens)
            fresh_tok["iso"] += qlen
            t_arm["iso"] += ti

            row = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype, "hop": hb,
                   "q": q, "gold": gold, "sel_tok": sel_tok, "total_tok": total_len}
            for arm, lr in arep.items():
                row["nrep_" + arm] = lr
            for a in ARMS:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            sel_tok_total += sel_tok
            traj_tok_total += total_len
            for tbl, key in ((byhop, hb), (bydom_t, ep.domain)):
                d = tbl.setdefault(key, {a: 0 for a in ARMS})
                d.setdefault("_n", 0)
                d["_n"] += 1
                for a in ARMS:
                    d[a] += row[a]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": ARMS, "n": n, "acc": acc, "fresh_tok": fresh_tok,
                   "t_arm": t_arm, "sel_tok_total": sel_tok_total,
                   "traj_tok_total": traj_tok_total, "byhop": byhop, "bydom": bydom_t},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} done | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] OW QUEUE over {len(eps)} episodes ({args.min_tokens}<tok<={args.ow_max}, "
              f"{len(bydom)} domains: {sorted(bydom)})", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            run_episode(eps[i])
    else:
        for ep in eps[args.shard::args.nshards]:
            run_episode(ep)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 76)
    print(f"OW w{wid}: {n} QA | mean traj {traj_tok_total//nn} tok, view {sel_tok_total//nn} tok")
    for a in ARMS:
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%  fresh-tok/QA {fresh_tok[a]//nn:6d}  "
              f"t/QA {t_arm[a]/nn:5.2f}s")
    print(f"OW_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
