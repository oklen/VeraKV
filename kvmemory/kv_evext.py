"""kv_evext.py -- evidence-extension control (frontier v2): is the dependency graph's value
on the ROUTING side? 2x3 factorial, all paired per QA:

  serving path x evidence set
    tx_base   / tx_graph   / tx_rand      (text re-prefill)
    bh_base   / bh_graph   / bh_rand      (anchored stores + rblk + fresh last-2)

  kept_base  = picked(K=5) ∪ last2                  (the b_hot operating point)
  kept_graph = kept_base ∪ top-3 signature-graph bridges   (fr-run extension)
  kept_rand  = kept_base ∪ count-matched random non-kept events (seeded per QA)

Pre-registered readings: (1) tx_graph vs tx_rand isolates the graph's routing value for the
text path; (2) bh_* tests whether ANY extension helps the cached path (C6 predicts no/harm
at 8B); (3) the interaction (same marginal events, text vs cache) is the cleanest C6 instance.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_evext --shard 0 --queue_dir ./out/ev_q
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
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_frontier import build_sigs, wov, qsig_of

ARMS = ["tx_base", "tx_graph", "tx_rand", "bh_base", "bh_graph", "bh_rand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--nbridge", type=int, default=3)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/ev.json")
    ap.add_argument("--ans_out", default="./out/ev_ans.jsonl")
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
        last2 = set(range(max(0, n_seg - 2), n_seg))
        lastidx = n_seg - 1
        old = [s for i, s in enumerate(ep.segments) if i not in last2]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)
        sigs = build_sigs(segment_texts)

        plans = []
        need = set()
        for qi, qa in enumerate(qas):
            q = qa["question"]
            picked = {id2idx[p] for p in router.select(q, old, args.k) if p in id2idx}
            qs = qsig_of(q)
            base = sorted(set(picked) | last2)
            cand = [v for v in range(n_seg) if v not in set(base)]
            conn = {v: (sum(wov(sigs[v], sigs[e]) for e in picked)
                        + wov(sigs[v], sigs[lastidx]) + wov(sigs[v], qs)) for v in cand}
            bridges = [v for v, s in sorted(conn.items(), key=lambda x: -x[1])[: args.nbridge]
                       if s > 0]
            rng = random.Random(31337 * int(ep.episode_id) + qi)
            pool = [v for v in cand if v not in set(bridges)]
            rand_ext = rng.sample(pool, min(len(bridges), len(pool))) if bridges else []
            kept_graph = sorted(set(base) | set(bridges))
            kept_rand = sorted(set(base) | set(rand_ext))
            plans.append((q, qa, base, kept_graph, kept_rand, bridges, rand_ext))
            need |= set(kept_graph) | set(kept_rand)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            st[i] = encode_block(
                llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0)),
                keep_a=H + alen)
        hkb = (header_kv, list(range(H)))

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

        def bh(kept, qids_):
            fresh = last2 & set(kept)
            keep = [i for i in kept if i not in fresh]
            c, p = build([hkb, rblk_for(kept)]
                         + [(st[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            if fresh:
                ii = sorted(fresh)
                fids = [t for i in ii for t in all_seg_ids[i]]
                fpos = [pp for i in ii for pp in range(spans[i][0], spans[i][1])]
                c = prefill_fresh(llm, c, p.shape[0], fids, fpos)
                p = torch.cat([p, torch.tensor(fpos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        def txt(kept, qtext_):
            text = header + "".join(segment_texts[i] for i in kept) + qtext_
            out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            return out

        for (q, qa, base, kept_graph, kept_rand, bridges, rand_ext) in plans:
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans = {}
            try:
                if "tx_base" in RUN:
                    ans["tx_base"] = txt(base, qtext)
                if "tx_graph" in RUN:
                    ans["tx_graph"] = txt(kept_graph, qtext)
                if "tx_rand" in RUN:
                    ans["tx_rand"] = txt(kept_rand, qtext)
                if "bh_base" in RUN:
                    ans["bh_base"] = bh(base, qids)
                if "bh_graph" in RUN:
                    ans["bh_graph"] = bh(kept_graph, qids)
                if "bh_rand" in RUN:
                    ans["bh_rand"] = bh(kept_rand, qids)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold,
                   "bridges": bridges, "rand_ext": rand_ext,
                   "btok": sum(spans[i][1] - spans[i][0] for i in bridges),
                   "rtok": sum(spans[i][1] - spans[i][0] for i in rand_ext)}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] EVEXT QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"EVEXT_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
