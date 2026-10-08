"""kv_review.py -- review-response controls (2026-07-13 external review).

Resolves the three soundness challenges with paired, single-variable arms:

(1) EVIDENCE MATCHING -- b_anch/b_hot serve anchor-residue rows (first-w tokens not covered
    by selected spans) that tx never sees. Arms:
      tx        header + selected span texts + q                      (baseline, replicates)
      tx_plus   header + position-ordered union(anchor region, selected spans) as TEXT + q
                -> text path sees EXACTLY the cached path's visible set
      b_hot     stores(+rblk anchor rows) + fresh tail                (replicates)
      b_noR     b_hot WITHOUT rblk: anchor is write-time conditioner only; visible content == tx
    If b_hot>tx survives as b_noR>tx and/or b_hot>tx_plus, the win is not extra evidence.

(2) CONDITIONER CONTENT -- is anchoring semantic (needs the true episode prefix) or structural
    boundary repair (any natural/degenerate prefix works)? Equal-length write-time conditioners,
    all served b_noR-style (no rblk, fresh tail):
      b_noR     true episode first-w tokens        (the chain's top)
      c_other   ANOTHER episode's tokens (donor = previous episode, tiled)
      c_shuf    true tokens, order shuffled (seed=episode_id)
      c_dummy   one token repeated
      c_none    no conditioner at all (= iso + fresh tail)
(3) COMPLETE-READING REFERENCE -- `full`: per-episode full prefill, the query attends to the
    ENTIRE episode (glob was gather-from-full, not this). The honest ceiling probe.
(4) TAIL DIAL AT FIXED EVIDENCE -- kept6 = picked ∪ last-6 for ALL rows; only the number of
    freshly recomputed tail events varies: b6_f0 / b6_f2 / b6_f6. (T2's dial co-varied the
    forced-recent evidence set; this one does not.)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_review --shard 0 --queue_dir ./out/rv_q
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
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_floor import prefill_fresh

ARMS = ["tx", "tx_plus", "b_anch", "b_hot", "b_noR", "c_other", "c_shuf", "c_dummy", "c_none",
        "full", "b6_f0", "b6_f2", "b6_f6"]
COND = {"b_noR": "true", "c_other": "other", "c_shuf": "shuf", "c_dummy": "dummy", "c_none": "none"}


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
    ap.add_argument("--out", default="./out/rv.json")
    ap.add_argument("--ans_out", default="./out/rv_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]
    need_conds = sorted({COND[a] for a in RUN if a in COND})
    need_b6 = any(a.startswith("b6_") for a in RUN)

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

    def seg_token_ids(ep):
        stexts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        return [list(llm.tok(t, add_special_tokens=False).input_ids) for t in stexts], stexts

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        all_seg_ids, segment_texts = seg_token_ids(ep)
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        spans, cur = [], H
        for sids in all_seg_ids:
            spans.append((cur, cur + len(sids)))
            cur += len(sids)
        total = cur
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        hot6_idx = set(range(max(0, n_seg - 6), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)

        # conditioner token pools (equal length w_anch, deterministic)
        pools = {"true": traj_flat[:w_anch]}
        if "other" in need_conds:
            donor = eps[(ei - 1) % len(eps)]
            dseg, _ = seg_token_ids(donor)
            dflat = [t for sids in dseg for t in sids]
            while len(dflat) < w_anch:
                dflat = dflat + dflat
            pools["other"] = dflat[:w_anch]
        if "shuf" in need_conds:
            sh = list(traj_flat[:w_anch])
            random.Random(1234 + int(ep.episode_id)).shuffle(sh)
            pools["shuf"] = sh
        if "dummy" in need_conds:
            tid = llm.tok(" the", add_special_tokens=False).input_ids[0]
            pools["dummy"] = [tid] * w_anch

        routed, need5 = [], set()
        for qa in qas:
            p5 = {id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx}
            k5 = sorted(hot_idx | p5)
            routed.append((k5, p5))
            need5 |= set(k5)
        need_all = set(need5)
        routed6 = []
        if need_b6:
            for (k5, p5) in routed:
                k6 = sorted(hot6_idx | p5)
                routed6.append(k6)
                need_all |= set(k6)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = None
        if any(a in RUN for a in ("b_anch", "b_hot")) or need_b6:
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)

        # stores: st[cond][i]; "true" also serves b_anch/b_hot/b6; "none" = [H; span]
        st = {c: {} for c in (set(need_conds) | {"true"})}
        for i in sorted(need_all):
            s0, e0 = spans[i]
            span_ids = all_seg_ids[i]
            alen = min(w_anch, s0 - H)
            for c in st:
                if c == "none":
                    if i in need5:
                        st[c][i] = encode_block(llm, header_ids + span_ids,
                                                list(range(H)) + list(range(s0, e0)), keep_a=H)
                    continue
                if c != "true" and i not in need5:
                    continue
                st[c][i] = encode_block(
                    llm, header_ids + pools[c][:alen] + span_ids,
                    list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0)),
                    keep_a=H + alen)

        hkb = (header_kv, list(range(H)))

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

        def fresh_ids_pos(idxs):
            ii = sorted(idxs)
            return ([t for i in ii for t in all_seg_ids[i]],
                    [p for i in ii for p in range(spans[i][0], spans[i][1])])

        def hot_style(cond, k5, fresh_set, rbl):
            """assemble [header (+rblk)] + stores(k5 minus fresh) + fresh(fresh_set), answer."""
            keep = [i for i in k5 if i not in fresh_set]
            blocks = [hkb] + ([rbl] if rbl else []) + \
                     [(st[cond][i], list(range(spans[i][0], spans[i][1]))) for i in keep]
            c, p = build(blocks)
            if fresh_set:
                fids, fpos = fresh_ids_pos(fresh_set)
                c = prefill_fresh(llm, c, p.shape[0], fids, fpos)
                p = torch.cat([p, torch.tensor(fpos, dtype=torch.long, device=llm.device)])
            return c, p

        full_cache, full_len = None, 0
        if "full" in RUN:
            full_cache, full_len, _ = prefill_chunked(llm, header, segment_texts)

        for qi, (qa, (k5, p5)) in enumerate(zip(qas, routed)):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans = {}
            hotset = set(k5) & hot_idx
            r5 = rblk_for(k5) if any(a in RUN for a in ("b_anch", "b_hot")) else None

            if "tx" in RUN:
                text = header + "".join(segment_texts[i] for i in k5) + qtext
                ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)

            if "tx_plus" in RUN:
                keep_tok = [False] * (total - H)
                for j in range(w_anch):
                    keep_tok[j] = True
                for i in k5:
                    s0, e0 = spans[i]
                    for pth in range(s0, e0):
                        keep_tok[pth - H] = True
                ids_plus = [traj_flat[j] for j in range(total - H) if keep_tok[j]]
                textp = header + llm.tok.decode(ids_plus) + qtext
                ans["tx_plus"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(textp),
                                                   args.ans_tokens)

            if "b_anch" in RUN:
                c, p = hot_style("true", k5, set(), r5)
                ans["b_anch"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c
            if "b_hot" in RUN:
                c, p = hot_style("true", k5, hotset, r5)
                ans["b_hot"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c

            for arm, cond in COND.items():
                if arm not in RUN:
                    continue
                c, p = hot_style(cond, k5, hotset, None)
                ans[arm], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c

            if "full" in RUN:
                ans["full"], _, _ = llm._greedy(full_cache, full_len, qids, args.ans_tokens)
                full_cache.crop(full_len)

            if need_b6:
                k6 = routed6[qi]
                r6 = rblk_for(k6)
                for arm, fn in (("b6_f0", 0), ("b6_f2", 2), ("b6_f6", 6)):
                    if arm not in RUN:
                        continue
                    fresh = set(sorted(hot6_idx)[len(hot6_idx) - fn:]) if fn else set()
                    c, p = hot_style("true", k6, fresh & set(k6), r6)
                    ans[arm], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                    del c

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
        del st, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] REVIEW QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"REVIEW_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
