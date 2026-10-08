"""kv_depanchor.py -- the WRITE-SIDE dependency-conditioned store: the one cell our read-side
nulls do not cover, and the candidate "new repair beyond anchor+tail".

Motivation chain (all our own data): fx showed same-episode conditioner CONTENT is worth a
real 2-3.4pp; dx showed the conditioner's geometry must be TRUTHFUL (native positions win);
the frontier/evext/ev2 nulls killed dependency selection on the READ side (fresh-budget and
routing) but never touched the WRITE side. Hypothesis: conditioning each event's store on its
top causal-upstream events (signature-dependency graph, query-free) -- which sit at their TRUE
native positions, so truthful geometry is automatic -- beats the generic episode opening.

Arms (write-time conditioning of the SERVED stores is the only variable family; every arm
serves the same routed events, the same fresh last-2 tail, and a ~volume-matched residue
block; residue rows always come from opening-anchored encodes so residue quality is a
constant, not a confound):

  bh       stores conditioned on episode opening (w=4096) + rblk residue     = deployed b_hot
           (gate: must reproduce .2209 = 182/824, 10th replication)
  dp       stores conditioned on top graph-upstream prior events (<=4096 tok, <=6 ev,
           native pos) + upstream residue (<=4096 tok)                        <- the method
  dp_noR   dp stores, NO residue served (write-only effect; Law-II serve-side component)
  dp_rand  stores conditioned on RANDOM prior events (budget-matched, seeded) + their
           residue                                                            <- targeting control
  hy       hybrid: opening[:2048] + graph-upstream (<=2048 tok, events starting >= H+2048)
           + rblk2048 + upstream residue (<=2048 tok)                         <- stacking probe

Pre-registered: P1 dp-bh (PRIMARY; >=+2pp p<.05 => new-method route / paper split);
P2 dp-dp_rand (targeting specificity); P3 dp-dp_noR (served-consistency component);
P4 hy-bh, hy-dp (stacking); P5 gate bh==.2209 byte-level.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_depanchor --shard 0 --queue_dir ./out/da_q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

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
from kvmemory.kv_frontier import build_sigs, wov

ARMS = ["bh", "dp", "dp_noR", "dp_rand", "hy"]


def pick_budget(cands, seg_lens, budget, max_ev):
    """Greedy: walk cands in given order, add whole events while they fit."""
    out, tot = [], 0
    for j in cands:
        if len(out) >= max_ev:
            break
        if tot + seg_lens[j] <= budget:
            out.append(j)
            tot += seg_lens[j]
    return sorted(out), tot


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
    ap.add_argument("--dep_budget", type=int, default=4096)
    ap.add_argument("--dep_maxev", type=int, default=6)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/da.json")
    ap.add_argument("--ans_out", default="./out/da_ans.jsonl")
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

        # dependency graph over events (structural, query-free)
        sigs = build_sigs(segment_texts)
        wmat = {}
        for i in range(n_seg):
            for j in range(i):
                w = wov(sigs[i], sigs[j])
                if w > 0:
                    wmat[(i, j)] = w

        ups, rups, hyups = {}, {}, {}
        for i in sorted(need):
            cands = sorted((j for j in range(i) if wmat.get((i, j), 0) > 0),
                           key=lambda j: -wmat[(i, j)])
            if not cands:
                cands = list(range(i - 1, -1, -1))  # recency fallback (deterministic)
            ups[i], _ = pick_budget(cands, seg_lens, args.dep_budget, args.dep_maxev)
            rng = random.Random(9173 * ei + 31 * i)
            rc = list(range(i))
            rng.shuffle(rc)
            rups[i], _ = pick_budget(rc, seg_lens, args.dep_budget, args.dep_maxev)
            hyc = [j for j in cands if spans[j][0] >= H + 2048]
            hyups[i], _ = pick_budget(hyc, seg_lens, args.dep_budget // 2,
                                      args.dep_maxev // 2)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)

        # opening-anchored encodes: bh's stores AND the residue-row source for all arms
        anch_need = set(need)
        for i in need:
            anch_need |= set(ups[i]) | set(rups[i]) | set(hyups[i])
        st_anch = {}
        for i in sorted(anch_need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                      pos, keep_a=H + alen)

        def cond_encode(i, cond_events, opening_len):
            """[header; opening[:opening_len]; cond events at native pos; event] keep event."""
            s0, e0 = spans[i]
            ol = min(opening_len, s0 - H)
            ids = header_ids + traj_flat[:ol]
            pos = list(range(H)) + list(range(H, H + ol))
            for j in cond_events:
                if spans[j][0] < H + ol or j >= i:
                    continue
                ids += all_seg_ids[j]
                pos += list(range(spans[j][0], spans[j][1]))
            keep_a = len(ids)
            ids += all_seg_ids[i]
            pos += list(range(s0, e0))
            return encode_block(llm, ids, pos, keep_a=keep_a)

        st_dp, st_rand, st_hy = {}, {}, {}
        for i in sorted(need):
            if any(a in RUN for a in ("dp", "dp_noR")):
                st_dp[i] = cond_encode(i, ups[i], 0)
            if "dp_rand" in RUN:
                st_rand[i] = cond_encode(i, rups[i], 0)
            if "hy" in RUN:
                st_hy[i] = cond_encode(i, hyups[i], 2048)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept, wlen):
            wl = min(wlen, w_anch)
            keep_mask = [True] * wl
            for i in kept:
                s0, e0 = spans[i]
                for pth in range(max(H, s0), min(H + wl, e0)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(wl) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def residue_for(kept, upsmap, budget, minpos):
            """Serve upstream events of the served cold stores as evidence rows (from
            opening-anchored encodes), weight-ranked, whole events, <= budget tokens."""
            cold = [i for i in kept if i not in hot_idx]
            cand = {}
            for i in cold:
                for j in upsmap[i]:
                    if j in kept or j in hot_idx or spans[j][0] < minpos:
                        continue
                    cand[j] = max(cand.get(j, 0.0), wmat.get((i, j), 0.0))
            order = sorted(cand, key=lambda j: -cand[j])
            picked, tot = [], 0
            for j in order:
                if tot + seg_lens[j] <= budget:
                    picked.append(j)
                    tot += seg_lens[j]
            return [(st_anch[j], list(range(spans[j][0], spans[j][1])))
                    for j in sorted(picked)], tot

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def serve(st, kept, rbl, extra, qids_):
            keep = [i for i in kept if i not in hot_idx]
            c, p = build([hkb] + ([rbl] if rbl else []) + list(extra)
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
            ans, vols = {}, {}
            try:
                if "bh" in RUN:
                    ans["bh"] = serve(st_anch, k5, rblk_for(k5, args.w), [], qids)
                if "dp" in RUN:
                    ex, tot = residue_for(k5, ups, args.dep_budget, H)
                    vols["dp_res"] = tot
                    ans["dp"] = serve(st_dp, k5, None, ex, qids)
                if "dp_noR" in RUN:
                    ans["dp_noR"] = serve(st_dp, k5, None, [], qids)
                if "dp_rand" in RUN:
                    ex, tot = residue_for(k5, rups, args.dep_budget, H)
                    vols["rand_res"] = tot
                    ans["dp_rand"] = serve(st_rand, k5, None, ex, qids)
                if "hy" in RUN:
                    ex, tot = residue_for(k5, hyups, args.dep_budget // 2, H + 2048)
                    vols["hy_res"] = tot
                    ans["hy"] = serve(st_hy, k5, rblk_for(k5, 2048), ex, qids)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            cold = [i for i in k5 if i not in hot_idx]
            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold,
                   "ups_n": [len(ups[i]) for i in cold],
                   "ups_tok": [sum(seg_lens[j] for j in ups[i]) for i in cold],
                   "vols": vols}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, st_dp, st_rand, st_hy, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] DEPANCHOR QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"DEPANCHOR_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
