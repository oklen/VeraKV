"""kv_epic.py -- the same-harness repair-family baseline the review asked for: EPIC/LegoLink-
style STATIC BOUNDARY RECOMPUTE (recompute the first k tokens of every served chunk at query
time, attending to everything positionally before it), run against our write-time anchoring.

Fidelity notes vs LegoLink: (a) their chunks are encoded from position 0; our iso stores are
already stronger -- header-conditioned and at NATIVE positions -- so the comparison is
conservative-fair (their repair gets our best isolated store as its base). (b) k is static
(16 / 32, their range), query-independent, one tiny forward per chunk, sequential in position
order so a chunk's boundary tokens never see rows that lie positionally after them (rblk rows
are drip-fed by position for the evidence-matched arm).

Arms (all paired per QA, same routed evidence K=5 + last-2, mechanism subset n~824):

  iso              [header; iso stores]                                naive floor (T5 repl.)
  epic16           iso + boundary-16 recompute per served event        EPIC/LegoLink as-is
  epic32           iso + boundary-32                                   dose control
  epic16_hot       epic16 + fresh last-2 tail                          repair + integration
  epic16_hot_rblk  epic16_hot + serve anchor-residue rows              MONEY: everything b_hot
                                                                       has except anchor WRITES
  b_anch           anchored stores + rblk, no tail                     gate (~.182, T1)
  b_hot            deployed reference                                  gate (.2209, 7th repl.)

Pre-registered contrasts:
  Q1 epic16 - iso            does token-level boundary repair work in agent-memory serving?
  Q2 epic32 - epic16         dose response.
  Q3 epic16_hot - epic16     what the causal tail adds ON TOP of token repair.
  Q4 b_hot - epic16_hot_rblk value of write-time anchor conditioning beyond query-time token
                             repair, at matched evidence and matched tail. Prediction from fx:
                             +2~4pp (the same-episode content component token repair can't buy).
     If instead ~0: iso-store + boundary recompute matches b_hot -> write cost collapses 5x->1x
     and the paper's recipe changes.
  Q5 gates: b_hot must reproduce .2209 byte-identically; b_anch ~ .182.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_epic --shard 0 --queue_dir ./out/ep_q
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
from kvmemory.kv_matrix import encode_block, assemble, slice_kv
from kvmemory.kv_floor import prefill_fresh

ARMS = ["iso", "epic16", "epic32", "epic16_hot", "epic16_hot_rblk", "b_anch", "b_hot"]


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
    ap.add_argument("--kb_small", type=int, default=16)
    ap.add_argument("--kb_big", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/ep.json")
    ap.add_argument("--ans_out", default="./out/ep_ans.jsonl")
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

    NEED_ANC = any(a in RUN for a in ("b_anch", "b_hot"))
    NEED_ISO = any(a.startswith(("iso", "epic")) for a in RUN)
    NEED_R = any(a in RUN for a in ("b_anch", "b_hot", "epic16_hot_rblk"))

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
        R_kv = None
        if NEED_R:
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
        st_iso, st_anc = {}, {}
        for i in sorted(need):
            s0, e0 = spans[i]
            span_ids = all_seg_ids[i]
            if NEED_ISO:
                st_iso[i] = encode_block(llm, header_ids + span_ids,
                                         list(range(H)) + list(range(s0, e0)), keep_a=H)
            if NEED_ANC:
                alen = min(w_anch, s0 - H)
                pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
                st_anc[i] = encode_block(llm, header_ids + traj_flat[:alen] + span_ids,
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

        def append_block(cache, kv):
            for li, (K, V) in enumerate(kv):
                cache.update(K.to(llm.device, non_blocking=True),
                             V.to(llm.device, non_blocking=True), li)

        def serve_plain(st, kept, rbl, qids_, tail_on):
            keep = [i for i in kept if not (tail_on and i in hot_idx)]
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            if tail_on:
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                p = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        def serve_epic(st, kept, rbl, qids_, kb, tail_on):
            cold = [i for i in sorted(kept) if not (tail_on and i in hot_idx)]
            c, p0 = build([hkb])
            pos_all = list(range(H))
            plen = H
            rkv, rpos, rptr = None, [], 0
            if rbl is not None:
                rkv, rpos = rbl

            def feed_rbl(upto):
                nonlocal plen, rptr
                if rkv is None or rptr >= len(rpos):
                    return
                cut = rptr
                while cut < len(rpos) and rpos[cut] < upto:
                    cut += 1
                if cut == rptr:
                    return
                append_block(c, [(K[:, :, rptr:cut], V[:, :, rptr:cut]) for K, V in rkv])
                pos_all.extend(rpos[rptr:cut])
                plen += cut - rptr
                rptr = cut

            for i in cold:
                s0, e0 = spans[i]
                feed_rbl(s0)
                kb_i = min(kb, e0 - s0)
                prefill_fresh(llm, c, plen, all_seg_ids[i][:kb_i],
                              list(range(s0, s0 + kb_i)))
                pos_all.extend(range(s0, s0 + kb_i))
                plen += kb_i
                if e0 - s0 > kb_i:
                    append_block(c, slice_kv(st[i], kb_i, e0 - s0))
                    pos_all.extend(range(s0 + kb_i, e0))
                    plen += e0 - s0 - kb_i
            feed_rbl(1 << 30)
            if tail_on:
                prefill_fresh(llm, c, plen, hot_ids, hot_pos)
                pos_all.extend(hot_pos)
                plen += len(hot_ids)
            out, _, _ = llm._greedy_pos(
                c, torch.tensor(pos_all, dtype=torch.long, device=llm.device),
                qids_, args.ans_tokens)
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
                if "iso" in RUN:
                    ans["iso"] = serve_plain(st_iso, k5, None, qids, False)
                if "epic16" in RUN:
                    ans["epic16"] = serve_epic(st_iso, k5, None, qids, args.kb_small, False)
                if "epic32" in RUN:
                    ans["epic32"] = serve_epic(st_iso, k5, None, qids, args.kb_big, False)
                if "epic16_hot" in RUN:
                    ans["epic16_hot"] = serve_epic(st_iso, k5, None, qids, args.kb_small, True)
                if "epic16_hot_rblk" in RUN:
                    ans["epic16_hot_rblk"] = serve_epic(st_iso, k5, rbl, qids,
                                                        args.kb_small, True)
                if "b_anch" in RUN:
                    ans["b_anch"] = serve_plain(st_anc, k5, rbl, qids, False)
                if "b_hot" in RUN:
                    ans["b_hot"] = serve_plain(st_anc, k5, rbl, qids, True)
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
        del st_iso, st_anc, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] EPIC QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"EPIC_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
