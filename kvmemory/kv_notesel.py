"""kv_notesel.py -- what exactly does the served anchor region carry: memoized NOTES
(conclusions parked on delimiter/aggregator tokens; Li 2606.17107) or a generic
structural scaffold?

Our Regularity II found the anchor's value is largely content-free (true ~ shuffled ~
other-episode > none > degenerate fillers). The kvnotes lens sharpens the question:
if what serving the anchor region really provides is the trajectory's memoized
inference state, then the anchor's DELIMITER/AGGREGATOR rows alone (sentence ends,
newlines, step-tag closers --- where prefill parks conclusions) should carry most of
its value, and matched-count RANDOM anchor rows should not. If instead random rows do
as well, the scaffold reading stands (consistent with the shuffle result).

Arms (all else identical to the b_hot recipe; n=824 mechanism subset, 8B):
  bh       full anchor-residue rows served (the .2209 gate)
  nt64     only 64 note rows of the anchor residue (delimiter positions, even-spread)
  rd64     64 seeded-random anchor-residue rows       (matched-budget control)
  nt256    256 note rows
  rd256    256 random rows
  norblk   no anchor-residue rows at all              (floor reference)

Pre-registered readings:
  nt_K ~ bh  and  rd_K << nt_K  -> notes ARE the payload (memoized-inference account)
  nt_K ~ rd_K (both between norblk and bh, scaling with K) -> scaffold/volume account
  both ~ bh even at K=64 -> tiny-budget repair; either account, deployment-relevant

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_notesel --shard 0 --queue_dir ./out/nt_q
"""
from __future__ import annotations

import argparse
import hashlib
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
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh

ARMS = ["bh", "nt64", "rd64", "nt256", "rd256", "norblk"]
NOTE_CHARS = ("\n", ".", ":", ";", ">", ")")


def note_positions(llm, ids, lo, hi):
    """Anchor-window positions whose TOKEN text ends at a delimiter/aggregator
    boundary --- where prefill parks per-segment conclusions."""
    toks = llm.tok.convert_ids_to_tokens(ids[lo:hi])
    out = []
    for j, t in enumerate(toks):
        s = llm.tok.convert_tokens_to_string([t])
        if s and (s.rstrip(" ") and s.rstrip(" ")[-1] in NOTE_CHARS):
            out.append(lo + j)
    return out


def spread(xs, k):
    """Deterministic even-spread subsample to k items (order preserved)."""
    if len(xs) <= k:
        return list(xs)
    step = len(xs) / k
    return [xs[int(i * step)] for i in range(k)]


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
    ap.add_argument("--out", default="./out/nt.json")
    ap.add_argument("--ans_out", default="./out/nt_ans.jsonl")
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
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st_anch = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                      pos, keep_a=H + alen)
        all_ids = header_ids + traj_flat
        notes_all = note_positions(llm, all_ids, H, H + w_anch)
        print(f"[w{wid}] ep {ep.episode_id} total={total} need={len(need)} "
              f"qa={len(qas)} notes_in_window={len(notes_all)}", flush=True)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def residue_rows(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                s0, e0 = spans[i]
                for pth in range(max(H, s0), min(H + w_anch, e0)):
                    keep_mask[pth - H] = False
            return [H + j for j in range(w_anch) if keep_mask[j]]

        def rbl_subset(rows_abs):
            if not rows_abs:
                return None
            rt = torch.tensor([p - H for p in rows_abs], dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    list(rows_abs))

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def serve(kept, rbl, qids_):
            keep = [i for i in kept if i not in hot_idx]
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
            res_rows = residue_rows(k5)
            res_set = set(res_rows)
            notes_res = [p for p in notes_all if p in res_set]
            rng = random.Random(int(hashlib.md5(
                f"{ep.episode_id}|{qi}".encode()).hexdigest()[:8], 16))
            ans, meta = {}, {"n_residue": len(res_rows), "n_notes": len(notes_res)}

            for a in RUN:
                if a == "bh":
                    rbl = rbl_subset(res_rows)
                elif a == "norblk":
                    rbl = None
                else:
                    kk = int(a[2:])
                    if a.startswith("nt"):
                        rows = spread(notes_res, kk)
                    else:
                        rows = sorted(rng.sample(res_rows, min(kk, len(res_rows))))
                    meta[f"k_{a}"] = len(rows)
                    rbl = rbl_subset(rows)
                ans[a] = serve(k5, rbl, qids)

            row = {"episode_id": ep.episode_id, "qi": qi, "q": q, "gold": gold,
                   "domain": ep.domain, **meta}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} done | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] NOTESEL QUEUE over {len(eps)} episodes", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM ep idx {i} skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 60)
    print(f"NOTESEL w{wid}: {n} QA")
    for a in RUN:
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%")
    print(f"NT_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
