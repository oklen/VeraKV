"""kv_vardecomp.py -- decisive 4-arm decomposition of the E4 var-track effect
(user-designed, 2026-07-16). On the SAME far-chain samples, separate the sources of the
harvest>>full gap:

  full_txt : full context + distractors, as text                         (baseline)
  sel_txt  : ONLY the gold chain events, fresh compact joint prefill      -> routing/denoising
  sel_ikv  : ONLY the gold chain events, each encoded independently
             [header; E_i], served at their ORIGINAL (spread) positions   -> independent KV
  harv_kv  : ONLY the gold chain events' rows, gathered from ONE joint
             prefill of the FULL trajectory, at their original positions  -> harvested KV

Decomposition (paired, same items):
  routing / denoising     = Acc(sel_txt) - Acc(full_txt)
  pre-digestion           = Acc(harv_kv) - Acc(sel_txt)     # cached-joint vs fresh re-encode
  joint-conditioning      = Acc(harv_kv) - Acc(sel_ikv)     # joint vs independent encoding

sel_ikv and harv_kv serve the SAME positions; only the encoding (independent vs joint)
differs. sel_txt is a clean compact re-encode of the same chain. Scoring reports exact
substring AND number-set recall (gold in the integers extracted from the output), so full
is not marked wrong merely for emitting an extra variable or a formatting artifact. A
truncation guard prints the max sequence length (must be << model context).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_vardecomp --items 16 --seed 300 --out ./out/vd.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, gen_item

ARMS = ("full_txt", "sel_txt", "sel_ikv", "harv_kv")


def numset(s):
    return set(re.findall(r"\d+", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="./out/vd.json")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    head_full = head + HDR
    H_ids = list(llm.tok(head_full, add_special_tokens=False).input_ids)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rng = random.Random(args.seed)

    rows = []
    maxseq = 0
    for hops in (1, 2, 4):
        for far in (0, 1):
            for it in range(args.items):
                ev, q, gold, slots = gen_item(rng, args.n_events, hops, bool(far))
                eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
                spans = []
                cur = H
                for e in eids:
                    spans.append((cur, cur + len(e)))
                    cur += len(e)
                total = cur
                flat = [t for e in eids for t in e]
                maxseq = max(maxseq, total)
                chain_ev = "".join(ev[i] for i in slots)
                out = {}
                # full_txt: full context + distractors
                o, _, _ = llm._greedy(DynamicCache(), 0,
                                      llm._ids(head_full + "".join(ev) + q + tail), 12)
                out["full_txt"] = o
                # sel_txt: chain events only, fresh compact joint prefill
                o, _, _ = llm._greedy(DynamicCache(), 0,
                                      llm._ids(head_full + chain_ev + q + tail), 12)
                out["sel_txt"] = o
                # harv_kv: joint prefill all, gather chain rows at original positions
                full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
                crows = [p for i in slots for p in range(*spans[i])]
                rt = torch.tensor([p - H for p in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c, full_kv
                out["harv_kv"] = o
                # sel_ikv: chain events each encoded independently, at ORIGINAL positions
                blocks = [(H_kv, list(range(H)))]
                for i in slots:
                    kv = encode_block(llm, H_ids + eids[i],
                                      list(range(H)) + list(range(*spans[i])), keep_a=H)
                    blocks.append((kv, list(range(*spans[i]))))
                c, p = assemble(llm, blocks)
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                out["sel_ikv"] = o
                torch.cuda.empty_cache()
                rec = {"hops": hops, "far": far, "it": it, "gold": gold,
                       "n_chain": len(slots)}
                for k in ARMS:
                    rec[k] = int(gold in out[k])
                    rec[k + "_rec"] = int(gold in numset(out[k]))
                rows.append(rec)
                print(f"[vd] h{hops} far{far} it{it} " +
                      " ".join(f"{k}={rec[k]}" for k in ARMS), flush=True)

    summary = {}
    for hops in (1, 2, 4):
        for far in (0, 1):
            g = [r for r in rows if r["hops"] == hops and r["far"] == far]
            summary[f"h{hops}_far{far}"] = {
                a: sum(r[a] for r in g) / len(g) for a in ARMS}
    json.dump({"rows": rows, "summary": summary, "maxseq": maxseq, "H": H,
               "n_events": args.n_events}, open(args.out, "w"), indent=1)
    print(f"\nMAXSEQ={maxseq} (H={H}); verify << model context (no truncation)")
    for k, v in summary.items():
        print(f"{k}: " + " ".join(f"{a}={v[a]:.2f}" for a in ARMS))
    print("VD_DONE", flush=True)


if __name__ == "__main__":
    main()
