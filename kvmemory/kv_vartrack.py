"""kv_vartrack.py -- dependency sweep over sparse event-KV reuse (RULER Variable
Tracking, agent-eventized; user-designed E4, 2026-07-16).

Each event is one assignment: either `register X initialized to <value>` or
`register Y copied from register X`. A hop-h chain needs h cross-event edges to
resolve the final register. We sweep:

  hops in {1, 2, 4}      chain depth
  span  in {near, far}   whether the chain's events sit adjacent or spread across
                         the whole trajectory (far = every edge crosses many events)

under four servings of the SAME trajectory (n_events=24, exact number scoring):

  full     one joint prefill, all rows                       (ceiling)
  harvest  joint prefill, gather ONLY the chain events' rows + head (holes)
  iso      each event encoded [H; E_i] independently, all events served
  anch     each event encoded [H; opening-anchor; E_i] (b_hot-style write), all served

Readings: where does iso break as hops grow (missing cross-event edges), does the
anchor's opening conditioning buy anything on chains it never saw, and does harvest
hold (rows carry pre-digested history) even when the chain is sparse and far.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=1 python -m kvmemory.kv_vartrack --out ./out/vt.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble

HDR = ("You are reviewing a completed agent trajectory. Use it to answer the "
       "question precisely.\n\nTask: run register maintenance and report values "
       "faithfully.\n\nTrajectory:\n")

NAMES = ["ALPHA", "BETA", "GAMMA", "DELTA", "OMEGA", "SIGMA", "KAPPA", "THETA",
         "LAMBDA", "EPSILON"]


def gen_item(rng, n_events, hops, far):
    """Returns (event_texts, question, gold)."""
    chain = rng.sample(NAMES, hops + 1)
    val = rng.randrange(100, 999)
    if far:
        slots = sorted(rng.sample(range(n_events), hops + 1))
        # force max spread: first and last slot pinned to the ends
        slots[0], slots[-1] = 0, n_events - 1
    else:
        s0 = rng.randrange(0, n_events - hops - 1)
        slots = list(range(s0, s0 + hops + 1))
    ev = []
    si = 0
    filler_regs = [n for n in NAMES if n not in chain]
    for i in range(n_events):
        if si <= hops and i == slots[si]:
            if si == 0:
                body = f"register {chain[0]} initialized to {val}"
            else:
                body = f"register {chain[si]} copied from register {chain[si-1]}"
            si += 1
        else:
            fr = rng.choice(filler_regs)
            body = f"register {fr} refreshed to {rng.randrange(100,999)}"
        ev.append(f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n")
    q = (f"\n\nQuestion: What is the value of register {chain[-1]}? "
         "Answer with the number only:")
    return ev, q, str(val), slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--w_anch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="./out/vt.json")
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
    for hops in (1, 2, 4):
        for far in (0, 1):
            for it in range(args.items):
                ev, q, gold, slots = gen_item(rng, args.n_events, hops, bool(far))
                eids = [list(llm.tok(t, add_special_tokens=False).input_ids)
                        for t in ev]
                spans = []
                cur = H
                for e in eids:
                    spans.append((cur, cur + len(e)))
                    cur += len(e)
                total = cur
                flat = [t for e in eids for t in e]
                arms = {}
                # full
                out, _, _ = llm._greedy(DynamicCache(), 0,
                                        llm._ids(head_full + "".join(ev) + q + tail), 12)
                arms["full"] = int(gold in out)
                # harvest: joint prefill rows of chain events only (+head)
                full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
                crows = [p for i in slots for p in range(*spans[i])]
                rt = torch.tensor([p - H for p in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                out, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                arms["harvest"] = int(gold in out)
                del full_kv
                # iso: [H; E_i] each, ALL events served
                blocks = [(H_kv, list(range(H)))]
                for i in range(args.n_events):
                    kv = encode_block(llm, H_ids + eids[i],
                                      list(range(H)) + list(range(*spans[i])), keep_a=H)
                    blocks.append((kv, list(range(*spans[i]))))
                c, p = assemble(llm, blocks)
                out, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                arms["iso"] = int(gold in out)
                # anch: [H; opening w; E_i]
                blocks = [(H_kv, list(range(H)))]
                w = min(args.w_anch, total - H)
                for i in range(args.n_events):
                    alen = min(w, spans[i][0] - H)
                    kv = encode_block(llm, H_ids + flat[:alen] + eids[i],
                                      list(range(H)) + list(range(H, H + alen))
                                      + list(range(*spans[i])), keep_a=H + alen)
                    blocks.append((kv, list(range(*spans[i]))))
                c, p = assemble(llm, blocks)
                out, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                arms["anch"] = int(gold in out)
                torch.cuda.empty_cache()
                rows.append({"hops": hops, "far": far, **arms})
                print(f"[vt] hops={hops} far={far} it={it} " +
                      " ".join(f"{k}={v}" for k, v in arms.items()), flush=True)

    print("\n===== SWEEP =====")
    summary = {}
    for hops in (1, 2, 4):
        for far in (0, 1):
            g = [r for r in rows if r["hops"] == hops and r["far"] == far]
            s = {a: sum(r[a] for r in g) / len(g) for a in ("full", "harvest", "iso", "anch")}
            summary[f"h{hops}_far{far}"] = s
            print(f"hops={hops} far={far} n={len(g)}: " +
                  " ".join(f"{a}={v:.2f}" for a, v in s.items()))
    json.dump({"rows": rows, "summary": summary}, open(args.out, "w"), indent=1)
    print("VT_DONE", flush=True)


if __name__ == "__main__":
    main()
