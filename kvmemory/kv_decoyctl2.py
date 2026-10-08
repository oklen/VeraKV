"""kv_decoyctl2.py -- completes the decoy-polarity table: the `none` cell, done right.

kv_decoyctl's `none` cell scored 0.000 for a design reason, not a finding: removing the decoy
event also removed the only occurrence of the answer vocabulary, so the model had no menu to
choose from (full_txt = .973 proved the task itself was fine). This rerun puts the menu in the
QUESTION ("Answer ONLINE or OFFLINE:") for all three modes, so `none` finally measures recall
with vocabulary but without any served state word to echo or eliminate.

Scoring must change with the menu: a substring test would credit echoes of the menu itself, so
we score the FIRST state word emitted (pick()), as kv_cogate does.

Modes (identical to kv_decoyctl otherwise; layout copied verbatim from kv_predigest_sweep):
  opposite      served decoy = negation of gold   (E4's flawed original)
  independent   served decoy drawn independently  (may equal gold)
  none          decoy event carries no state word

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_decoyctl2 --items 32 --seed 1200 --out ./out/d2.json
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
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]
MODES = ["opposite", "independent", "none"]


def pick(o):
    u = o.upper()
    a, b = u.find("ONLINE"), u.find("OFFLINE")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "OFFLINE"
    if b < 0:
        return "ONLINE"
    return "ONLINE" if a < b else "OFFLINE"


def gen_item(rng, n_events, mode):
    D, S, T = rng.sample(NAMES, 3)
    gm = rng.choice(STATUS2)
    if mode == "opposite":
        dm = STATUS2[1 - STATUS2.index(gm)]
        dec = f"register {D} set to state {dm}"
    elif mode == "independent":
        dm = rng.choice(STATUS2)
        dec = f"register {D} set to state {dm}"
    else:
        dm = None
        dec = f"register {D} completed a routine self-test"
    src = f"register {S} set to state {gm}"
    dst = f"register {T} copied from register {S}"
    q = (f"\n\nQuestion: What state is register {T} in? Answer ONLINE or OFFLINE:")
    a_idx, c_idx = 0, n_events - 1
    m = rng.randrange(n_events // 3, (2 * n_events) // 3)
    special = {a_idx: dec, m: src, c_idx: dst}
    others = [n for n in NAMES if n not in (D, S, T)]
    ev = []
    for i in range(n_events):
        body = special.get(i) or \
            f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
        ev.append(f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n")
    return ev, q, gm, dm, [a_idx, c_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1200)
    ap.add_argument("--out", default="./out/d2.json")
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
    for mode in MODES:
        for it in range(args.items):
            ev, q, gold, decoy, Sset = gen_item(rng, args.n_events, mode)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            rec = {"mode": mode, "it": it, "gold": gold, "decoy": decoy,
                   "dec_eq_gold": int(decoy == gold) if decoy else -1}

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev) + q + tail), 8)
            rec["full_txt"] = int(pick(o) == gold)
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev[i] for i in Sset)
                                           + q + tail), 8)
            rec["sel_txt"] = int(pick(o) == gold)

            crows = [pp for i in Sset for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 8)
            del c
            pk = pick(o)
            rec["harv_kv"] = int(pk == gold)
            rec["harv_dec"] = int(pk == decoy) if decoy else -1
            rec["harv_none"] = int(pk is None)

            iso = []
            for i in Sset:
                k = encode_block(llm, H_ids + eids[i],
                                 list(range(H)) + list(range(*spans[i])), keep_a=H)
                iso.append((k, list(range(*spans[i]))))
            c, p = assemble(llm, [(H_kv, list(range(H)))] + iso)
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 8)
            del c
            rec["sel_ikv"] = int(pick(o) == gold)

            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[d2] {mode} it{it} full={rec['full_txt']} sel={rec['sel_txt']} "
                  f"ikv={rec['sel_ikv']} harv={rec['harv_kv']}", flush=True)

    print("\n===== DECOY CONTROL v2 (menu in question, chance=0.50) =====")
    for mode in MODES:
        g = [r for r in rows if r["mode"] == mode]
        if not g:
            continue

        def m(k):
            return sum(r[k] for r in g) / len(g)
        print(f"{mode:12s} n={len(g):3d} | full={m('full_txt'):.3f} sel_txt={m('sel_txt'):.3f} "
              f"sel_ikv={m('sel_ikv'):.3f} harv={m('harv_kv'):.3f} "
              f"(harv_none={m('harv_none'):.2f})")
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("D2_DONE", flush=True)


if __name__ == "__main__":
    main()
