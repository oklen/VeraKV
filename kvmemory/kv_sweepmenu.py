"""kv_sweepmenu.py -- is the ~1-bit capacity cliff STORAGE or READOUT?

kv_decoyctl2 showed the K=2 recall rate was readout-limited, not storage-limited: with the
answer menu in the question, harvest recalls the dropped source's state at .984 (the cleanest
cell -- no state word served at all) vs .80 under the open-ended phrasing. But the entropy
boundary E4 reported (dir4 .07, name8 .01 -> "a served row carries ~1 bit") was measured with
the SAME open-ended phrasing. If readout suppressed K=2 by 18 points, how much of the K=4/8
collapse is artifact?

One change: the question enumerates the full answer menu. Decoys drawn INDEPENDENTLY (may equal
gold). drop_mid only (the source event is dropped; the answer exists nowhere in any served
token). Score = first menu word emitted.

  dir4 jumps  -> the cliff was readout; capacity >= 2 bits; the tex claim is rewritten
  dir4 stays  -> the cliff is storage; the 1-bit claim survives its strongest attack

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_sweepmenu --items 32 --seed 1400 --out ./out/sm.json
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
DIR4 = ["NORTH", "SOUTH", "EAST", "WEST"]
MODE8 = ["ALPHA", "BRAVO", "DELTA", "ECHO", "GOLF", "HOTEL", "KILO", "TANGO"]
POOLS = {"bin2": STATUS2, "dir4": DIR4, "name8": MODE8}
SETTINGS = ["bin2", "dir4", "name8"]


def menu(pool):
    return " Answer " + ", ".join(pool[:-1]) + f", or {pool[-1]}:"


def pick(o, pool):
    u = o.upper()
    hits = [(u.find(w), w) for w in pool if u.find(w) >= 0]
    return min(hits)[1] if hits else None


def gen_item(rng, n_events, setting):
    pool = POOLS[setting]
    D, S, T = rng.sample(NAMES, 3)
    gm = rng.choice(pool)
    dm = rng.choice(pool)                     # INDEPENDENT: may equal gm
    dec = f"register {D} set to state {dm}"
    src = f"register {S} set to state {gm}"
    dst = f"register {T} copied from register {S}"
    q = f"\n\nQuestion: What state is register {T} in?{menu(pool)}"
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
    ap.add_argument("--seed", type=int, default=1400)
    ap.add_argument("--out", default="./out/sm.json")
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
    for setting in SETTINGS:
        pool = POOLS[setting]
        for it in range(args.items):
            ev, q, gold, decoy, Sset = gen_item(rng, args.n_events, setting)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
            spans, cur = [], H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            rec = {"setting": setting, "it": it, "gold": gold, "decoy": decoy,
                   "dec_eq_gold": int(decoy == gold)}

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev) + q + tail), 10)
            rec["full_txt"] = int(pick(o, pool) == gold)
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev[i] for i in Sset)
                                           + q + tail), 10)
            rec["sel_txt"] = int(pick(o, pool) == gold)

            crows = [pp for i in Sset for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 10)
            del c
            pk = pick(o, pool)
            rec["harv_kv"] = int(pk == gold)
            rec["harv_dec"] = int(pk == decoy)
            rec["harv_none"] = int(pk is None)

            iso = []
            for i in Sset:
                k = encode_block(llm, H_ids + eids[i],
                                 list(range(H)) + list(range(*spans[i])), keep_a=H)
                iso.append((k, list(range(*spans[i]))))
            c, p = assemble(llm, [(H_kv, list(range(H)))] + iso)
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 10)
            del c
            rec["sel_ikv"] = int(pick(o, pool) == gold)

            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[sm] {setting} it{it} full={rec['full_txt']} sel={rec['sel_txt']} "
                  f"ikv={rec['sel_ikv']} harv={rec['harv_kv']}", flush=True)

    print("\n===== ENTROPY BOUNDARY, MENU READOUT (drop_mid, independent decoy) =====")
    for setting in SETTINGS:
        g = [r for r in rows if r["setting"] == setting]

        def m(k):
            return sum(r[k] for r in g) / len(g)
        print(f"{setting:6s} n={len(g):3d} chance={1/len(POOLS[setting]):.3f} | "
              f"full={m('full_txt'):.3f} sel_txt={m('sel_txt'):.3f} "
              f"sel_ikv={m('sel_ikv'):.3f} harv={m('harv_kv'):.3f} "
              f"(dec={m('harv_dec'):.2f} none={m('harv_none'):.2f})")
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("SM_DONE", flush=True)


if __name__ == "__main__":
    main()
