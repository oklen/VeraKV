"""kv_decoyctl.py -- did E4's bin2 headline (+1.00) measure MEMOIZATION or ELIMINATION?

kv_predigest_sweep gen_item does `gm, dm = rng.sample(pool, 2)`. For bin2 the pool is exactly
{ONLINE, OFFLINE}, so the served decoy is ALWAYS the negation of the gold answer. With K=2,
"recall the memoized gold" and "answer the opposite of the served decoy" are indistinguishable --
and E4's +1.00 / p=6e-39 rests entirely on that cell. kv_coherence, whose unrelated register's
state is drawn INDEPENDENTLY, saw the same structure produce only ~0.57-0.70. That arithmetic
(0.70 real + 0.30 by elimination = 1.00) fits the artifact too well to leave unchecked.

ONE variable moves. Everything else is copied verbatim from kv_predigest_sweep (24 events,
decoy@0, source@m in [8,16), carrier@23, same question, same drop_mid subset).

  opposite      dm = NOT gm            -- E4 exactly. MUST reproduce ~1.00 or this script is
                                          not a faithful replication and the contrast is void.
  independent   dm ~ U{ONLINE,OFFLINE} -- may equal gm. Elimination and echo are each worth
                                          0.50, so anything above 0.50 is real recall.
  none          the decoy event carries no state word at all -- nothing to eliminate or echo.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_decoyctl --items 32 --seed 900 --out ./out/dc.json
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
from kvmemory.kv_vartrack import HDR, NAMES

STATUS2 = ["ONLINE", "OFFLINE"]
MODES = ["opposite", "independent", "none"]


def match(o, ans):
    return int(ans.lower() in o.lower())


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
    q = (f"\n\nQuestion: What state is register {T} in? "
         "Answer with the single state word only:")
    a_idx, c_idx = 0, n_events - 1
    m = rng.randrange(n_events // 3, (2 * n_events) // 3)
    special = {a_idx: dec, m: src, c_idx: dst}
    others = [n for n in NAMES if n not in (D, S, T)]
    ev = []
    for i in range(n_events):
        body = special.get(i) or \
            f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
        ev.append(f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n")
    return ev, q, gm, dm, {"drop_mid": [a_idx, c_idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--out", default="./out/dc.json")
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
            ev, q, gold, decoy, subsets = gen_item(rng, args.n_events, mode)
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
                                  llm._ids(head_full + "".join(ev) + q + tail), 12)
            rec["full_txt"] = match(o, gold)
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            Sset = subsets["drop_mid"]

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev[i] for i in Sset)
                                           + q + tail), 12)
            rec["sel_txt"] = match(o, gold)

            crows = [pp for i in Sset for pp in range(*spans[i])]
            rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
            del c
            rec["harv_kv"] = match(o, gold)
            rec["harv_dec"] = match(o, decoy) if decoy else -1

            iso = []
            for i in Sset:
                k = encode_block(llm, H_ids + eids[i],
                                 list(range(H)) + list(range(*spans[i])), keep_a=H)
                iso.append((k, list(range(*spans[i]))))
            c, p = assemble(llm, [(H_kv, list(range(H)))] + iso)
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
            del c
            rec["sel_ikv"] = match(o, gold)

            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[dc] {mode} it{it} full={rec['full_txt']} sel_txt={rec['sel_txt']} "
                  f"sel_ikv={rec['sel_ikv']} harv={rec['harv_kv']}", flush=True)

    print("\n===== DECOY CONTROL (bin2, drop_mid, chance=0.50) =====")
    for mode in MODES:
        g = [r for r in rows if r["mode"] == mode]
        if not g:
            continue

        def m(k):
            return sum(r[k] for r in g) / len(g)
        print(f"{mode:12s} n={len(g):3d} | full_txt={m('full_txt'):.3f} "
              f"sel_txt={m('sel_txt'):.3f} sel_ikv={m('sel_ikv'):.3f} "
              f"harv_kv={m('harv_kv'):.3f}")
        if mode == "independent":
            for eq in (0, 1):
                h = [r for r in g if r["dec_eq_gold"] == eq]
                if h:
                    print(f"    decoy{'==' if eq else '!='}gold n={len(h):3d}: "
                          f"harv={sum(r['harv_kv'] for r in h)/len(h):.3f} "
                          f"sel_txt={sum(r['sel_txt'] for r in h)/len(h):.3f}")
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("DC_DONE", flush=True)


if __name__ == "__main__":
    main()
