"""kv_refcarrier.py -- the reference gradient: WHAT must a served row say to inherit a state?

kv_mater's hard null: an explicitly stated verdict at slot m+1 does NOT enter a neutral
carrier 10 steps later (.477, chance) -- yet E4/d2/sm's carrier ("T copied from / mirrors S")
inherits S's state at .93-.98, and nk2's note (whose own text points at the verdict) carries
it at .70. Hypothesis: memoization follows the served row's OWN reference semantics -- a row
stores the resolved meaning of its own sentence, and nothing else's.

One variable: the carrier's wording at step 24 (position fixed, same upstream, same question).
Upstream is mt's `explicit` tier verbatim: operands at m, the verdict stated at m+1, both
DROPPED. Served = {decoy@0, carrier@23}.

  c_neutral   "checkpoint saved for the sensor {S} review"            -- mt replication, ~.48
  c_outcome   "the outcome of the sensor {S} alert check was archived for the report"
                                                                       -- ABOUT the verdict,
                                                                          does not carry state
  c_state     "the alert status flag for sensor {S} was set according to the check result"
                                                                       -- a STATE-bearing row:
                                                                          the flag's state IS
                                                                          the verdict
  c_copy      "register {T} recorded the sensor {S} verdict"           -- explicit copy
                                                                          relation, question
                                                                          re-targeted to T

Prediction under the reference law: c_copy ~ c_state > c_outcome > c_neutral ~ chance.
sel_txt is the per-variant leak check (served text never contains the verdict word).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_refcarrier --items 32 --seed 1600 --out ./out/rc.json
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

CARRIERS = ["c_neutral", "c_outcome", "c_state", "c_copy"]
N_BODY = 24
DEC_IDX, CAR_IDX = 0, 23


def pick(o):
    u = o.upper()
    a, b = u.find("EXCEEDED"), u.find("NORMAL")
    if a < 0 and b < 0:
        return None
    if a < 0:
        return "NORMAL"
    if b < 0:
        return "EXCEEDED"
    return "EXCEEDED" if a < b else "NORMAL"


def gen_base(rng):
    D, S, T = rng.sample(NAMES, 3)
    gv = rng.randrange(100, 999)
    thr = rng.randrange(100, 999)
    while abs(gv - thr) < 30:
        thr = rng.randrange(100, 999)
    gold = "EXCEEDED" if gv > thr else "NORMAL"
    m = rng.randrange(N_BODY // 3, (2 * N_BODY) // 3 - 1)
    others = [n for n in NAMES if n not in (D, S, T)]
    fill = {}
    for i in range(N_BODY):
        if i in (DEC_IDX, m, m + 1, CAR_IDX):
            continue
        fill[i] = ("reg_op()", f"register {rng.choice(others)} refreshed to "
                               f"{rng.randrange(100, 999)}")
    car = {
        "c_neutral": ("save_checkpoint()",
                      f"checkpoint saved for the sensor {S} review"),
        "c_outcome": ("archive_result()",
                      f"the outcome of the sensor {S} alert check was archived for the "
                      "report"),
        "c_state": ("update_flag()",
                    f"the alert status flag for sensor {S} was set according to the check "
                    "result"),
        "c_copy": ("record_verdict()",
                   f"register {T} recorded the sensor {S} verdict"),
    }
    qS = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
          "Answer EXCEEDED or NORMAL:")
    qT = (f"\n\nQuestion: What verdict did register {T} record for sensor {S}? "
          "Answer EXCEEDED or NORMAL:")
    return {"D": D, "S": S, "T": T, "gv": gv, "thr": thr, "gold": gold, "m": m,
            "fill": fill, "car": car, "q": {"c_copy": qT, "_default": qS}}


def events(base, cvar):
    b = dict(base["fill"])
    b[DEC_IDX] = ("reg_op()", f"sensor {base['D']} logged a routine idle reading")
    b[base["m"]] = ("read_sensor()",
                    f"sensor {base['S']} measured {base['gv']}; the alert threshold is "
                    f"{base['thr']}")
    b[base["m"] + 1] = ("evaluate_alert()",
                        f"alert check result for sensor {base['S']}: threshold "
                        f"{base['gold']}")
    b[CAR_IDX] = base["car"][cvar]
    return [f"<step {i+1}>\naction: {b[i][0]}\nobservation: {b[i][1]}\n"
            for i in range(N_BODY)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1600)
    ap.add_argument("--out", default="./out/rc.json")
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

    for bi in range(args.items):
        base = gen_base(rng)
        evs = {c: events(base, c) for c in CARRIERS}
        eids = {c: [list(llm.tok(x, add_special_tokens=False).input_ids)
                    for x in evs[c]] for c in CARRIERS}
        budget = {CAR_IDX: max(len(eids[c][CAR_IDX]) for c in CARRIERS)}
        rec = {"it": bi, "gold": base["gold"]}
        for cvar in CARRIERS:
            ei = eids[cvar]
            q = base["q"].get(cvar, base["q"]["_default"])
            spans, cur = [], H
            for i, e in enumerate(ei):
                spans.append((cur, cur + len(e)))
                cur += budget.get(i, len(e))
            pos, rowmap, r = [], {}, 0
            for i in range(N_BODY):
                for pp in range(spans[i][0], spans[i][0] + len(ei[i])):
                    pos.append(pp)
                    rowmap[pp] = r
                    r += 1
            flat = [t for e in ei for t in e]
            full_kv = encode_block(llm, H_ids + flat, list(range(H)) + pos, keep_a=H)
            crows = [pp for i in (DEC_IDX, CAR_IDX)
                     for pp in range(spans[i][0], spans[i][0] + len(ei[i]))]
            rt = torch.tensor([rowmap[pp] for pp in crows], dtype=torch.long)
            sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full
                                           + "".join(evs[cvar][i] for i in
                                                     (DEC_IDX, CAR_IDX)) + q + tail), 10)
            rec[f"{cvar}_sel"] = int(pick(o) == base["gold"])
            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 10)
            del c
            rec[f"{cvar}_harv"] = int(pick(o) == base["gold"])
            del full_kv, sub
            torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[rc] it{bi} gold={base['gold'][:3]} | " + " ".join(
            f"{c[2:6]}:s{rec[f'{c}_sel']}h{rec[f'{c}_harv']}" for c in CARRIERS), flush=True)

    n = len(rows)
    print(f"\n===== REFERENCE GRADIENT (n={n}, chance=.50) =====")
    for c in CARRIERS:
        print(f"{c:10s} sel_txt={sum(r[f'{c}_sel'] for r in rows)/n:.3f} "
              f"harv={sum(r[f'{c}_harv'] for r in rows)/n:.3f}")
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("RC_DONE", flush=True)


if __name__ == "__main__":
    main()
