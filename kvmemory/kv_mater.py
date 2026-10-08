"""kv_mater.py -- the materialization boundary, three tiers (user-designed, stage 3).

E4-family results say a harvested row carries a resolved 1-bit state at ~.98 (menu readout,
d2) when an upstream event explicitly STATED it, but a verdict that was never computed stays
at chance (.48, menu) -- the model defers computation. The remaining question is WHAT makes a
conclusion enter the carrier: its answer token existing upstream, or the trajectory having
USED it? Three tiers, identical operands, identical final carrier, differing ONLY in what
happens at the materialization slot (m+1):

  operands    slot m+1 is a neutral event -- the verdict is merely DERIVABLE (447 vs 312 both
              in scope at write time), never stated, never used
  explicit    slot m+1 STATES the verdict ("alert check result for S: threshold EXCEEDED") --
              the answer token exists upstream; the copied-label channel may carry it
  behavioral  slot m+1 is an ACTION whose choice depends on the verdict (trigger_alarm vs
              log_routine), with a neutral observation -- the answer token NEVER exists,
              but the trajectory behaviorally committed to it

  step 0    : decoy sensor event (no verdict word; menu supplies vocabulary)
  step m    : operands   "sensor S measured {gv}; the alert threshold is {thr}"   -- DROPPED
  step m+1  : the tier slot                                                        -- DROPPED
  step 23   : carrier    "checkpoint saved for the sensor {S} review"              -- SERVED
  served = {0, 23}; question "Was sensor S above the alert threshold? Answer EXCEEDED or
  NORMAL:" -- same phrasing the verdict cells already use (menu, so readout-fair).

The three tiers are built from the SAME base item (same operands, same fillers, same names,
same positions -- the tier slot is length-padded to the max across tiers), so tier contrasts
are within-item. sel_txt is the leak check: served text is tier-independent and contains no
verdict word, so it must sit at chance for every tier.

If explicit recovers but behavioral does not: only answer-token copying survives -- "the
carrier stores tokens it saw, not states it used". If behavioral recovers too: write-time USE
materializes a conclusion into downstream rows even when its token never existed -- the
strongest form of "KV rows memoize computation". Write it as `behaviorally materialized`
either way; no claims about what the model "thought".

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_mater --items 32 --seed 1500 --out ./out/mt.json
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

TIERS = ["operands", "explicit", "behavioral"]
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
    D, S = rng.sample(NAMES, 2)
    gv = rng.randrange(100, 999)
    thr = rng.randrange(100, 999)
    while abs(gv - thr) < 30:
        thr = rng.randrange(100, 999)
    gold = "EXCEEDED" if gv > thr else "NORMAL"
    m = rng.randrange(N_BODY // 3, (2 * N_BODY) // 3 - 1)
    others = [n for n in NAMES if n not in (D, S)]
    fill = {}
    for i in range(N_BODY):
        if i in (DEC_IDX, m, m + 1, CAR_IDX):
            continue
        fill[i] = (f"reg_op()", f"register {rng.choice(others)} refreshed to "
                                f"{rng.randrange(100, 999)}")
    # the tier slot, (action, observation); verdict word appears ONLY in `explicit`
    alarm = gold == "EXCEEDED"
    slot = {
        "operands": ("log_status()",
                     "routine status entry recorded for the monitoring cycle"),
        "explicit": ("evaluate_alert()",
                     f"alert check result for sensor {S}: threshold {gold}"),
        "behavioral": ("trigger_alarm()" if alarm else "log_routine()",
                       "the requested operation completed and was recorded"),
    }
    q = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
         "Answer EXCEEDED or NORMAL:")
    return {"D": D, "S": S, "gv": gv, "thr": thr, "gold": gold, "m": m,
            "fill": fill, "slot": slot, "q": q}


def events(base, tier):
    b = dict(base["fill"])
    b[DEC_IDX] = ("reg_op()", f"sensor {base['D']} logged a routine idle reading")
    b[base["m"]] = ("read_sensor()",
                    f"sensor {base['S']} measured {base['gv']}; the alert threshold is "
                    f"{base['thr']}")
    b[base["m"] + 1] = base["slot"][tier]
    b[CAR_IDX] = ("save_checkpoint()",
                  f"checkpoint saved for the sensor {base['S']} review")
    return [f"<step {i+1}>\naction: {b[i][0]}\nobservation: {b[i][1]}\n"
            for i in range(N_BODY)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1500)
    ap.add_argument("--out", default="./out/mt.json")
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
        evs = {t: events(base, t) for t in TIERS}
        eids = {t: [list(llm.tok(x, add_special_tokens=False).input_ids)
                    for x in evs[t]] for t in TIERS}
        # pad the two tier-dependent slots to their cross-tier max so served positions match
        budget = {i: max(len(eids[t][i]) for t in TIERS)
                  for i in (base["m"] + 1,)}
        rec = {"it": bi, "gold": base["gold"], "gv": base["gv"], "thr": base["thr"]}

        for tier in TIERS:
            ei = eids[tier]
            spans, cur = [], H
            for i, e in enumerate(ei):
                spans.append((cur, cur + len(e)))
                cur += budget.get(i, len(e))
            total = cur
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
                                  llm._ids(head_full + "".join(evs[tier]) + base["q"]
                                           + tail), 10)
            rec[f"{tier}_full"] = int(pick(o) == base["gold"])

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full
                                           + "".join(evs[tier][i] for i in
                                                     (DEC_IDX, CAR_IDX))
                                           + base["q"] + tail), 10)
            rec[f"{tier}_sel"] = int(pick(o) == base["gold"])

            c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
            o, _, _ = llm._greedy_pos(c, p, llm._ids(base["q"] + tail), 10)
            del c
            rec[f"{tier}_harv"] = int(pick(o) == base["gold"])
            del full_kv, sub
            torch.cuda.empty_cache()
        rows.append(rec)
        print(f"[mt] it{bi} gold={base['gold'][:3]} | " + " | ".join(
            f"{t[:4]}: full={rec[f'{t}_full']} sel={rec[f'{t}_sel']} "
            f"harv={rec[f'{t}_harv']}" for t in TIERS), flush=True)

    n = len(rows)
    print(f"\n===== MATERIALIZATION BOUNDARY (n={n}, chance=.50) =====")
    for t in TIERS:
        print(f"{t:11s} full={sum(r[f'{t}_full'] for r in rows)/n:.3f} "
              f"sel_txt={sum(r[f'{t}_sel'] for r in rows)/n:.3f} "
              f"harv={sum(r[f'{t}_harv'] for r in rows)/n:.3f}")
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print("MT_DONE", flush=True)


if __name__ == "__main__":
    main()
