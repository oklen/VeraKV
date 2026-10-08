"""kv_noteknob2.py -- clean PAIRED retest of the note knob (user-designed, 2026-07-16).

kv_noteknob v1 had two flaws its own floor-control exposed: (a) note vs no-note were
DIFFERENT items (unpaired, low power); (b) the note was appended to the downstream event's
span, so the harvested span ended in the note instead of the base statement -- it diluted
the memoized signal AND distracted the model outright (bin2 full_txt fell 0.98->0.76).

This fixes both:
  * PAIRED: one item, two variants that differ in exactly ONE event slot.
  * The note gets its OWN event span at slot m+1 (right after the SOURCE, so it attends it),
    and the downstream statement at slot n-1 is left untouched.
  * The control variant puts a length-matched, neutral "routine note:" filler in the SAME
    slot -- so both variants have identical event counts, identical other fillers, and
    near-identical positions. The only difference is whether slot m+1 prompts a computation.

  step 0    : decoy
  step m    : SOURCE (defines the target / operands)   -- DROPPED in drop_mid
  step m+1  : NOTE (prompt-to-compute about S)  |  FILLER (neutral)   -- SERVED
  step n-1  : DOWNSTREAM (base statement, clean)                       -- SERVED
  served: drop_mid={0, m+1, n-1} ; complete={0, m, m+1, n-1}

If the note induces eager computation at prefill, the note event's rows carry the answer and
harvest(note) > harvest(filler) on the SAME items (McNemar). The note names neither the
answer nor the operands, so sel_txt stays a floor in both variants (asserted + measured).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_noteknob2 --items 24 --seed 700 --out ./out/nk2.json
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
SETTINGS = ["bin2", "num", "verdict"]
KCARD = {"bin2": 2, "num": 900, "verdict": 2}

# the note prompts a computation about S (just defined at slot m); it names neither the
# answer nor any operand.
NOTE_T = {
    "bin2": ("audit note: resolve and record the current state of register {S} here for "
             "the final report"),
    "num": ("audit note: resolve and record the current numeric value of register {S} here "
            "for the final report"),
    "verdict": ("audit note: determine the threshold verdict for sensor {S} now and record "
                "it here for the final report"),
}
FILL_T = ("routine note: the scheduled maintenance sweep completed its pass without "
          "incident for the auxiliary bus")


def match(o, ans):
    if ans.isdigit():
        return int(ans in re.findall(r"\d+", o))
    return int(ans.lower() in o.lower())


def gen_item(rng, n_events, setting):
    D, S, T = rng.sample(NAMES, 3)
    if setting == "num":
        dv = rng.randrange(100, 999)
        gv = rng.randrange(100, 999)
        while gv == dv:
            gv = rng.randrange(100, 999)
        dec = f"register {D} initialized to {dv}"
        src = f"register {S} initialized to {gv}"
        dst = f"register {T} copied from register {S}"
        q = (f"\n\nQuestion: What is the value of register {T}? "
             "Answer with the number only:")
        gold, decoy = str(gv), str(dv)
    elif setting == "bin2":
        gm, dm = rng.sample(STATUS2, 2)
        dec = f"register {D} set to state {dm}"
        src = f"register {S} set to state {gm}"
        dst = f"register {T} copied from register {S}"
        q = (f"\n\nQuestion: What state is register {T} in? "
             "Answer with the single state word only:")
        gold, decoy = gm, dm
    elif setting == "verdict":
        gv = rng.randrange(100, 999)
        thr = rng.randrange(100, 999)
        while abs(gv - thr) < 30:
            thr = rng.randrange(100, 999)
        gold = "EXCEEDED" if gv > thr else "NORMAL"
        decoy = "NORMAL" if gv > thr else "EXCEEDED"
        dec = f"sensor {D} logged a routine idle reading"
        src = f"sensor {S} measured {gv}; the alert threshold is {thr}"
        dst = f"the operator reviewed sensor {S} against the alert threshold"
        q = (f"\n\nQuestion: Was sensor {S} above the alert threshold? "
             "Answer EXCEEDED or NORMAL:")
    note = NOTE_T[setting].format(S=S, T=T)
    assert not match(note, gold) and not match(note, decoy), f"NOTE LEAKS: {note!r}"
    assert not match(FILL_T, gold) and not match(FILL_T, decoy), "FILLER LEAKS"
    m = rng.randrange(n_events // 3, (2 * n_events) // 3)
    others = [n for n in NAMES if n not in (D, S, T)]
    bodies = [f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
              for _ in range(n_events)]
    bodies[0] = dec
    bodies[m] = src
    bodies[n_events - 1] = dst
    return {"bodies": bodies, "m": m, "q": q, "gold": gold, "decoy": decoy, "note": note}


def build_ev(bodies, m, slot_text):
    b = list(bodies)
    b[m + 1] = slot_text
    return [f"<step {i+1}>\naction: reg_op()\nobservation: {b[i]}\n" for i in range(len(b))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=700)
    ap.add_argument("--out", default="./out/nk2.json")
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
        for it in range(args.items):
            item = gen_item(rng, args.n_events, setting)
            m, q, gold, decoy = item["m"], item["q"], item["gold"], item["decoy"]
            rec = {"setting": setting, "it": it}
            for vname, slot in (("note", item["note"]), ("fill", FILL_T)):
                ev = build_ev(item["bodies"], m, slot)
                eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
                spans = []
                cur = H
                for e in eids:
                    spans.append((cur, cur + len(e)))
                    cur += len(e)
                total = cur
                flat = [t for e in eids for t in e]
                rec[f"slotlen_{vname}"] = len(eids[m + 1])
                subsets = {"complete": [0, m, m + 1, args.n_events - 1],
                           "drop_mid": [0, m + 1, args.n_events - 1]}
                o, _, _ = llm._greedy(DynamicCache(), 0,
                                      llm._ids(head_full + "".join(ev) + q + tail), 12)
                rec[f"full_{vname}"] = match(o, gold)
                full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
                for cond, Sset in subsets.items():
                    if cond == "drop_mid":
                        o, _, _ = llm._greedy(
                            DynamicCache(), 0,
                            llm._ids(head_full + "".join(ev[i] for i in Sset) + q + tail), 12)
                        rec[f"sel_drop_{vname}"] = match(o, gold)
                    crows = [pp for i in Sset for pp in range(*spans[i])]
                    rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                    sub = [(K.index_select(2, rt), V.index_select(2, rt))
                           for K, V in full_kv]
                    c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                    o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                    del c
                    key = "harv_complete" if cond == "complete" else "harv_drop"
                    rec[f"{key}_{vname}"] = match(o, gold)
                    if cond == "drop_mid":
                        rec[f"harv_drop_{vname}_dec"] = match(o, decoy)
                del full_kv
                torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[nk2] {setting} it{it} | NOTE full={rec['full_note']} "
                  f"harvD={rec['harv_drop_note']} selD={rec['sel_drop_note']} "
                  f"|| FILL full={rec['full_fill']} harvD={rec['harv_drop_fill']} "
                  f"selD={rec['sel_drop_fill']}", flush=True)

    print("\n===== PAIRED NOTE-KNOB =====")
    for s in SETTINGS:
        g = [r for r in rows if r["setting"] == s]
        if not g:
            continue
        def mn(k):
            return sum(r[k] for r in g) / len(g)
        bw = sum(1 for r in g if r["harv_drop_note"] == 1 and r["harv_drop_fill"] == 0)
        cw = sum(1 for r in g if r["harv_drop_note"] == 0 and r["harv_drop_fill"] == 1)
        print(f"{s:8s} n={len(g):3d} chance={1/KCARD[s]:.2f} slotlen note/fill="
              f"{mn('slotlen_note'):.0f}/{mn('slotlen_fill'):.0f} | "
              f"full {mn('full_fill'):.2f}->{mn('full_note'):.2f} | "
              f"selD {mn('sel_drop_fill'):.2f}->{mn('sel_drop_note'):.2f} | "
              f"harvD {mn('harv_drop_fill'):.2f}->{mn('harv_drop_note'):.2f} "
              f"(disc +{bw}/-{cw})")
    json.dump({"rows": rows, "kcard": KCARD}, open(args.out, "w"), indent=1)
    print("NK2_DONE", flush=True)


if __name__ == "__main__":
    main()
