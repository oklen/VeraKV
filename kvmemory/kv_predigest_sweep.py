"""kv_predigest_sweep.py -- WHEN does harvested KV carry a usable pre-computed conclusion?
(user-designed sweep, 2026-07-16). kv_predigest found a flat 0 at the hardest corner
(exact 3-digit value + copy-pointer + source dropped). This sweeps the two axes that should
move the boundary, all under the same "drop the source event" structure:

  answer entropy:  bin2 (K=2) < dir4 (K=4) < name8 (K=8) < num (K~=900)
  downstream type: copy-pointer ("DST copied from SRC")  vs  computed verdict
                   ("operator reviewed SRC against threshold" -- verdict NOT in the text)

Per setting: decoy@step0, SRC-defining event@middle (DROPPED in drop_mid), downstream@last.
Arms x {complete, drop_mid}: sel_txt (fresh compact), sel_ikv (independent KV, orig pos),
harv_kv (gathered from one full-trajectory prefill). Decisive: harv_kv(drop) vs sel(drop),
read against the 1/K chance floor. Where harv rises above chance and above sel = where the
memoized conclusion is recoverable.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_predigest_sweep --items 16 --seed 500 --out ./out/pds.json
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
DIR4 = ["NORTH", "SOUTH", "EAST", "WEST"]
MODE8 = ["IDLE", "BUSY", "LOCKED", "PAUSED", "SEALED", "PINNED", "FLUSHED", "DRAINED"]
SETTINGS = ["bin2", "dir4", "name8", "num", "verdict"]
KCARD = {"bin2": 2, "dir4": 4, "name8": 8, "num": 900, "verdict": 2}


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
    elif setting in ("bin2", "dir4", "name8"):
        pool = {"bin2": STATUS2, "dir4": DIR4, "name8": MODE8}[setting]
        gm, dm = rng.sample(pool, 2)
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
    a_idx, c_idx = 0, n_events - 1
    m = rng.randrange(n_events // 3, (2 * n_events) // 3)
    special = {a_idx: dec, m: src, c_idx: dst}
    others = [n for n in NAMES if n not in (D, S, T)]
    ev = []
    for i in range(n_events):
        body = special.get(i) or \
            f"register {rng.choice(others)} refreshed to {rng.randrange(100, 999)}"
        ev.append(f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n")
    return ev, q, gold, decoy, {"complete": [a_idx, m, c_idx], "drop_mid": [a_idx, c_idx]}


def match(o, ans):
    if ans.isdigit():
        return int(ans in re.findall(r"\d+", o))
    return int(ans.lower() in o.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default="./out/pds.json")
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
            ev, q, gold, decoy, subsets = gen_item(rng, args.n_events, setting)
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
            spans = []
            cur = H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            rec = {"setting": setting, "it": it}

            def sc(o, key, drop=False):
                rec[key] = match(o, gold)
                if drop:
                    rec[key + "_dec"] = match(o, decoy)

            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev) + q + tail), 12)
            sc(o, "full_txt")
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            for cond, Sset in subsets.items():
                drop = cond == "drop_mid"
                o, _, _ = llm._greedy(DynamicCache(), 0,
                                      llm._ids(head_full + "".join(ev[i] for i in Sset)
                                               + q + tail), 12)
                sc(o, f"sel_txt_{cond}", drop)
                blocks = [(H_kv, list(range(H)))]
                for i in Sset:
                    kv = encode_block(llm, H_ids + eids[i],
                                      list(range(H)) + list(range(*spans[i])), keep_a=H)
                    blocks.append((kv, list(range(*spans[i]))))
                c, p = assemble(llm, blocks)
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                sc(o, f"sel_ikv_{cond}", drop)
                crows = [pp for i in Sset for pp in range(*spans[i])]
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                sc(o, f"harv_kv_{cond}", drop)
            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[pds] {setting} it{it} full={rec['full_txt']} "
                  f"| C sel={rec['sel_txt_complete']} harv={rec['harv_kv_complete']} "
                  f"| DROP sel={rec['sel_txt_drop_mid']} ikv={rec['sel_ikv_drop_mid']} "
                  f"harv={rec['harv_kv_drop_mid']}", flush=True)

    print("\n===== SWEEP =====")
    for setting in SETTINGS:
        g = [r for r in rows if r["setting"] == setting]
        def mn(k):
            return sum(r[k] for r in g) / len(g)
        print(f"{setting:8s} K={KCARD[setting]:3d} chance={1/KCARD[setting]:.2f} n={len(g)} | "
              f"full={mn('full_txt'):.2f} || COMPLETE sel={mn('sel_txt_complete'):.2f} "
              f"harv={mn('harv_kv_complete'):.2f} || DROP sel_txt={mn('sel_txt_drop_mid'):.2f} "
              f"sel_ikv={mn('sel_ikv_drop_mid'):.2f} harv={mn('harv_kv_drop_mid'):.2f} "
              f"(harv_dec={mn('harv_kv_drop_mid_dec'):.2f})")
    json.dump({"rows": rows, "kcard": KCARD}, open(args.out, "w"), indent=1)
    print("PDS_DONE", flush=True)


if __name__ == "__main__":
    main()
