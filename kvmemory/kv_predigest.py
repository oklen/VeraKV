"""kv_predigest.py -- does harvested KV carry a *usable* pre-computed conclusion when the
served subset DROPS the intermediate that introduced it? (user-designed, 2026-07-16).

This removes the ceiling confound of kv_vardecomp: there, perfect routing to the exact
chain made the selected task trivial, so pre-digestion had no headroom. Here the served
subset is deliberately INCOMPLETE -- the step that introduced the gold value is dropped --
so the gold appears in NO served token, and only a row that memoized it during a full joint
prefill can still deliver it.

Structure (far):
  step 0        : register DECOY initialized to <dval>      (decoy number, a distractor value)
  step m (mid)  : register SRC   initialized to <gval>      (GOLD source; dropped in drop_mid)
  step n-1      : register DST   copied from register SRC    (memoizes SRC during full prefill)
  Q: value of register DST?   gold = gval   (gval appears ONLY at step m)

Served subsets: complete = {0, m, n-1} ; drop_mid = {0, n-1}  (SRC's defining step removed).
Arms (each subset S): sel_txt(S) fresh compact text ; sel_ikv(S) independent KV at original
positions ; harv_kv(S) rows gathered from one joint prefill of the full trajectory. Plus
full_txt (all events, distracted). Decisive: harv_kv(drop_mid) vs sel_txt/sel_ikv(drop_mid)
--- sel cannot produce gval (absent from served tokens); a memoized harvested row can.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_predigest --items 24 --seed 400 --out ./out/pd.json
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


def numset(s):
    return set(re.findall(r"\d+", s))


def gen_item(rng, n_events, far):
    decoy, src, dst = rng.sample(NAMES, 3)
    dval = rng.randrange(100, 999)
    gval = rng.randrange(100, 999)
    while gval == dval:
        gval = rng.randrange(100, 999)
    a_idx, c_idx = 0, n_events - 1
    m = rng.randrange(n_events // 3, (2 * n_events) // 3) if far else n_events - 2
    special = {a_idx: f"register {decoy} initialized to {dval}",
               m: f"register {src} initialized to {gval}",
               c_idx: f"register {dst} copied from register {src}"}
    fillers = [n for n in NAMES if n not in (decoy, src, dst)]
    ev = []
    for i in range(n_events):
        body = special.get(i) or \
            f"register {rng.choice(fillers)} refreshed to {rng.randrange(100, 999)}"
        ev.append(f"<step {i+1}>\naction: reg_op()\nobservation: {body}\n")
    q = (f"\n\nQuestion: What is the value of register {dst}? "
         "Answer with the number only:")
    return ev, q, str(gval), str(dval), {"complete": [a_idx, m, c_idx],
                                         "drop_mid": [a_idx, c_idx]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_events", type=int, default=24)
    ap.add_argument("--items", type=int, default=24)
    ap.add_argument("--seed", type=int, default=400)
    ap.add_argument("--out", default="./out/pd.json")
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
    for far in (1,):
        for it in range(args.items):
            ev, q, gold, dval, subsets = gen_item(rng, args.n_events, bool(far))
            eids = [list(llm.tok(t, add_special_tokens=False).input_ids) for t in ev]
            spans = []
            cur = H
            for e in eids:
                spans.append((cur, cur + len(e)))
                cur += len(e)
            total = cur
            flat = [t for e in eids for t in e]
            maxseq = max(maxseq, total)
            rec = {"far": far, "it": it, "gold": gold, "dval": dval}

            def score(o, key, drop=False):
                rec[key] = int(gold in o)
                rec[key + "_rec"] = int(gold in numset(o))
                if drop:
                    rec[key + "_dec"] = int(dval in numset(o))

            # full_txt (all events, distracted)
            o, _, _ = llm._greedy(DynamicCache(), 0,
                                  llm._ids(head_full + "".join(ev) + q + tail), 12)
            score(o, "full_txt")
            # one joint prefill reused for both harvest subsets
            full_kv = encode_block(llm, H_ids + flat, list(range(total)), keep_a=H)
            for cond, S in subsets.items():
                drop = cond == "drop_mid"
                # sel_txt: fresh compact text of the subset
                o, _, _ = llm._greedy(DynamicCache(), 0,
                                      llm._ids(head_full + "".join(ev[i] for i in S)
                                               + q + tail), 12)
                score(o, f"sel_txt_{cond}", drop)
                # sel_ikv: each subset event encoded independently, original positions
                blocks = [(H_kv, list(range(H)))]
                for i in S:
                    kv = encode_block(llm, H_ids + eids[i],
                                      list(range(H)) + list(range(*spans[i])), keep_a=H)
                    blocks.append((kv, list(range(*spans[i]))))
                c, p = assemble(llm, blocks)
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                score(o, f"sel_ikv_{cond}", drop)
                # harv_kv: gather subset rows from the joint prefill, original positions
                crows = [pp for i in S for pp in range(*spans[i])]
                rt = torch.tensor([pp - H for pp in crows], dtype=torch.long)
                sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
                c, p = assemble(llm, [(H_kv, list(range(H))), (sub, crows)])
                o, _, _ = llm._greedy_pos(c, p, llm._ids(q + tail), 12)
                del c
                score(o, f"harv_kv_{cond}", drop)
            del full_kv
            torch.cuda.empty_cache()
            rows.append(rec)
            print(f"[pd] it{it} full={rec['full_txt']} "
                  f"| complete sel_txt={rec['sel_txt_complete']} "
                  f"ikv={rec['sel_ikv_complete']} harv={rec['harv_kv_complete']} "
                  f"| DROP sel_txt={rec['sel_txt_drop_mid']} "
                  f"ikv={rec['sel_ikv_drop_mid']} harv={rec['harv_kv_drop_mid']}", flush=True)

    keys = ["full_txt"] + [f"{a}_{c}" for c in ("complete", "drop_mid")
                           for a in ("sel_txt", "sel_ikv", "harv_kv")]
    summ = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    json.dump({"rows": rows, "summary": summ, "maxseq": maxseq, "H": H}, open(args.out, "w"), indent=1)
    print(f"\nMAXSEQ={maxseq} (H={H})")
    print("== COMPLETE (control) ==", {k: f"{summ[k]:.2f}" for k in keys if "complete" in k})
    print("== DROP_MID (decisive) ==", {k: f"{summ[k]:.2f}" for k in keys if "drop" in k})
    print(f"full_txt={summ['full_txt']:.2f}")
    print("PD_DONE", flush=True)


if __name__ == "__main__":
    main()
