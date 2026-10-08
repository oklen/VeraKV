"""kv_pfx.py -- the fair within-window efficiency baseline the review asked for: a RESIDENT
FULL-TRAJECTORY PREFIX CACHE (standard prefix caching) also re-prefills only the question, so
the honest comparison triple per query is:

  tx    re-prefill [header + selected evidence as text + question]   (the paper's fig baseline)
  sub   resident full cache -> gather E selected rows -> prefill question over E rows (ours)
  pfx   resident full cache -> prefill question over ALL N rows       (prefix-cache baseline)

Sub-selection's saving vs pfx is the attention context (E vs N) at question-prefill and
decode, not the trajectory prefill. Measured: TTFT (question feed -> first token) and decode
s/tok over 32 tokens, CUDA-synchronized, rep-0 warmup + median of 3 measured reps.
Accuracy is NOT re-measured here (pfx serving == the full arm, already reported).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_pfx --out ./out/px.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_globhot import gather_rows
from kvmemory.kv_ow import prefill_chunked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=12)
    ap.add_argument("--max_qa", type=int, default=4)
    ap.add_argument("--ks", default="2,5,8,12")
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--dec_tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="./out/px.json")
    args = ap.parse_args()
    KS = [int(x) for x in args.ks.split(",")]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())

    rows = []

    def timed_greedy_pos(cache, pos, qids, ntok):
        torch.cuda.synchronize()
        t0 = time.time()
        out, tft, ttot = llm._greedy_pos(cache, pos, qids, ntok)
        torch.cuda.synchronize()
        wall = time.time() - t0
        return out, wall, tft, ttot

    for ep in eps:
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        all_seg_ids, spans, cur = [], [], H
        for txt in segment_texts:
            sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
            all_seg_ids.append(sids)
            spans.append((cur, cur + len(sids)))
            cur += len(sids)
        total = cur
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts, chunk=4096)
        assert ftot == total

        for qa in ep.qa[: args.max_qa]:
            q = qa["question"]
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            for k in KS:
                picked = {id2idx[p] for p in router.select(q, old, k) if p in id2idx}
                kept = sorted(hot_idx | picked)
                etok = sum(spans[i][1] - spans[i][0] for i in kept)
                idx = list(range(H)) + [p for i in kept
                                        for p in range(spans[i][0], spans[i][1])]
                text_sel = header + "".join(segment_texts[i] for i in kept) + qtext

                meas = {"tx": [], "sub": [], "pfx": [], "sub_gather": []}
                dec = {"tx": [], "sub": [], "pfx": []}
                for rep in range(args.reps + 1):
                    # tx: fresh full prefill of evidence text + question
                    torch.cuda.synchronize(); t0 = time.time()
                    _, tft, ttot = llm._greedy(DynamicCache(), 0, llm._ids(text_sel),
                                               args.dec_tokens)
                    torch.cuda.synchronize()
                    if rep:
                        meas["tx"].append(tft)
                        dec["tx"].append((ttot - tft) / max(1, args.dec_tokens - 1))
                    # sub: gather + question over E rows
                    torch.cuda.synchronize(); t0 = time.time()
                    c = gather_rows(llm, full_cache, idx)
                    torch.cuda.synchronize()
                    g = time.time() - t0
                    p = torch.tensor(idx, dtype=torch.long, device=llm.device)
                    _, tft, ttot = llm._greedy_pos(c, p, qids, args.dec_tokens)
                    torch.cuda.synchronize()
                    del c
                    if rep:
                        meas["sub"].append(g + tft)
                        meas["sub_gather"].append(g)
                        dec["sub"].append((ttot - tft) / max(1, args.dec_tokens - 1))
                    # pfx: question over ALL N rows of the resident cache
                    fp = torch.arange(total, device=llm.device)
                    _, tft, ttot = llm._greedy_pos(full_cache, fp, qids, args.dec_tokens)
                    full_cache.crop(total)
                    torch.cuda.synchronize()
                    if rep:
                        meas["pfx"].append(tft)
                        dec["pfx"].append((ttot - tft) / max(1, args.dec_tokens - 1))
                rows.append({
                    "episode_id": ep.episode_id, "total": total, "k": k, "etok": etok,
                    "ttft_tx": st.median(meas["tx"]), "ttft_sub": st.median(meas["sub"]),
                    "ttft_pfx": st.median(meas["pfx"]),
                    "gather": st.median(meas["sub_gather"]),
                    "dec_tx": st.median(dec["tx"]), "dec_sub": st.median(dec["sub"]),
                    "dec_pfx": st.median(dec["pfx"]),
                })
                print(f"ep{ep.episode_id} N={total} k={k} E={etok} | "
                      f"ttft tx {rows[-1]['ttft_tx']*1e3:.0f}ms sub "
                      f"{rows[-1]['ttft_sub']*1e3:.0f}ms pfx {rows[-1]['ttft_pfx']*1e3:.0f}ms"
                      f" | dec sub {rows[-1]['dec_sub']*1e3:.1f} pfx "
                      f"{rows[-1]['dec_pfx']*1e3:.1f} ms/tok", flush=True)
        del full_cache
        torch.cuda.empty_cache()
        json.dump(rows, open(args.out, "w"))
    json.dump(rows, open(args.out, "w"))
    print(f"PFX_DONE n={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
