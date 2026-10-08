"""Benchmark the framework's efficiency claim: compress-once-answer-many via KV reuse.

An AMA episode = one trajectory answering many QA. The query-independent arms (`full`, `gist`) build
the SAME context for every question, so their KV can be prefilled ONCE and reused across the
episode's QA — instead of re-prefilling the whole context per question (what a stateless server does).

For each episode this runs the QA two ways and reports the gap:
  reuse=True   prefill the trajectory KV once, crop-back between queries (this framework)
  reuse=False  re-prefill trajectory+question from scratch per query (the baseline cost)

It asserts the two produce token-identical answers (KV reuse must be lossless), then reports
wall-clock speedup and prefill-tokens saved. Accuracy is unchanged by construction — this measures
the efficiency, which is the data-independent contribution.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import AutoGistSummarizer
from kvmemory.core import KVMemory, KVMemoryConfig
from kvmemory.llm_hf import HFBackend
from kvmemory.run_replay import answer_prompt_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--ids", default="")
    ap.add_argument("--arm", default="full", choices=["full", "gist"])  # query-independent arms
    ap.add_argument("--max_tokens", type=int, default=16000)  # episode context cap (fits A3B)
    ap.add_argument("--min_qa", type=int, default=4)          # need amortization to be meaningful
    ap.add_argument("--max_qa", type=int, default=12)
    ap.add_argument("--max_ep", type=int, default=4)
    ap.add_argument("--ans_tokens", type=int, default=64)     # answer length (same for both modes)
    ap.add_argument("--hot_turns", type=int, default=4)
    ap.add_argument("--out", default="./out/bench_kv.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()  # compile CUDA kernels before any timed run (fair cold-start)
    head, tail = llm.split_wrap()
    summ = AutoGistSummarizer()

    ids = set(int(x) for x in args.ids.split(",") if x) if args.ids else None
    eps = load_episodes(args.data, max_tokens=args.max_tokens)
    if ids:
        eps = [e for e in eps if e.episode_id in ids]
    eps = [e for e in eps if len(e.qa) >= args.min_qa]
    eps = eps[:args.max_ep]
    print(f"benchmarking {len(eps)} episodes | arm={args.arm} ans_tokens={args.ans_tokens}", flush=True)

    rows = []
    for ep in eps:
        for s in ep.segments:
            s.n_tok = llm.count_tok(s.text)
            s.gist = summ.gist(s)
        mem = KVMemory(KVMemoryConfig(arm=args.arm, hot_turns=args.hot_turns),
                       summarizer=summ, count_tok=llm.count_tok)
        for s in ep.segments:
            mem.append(s)
        ctx = mem.assemble("")  # query-independent for full/gist → same prefix for every QA
        qas = ep.qa[:args.max_qa]
        prefix_text = head + answer_prompt_parts(ep.task, ctx, qas[0]["question"])[0]
        suffixes = [answer_prompt_parts(ep.task, ctx, qa["question"])[1] + tail for qa in qas]

        ans_r, tr = llm.answer_many(prefix_text, suffixes, max_tokens=args.ans_tokens, reuse=True)
        ans_b, tb = llm.answer_many(prefix_text, suffixes, max_tokens=args.ans_tokens, reuse=False)
        match = sum(a == b for a, b in zip(ans_r, ans_b))

        ttft_speedup = tb["t_ingest_mean"] / tr["t_ingest_mean"] if tr["t_ingest_mean"] else 0.0
        e2e_speedup = tb["t_total"] / tr["t_total"] if tr["t_total"] else 0.0
        tok_saved = tb["prefill_tok_processed"] - tr["prefill_tok_processed"]
        row = {
            "episode_id": ep.episode_id, "domain": ep.domain, "n_qa": len(qas),
            "prefix_tok": tr["prefix_tok"], "match": match,
            "base_ttft": round(tb["t_ingest_mean"], 3),         # baseline TTFT = re-prefill the context
            "reuse_ttft": round(tr["t_ingest_mean"], 3),        # reuse TTFT = encode the short suffix
            "decode_mean": round(tb["t_decode_mean"], 3),       # shared decode (mode-independent)
            "reuse_setup": round(tr["t_setup"], 2),             # one-time prefix prefill
            "base_t_total": round(tb["t_total"], 2),
            "reuse_t_total": round(tr["t_total"], 2),
            "ttft_speedup": round(ttft_speedup, 1),
            "e2e_speedup": round(e2e_speedup, 2),
            "prefill_tok_base": tb["prefill_tok_processed"],
            "prefill_tok_reuse": tr["prefill_tok_processed"],
            "prefill_tok_saved": tok_saved,
        }
        rows.append(row)
        print(f"EP {ep.episode_id} {ep.domain} | {len(qas)} QA | prefix {row['prefix_tok']}t | "
              f"match {match}/{len(qas)} | TTFT {row['base_ttft']}s→{row['reuse_ttft']}s "
              f"({row['ttft_speedup']}x) | decode {row['decode_mean']}s/q | "
              f"e2e {row['base_t_total']}s→{row['reuse_t_total']}s ({row['e2e_speedup']}x) | "
              f"prefill-tok saved {tok_saved:,}", flush=True)
        json.dump(rows, open(args.out, "w"), indent=2)

    if rows:
        n_ep = len(rows)
        tot_match = sum(r["match"] for r in rows)
        tot_qa = sum(r["n_qa"] for r in rows)
        mean_ttft = sum(r["base_ttft"] for r in rows) / n_ep, sum(r["reuse_ttft"] for r in rows) / n_ep
        tot_saved = sum(r["prefill_tok_saved"] for r in rows)
        agg_e2e = sum(r["base_t_total"] for r in rows) / max(1e-9, sum(r["reuse_t_total"] for r in rows))
        agg_ttft = mean_ttft[0] / max(1e-9, mean_ttft[1])
        print(f"\n=== AGGREGATE ({n_ep} eps, {tot_qa} QA, arm={args.arm}, ans_tokens={args.ans_tokens}) ===")
        print(f"exact-string answer match (reuse vs re-prefill): {tot_match}/{tot_qa}  "
              f"(<100% = bf16 greedy nondeterminism, split vs joint prefill; see kv_correctness.py)")
        print(f"per-query TTFT: {mean_ttft[0]:.2f}s (re-prefill) → {mean_ttft[1]:.2f}s (reuse) = {agg_ttft:.1f}x faster")
        print(f"end-to-end (incl. shared decode): {agg_e2e:.2f}x faster")
        print(f"prefill tokens saved (sum over {tot_qa} QA): {tot_saved:,}")
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
