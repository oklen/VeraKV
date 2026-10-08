"""Efficiency curves (review #10): KV-cache STORAGE cost + per-query TTFT / decode-throughput / peak GPU
memory as the SELECTED-KV grows to 64k tokens, on a single 80GB A100. Table 7 stopped at 4.4k and reported
only TTFT; the reviewer asks how far the flat-TTFT holds and what memory it costs. We prefill one long
trajectory once (YaRN-extended context), then per target select a prefix summing to ~4k/16k/32k/64k tokens
and compare:
  * text path : re-prefill header + selected text, time to first token (grows with selected tokens)
  * sub-select: gather the selected spans' cached KV + decode the question (flat TTFT, but decode attends
                over the selected KV -> peak memory and decode cost grow with selected-KV length)
We also report the analytic cache cost (bytes/token from the model config). Production paged-attention /
serving comparison is out of scope for this research prototype (stated as future work).

  SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ROPE_FACTOR=4.0 SPRAG_MAX_CTX=100000 PYTHONPATH=. \
    CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_efficiency --n_steps 2200
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend  # noqa: E402
from kvmemory.kv_select_smoke import build_trajectory, build_full_ids, split_wrap_nothink  # noqa: E402


def cache_bytes_per_token(model):
    c = model.config
    L = c.num_hidden_layers
    kvh = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    return 2 * L * kvh * hd * 2  # K + V, bf16 (2 bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="4000,16000,32000,64000")
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--n_steps", type=int, default=2200)
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)

    bpt = cache_bytes_per_token(llm.model)
    print(f"MODEL {os.environ.get('SPRAG_MODEL_PATH','?')}  attn={os.environ.get('SPRAG_ATTN_IMPL','default')}"
          f"  rope_factor={os.environ.get('SPRAG_ROPE_FACTOR','1')}", flush=True)
    print(f"CACHE COST: {bpt/1024:.1f} KB/token  |  {bpt/1e9*1000:.3f} GB per 1k tokens  |  "
          f"{bpt/1e9*64000:.1f} GB for 64k tokens", flush=True)

    targets = [int(x) for x in args.targets.split(",")]
    steps = build_trajectory(args.n_steps)
    header = head + "You are reviewing a completed agent trajectory. Answer precisely.\n\nTrajectory:\n"
    full_cache, spans, total = llm.prefill_full(steps, header)
    header_len = spans[0][0]
    print(f"prefilled {len(steps)} steps = {total} tokens (need >= {max(targets)})", flush=True)

    q = "\n\nQuestion: summarize what happened.\nAnswer concisely:" + tail
    qids = llm._ids(q)

    print(f"\n{'sel_tok':>8} {'ttft_text_ms':>12} {'ttft_sub_ms':>11} {'speedup':>7} "
          f"{'dec_tok/s':>9} {'peak_GB':>7}", flush=True)
    for tgt in targets:
        kept, acc = [], 0
        for i, (s, e) in enumerate(spans):
            if i == 0:
                continue  # header span
            kept.append(i)
            acc += (e - s)
            if acc >= tgt:
                break
        if acc < tgt * 0.8:
            print(f"{acc:8d}  (trajectory too short for target {tgt}; skip)", flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
        _, ttft_sub, tdec = llm._greedy_pos(sub_cache, kpos, qids, args.decode)
        peak = torch.cuda.max_memory_allocated() / 1e9
        dtps = args.decode / tdec if tdec > 0 else 0.0
        cids = build_full_ids(llm, header, steps, kept, q)
        _, ttft_text, _ = llm._greedy(DynamicCache(), 0, cids, 1)
        sp = ttft_text / ttft_sub if ttft_sub > 0 else 0.0
        print(f"{acc:8d} {ttft_text*1000:12.0f} {ttft_sub*1000:11.0f} {sp:7.1f} {dtps:9.1f} {peak:7.1f}",
              flush=True)
    print("EFF_DONE", flush=True)


if __name__ == "__main__":
    main()
