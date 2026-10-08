# Environment (versions used for the results in this repo)

**Official AMA-Bench harness (text pipeline; `scripts/run_ama.sh`, `ama/maxutil_run.sh`)**

- Hardware: 8× NVIDIA A100 80GB per worker (vLLM serving 4×TP=2)
- Python 3.10.20
- torch==2.6.0, vllm==0.8.5.post1, transformers==4.51.3, tokenizers==0.21.4,
  safetensors==0.8.0, accelerate==1.14.0, numpy==2.2.6, triton==3.2.0, xformers==0.0.29.post2
- The last full-set batches (2026-07-08: REDEFF3B, PROTX3F, PROTX2F2, RESTRF4) ran on a second worker
  pair with vLLM 0.10.2; the paper keeps every same-batch table on one stack.
- Models: Qwen3-32B (reader + judge, Track B), Qwen3-Embedding-0.6B (embedding router),
  Llama-3.1-70B-Instruct (cross-family re-judge, `ama/configs/judge_llama70b.yaml`)

**KV experiments (`kvmemory/kv_*.py`, HF transformers backend `kvmemory/llm_hf.py`)**

- Hardware: A100/A800 80GB, one model copy per GPU (Qwen3-32B: one GPU for view-level runs, or
  `SPRAG_DEVICE_MAP=auto` across several)
- Python 3.12, torch==2.4.1 (CUDA 12.1 build), transformers==4.55.2, accelerate
- Earlier KV microbenchmarks (KV-equivalence 576, privacy probe) ran on the harness stack above
- Models: Qwen3-8B (mechanism subset, probes), Qwen3-32B (scale checks, official-harness bridge),
  Llama-3.1-8B-Instruct (harvest replication, cross-backbone gate, cross-family judge),
  Qwen3-Embedding-0.6B (hybrid router of the bridge)
- Official judge for the bridge answers (`ama/br_judge.py`, `ama/bg_judge.py`): AMA-Bench's
  `evaluate_batch` against a Qwen3-32B vLLM server (vllm==0.9.2, transformers==4.52.4, thinking on)

The KV sub-selection code paths handle both `DynamicCache` APIs (`.layers` vs
`.key_cache/.value_cache`), so nearby transformers versions work; the versions above are
the ones actually run.
