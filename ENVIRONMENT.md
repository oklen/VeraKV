# Environment (exact versions used for every result in this repo)

- Hardware: 8× NVIDIA A100 80GB PCIe per worker (vLLM serving 4×TP=2; probes single-GPU)
- Python 3.10.20
- torch==2.6.0, vllm==0.8.5.post1, transformers==4.51.3, tokenizers==0.21.4,
  safetensors==0.8.0, accelerate==1.14.0, numpy==2.2.6, triton==3.2.0, xformers==0.0.29.post2
- Models: Qwen3-32B (AMA reader+judge, official Track B), Qwen3-8B (KV/privacy microbenchmarks),
  Qwen3-Embedding-0.6B (embedding router), Llama-3.1-8B/70B-Instruct (cross-family re-judging)
- Serving recipe (ports, TP, sharding, env): `ama/maxutil_run.sh`

The KV sub-selection code paths handle both `DynamicCache` APIs (`.layers` vs
`.key_cache/.value_cache`), so nearby transformers versions work; the versions above are
the ones actually run.
