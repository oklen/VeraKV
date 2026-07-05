# Result status: what is officially verified vs locally run

| Status | Meaning | Which results |
|---|---|---|
| **Submitted for official verification** | Our best run's per-question answers are on the official AMA-Bench leaderboard queue (`submissions/verakv_ama_submission.jsonl`); the organizers re-judge with their own LLM-as-judge. Expect the verified score to differ from our local 0.6478 by roughly judge-agreement noise (±1–2pp). | FLSA+FLSB (0.6478 local; official-run headline 0.6466) |
| **Local run of the official harness** | Full official `run.py` end-to-end (memory build → retrieval → answer → Qwen3-32B judge) on our GPUs, unmodified except the disclosed, config-gated answer-instruction swap. Raw per-question predictions + judge scores released for every tag. | Everything in `results/manifest.json` |
| **Microbenchmark** | Stripped single-request prototype (KV-equivalence, efficiency, privacy probes) — parity/mechanism claims only, not deployment numbers. | `results/kv_equiv_576/`, `results/w5_privacy.json`, §Efficiency tables |

Judge caveat (disclosed in the paper): the official docs report Qwen3-32B is a lenient judge in absolute terms;
all leaderboard comparisons share that judge. An independent Llama-3.1-70B re-judge of our identical headline
predictions scores it higher (0.677, 84% item agreement) — supportive, not definitive.

Reproduce: `scripts/run_ama.sh` (needs 8×A100 + the models staged; see `ama/maxutil_run.sh` for the exact
serving/launch recipe). CIs: `analysis/cluster_ci.py` over `results/mu_merged_*.json`.
