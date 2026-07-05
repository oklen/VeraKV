# Results manifest — tag → configuration → paper location

Every row is a run of the **official AMA-Bench harness** (Qwen3-32B reader + judge,
open-ended split). `full` = all 2,496 QA (both halves merged where two tags listed);
`h0`/`h1` = the two episode-halves (n=1,248); `SW` = SOFTWARE domain (n=432).
Raw per-question predictions + judge scores: `mu_merged_<TAG>.json`.
Run manifests (n / accuracy / bootstrap CI): `mu_<TAG>_done`.

## Factorial (paper Table: memory × reader)

| Tag(s) | Memory | Reader prompt | Split | Acc | Paper |
|---|---|---|---|---|---|
| R0 | lexical | default | full | 0.5601 | factorial table |
| R1 | lexical | structured | full | 0.6154 | factorial table |
| R2 | hybrid | structured | full | 0.6294 | ladder / router table |
| FPA + FPB | flagship (causal+hybrid+pin) | default | full | **0.5954** | factorial table (memory-only headline) |
| FLSA + FLSB | flagship | structured | full | 0.6478 | reproduction of the 0.6466 official-run headline |

## Same-day anchors (h0, used by the ablations below)

| Tag | Config | Acc |
|---|---|---|
| REDEF | flagship, default prompt | 0.5970 |
| RESTR | flagship, structured | 0.6290 |
| FSANC | flagship, structured (earlier wave) | 0.6394 |

## Context-budget sweep (paper: "fill the budget")

| Tag | max_ctx_tokens | Acc |
|---|---|---|
| B6 / B6C | 6k | 0.5497 / 0.5425 |
| B14 | 14k | 0.6242 |
| B22 | 22k | 0.6442 |

## Payload ablations (paper: "verbatim, or nothing")

| Tag | What | Acc | Δ vs anchor |
|---|---|---|---|
| GIST | overview only, no verbatim appendix (k_rehydrate=0) | 0.5865 | −4.3 vs RESTR |
| LOSSY | verbatim appendix → LLM summaries (same steps) | 0.5777 | −5.1 vs RESTR |

## Reader-side mechanism ablations (paper Table: reader-side machinery)

| Tag | Mechanism | Acc | Δ |
|---|---|---|---|
| PLDEF | EvidencePlan + default reader | 0.5409 | −5.6 vs REDEF (anti-transfer) |
| PLSTR | EvidencePlan + structured reader | 0.6018 | −2.7 vs RESTR |
| DEC6 / DEC14 | separate sufficiency gate + re-retrieval @6k/@14k budget | 0.5385 / 0.6338 | −0.4 / +1.0 (cannot repair breadth) |
| TSW / CSW | deterministic lookup tools on SW / control | 0.4444 / 0.4375 | +0.7 ≈ 0 |
| QSW / CSW2 | quote-first instruction on SW / control | 0.3634 / 0.4630 | −10.0 (format is not the gap) |

(select / state / gated / gate-only rows of the paper table come from the earlier
wave whose merged files are on the experiment host; they ship in the next artifact push.)

## Structural address resolution (paper: robustness paragraph)

| Tag | Corruption | Acc | Δ vs RESTR |
|---|---|---|---|
| SHUF | every pin shifted ±3 steps | 0.6298 | ±0.0 |
| SHUFFAR | every pin → far-random wrong step | 0.6114 | −1.8 |

## Reproducibility notes

- Same-day replicate noise ≈ 0.5pp (B22 vs FSANC); cross-day ≈ 2pp; SOFTWARE batch noise ±4pp (CSW 0.4375 vs CSW2 0.4630 vs FLSA-batch 0.4769, same config) — compare within batch only.
- `analysis/cluster_ci.py` reproduces QA-level and episode-cluster CIs from these files.
- `analysis/pin_split.py` reproduces the step-pin mechanism signature (+2.5pp on step-citing questions, +0.0 elsewhere).

## KV-vs-text equivalence at scale (results/kv_equiv_576/, paper §Efficiency)

576 QA / 72 episodes (≤26k tokens, five domains; Qwen3-8B backbone, stripped
microbenchmark reader — parity is the claim, not absolute accuracy):
mask-oracle gate 24/24; kv 0.175 vs text 0.165 (+1.0pp, same-model judge);
+0.7pp under an independent Qwen3-32B re-grade; first-token agreement 81%.
`w4_ans_s*.jsonl` = per-QA answers for both arms; `w4_judged.jsonl` adds 32B grades.

## Privacy / deletion probe (results/w5_privacy.json, paper §Limitations)

Planted-secret probe (`kvmemory/kv_privacy.py`; Qwen3-8B, greedy, 40 trials per
condition). A secret credential is planted at one step, that step is erased, and
only post-secret steps are served — either as gathered KV from the
full-trajectory prefill (contextual carryover present) or as a compact
re-prefill of the same text (secret information-theoretically absent).
Deletion setting: kv arm leaks **0/120** across gap distances 1/3/10; the
secret's first-token logprob sits at the text arm's noise floor (≈−31 nats,
kv within 1.1 nats, direction *below* text). Positive control (secret restated
in a *selected* step) leaks in both arms (kv 120/120, tx 83/120), so the probe
is sensitive. The kv arm's higher positive-control recall confounds restatement
with carryover — we do not claim it as a fidelity advantage.
