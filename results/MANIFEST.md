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

## Wave-6 (payload family completion + full-context control + SOFTWARE loop)

| Tag | What | Split | Acc | Δ (same-batch anchor) |
|---|---|---|---|---|
| EXTLW | appendix → query-relevant ORIGINAL lines (deterministic, verbatim) | h0 | 0.6154 | −2.6 vs RESTR2 — paraphrase, not compression, is the poison |
| FACTSW | appendix → extracted atomic facts (Mem0-style) | h0 | 0.5825 | −5.9 vs RESTR2 (≤ gist-only) |
| FULLTR | full trajectory, recency-truncated @22k (arm=full) | h0 | 0.5577 | −8.4 vs RESTR2 |
| LOOPSW | iterative ReAct lookup loop REPLACING the answer pass | SW-432 | 0.3773 | −9.0 vs CSW3 |
| LOOPS2 | same loop as EVIDENCE-GATHERER, structured answer pass kept | SW-432 | 0.4676 | ±0.0 vs CSW3 (0.4676) — iteration adds nothing |
| CSW3 | plain structured control (same batch as the loop runs) | SW-432 | 0.4676 | — |
| ADAPT | type-routed specialized instructions (rules from 26-case study; analysis/adapt_case_study.md) | h0 | 0.6122 | −3.0 vs RESTR2; per-rule: R6(identical default, n=913) −0.3 replicates anchor, routed classes R1 −6.8 / R2 −3.4 / R3 +12.5(n=16) / R4 −23.3 / R5 −8.0 — every fragment ablates the derive→cite→compose loop and loses |
| RESTR2 | flagship structured anchor for this wave (RESTR drifted 0.6290→0.6418, ~known cross-day noise) | h0 | 0.6418 | anchor |

## RAMP MVP (flip-set probe: can memory PRESENTATION absorb the prompt gain?)

Flip set = 316 F+ questions (default-wrong & structured-right on the full runs) + 299
both-right stability sample; both arms same batch, DEFAULT reader (`analysis/flip_mvp.py`).

| Tag | Arm | F+ recovery | S retention |
|---|---|---|---|
| PKTD | appendix → reasoning-affordant view (need groups / occurrences / timelines, no conclusions) | 0.326 | 0.786 |
| RFLIP | raw appendix control (spontaneous regression-to-mean) | 0.323 | 0.920 |

| PKT2H0 | v2 packet + guard, FULL h0 deployment | h0 | 0.5617 | −0.2 vs REDEF2 (same batch) = exactly null; serve rate 31% |
| REDEF2 | raw + default same-batch anchor (REDEF drifted 0.597→0.564: default reader cross-day noise > structured's) | h0 | 0.5641 | anchor |

Verdict: the view recovers nothing (+0.3pp over 32% spontaneous) and breaks 13.4pp of
previously-correct questions — its missing-evidence panels assert appendix-scoped absences as
data-wide facts and the reader trusts them over the overview (plan-mode inheritance in a new
guise). Presentation does not transfer; the instruction's value stays in the answering pass.

**v2 (PKTD2): hardened compiler + deterministic verbatim-containment guard.** Guard rejected 69%
of compilations (instructions alone do not enforce fidelity — fabricated/mis-attributed quotes
persist under explicit verbatim rules). Surviving packets, net of same-batch re-roll churn
(measured on the fallback subset, both arms raw): **+0.8pp on F+ / −5.3pp on S** — fidelity can
be guarded, usefulness cannot be compiled. Case-1 fabrication mechanism (available-actions menu
promoted to executed events) audited against the raw trajectory in `analysis/ramp_case_study.md`.

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

**Deployed-layout variant (results/kv_equiv_deployed/):** the KV arm serves the
deployed overview+appendix layout (overview prefilled once per episode as a
post-trajectory block at fixed positions; selected spans gathered; only the
question fresh). 576 QA, ≤24k, four domains: mask-oracle gate 24/24, arms tied
within noise (kv 0.222 vs tx 0.240, ±5pp band, 90% first-token agreement); the
overview lifts both arms ~5 points over the compact microbenchmark.
`kv_equiv500.py --layout deployed` reproduces it.

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
