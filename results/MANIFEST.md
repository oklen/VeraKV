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
| FLSA + FLSB | flagship | structured | full | 0.6478 | the submitted entry (leaderboard-verified 0.6478) |

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

The select / state / gated / gate-only / decouple rows of the paper table come from the 2026-07-04
reader-side wave; their merged files are now included (section "Reader-side wave" below).

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

## Goal investigation: why memory-side reasoning fails (the relay ladder)

All same-batch h0; anchors RESTR2 (structured, 0.6400 joined) / REDEF2 (default, 0.5617).
Diagnosis chain in `analysis/plan_poisoning_case_study.md`.

| Tag | Arm | Acc | P(right \| structured-right) | poisoned-pool (88) |
|---|---|---|---|---|
| PLAN2 | few-shot-hardened planner (anti-absence, modality-mirroring, full budget) + default reader | 0.5400 | — | 46/88 — halves plan poisoning (−5.6→−2.1), sign unchanged |
| HOFA | memory side runs FULL structured pass; artifact (99% = answer+evidence, derivation locked in <think>) + default reader | 0.5666 | 0.797 | 57/88 |
| HOFS | intended answer-stripped arm — strip emptied 98.7% of artifacts (answer-first format); de facto a third default re-roll anchor | 0.5811 | 0.793 | 58/88 |
| HOFT | HOFA + one-sentence trust-relay instruction | 0.5932 | 0.816 | 62/88 |

Ladder: no artifact 0.562 → answer-in-context 0.567 → answer+trust 0.593 → structured-in-place
0.640. Monotone in how much of the answering pass is restored to the reader; saturates only at
all of it. Three independent blockers: (1) unaccountable generation reasons wrongly (five faces);
(2) accountable generation locks its derivation in the thinking block (not exportable);
(3) even the correct answer in context is barely picked up (+0.4pp; +3.2 with explicit trust).

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

## Export-tax probe: forcing the derivation out of the thinking block (analysis/export_tax_probe.md)

Small-set screening wave (n=144 QA, 12 episodes stratified 2/domain from h0, all
arms same-batch on identical questions; ±7pp per-arm CIs — directional, not
headline). Follow-up to the relay ladder: prompt+few-shot CAN export the
derivation (99% format compliance vs ~1% default) and a verify-with-named-flaw
reader instruction CAN drive adoption to 94% — but the export demand itself
taxes the answering pass **−7.7pp with no relay involved** (PRSXP 0.587 vs
PRSTR 0.664, answered in place; the only change is "your visible reply must lay
out the derivation"). Fingerprint: all-quotes-verbatim collapses to 19–23% on
derivation-demand arms (~50–57% elsewhere) — the model re-writes evidence from
memory of its thinking instead of re-copying; victims concentrate in
counterfactual/strategic types. Even a one-line citation demand nets −6.3;
verify-framing on a bare answer underperforms blanket trust; nothing beats the
dumbest relay (bare answer + trust, −2.8). Revised blocker-2 verdict: a
tax-free export does not exist — the thinking block is not hiding the
derivation, it is protecting it. Rows: `mu_merged_PR{STR,DEF,TRU,AV,X1,XC,XD,SXP}.json`
(0.664 / 0.671 / 0.636 / 0.615 / 0.608 / 0.601 / 0.573 / 0.587), artifact dumps
`sel_PR*_full.jsonl` (per-question q/art/up_ans/quotes_ok/ans).

**Full-scale confirmation (n=1248 h0, same-batch, 2026-07-06):** RESTR4 0.6354
(anchor, in band) / SXPH 0.5192 / HOFX 0.5304. Tax = **−11.8pp [−14.4,−9.1]**
paired (fixed 74 / broke 220) — overshoots the entire structured gain, landing
below every default anchor. Relay adds +1.1 [−1.1,+3.5] (faithful, zero-incl.);
HOFX ≪ HOFT 0.593. Mechanism replicates: 99% compliance / 95% adoption / 21%
verbatim quotes. Tax by type: B −20.6, A −11.1, D −9.1, C −6.0; on the F+ pool
0.722→0.411. Rows `mu_merged_{RESTR4,SXPH,HOFX}.json`, dumps
`sel_{SXPH,HOFX}_full.jsonl`, analysis `analysis/w8_verdict.py`. Integrated
into the paper's relay-ladder passage + law ("nor observed without corrupting
it").

**Decomposition correction (same day):** SXPH's answers were 3.5× shorter than
anchor (869→247 med chars) — completeness confound. Controls: SBRF (brevity-only,
no derivation) −7.5 [−10.1,−4.9] @389 chars; SXPF (export + complete-answer
demand) −5.3 [−8.0,−2.7] @513 chars; penalty monotone in reply compression; F+
pool loses identically under brevity-only vs full export (0.72→0.50 vs 0.51) →
the tax is carried by answer completeness, not derivation corruption; −5.3 =
upper bound on any pure observation effect. Paper passage + law clause
corrected ("its reply cannot be re-shaped"). Rows `mu_merged_{SBRF,SXPF}.json`,
dumps `sel_{SBRF,SXPF}_full.jsonl`, `analysis/w9_verdict.py`, `analysis/w8_audit.py`.

**Endgame wave (7/6-7/7 night):** SXPV −1.1 (export tax fully recovered at 786
chars) / RVERB +0.2 (flat above natural) → one compression curve; relay fixed
point 0.593 = reader-side compression (HOFV=HOFT2=0.5900 @522-666 chars);
completeness-protected relay HOFTW 0.6586 / HOFTW2 0.6635 / full-set HOFTWF
0.6631 (+1.7 [+0.1,+3.3] vs RESTRF 0.6454) — three zero-excluded, F+ full
transfer + F− repair; DVERB2 (default+completeness, prefix replaced) 0.6320
ties structured → gap = completeness constant + net-zero reasoning trade;
DMANI/DSCAF memory-side presentation nulls (length unmoved; worksheet guard
rejects 74%). Full detail: analysis/export_tax_probe.md Endgame section.

**Gate arm (HOFTC, 7/7):** trust+complete relay + named-step override gate =
0.6747 [0.649,0.701], +4.0 [+1.5,+6.5] vs RESTR4 — project best. Override rate
unchanged, precision up (F+ 0.767 > anchor 0.722; F− 0.554). Rows
mu_merged_HOFTC.json, dump sel_HOFTC_full.jsonl.

**Overnight finale (7/7):** full-set HOFTCF 0.6715 (+2.6 vs RESTRF) / DVERB2F
0.6414 / REDEFF 0.5809; Llama-8B re-grade preserves all orderings (relay +12.3,
completeness +12.7, curve monotone; xjudge_out{,2}.json); mini case rig 11×4
(minicase_out.txt); ROUTER AUDIT: deployed hybrid={lexical,model-pick} with
truncated pick-stubs, lexical-only beats it +2.7 [+0.5,+5.0] (RESTRLEX 0.6506 /
RESTREMB 0.6410 / RESTR5 0.6226) — paper corrected, recommended default = pin
over lexical.

## Added in the October 2026 release

### Reader-side wave (2026-07-04; paper App. "reader-side falsification map")

Same harness and judge; `h0`/`h1` halves. Agentic reader modes are set via `AMA_AGENTIC_MODE`
(`ama/agentic_reader.py`), run through `ama/maxutil_run.sh`.

| Tag(s) | Memory / reader mode | Split | Acc | Paper row |
|---|---|---|---|---|
| FSELA + FSELB | model-selected evidence compression (`select`) | full | 0.5401 / 0.5144 | evidence compression −3.0 |
| STA + STB | flagship, entity-state timeline scaffold (`state`) | full | 0.6178 / 0.6450 | timeline scaffold −1.6 vs FLSA+FLSB |
| FLGA + FLGB | flagship, sufficiency gate fused into the answer prompt (`gated`) | full | 0.6378 / 0.6258 | fused gate −1.6 |
| GOFA / GOLA / GOCA | escape clause only, never re-retrieves (`gateonly`) — flagship / lexical / lexical+pin | h0 | 0.6338 / 0.5978 / 0.6282 | escape clause alone −2.6 (GOFA vs FLSA) |
| DECA + DECB | flagship, gate as a separate call (`decouple`) | full | 0.6514 / 0.6442 | separate gate ±0.0 |
| FSTRA / FGATEA / FGATEB | lexical router: structured base / fused gate (h0, h1) | h0, h1 | 0.5929 / 0.6058 / 0.5994 | re-retrieval under a weak router +1.3 (FGATEA vs FSTRA) |
| CSA, CSB / CGA, CGB | lexical+pin: structured base / fused gate | h0, h1 | 0.6370, 0.6530 / 0.6362, 0.6346 | gate attribution on lexical+pin |
| TH0 | deterministic lookup tools, full domain mix (`tools`) | h0 | 0.6426 | tools (indicative; SW contrast is TSW vs CSW) |
| SWRECALL | flagship, structured, gate-only reader with step-recall logging (asked step present 268/268) | SW | 0.4259 | SOFTWARE dissection: not retrieval (App. E) |
| SWBASE / SWSEL / SWSEL2 / SWAG / SW3 | further SOFTWARE-domain development probes of the reader modes (SW3: one shard missing, n=324) | SW | 0.4282 / 0.3889 / 0.3727 / 0.3588 / 0.3889 | not cited by number |
| PROBE / PROBEG / PROBEP / PROBES / STPROBE | 48-QA smoke probes of the reader modes | 48 QA | 0.52–0.60 | development only |
| EXTLW / FACTSW / FULLTR | appendix → original lines / extracted facts; full trajectory recency-truncated (`cfg_full22.json`) | h0 | 0.6154 / 0.5825 / 0.5577 | Table "family" (vs RESTR2 / RESTR) |
| LOOPSW / LOOPS2 / CSW3 | iterative lookup loop replacing the answer pass / as evidence gatherer / same-batch control | SW | 0.3773 / 0.4676 / 0.4676 | loop rows (−9.0 / ±0.0) |

### Memory assembly 2×2 (paper App. D)

Full set, structured reader, lexical router, same batch. `analysis/assembly_2x2.py`.

| Tag | Routed verbatim selection | Gist overview | Config | Acc | SOFTWARE |
|---|---|---|---|---|---|
| ASM_RECENT | – | – | `ama/configs/cfg_arm_recent.json` | 0.6102 | 0.4537 |
| ASM_GIST | – | ✓ | `cfg_arm_gist.json` | 0.6042 | 0.4375 |
| ASM_RETRIEVAL | ✓ | – | `cfg_arm_retrieval.json` | 0.6114 | 0.4815 |
| ASM_KVLEX | ✓ | ✓ | `cfg_kvmem_lex.json` | 0.6130 | 0.4792 |

### Router audit tags (paper App. F)

`RESTRLEXF` = lexical + step-pin (`cfg_kvmem_causal.json`), structured, full set: 0.6466 (2026-07-07 batch;
paired with `RESTRF2` in the same batch: −0.8pp [−2.3, +0.8]).
`RESTRF4` = model-pick hybrid + step-pin (`cfg_flagship.json`), structured, full set: 0.6410 (2026-07-08,
upgraded serving stack; not paired with `RESTRLEXF`).
`RESTRF` / `RESTRF2` / `RESTRF3` = same-config replicates of the flagship configuration
(0.6454 / 0.6538 / 0.6466). `RESTRF3` and `RESTRLEXF` score the same 0.6466 by coincidence; they are different
configurations.

### Judge robustness (paper App. "Judge robustness")

`judge_llama70b/ama_rejudged.json`: per-question predictions of a full official-harness re-run with the
original Qwen3-32B judge score (`qwen_score`, 0.6426) and an independent Llama-3.1-70B-Instruct judge score
(`score`, 0.6767 [0.6583, 0.6947]); 84.4% item agreement. `python analysis/rejudge_llama70b.py --summary
results/judge_llama70b/ama_rejudged.json`.

### KV serving (paper Sec. 3.5, App. "KV-serving correctness")

| Path | What | Script |
|---|---|---|
| `kv_equiv_144/kv_equiv.json` | 144 QA / 24 episodes: kv 17 vs text 18 correct, mask-oracle gate 24/24 | `kvmemory/kv_equiv.py` |
| `kv_equiv_896/kvq_merged.json` (+ per-shard) | 896 QA / 112 episodes: kv 155 vs text 156, argmax agreement 747, gate 126/128 | `kvmemory/kv_equiv.py` (4 shards) |
| `kv_sidechannel/kv_sidechannel_full.json` | perturb unselected upstream turns: gathered-KV answer changes 111/168, text 0, gate 24/24 | `kvmemory/kv_sidechannel.py` |
| `sw_mechanism/sw_mechanism_{8b,32b}.json` | 74 temporal-scan questions: selection / +gold step / +complete index (8B 1, 9, 12; 32B 6, 9, 10 correct); gold step in selection 4/74 | `kvmemory/sw_mechanism.py` |
| `kv_cost/ttft_{8b,32b}.jsonl` | stage-resolved query cost and store-write time, 12 episodes × 4 QA | `kvmemory/kv_ttft.py`, `analysis/ttft_stages.py` |
| `kv_cost/pfx_8b.json` | sub-selection vs resident prefix cache vs text re-prefill, K = 2–12 | `kvmemory/kv_pfx.py`, `analysis/pfx_ttft.py` |

### Out-of-window and KV-serving runs (paper Sec. 3.6, Sec. 4) — `oow/`

One gzipped JSONL per run; each row is one question with a 0/1 field per arm (and `ans_<arm>` /
`pred_<arm>` answers). `oow/index.json` lists file, script, n and arms; `analysis/oow_paired.py` computes
the paired statistics.

| File | n | Script | What |
|---|---|---|---|
| `official_k12.jsonl.gz` | 2431 | `kv_ama_bridge` + `ama/br_judge.py` | official harness K=12: tx .3974 / iso .2653 / b_hot .3603 |
| `official_k5.jsonl.gz` | 2403 | `kv_ama_bridge --k 5` + `br_judge` | K=5: tx .3679 / b_hot .3600 |
| `official_deploymap.jsonl.gz` | 2496 | `kv_ama_bridge --arms gsk,g_txt` + `ama/bg_judge.py` | deployment map (pair with K=12: gsk .3965 vs .3603 / .3974) |
| `owf8`, `owf32` | 527, 474 | `kv_owfloor` | over-window cell, 8B / 32B |
| `rv`, `rv32` | 824, 616 | `kv_review` | evidence matching, conditioner content, full-reading ceiling, tail dial |
| `fl` | 824 | `kv_floor` (hot=4) | floor arms incl. generated-digest conditioner |
| `gh`, `gh_llama` | 824 | `kv_globhot` | harvest 2×2, Qwen3-8B / Llama-3.1-8B-Instruct |
| `hx`, `hx32` | 824 | `kv_hxfer` | agent-style cache header (8B); harvest vs store vs full (32B) |
| `fr`, `kp` | 824 | `kv_frontier`, `kv_keepsel` | fresh-budget selectors; KEEP-style selector |
| `ev` | 824 | `kv_evext` | Regularity I factorial (use `--nonempty bridges`, n=727) |
| `fx`, `dx`, `ep`, `da`, `nt` | 824 | `kv_fixanchor`, `kv_distanchor`, `kv_epic`, `kv_depanchor`, `kv_notesel` | Regularity II dials |
| `kl` | 824 | `kv_klprobe` | first-token KL autopsy (`analysis/kl_probe.py`) |
| `sk`, `sl` | 800 | `kv_skill`, `kv_skillib` | skill block; skill library |
| `loc8`, `loc32` | 1150 | `kv_locomo` | LOCOMO, sessions as events |
| `mx` | 824 | `kv_matrix` | 12-arm conditioner matrix (input to `analysis/vote_analysis.py`, `analysis/cascade_ceiling.py`) |

### Synthetic probes (paper Sec. 4, "Four synthetic probes") — `probes/`

Per-shard outputs (8 shards, Qwen3-8B) of `kv_phantom` + `kv_vartrack` (`phvt/`), `kv_vardecomp` (`vd/`),
`kv_predigest` (`pd/`), `kv_predigest_sweep` (`pds/`), `kv_decoyctl` (`dc/`), `kv_decoyctl2` (`d2/`),
`kv_sweepmenu` (`sm/`), `kv_mater` (`mt/`), `kv_refcarrier` (`rc/`), `kv_noteknob2` (`nk2/`). Analysis:
`analysis/probes/{phvt,vd,pd,pds,nk2}_analyze.py <dir>`.
