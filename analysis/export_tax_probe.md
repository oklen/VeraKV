# The export tax: why memory-side reasoning cannot be exported without paying for it

**Probe wave (2026-07-06), small-set screening. n=144 QA (12 episodes, 2/domain stratified
from the h0 half, seed 0), Qwen3-32B reader+judge, flagship memory (hybrid+pin), all arms
same batch on identical questions (143 common after joining). CIs are ±7pp per arm — every
delta below is a paired, directional reading, not a headline number.**

## Question

The relay ladder (`plan_poisoning_case_study.md`) ended at: the structured pass's derivation
never leaves the thinking block (99% of artifacts begin `Answer[1]:`), and even the correct
answer barely transfers to a default reader. Follow-up: can careful prompt + few-shot work
force the derivation *out* (blocker 2), and a verification-framed reader instruction make it
*adopted* (blocker 3)?

## Arms

| arm | upstream instruction | artifact | reader instruction | acc | Δ vs PRSTR (paired) |
|---|---|---|---|---|---|
| PRDEF | — (in place) | — | default | 0.671 | +0.007 |
| PRSTR | structured (in place) | — | — | 0.664 | — |
| PRTRU | structured | bare answer | trust | 0.636 | −2.8 |
| PRAV | structured | bare answer | verify-lite | 0.615 | −4.9 |
| PRX1 | structured + **export demand** | Evidence/Derivation/Answer | verify (named-flaw) | 0.608 | −5.6 |
| PRXC | structured + citation demand | answer + `[evidence: steps N,M]` | verify-lite | 0.601 | −6.3 |
| PRXD | structured + export demand | Evidence/Derivation/Answer | trust | 0.573 | −9.1 |
| **PRSXP** | structured + **export demand** | **answered in place — no relay** | — | **0.587** | **−7.7** |

Export demand = format requirement + schematic few-shot: visible reply must contain
`Evidence:` (verbatim quotes with step numbers), `Derivation:` (numbered steps ending in a
no-later-step-overturns check), `Answer[1]:`.

## Mechanism readings (from per-question artifact dumps)

| arm | format compliance | all-quotes-verbatim | reader adoption |
|---|---|---|---|
| PRTRU (bare answer) | 0% (n/a) | 54% | 39% |
| PRAV (bare answer) | 0% (n/a) | 50% | 41% |
| PRX1 (export+verify) | **99%** | **23%** | **94%** |
| PRXD (export+trust) | 99% | 19% | 67% |
| PRXC (citation only) | n/a | 57% | 83% |
| PRSXP (export, in place) | 99% | — | 100% (by construction) |

(Adoption is a string heuristic between the upstream answer and the reader's final answer; it
under-detects on verbose bare-answer artifacts — compare within-column, not across.)

## Findings

1. **Both original blockers fall mechanically to prompting.** The format demand + one
   schematic few-shot exports the derivation in 99% of artifacts (default behavior: ~1%).
   The verify-with-named-flaw instruction achieves 94% adoption (blanket trust: 67% on the
   same artifacts). Prompt engineering *can* pierce the thinking block and *can* make the
   reader adopt.

2. **But the export demand itself taxes the answering pass: −7.7pp with no relay involved**
   (PRSXP vs PRSTR, same batch, paired). This is the cleanest single number of the wave:
   identical memory, identical questions, identical structured instruction — the only change
   is "your visible reply must lay out the derivation," and the in-place score drops from
   0.664 to 0.587. PRX1 (0.608) ≈ the taxed upstream faithfully relayed (94% adoption); its
   deficit is inherited, not added.

3. **The tax has a fingerprint: quote decay and soft-type victims.** Only the two
   derivation-demand arms collapse to 19–23% all-quotes-verbatim (every other arm ~50–57%):
   the model derives in `<think>`, then *re-writes* the Evidence section from memory of its
   thinking rather than re-copying from context. Judged victim cases (PRX1 adopted-wrong ∧
   PRSTR-right) concentrate in counterfactual / strategic-next-action question types, where
   committing prose sub-conclusions early locks phrasing that the rubric then rejects.

4. **Even the minimal demand pays.** A one-line citation requirement
   (`[evidence: steps N,M]`) — no quotes, no derivation prose — still nets −6.3 relative to
   in-place structured and −3.5 relative to the bare-answer relay, despite buying 83%
   adoption.

5. **Verification framing does not help bare answers.** PRAV (verify-lite) ≤ PRTRU (trust)
   (−2.1, directional): a verification procedure on an artifact with nothing checkable
   invites re-derivation; blanket trust adopts more. The verify instruction's 94% only
   appears when there is a derivation to check — which only exists if you pay the tax.

6. **Nothing beats the dumbest relay.** Bare answer + blanket trust (PRTRU, −2.8) tops every
   artifact enrichment tried; and it in turn loses to just answering in place.

## Revised verdict on blocker 2

The original phrasing — "the derivation is architecturally non-exportable" — was about
default behavior. The corrected, stronger form: **a tax-free export does not exist.** The
visible derivation either does not exist (default), or exists already corrupted by the
demand that it exist. The thinking block is not hiding the derivation from the pipeline; it
is *protecting* it — the freedom to derive without committing prose is where the structured
instruction's value lives, and any demand that pierces that veil pays back the gain it was
trying to move. This is the export-side dual of the relay ladder's conclusion: reasoning
must happen in the answering pass, *unobserved*.

## Caveats

- Small-set screening (n=144, single batch): per-arm CIs ±7pp; the 8-arm consistency, the
  quote-decay signature, and the victim-type concentration carry the story, not any single
  delta. A full-half (n=1248) PRSXP-vs-anchor run would make the tax number publishable.
- This stratified subset shows **no default→structured gap** (PRDEF 0.671 ≈ PRSTR 0.664;
  on the full h0 the gap is +7.8pp) — equal domain weights dilute TEXT2SQL and n is small —
  so this set measures relay/export *friction*, not gap *recovery*. Do not read PRDEF≈PRSTR
  as contradicting the factorial.
- Files: `results/mu_merged_PR{STR,DEF,TRU,AV,X1,XC,XD,SXP}.json`,
  `results/sel_PR{TRU,AV,X1,XC,XD,SXP}_full.jsonl`; analysis in
  `analysis/s1_verdict.py`-style scripts (worker `stage_w7/`).

---

# Full-scale confirmation (n=1248, test_h0, same-batch — 2026-07-06 15:18)

The small-set screening under-estimated the tax. Same-batch full-half run,
paired on 1,239 common questions:

| arm | config | acc |
|---|---|---|
| RESTR4 | structured, in place (anchor; band 0.629–0.642) | **0.6354** |
| SXPH | structured + export demand, in place, no relay | **0.5192** |
| HOFX | export demand + verify-adopt relay | **0.5304** |

**1. The tax at scale: −11.8pp, CI[−14.4, −9.1]** (paired bootstrap; fixed 74 /
broke 220). Not only does the export demand refund the structured instruction's
entire +7–8pp gain — it **overshoots into net harm**: SXPH 0.519 sits *below
every default-instruction anchor* (REDEF2 0.564, HOFS 0.583, REDEF 0.597).
Demanding a visible derivation makes the structured instruction worse than the
one-line default.

**2. The mechanism numbers replicate exactly**: compliance 98–99%, reader
adoption 95%, all-quotes-verbatim 21% (small set: 99% / 94% / 23%).

**3. The relay is faithful and irrelevant: HOFX − SXPH = +1.1pp, CI[−1.1, +3.5]**
(zero-inclusive). The verify reader adopts what it is given and inherits the
damage. HOFX 0.528 ≪ HOFT 0.593 (trust-relay of the *untaxed* pass's bare
answer, joint n=1239): adoption was never the binding constraint — upstream
quality is.

**4. The tax lands on exactly the questions the instruction exists to fix.**
On the F+ pool (default-wrong ∧ structured-right, n=180): RESTR4 re-roll 0.722
→ SXPH **0.411** (−31pp; HOFX 0.433) — the demand destroys ~43% of what
structured wins there, while HOFT holds 0.589. On F− (n=83) SXPH ≈ anchor
(0.422 vs 0.434): pure harm, no compensating class.

**5. Concentration confirms the small-set fingerprint.** By qa_type, Type B
(strategic/reasoning) pays −20.6pp, A (exact recall) −11.1, D −9.1, C −6.0.
By domain: TEXT2SQL −15.7, OPENWORLD −15.0, SOFTWARE −14.4 vs EMBODIED −4.7,
WEB −6.1.

**Verdict, final form.** Forcing the derivation into the visible reply is not a
lossy export of the reasoning — it is a *different, worse reasoning process*:
the model commits prose sub-conclusions as it thinks (Type B pays most),
re-writes evidence from memory instead of re-copying (21% quote fidelity), and
the visible-format constraint binds precisely where the instruction's freedom
was producing the gain (F+). The thinking block is not hiding the derivation
from the pipeline; it is protecting the conditions under which the derivation
is any good. Reasoning must happen in the answering pass, unobserved.

Files: `results/mu_merged_{RESTR4,SXPH,HOFX}.json`,
`results/sel_{SXPH,HOFX}_full.jsonl`; analysis `analysis/w8_verdict.py`
(paired-bootstrap CIs, F± pools, domain/type splits).
