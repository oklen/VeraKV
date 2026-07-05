# VeraKV — research brief (3-minute read)

**Problem.** Long-horizon agents (coding, web, embodied) accumulate histories that overflow any
context window. Deployed memory systems *construct* a lossy store — extract facts, build graphs,
summarize — and retrieve from it. On dense agent trajectories this destroys exactly the evidence
questions need: on AMA-Bench, every published construction-based system scores *below* a
no-memory baseline.

**Core finding — payload fidelity dominates the memory side.** VeraKV constructs *indices, not
answer payloads*: raw turns kept verbatim, a cheap extractive gist index + recency overview over
them, per-query routing, selected spans rehydrated byte-for-byte. On the official AMA-Bench
harness (Qwen3-32B reader+judge, all 2,496 QA):

| | Acc |
|---|---|
| **VeraKV memory stack, harness default reader** | **0.5954** |
| AMA-Agent (purpose-built for the benchmark) | 0.5722 |
| Best prior published memory system | 0.4606 |
| VeraKV + one-sentence structured answer instruction (end-to-end recipe) | 0.6466* |

*\*submitted for official leaderboard verification; above the prior leaderboard best 0.6246.*

A same-pipeline attribution ladder makes the payload's role causal (same reader, prompt, budget,
selected steps — only the payload form varies): recency-truncated full context **−8.4pp** →
LLM summaries **−5.1** → extracted facts **−5.9** (both *below* serving no evidence at all,
−4.3) → deterministic *original-lines* extraction **−2.6** → verbatim anchor. **The poison is
paraphrase, not compression.**

**The confound audit (what usually goes unmeasured).** A memory×reader factorial shows the single
largest knob in the whole system is the *reader instruction* (+5.1–5.8pp at either router) —
larger than every memory-side refinement combined (+3.1–3.5). So end-to-end memory comparisons
that don't control the reader largely measure reader differences. We report both axes separately,
disclose the instruction verbatim, and characterize its churn honestly (it fixes 318 questions
and breaks 184; part of the gain is judge-facing style, measured by case analysis).

**A predictive law, stress-tested 14 ways.** Twelve reader-side mechanisms over a fixed memory
(evidence compression, state scaffolds, sufficiency gates, deterministic lookup tools, iterative
ReAct loops, pre-computed reasoning plans, type-routed specialized instructions, re-retrieval
under three budgets) yield one law: *a
mechanism helps only when it injects specific, query-addressable information the context is
missing; reasoning must happen in the answering pass.* Pre-computed reasoning is
anti-transferable (−5.6pp); a prompted iterative loop adds exactly nothing as an evidence-gatherer
(±0.0 same-batch) and costs −9pp when it displaces the answering pass; and a reasoning-affordant
re-presentation of the same verbatim evidence (need-grouped quotes, timelines, occurrence panels —
no conclusions) recovers none of the instruction's per-question fixes (+0.3pp vs a 32% spontaneous-
recovery control) while breaking 13% of previously-correct questions; even carving the
instruction itself into question-type-specialized fragments loses (−3.0pp; routed classes −7 to
−23) — the eliciting instruction survives as an indivisible whole. The one domain we lose
(SOFTWARE, −17 vs AMA-Agent) is dissected to a five-layer elimination — not retrieval (100%
recall), not access, not format, not iteration — isolating the residual to *code execution over
the raw trajectory* + construction-time indexing: a complementary paradigm, stated as a boundary.

**The serving mechanism.** Because the store is verbatim KV, selection is a cache operation:
prefill the trajectory once, gather selected spans' KV at their original positions, re-prefill
only the question. Exact against a full-prefill-then-mask oracle (24/24 episodes, two model
families, YaRN-extended RoPE), accuracy-tied with the text pipeline at n=576 — including serving
the *deployed* overview+appendix layout — with flat TTFT as evidence grows (1.6–10.9×, reaching
32× at 32k selected tokens). A planted-secret probe measures the privacy boundary: 0/120
extractable leakage from contextual carryover, positive control confirms probe sensitivity.

**Boundaries, stated not hidden.** Full context wins when history fits the window and the reader
can exploit it (LongMemEval crossover mapped per question type); construction-time extraction wins
for weak readers on tiny isolated facts; verbatim storage has governance costs (deletion semantics
measured at both text and KV layers).

**Artifacts.** Everything is released and re-runnable: 30+ tagged runs with raw per-question
predictions and judge scores, machine-readable manifest, exact prompts, analysis scripts
(bootstrap/cluster CIs, flip-set case studies), serving/launch recipes, and the leaderboard
submission file. https://github.com/oklen/VeraKV
