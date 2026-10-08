# VeraKV — research brief (3-minute read)

**Problem.** Long-horizon agents accumulate histories that outgrow the context window. Most memory
systems keep the raw history but hand the reader what they constructed from it — facts, notes,
summaries, graphs — so which details the reader can ever see is fixed at write time. On dense agent
trajectories that loses the exact values and steps questions ask for: on AMA-Bench, purpose-built
dialogue-memory systems (Mem0 0.21, A-Mem 0.32, MemGPT 0.33, MemoryBank 0.34) score below a no-memory
long-context baseline (0.52). Serving the raw history from a cheap index is by now an established
pattern; what was missing is measurement — which property of the raw payload carries the benefit, and
when the trajectory's own KV cache can stand in for re-encoded text.

**The system.** VeraKV keeps the raw turns verbatim, indexes them with extractive gists and a recency
overview, selects spans per query (cheap router + deterministic step-address pin), and serves them
verbatim — as re-prefilled text in every headline number, or as gathered KV within the window. On the
official AMA-Bench harness (Qwen3-32B reader + judge, all 2,496 QA):

| | Acc |
|---|---|
| **VeraKV (deployed router + step-pin memory, structured answer instruction)** — leaderboard-verified, #1 as of 2026-07-15, #2 since a 2026-09 entry (0.6975); the released lexical + step-pin default scores 0.6466 locally | **0.6478** |
| VeraKV memory stack, harness default reader | 0.5954 |
| AMA-Agent (purpose-built for the benchmark) | 0.5722 |
| Best other published memory system (MemoRAG) | 0.4606 |

**Payload fidelity, attributed.** Matched evidence, reader and judge, only the payload form varies:
serving an LLM summary or extracted facts instead of the verbatim cited step costs 8–14pp (910 QA,
replicated under a Llama-3.1-8B judge). In the deployed pipeline, replacing the verbatim appendix with
LLM summaries costs −5.1pp and with extracted facts −5.9pp — both at or below serving no appendix — while
a deterministic swap to the query-relevant original lines costs −2.6pp: paraphrase hurts about twice as
much as shortening. A cheap router matters little on step-indexed agent traces (a model-pick fusion
audited score-neutral) and a lot on diffuse dialogue.

**The reader is a separable axis.** A same-batch memory × reader factorial is additive: routing
+3.1–3.5pp at either reader, a one-sentence structured answer instruction +5.1–5.5pp at either router.
Fourteen reader-side mechanisms (plans, compiled views, checklists, re-retrieval, type-routed
instructions, lookup loops) come back null or negative over a strong router. The instruction itself can
ride in the memory's return value, and as a rotated skill-KV block it ties the text instruction; a
query-routed library of additive per-class hints adds +2.1pp overall (+6.3pp where a rule fires).

**KV serving, within the window.** Gathering the selected spans' cached KV at original positions
reproduces a full-prefill-then-mask oracle (identical 256-token greedy sequences) and is
accuracy-equivalent to re-prefilling the same spans as text (17.3% vs 17.4%, 896 QA). The benefit is cost:
the question's first-token latency stays flat while text re-prefill grows with evidence (1.6–10.9×), and
it beats a resident full-trajectory prefix cache 1.4–1.9× on TTFT.

**Beyond the window.** Independently encoded event stores collapse on the official harness (−13.2pp
paired, n=2431). Anchored stores plus a fresh two-event tail recover 72%, and at a lean K=5 budget the gap
is −0.8pp, CI [−3.0, +1.4]. Two scoped regularities organize the rest: evidence converts to accuracy only
through joint encoding (text gains from more served events, the cache does not), and a store's
write-time conditioning must match what is served, in content and in position. Query-time token repair
(EPIC-style), dependency-graph conditioners and six fresh-budget selectors including a KEEP-style one
all fail to beat the simple recipe. Where the agent's own rollout cache survives, harvesting it replaces
the ~5× store-write cost at parity across scales (8B/32B), benchmarks (AMA-Bench/LOCOMO) and backbones
(Qwen/Llama); a mixed deployment policy ties the routed-text carrier baseline on the full official harness
(.397 vs .397).

**Boundaries, stated.** Full context can win when the history fits and the reader exploits it
(LongMemEval-S at 128k); on AMA-Bench SOFTWARE, temporal-scan questions ("which step first / last") hit a
coverage ceiling — supplying the answer-bearing step verbatim lifts an 8B or 32B reader only to ~0.12,
because verifying an extremum needs the whole 60–100k-token trajectory; exact KV-layer deletion needs
downstream invalidation (a planted-secret probe leaks 0/120); the out-of-window dials are 8B-only.

**Artifacts.** Code, configs, prompts, raw per-question predictions with judge outputs, the per-question
outputs of the KV and out-of-window runs, analysis scripts and the leaderboard submission:
https://github.com/oklen/VeraKV
