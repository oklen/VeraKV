# VeraKV vs AMA-Agent, paired (2026-10-08)

The leaderboard numbers in the README compare VeraKV's per-question results with AMA-Agent's *published*
numbers. This page reports a paired re-run on 2,136 questions: same questions, same server and same judge for
both systems. Two follow-up arms test why AMA-Agent loses; both were pre-registered before they ran.

**In short**
- **Against AMA-Agent as released, VeraKV wins:** 0.579 vs 0.537, **+4.2 [+1.8, +6.6]**. On state-update
  questions the gap is **+13.7 [+8.7, +18.6]**, 117 questions won and 43 lost.
- **AMA-Agent loses because it stops early, not because of its context cap.**
  - On 84% of questions its "is this enough?" check answers immediately, after reading only the 5 most
    similar raw steps (plus any steps the question names) and none of its state summary.
  - The prompt behind its answer is a median 3,046 tokens; VeraKV's is 9,338.
  - Counting its context cap in tokens instead of characters changes nothing (+0.2).
- **Forced past that early answer (arm A3), AMA-Agent gains +6.2 overall and +10.7 on state update.** That
  closes 74% of VeraKV's state-update lead and ties VeraKV overall: 0.598 vs 0.579, a difference of −2.0
  [−4.4, +0.4]. On code it beats VeraKV by 9.3.
- **So against an AMA-Agent that keeps going, VeraKV's advantage is cost.**
  - VeraKV: 2 model calls per question (one of them a step-selection call capped at 32 output tokens),
    0.58k generated tokens, about 1 minute, and no model calls while writing memory.
  - A3: about 3 calls, 4.3k generated tokens and 3.5 minutes per question, plus about one model call per
    8,192 characters of trajectory when building its memory.

## Setup

- **Questions:** AMA-Bench open-ended. Open-world is left out; the other 5 domains give 178 trajectories and
  2,136 questions, 540 of them state-update.
- **Serving:** one 8×A100 node running 4 Qwen3-32B instances (vLLM 0.28, tensor parallel 2, the default
  thinking mode).
  - Each instance serves a quarter of the trajectories.
  - All arms of a trajectory, and their judging, run on the same instance.
- **Reader and judge, the same for every arm:**
  - reader prompt: the harness default ("Provide a direct and concise answer.");
  - judge: the official `compute_llm_as_judge` on the same Qwen3-32B.

| Arm | What it is |
|---|---|
| A1 | AMA-Agent as released (`configs/ama_agent.yaml`: Qwen3-Embedding-4B top-5, the sufficiency loop, keyword search), answered through the harness's `answer_question`, including its direct-answer fast path. |
| A2 | A1 with one change: the final context cap (23,808 = 32,000 − 8,192) counted in reader tokens instead of characters. It shares A1's memory, retrieval and sufficiency calls. |
| A3 | A1 with one change: the sufficiency check may never end with a direct answer. Where the released code would answer, A3 runs the keyword-search round (same 60-second budget), assembles the final context (summary + named steps + evidence) and lets the reader answer. A1r, recorded in the same run, is what the released code would have answered at that point. |
| V1 | VeraKV, the 0.5954 configuration (`ama/configs/cfg_flagship.json`). |
| V2 | VeraKV, lexical router + step pinning (`ama/configs/cfg_kvmem_causal.json`). |

- Every model call was logged: the exact prompt, the raw response and the server-side token counts.
- No arm had an infrastructure failure.
- Statistics:
  - bootstrap over trajectories (episode-clustered, 4,000 resamples, seed 20261008), plus exact McNemar on
    the discordant pairs;
  - Holm correction over each test's two primary endpoints (all questions, state update).
- Two environment fixes, neither of which changes a method:
  - AMA-Agent's keyword search relies on Python 3.12's `tempfile.mkdtemp`, which returns an absolute path
    for a relative directory. On 3.11 every search failed with "can't open file", so `run.py` restores the
    3.12 behaviour. In the full run, 212 of 257 searches returned results.
  - vLLM 0.28 on Python 3.11 needed a one-line flashinfer annotation patch for tensor parallelism, and the
    native sampler. All our calls are greedy.

## Results

| | A1 | A2 | V1 | V2 |
|---|---|---|---|---|
| All | .537 | .539 | **.579** | .575 |
| State update | .507 | .504 | **.644** | .641 |
| Web | .583 | .581 | **.688** | .667 |
| Embodied AI | .464 | .478 | **.539** | .542 |
| Game | .669 | .672 | .675 | .675 |
| Text2SQL | .590 | .587 | .605 | .598 |
| Software | .373 | .375 | .400 | .407 |

Paired differences (95% interval; questions won/lost):

| Slice | V1 − A1 | A2 − A1 | V1 − A2 |
|---|---|---|---|
| **All** (2,136) | **+4.2** [+1.8, +6.6], 335/246 | +0.2 [−0.5, +0.9], 30/26 | +4.0 [+1.6, +6.4] |
| **State update** (540) | **+13.7** [+8.7, +18.6], 117/43 | −0.4 [−1.7, +1.1] | +14.1 [+9.1, +18.9] |
| Web (372) | +10.5 [+3.8, +16.9], 69/30 | −0.3 [−2.7, +1.9] | +10.8 [+3.8, +17.5] |
| Embodied AI (360) | +7.5 [+2.8, +12.8], 60/33 | +1.4 [−0.3, +3.1] | +6.1 [+1.4, +11.1] |
| Game (360) | +0.6 [−5.0, +6.4] | +0.3 | +0.3 |
| Text2SQL (612) | +1.5 [−2.0, +5.1] | −0.3 | +1.8 |
| Software (432) | +2.8 [−2.3, +8.1] | +0.2 | +2.5 |
| Recall (741) | +0.9 [−2.7, +4.6] | +0.3 | +0.7 |
| Causal (501) | −2.2 [−6.7, +2.4] | −0.6 | −1.6 |
| State abstraction (354) | +5.4 [+0.0, +10.9] | +2.0 [0.0, +4.0] | +3.4 |

- **Both primaries hold after Holm correction:** exact McNemar p = 2.6e-4 overall and 4e-9 on state update.
- **V2 ≈ V1** (−0.4 [−2.0, +1.3]).
- **V1's step-selection call did not work as designed in this run.**
  - It is capped at 32 output tokens and stopped at the cap on all 2,136 questions.
  - In thinking mode it returned only the opening of the model's reasoning: all 2,136 responses were an
    unfinished `<think>` block.
  - The parser took whatever numbers appeared there as step picks; 61% of the responses had one.
  - V2 makes no such call and scores the same, so the comparison with AMA-Agent does not depend on it.
  - The paper's router audit had already seen truncated pick calls and found the signal score-neutral.
    A follow-up re-run with the call fixed scores the same as well: [MODEL_PICK_FIX.md](MODEL_PICK_FIX.md).

### What AMA-Agent read

- **On 84% of questions (1,793 of 2,136) it answered at the sufficiency check.**
  - That prompt holds only the top-5 similar steps and the steps the question names, each observation cut
    to 3,000 characters, and no state summary.
  - The share by domain: Game 95%, Software 91%, Text2SQL 86%, Embodied AI 76%, Web 69%.
- **The check almost never iterates:** it ran a second round on 41 of 2,136 questions. One check takes a
  median 62 s in thinking mode (p10 32 s, p90 132 s), and the whole loop has a 60-second budget.

| Prompt that produced the answer (server tokens) | Median | p90 |
|---|---|---|
| A1 | 3,046 | 7,733 |
| V1 | 9,338 | 19,218 |

- **The character cap is real, but it barely matters.**
  - A1's assembled context is a median 6,092 tokens, and 20% of contexts are cut at 23,808 characters.
  - That context is read on only 16% of questions.
  - On the 142 questions where it was both read and cut, counting the cap in tokens gains +6.3 (18 vs 9,
    p = .12).

### Where the evidence was (model labels)

- **What was labelled:** all 581 disagreements between V1 and A1, plus the 27 between A2 and A1 on state
  update and Web.
- **Labeller:** GPT-5.6-Sol at high reasoning effort.
- **What it saw:** the question, the reference answer, both answers and both sides' complete reader prompts,
  with no indication of which system was which.
- **Format:** fixed categories and a JSON reply (`ama/paired/label.py`).
- **Hand check:** 25 labels drawn with seed 7 matched on the error cause in 25 of 25.

| | V1 right, A1 wrong (335) | A1 right, V1 wrong (246) |
|---|---|---|
| **Wrong side's context held the complete evidence** | **48%** (partial 46%, none 5%) | **86%** (partial 12%, none 2%) |
| Right side's context held it | 98% | 76% |
| Error: wrong value | 43% | 42% |
| Error: wrong reasoning | 22% | 22% |
| Error: partial answer | 20% | 19% |
| Error: judge noise | 11% | 17% |
| Question needs a scan of the whole trajectory | 30% | 17% |

- **VeraKV's wins are about evidence coverage.** Where it won, AMA-Agent's context held only part of the
  evidence half the time; on state update, 54% partial and 7% none.
- **VeraKV's losses are misreadings.** Where it lost, 86% of the time the evidence was in its own context.

### A3: no answer from the first check

| | A1r (as released) | A3 | V1 |
|---|---|---|---|
| All | .536 | **.598** | .579 |
| State update | .500 | **.607** | .644 |
| Web | .586 | .656 | .688 |
| Embodied AI | .461 | .556 | .539 |
| Game | .664 | .683 | .675 |
| Text2SQL | .587 | .613 | .605 |
| Software | .377 | **.493** | .400 |

| Slice | A3 − A1r | V1 − A3 |
|---|---|---|
| **All** | **+6.2** [+4.2, +8.3], 303/170 | −2.0 [−4.4, +0.4] |
| **State update** | **+10.7** [+6.8, +14.6], 92/34 | +3.7 [−0.9, +8.3] |
| Web | +7.0 [+2.7, +12.1] | +3.2 [−1.9, +8.3] |
| Embodied AI | +9.4 [+5.8, +13.3] | −1.7 [−6.9, +3.6] |
| Game | +1.9 [−3.6, +7.2] | −0.8 |
| Text2SQL | +2.6 [−1.5, +6.7] | −0.8 |
| Software | **+11.6** [+7.4, +16.2] | **−9.3** [−14.8, −3.7] |
| State abstraction | +10.2 [+5.3, +15.2] | −4.2 [−9.6, +1.1] |
| Causal | +2.4 | −5.2 [−9.9, −0.4] |
| Questions the released code answered at the check (1,813) | +7.3 [+4.9, +9.8] | −2.9 [−5.4, −0.3] |
| … of which state update (434) | +13.4 [+8.5, +18.1] | +1.6 [−3.3, +6.7] |
| Questions the released code sent on (323) | 0 (identical by construction) | +3.1 [−3.3, +9.5] |
| … of which state update (106) | 0 | +12.3 [+2.6, +22.3] |

- **Both primaries hold after Holm correction:** McNemar p = 1e-9 overall and 2e-7 on state update.
- **A3 closes 74% of V1's state-update lead** (interval 50% to 108%).
- **Batch noise is small.** The same-run A1r and the full-run A1 differ by −0.1 [−1.8, +1.6], with 83.4% of
  verdicts identical. A1r answered at the check on 1,813 questions, the full-run A1 on 1,793.
- **What A3 actually did:**
  - The reader answered every question.
  - Its final context held keyword-search results on 83% of questions (1,775 of 2,136). The search failed
    on 107, came back empty on 16 and left nothing in the context on 238.
  - The final context was a median 6,377 tokens, and 28% of contexts were cut by the character cap.
- **A3's change brings in two things at once:** the search results, and the reader seeing the state summary.
  The design cannot separate them. Where the released code answered at the check and A3's search then
  failed or came back empty (100 questions), A3 still gained +4.0 (13 vs 9, not significant).

### Cost per question

Cost counts the answering calls only, without the judge. Wall time was measured under each run's own load,
so read it as an order of magnitude.

| | Model calls | Input tokens (median) | Generated tokens (median) | Wall time (median) |
|---|---|---|---|---|
| V2 (lexical router) | 1 | 9.3k | 0.55k | 54 s |
| V1 | 2 (one of them 32 tokens long) | 13.6k | 0.58k | 63 s |
| A1 | 1.3 | 3.1k | 0.76k | 71 s |
| **A3** | **3.0** | 11.3k | **4.3k** | **207 s** |

Building AMA-Agent's memory also costs about one model call per 8,192 characters of trajectory. VeraKV
makes no model call when writing.

## Reproduction checks

- **V1** agrees with the July run on 83.8% of verdicts over the same 2,136 questions: 0.579 now, 0.587 then.
  The vLLM version changed in between.
- **A1, overall:** 0.537, against AMA-Agent's published numbers for these 5 domains (weighted by questions)
  of 0.586. The difference of −4.9 is within the 5-point tolerance set before the run. Web and Game come out
  higher than published; Embodied AI and Text2SQL are a few points lower, all within ±7.
- **A1, Software:** 0.373 against the published 0.636, a 26-point gap we cannot explain. It is not the
  60-second budget: A1 answered 91% of code questions at the first check and used the keyword search on
  only 7%. Code version, serving or judging differences are all possible; we could not check them.

## What this supports, and what it doesn't

**Supported**
- On the same questions, server and judge, VeraKV beats AMA-Agent as released: +4.2 overall and +13.7 on
  state update.
- AMA-Agent as released loses mainly because its first sufficiency check says "enough" too early. Forced to
  keep going, it closes three quarters of the state-update gap and ties VeraKV overall.
- In that tie, VeraKV is cheaper: about 3× less time and 7× fewer generated tokens per question, and no
  model calls while writing memory.

**Not supported**
- That VeraKV beats AMA-Agent's *published* scores. Our A1 is 4.9 below them, and 26 below on code.
- That VeraKV beats AMA-Agent as such. A3 ties it overall and wins on code and causal questions.
- Anything about the open-world domain, which this run leaves out.
- Anything beyond this one serving setup: Qwen3-32B in thinking mode on vLLM 0.28. AMA-Agent's loop has a
  60-second budget, so slower serving stops it earlier. It decides "enough" in the first round on 84% of
  questions, though, so the budget affects few of them.

## Files

- `ama/paired/`: the run script (all arms), the A2 and A3 patch generators, the analyses and the labeller
  ([how to run](../ama/paired/README.md)).
- `results/ama_paired/`:
  - `full/s{0..3}/q.jsonl.gz`: arms A1, A2, V1 and V2, 2,136 questions each;
  - `a3/s{0..3}/q.jsonl.gz`: arms A3 and A1r;
  - `labels.jsonl`: the 608 labels;
  - `handcheck_sample.jsonl`: the 25 hand-checked labels;
  - `analysis.json` and `a3_analysis.json`: every number on this page.

  Each record has the question, the reference answer, the answer, the verdict, AMA-Agent's path, the token
  counts and per-call usage. The full reader prompts are left out for size (about 74 MB compressed); ask if
  you need them.
