# The model-pick call, fixed and re-run (2026-10-09)

- **What model-pick is:** the deployed configuration (`cfg_flagship.json`, behind the 0.6478 entry) fuses
  lexical overlap with a "model-pick" signal, a short model call that names the past steps a question needs.
- **What was already known:** the paper's router audit (appendix "The deployed router") had seen these
  32-token calls return truncated stubs on a thinking model, and found the signal score-neutral.
- **What the call logs of the [paired re-run](AMA_AGENT_PAIRED.md) add:** it is every call, not most. The
  October revision of the paper now says so in that appendix, and reports the re-run below.
  - AMA-Bench's client never passes the config's `enable_thinking`, so Qwen3 thinks by default.
  - Each call returns only the opening of an unfinished `<think>` block.
  - The parser takes whatever numbers appear there as step picks.
- **This page:** a re-run of the deployed configuration with the call fixed, pre-registered before running.

**In short**
- **Fixing the call does not change the score.** Against the deployed configuration, the fixed one scores
  +0.2 [−1.5, +2.1] with the harness's default reader. With the structured reader of the 0.6478 entry it
  scores +1.4 [−0.2, +3.2] (Holm-adjusted p = 0.19). Both comparisons cover 2,496 questions.
- **The fix works as intended.**
  - All 2,496 fixed calls returned step lists instead of reasoning.
  - The fix changed which steps were served on 72% of questions.
- **The differences stay within the noise floor.** On questions where both arms served identical contexts, the
  arms still differ by −1.7 [−4.4, +1.0], because serving is not deterministic.
- **The deployed configuration reproduces its published numbers in this run:** 0.5992 against 0.5954 with
  the default reader, and 0.6486 against 0.6478 with the structured reader.
- **The paper's conclusion holds for a working signal too.** Lexical + step-pin remains the recommended
  configuration, and it needs one model call fewer per question.

## Setup

| Arm | Configuration |
|---|---|
| V1 | `cfg_flagship.json` as published: the pick call runs in Qwen3's default thinking mode |
| V1f | `cfg_flagship_nothink.json`, which is V1 plus `"pick_thinking": false`. The pick call asks vLLM's chat template for the non-thinking mode; its prompt, 32-token limit and greedy decoding are unchanged. |
| V2 | `cfg_kvmem_causal.json`: lexical + step-pin, no pick call |

- **Readers:**
  - the harness default ("Provide a direct and concise answer."), as in the 0.5954 run;
  - the structured instruction of the 0.6478 entry (`ama/dec_instr.txt`).
- **Questions:** all 2,496 open-ended questions, in six domains.
- **Serving:** one 8×A100 node running 4 Qwen3-32B instances.
  - vLLM 0.28.0, tensor parallel 2.
  - Thinking mode stays on for every other call.
  - The official judge runs on the same instance.
  - Each instance serves a quarter of the trajectories.
  - The three arms of a question run back to back in one driver.
- **Statistics,** as in the paired re-run:
  - bootstrap over trajectories (4,000 resamples, seed 20261009);
  - exact McNemar;
  - Holm correction over the two primary comparisons (V1f − V1 with each reader).
- **Gates, checked before reading any difference** (`ama/paired/modelpick_gates.py`):
  - every V1f pick call carried `enable_thinking: false`, none returned reasoning, and 99.8% returned step
    numbers;
  - every V1 pick call still stopped inside `<think>`;
  - each reader got its own instruction;
  - infrastructure failures stayed at or below 0.4% per arm.

## Results

| | V1 (as published) | V1f (fixed) | V2 (lexical + pin) |
|---|---|---|---|
| Default reader | .599 | .603 | .586 |
| Structured reader | .649 | .663 | .655 |

Paired differences (95% interval; questions won/lost):

| | Default reader | Structured reader |
|---|---|---|
| **V1f − V1** (primary) | **+0.2** [−1.5, +2.1], 234/228 | **+1.4** [−0.2, +3.2], 246/210 |
| … where the fix changed the selected steps (≈1,776 questions) | +1.0 [−1.2, +3.3] | +1.7 [−0.4, +4.0] |
| … where it did not: identical contexts, so this is the noise floor (≈710) | −1.7 [−4.4, +1.0] | +0.7 [−2.4, +3.7] |
| V1f − V2 | +1.8 [+0.0, +3.6], 259/214 | +0.8 [−0.8, +2.4] |
| V1 − V2 | +1.4 [−0.2, +2.9] | −0.6 [−2.1, +0.8] |

V1f − V1 by domain:

| Domain | Default reader | Structured reader |
|---|---|---|
| Web | −0.8 | +4.0 |
| Embodied AI | +2.5 | −0.8 |
| Game | +4.9 [+0.0, +9.8] | −0.3 |
| Text2SQL | +0.7 | +0.7 |
| Software | −3.9 [−8.1, +0.7] | +3.0 |
| Open world | −1.1 | +2.2 |

- **No domain moves the same way under both readers.**
- **The secondary V1f − V2 comparison is not established:** +1.8 with the default reader but +0.8 with the
  structured one. It is uncorrected.

## What the call returns

- **V1:** all 2,496 replies are an unfinished `<think>` block, for example "<think>\nOkay, let's try to figure
  out which steps are needed…". 60% of them contain a number, which the parser takes as a step.
- **V1f:** step lists, often the recent steps in descending order ("14,13,12,11,…"). 48% use all 32 tokens,
  because the prompt asks for up to 20 steps.
- **Steps the question names make up 13% of the selected steps in both arms.** The router pins those steps
  anyway.

## Reproduction checks

- **Against the published numbers:** V1 scores 0.5992 against 0.5954 (default reader) and 0.6486 against
  0.6478 (structured reader).
- **Against the paired re-run,** default reader, same 2,136 questions:

  | | Verdict agreement | Accuracy here | Accuracy in the paired re-run |
  |---|---|---|---|
  | V1 | 85.4% | 0.5939 | 0.5788 |
  | V2 | 86.2% | 0.5814 | 0.5763 |

## What this supports, and what it doesn't

**Supported**
- In the official harness the model-pick call never worked as designed. It returned unfinished reasoning on
  every call of two runs.
- With either reader, a working model-pick signal does not measurably change the deployed configuration's
  score.

**Not supported**
- That the working signal beats lexical + step-pin (see the secondary comparison above).
- Anything about the call with thinking on and a larger token budget, which was not run.
- Anything checked call by call for the July runs behind the published numbers. Their call logs were not
  kept. They used the same harness and adapter, and the paper's audit saw the same truncated stubs.

## Files

- **`ama/harness/method_kvmemory.py`:** the `pick_thinking` switch. The default is the published behaviour, so
  the published configurations still reproduce the published numbers.
- **`ama/configs/cfg_flagship_nothink.json`:** the fixed configuration.
- **`ama/paired/`:**
  - `run.py` with arm V1f;
  - `modelpick_gates.py`, which needs a fresh run's records because it reads the reader prompts;
  - `modelpick_analyze.py`.

  How to run them is in [ama/paired/README.md](../ama/paired/README.md).
- **`results/modelpick_fix/{def,str}/s{0..3}/q.jsonl.gz`:** 2,496 questions × 3 arms per reader.
  - Each record has the answer, the verdict, the pick-call replies and the step numbers of the served
    appendix (`picked_steps`).
  - The full reader prompts are left out.
- **`results/modelpick_fix/modelpick_analysis.json`:** every number on this page. It is reproduced by:

```bash
python ama/paired/modelpick_analyze.py results/modelpick_fix --paired1008 results/ama_paired/full
```
