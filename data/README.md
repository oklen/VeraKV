# Data

Benchmark files are not shipped; place them here (paths are the modules' defaults).

- **AMA-Bench** — `data/ama_test.jsonl`: the open-ended test set, i.e. AMA-Bench's
  `dataset/test/open_end_qa_set.jsonl` copied verbatim (208 episodes, 2,496 QA; the file used here has
  md5 `245c87b291c259670cb68a4c7d2fe367`). One episode per line: `episode_id`, `domain`, `task_type`,
  `num_turns`, `total_tokens`, `success`, `trajectory` of `{turn_idx, action, observation}`, `qa_pairs` of
  `{question, answer, type, question_uuid}`. The KV modules read it with `kvmemory.ama_bench.load_episodes`;
  the mechanism subset is 103 episodes with at most 24,000 tokens, drawn round-robin across domains
  (domains in sorted order, episodes in file order; `--max_ep 103 --max_tokens 24000`).
  The official-harness runs read the same file from the AMA-Bench checkout instead.
- **LOCOMO** — `data/locomo10.json` (the public LoCoMo release). Used by `kvmemory/locomo_eval.py`,
  `kvmemory/locomo_gpt4omini.py` and `kvmemory/kv_locomo.py`.
- **LongMemEval-S** — `data/longmemeval_s_cleaned.json` from the official `xiaowu0162/longmemeval`
  release. Used by `kvmemory/longmemeval_eval.py`; the oracle evidence-session labels are used only for
  error attribution.

The synthetic probes (`kv_phantom`, `kv_vartrack`, `kv_vardecomp`, `kv_predigest*`, `kv_decoyctl*`,
`kv_sweepmenu`, `kv_mater`, `kv_refcarrier`, `kv_noteknob2`, `kv_select_smoke`, `kv_faithful`) generate
their own data.
