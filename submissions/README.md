# Leaderboard submission

`verakv_ama_submission.jsonl` — one line per episode (208 episodes, 2,496 answers),
built from the released raw predictions `results/mu_merged_FLSA.json` + `mu_merged_FLSB.json`
(flagship config `ama/configs/cfg_flagship.json`, structured answer instruction `ama/dec_instr.txt`,
official harness, Qwen3-32B reader + judge). Self-reported judge scores give 0.6478;
official verification will re-judge independently.
