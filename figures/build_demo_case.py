"""Build figures/demo_case.json, the data behind the README demo (figures/make_demo.py).

    python figures/build_demo_case.py <case.json> figures/demo_case.json

<case.json> holds the raw paired re-run rows for one question (all arms, with the full prompts and raw model
outputs that results/ama_paired/ leaves out because of their size) and that question's AMA-Bench episode:
{"rows": {arm: row}, "episode": <line of the AMA-Bench test file>}. The question shown is episode 34
(alfworld), question 51a13dd0-f7f2-4599-b920-dedcf31fcd17, rows ep=34 qi=3 of the full run.

Every string the demo shows is copied from these inputs, never written by hand: the question and reference
answer, the steps each system put in front of its reader (parsed from its logged prompt), AMA-Agent's
check output and two sentences of its reasoning, both answers and both judge verdicts."""
import json
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding="utf-8"))
ep, a1, v2 = d["episode"], d["rows"]["A1"], d["rows"]["V2"]
qa = next(q for q in ep["qa_pairs"] if q["question"] == a1["question"])
traj = {t["turn_idx"]: t for t in ep["trajectory"]}


def first_par(text):
    """An observation up to its list of available actions: what happened at that step."""
    return text.split("\n\nThe current available actions")[0].strip()


def step(n):
    return {"step": n, "action": traj[n]["action"], "observation": first_par(traj[n]["observation"])}


# AMA-Agent: the turns in the prompt of its sufficiency check (it answered from that call).
a1_prompt = a1["answer_prompt"]
a1_turns = [int(x) for x in re.findall(r"(?m)^Turn (\d+):", a1_prompt)]
named = [int(x) for x in re.findall(r"(?i)\bstep (\d+)\b", qa["question"])]
resp = a1["answer_response"]
verdict = resp.split("</think>")[-1].strip().split("\n")[0].strip()
quotes = ["But the retrieved turns don't include turn 5.",
          "we might assume that drawer 1 remained closed from turn 6 onwards until step 77."]
for q in quotes:
    assert q in resp, q
check_q = ("can you answer the question COMPLETELY and ACCURATELY using only the retrieved turns above "
           "(plus general knowledge)?")
assert check_q in a1_prompt

# VeraKV: a short head of every step, the latest steps verbatim, and the selected steps in full.
v2_prompt = v2["answer_prompt"]
full_at = v2_prompt.index("Full text of the most relevant earlier steps")
heads = [int(x) for x in re.findall(r"<step (\d+)> \[summary\]", v2_prompt[:full_at])]
hot = [int(x) for x in re.findall(r"(?m)^<step (\d+)>\n", v2_prompt[:full_at])]
full = [int(x) for x in re.findall(r"(?m)^<step (\d+)>", v2_prompt[full_at:])]
head_text = {int(n): t.strip() for n, t in re.findall(r"<step (\d+)> \[summary\] ([^\n]*)", v2_prompt)}

case = {
    "source": {"benchmark": "AMA-Bench test split (MIT)", "episode": ep["episode_id"], "domain": ep["domain"],
               "task_type": ep["task_type"], "question_uuid": qa["question_uuid"],
               "run": "paired re-run 2026-10-08, rows ep=34 qi=3 (docs/AMA_AGENT_PAIRED.md)"},
    "task": ep["task"], "num_steps": ep["num_turns"], "trajectory_tokens": ep["total_tokens"],
    "question": qa["question"], "reference": qa["answer"], "question_type": qa["type"],
    "deciding_steps": [step(5), step(6), step(17)],
    "ama_agent": {"arm": "A1", "retrieved": [n for n in a1_turns if n not in named], "named": named,
                  "steps": [step(n) for n in a1_turns], "prompt_tokens": a1["answer_prompt_tokens"],
                  "check_question": check_q, "check_output": verdict, "reasoning_quotes": quotes,
                  "answer": a1["pred"], "judge": a1["judge_response"].split("</think>")[-1].strip(),
                  "score": a1["score"]},
    "verakv": {"arm": "V2", "config": "ama/configs/cfg_kvmem_causal.json (lexical router + step pin)",
               "heads": [min(heads), max(heads)], "hot": hot, "full": full,
               "head_lines": {str(n): head_text[n] for n in (5, 6, 17)},
               "prompt_tokens": v2["answer_prompt_tokens"], "model_calls_in_memory": 0,
               "answer": v2["pred"], "judge": v2["judge_response"].split("</think>")[-1].strip(),
               "score": v2["score"]},
}
assert case["ama_agent"]["score"] == 0.0 and case["verakv"]["score"] == 1.0
json.dump(case, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", dst, "| AMA-Agent turns", a1_turns, "| VeraKV heads %d-%d, hot %s, full %s" % (min(heads), max(heads), hot, full))
