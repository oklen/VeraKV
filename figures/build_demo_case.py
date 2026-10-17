"""Build figures/demo_case.json, the data behind the README demo (figures/make_demo.py).

    python figures/build_demo_case.py <case.json> <ama_memory.json> figures/demo_case.json

<case.json> holds the raw paired re-run rows for one question (all arms, with the full prompts and raw model
outputs that results/ama_paired/ leaves out because of their size) and that question's AMA-Bench episode:
{"rows": {arm: row}, "episode": <line of the AMA-Bench test file>}. The question shown is episode 34
(alfworld), question 51a13dd0-f7f2-4599-b920-dedcf31fcd17, rows ep=34 qi=3 of the full run.

<ama_memory.json> lists the turn ranges of the state summaries in AMA-Agent's stored memory for that
episode, from the same run: one LLM call per summary when the memory was written.

Every string and number the demo shows is copied from these inputs, never written by hand: the question
and reference answer, the steps each reader got (parsed from its logged prompt), AMA-Agent's check
output and a sentence of its reasoning, the answers, judge verdicts, model calls and wall-clock seconds."""
import json
import re
import sys

src, mem_src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(src, encoding="utf-8"))
mem = json.load(open(mem_src, encoding="utf-8"))
ep, a1, v2, a3 = d["episode"], d["rows"]["A1"], d["rows"]["V2"], d["rows"]["A3"]
qa = next(q for q in ep["qa_pairs"] if q["question"] == a1["question"])
traj = {t["turn_idx"]: t for t in ep["trajectory"]}


def first_par(text):
    """An observation up to its list of available actions: what happened at that step."""
    return text.split("\n\nThe current available actions")[0].strip()


def step(n):
    return {"step": n, "action": traj[n]["action"], "observation": first_par(traj[n]["observation"])}


def run(r):
    return {"arm": r["arm"], "model_calls": r["n_calls"], "call_kinds": [c[0] for c in r["calls"]],
            "generated_tokens": sum(c[2] for c in r["calls"]), "seconds": r["dt"], "answer": r["pred"],
            "judge": r["judge_response"].split("</think>")[-1].strip(), "score": r["score"]}


# AMA-Agent: the turns in the prompt of its sufficiency check (it answered from that call).
a1_prompt, resp = a1["answer_prompt"], a1["answer_response"]
a1_turns = [int(x) for x in re.findall(r"(?m)^Turn (\d+):", a1_prompt)]
named = [int(x) for x in re.findall(r"(?i)\bstep (\d+)\b", qa["question"])]
verdict = resp.split("</think>")[-1].strip().split("\n")[0].strip()
quote = "But the retrieved turns don't include turn 5."
assert quote in resp

# VeraKV: the first lines of every step (its index), read at answer time with no model call.
v2_prompt = v2["answer_prompt"]
head_text = {int(n): t.strip() for n, t in re.findall(r"<step (\d+)> \[summary\] ([^\n]*)", v2_prompt)}
ranges = mem["turn_ranges"]
assert ranges[0][0] == 0 and ranges[-1][1] == ep["num_turns"] - 1
assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:])), "summaries must tile the trajectory"

case = {
    "source": {"benchmark": "AMA-Bench test split (MIT)", "episode": ep["episode_id"], "domain": ep["domain"],
               "task_type": ep["task_type"], "question_uuid": qa["question_uuid"],
               "run": "paired re-run 2026-10-08, rows ep=34 qi=3 (docs/AMA_AGENT_PAIRED.md)",
               "memory": mem["source"]},
    "num_steps": ep["num_turns"], "question": qa["question"], "reference": qa["answer"],
    "deciding_steps": [step(5), step(6), step(17)],
    "memory_build": {"ama_agent_llm_calls": len(ranges), "ama_agent_turn_ranges": ranges,
                     "ama_agent_session_chars": mem["session_size"], "verakv_model_calls": 0},
    "ama_agent": dict(run(a1), retrieved=[n for n in a1_turns if n not in named], named=named,
                      steps=[step(n) for n in a1_turns], check_output=verdict, reasoning_quote=quote),
    "ama_agent_no_early_answer": run(a3),
    "verakv": dict(run(v2), head_lines={str(n): head_text[n] for n in (5, 6, 17)}),
}
assert case["ama_agent"]["score"] == 0.0 and case["verakv"]["score"] == 1.0
assert case["ama_agent_no_early_answer"]["score"] == 0.0
json.dump(case, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", dst)
for k in ("ama_agent", "verakv", "ama_agent_no_early_answer"):
    r = case[k]
    print("%-26s calls %d %s, generated %d tokens, %.2f s, judge %s" % (k, r["model_calls"], r["call_kinds"],
                                                                       r["generated_tokens"], r["seconds"], r["judge"]))
print("memory build: AMA-Agent %d LLM calls, VeraKV 0" % len(ranges))
