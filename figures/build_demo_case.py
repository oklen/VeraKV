"""Build figures/demo_case.json, the data behind the README demo (figures/make_demo.py, figures/make_context.py).

    python figures/build_demo_case.py <case.json> <ama_memory.json> <ama_build_calls.json> <verakv_build.json> \
        figures/demo_case.json

<ama_build_calls.json>: the 14 logged LLM calls that wrote AMA-Agent's memory for this episode (start time,
seconds, prompt and generated tokens, and the steps each one summarised, read from its prompt) and the logged
wall time of that build. <verakv_build.json>: VeraKV's
memory build for the same episode, timed with figures/time_index_build.py.

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

src, mem_src, calls_src, vbuild_src, dst = sys.argv[1:6]
d = json.load(open(src, encoding="utf-8"))
mem = json.load(open(mem_src, encoding="utf-8"))
bcalls = json.load(open(calls_src, encoding="utf-8"))
vbuild = json.load(open(vbuild_src, encoding="utf-8"))
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
full_at = v2_prompt.index("Full text of the most relevant earlier steps")
heads = [int(x) for x in re.findall(r"<step (\d+)> \[summary\]", v2_prompt[:full_at])]
hot = [int(x) for x in re.findall(r"(?m)^<step (\d+)>\n", v2_prompt[:full_at])]
full = [int(x) for x in re.findall(r"(?m)^<step (\d+)>", v2_prompt[full_at:])]
ranges = mem["turn_ranges"]
assert ranges[0][0] == 0 and ranges[-1][1] == ep["num_turns"] - 1
assert all(b[0] == a[1] + 1 for a, b in zip(ranges, ranges[1:])), "summaries must tile the trajectory"

calls = sorted(bcalls["calls"], key=lambda c: c["steps"][0])
assert [c["steps"] for c in calls] == ranges, "one logged call per stored summary"
starts = [c["t0"] for c in calls]
assert max(starts) - min(starts) < 1.0, "the summary calls were issued together"


def lines_of(block, keep):
    """The first `keep` lines of a prompt block, each cut after its first sentence that ends a line."""
    out = []
    for line in block.split("\n"):
        if line.strip():
            out.append(line.rstrip())
        if len(out) == keep:
            break
    return out


def cut(line, marker):
    """A prompt line up to and including `marker`, with '…' when the line goes on."""
    i = line.index(marker) + len(marker)
    return line[:i] + (" …" if line[i:].strip() else "")


# AMA-Agent's reader prompt, section by section (verbatim lines, long ones cut with '…').
a_lines = a1_prompt.split("\n")
turn_blocks = []
for n in a1_turns:
    i = a_lines.index("Turn %d:" % n)
    act, obs = a_lines[i + 1].rstrip(), a_lines[i + 2].rstrip()
    if len(obs) > 74:                                   # keep the first sentence of a long observation
        obs = obs.split(". ")[0] + ". …"
    turn_blocks.append(["Turn %d:" % n, act, obs])
proc = [l for l in a_lines if l.startswith("Step ")]
arrows = [l.strip() for l in a_lines if l.strip().startswith("→")]
a_ctx = [
    {"title": "Role", "lines": [a_lines[0]]},
    {"title": "Query", "lines": [cut(next(l for l in a_lines if l.startswith("Query:")), "step 77?")]},
    {"title": "Retrieved turns", "size": "each turn also lists its available actions, left out here",
     "header": next(l for l in a_lines if l.startswith("Retrieved Turns")), "turns": turn_blocks},
    {"title": "Note", "lines": [cut(next(l for l in a_lines if l.startswith("IMPORTANT:")), "selected by similarity.")]},
    {"title": "Decision procedure", "header": next(l for l in a_lines if l.startswith("# Decision procedure")),
     "lines": ["…", cut(proc[1], "retrieved turns above"), arrows[1], cut(proc[2], "inspect to answer?"), arrows[2],
               "…"]},
    {"title": "Output formats", "header": next(l for l in a_lines if l.startswith("# Output formats")),
     "lines": ["SUFFICIENT", "ANSWER: <your complete and accurate answer>", "NEED_GRAPH: <spec>", "…",
               "NEED_CODE: <short description of what to compute>"]},
]
for s in a_ctx:
    for l in s.get("lines", []) + [s.get("header", "")]:
        assert l == "…" or l.rstrip(" …") in a1_prompt, l

# VeraKV's reader prompt, section by section.
v_lines = v2_prompt.split("\n")
over = {int(n): l for l in v_lines for n in re.findall(r"^<step (\d+)> \[summary\]", l)}


def over_line(n):
    l = over[n]
    obs = l.split(" obs=", 1)[1]
    two = ". ".join(obs.split(". ")[:2]) + "."
    first = two if len(two) <= 50 else obs.split(". ")[0] + "."
    return l.split(" obs=", 1)[0] + " obs=" + first + " …"


hot0 = v_lines.index("<step %d>" % hot[0])
full_hdr = next(l for l in v_lines if l.startswith("Full text of the most relevant earlier steps"))
fi = v_lines.index("<step 77>", v_lines.index(full_hdr))
v_ctx = [
    {"title": "Overview: one line per step", "size": "steps %d–%d" % (min(heads), max(heads)),
     "lines": [over_line(n) for n in (0, 4, 5, 6, 16, 17, max(heads))], "gaps_after": [0, 6, 17]},
    {"title": "The last %d steps in full" % len(hot), "size": "steps %d–%d" % (hot[0], hot[-1]),
     "lines": [v_lines[hot0], v_lines[hot0 + 1], cut(v_lines[hot0 + 2], "countertop 2.")]},
    {"title": "Selected steps in full", "header": full_hdr, "size": "steps %s" % ", ".join(map(str, full)),
     "lines": [v_lines[fi], v_lines[fi + 1], cut(v_lines[fi + 2], "countertop 2.")]},
    {"title": "Question", "lines": [l for l in v_lines if l in ("## Questions",)] +
     [cut(next(l for l in v_lines if l.startswith("Question 1:")), "step 77?")]},
    {"title": "Instruction", "lines": ["## Instructions", next(l for l in v_lines if l.startswith("Provide a"))]},
]
for s in v_ctx:
    for l in s.get("lines", []) + [s.get("header", "")]:
        assert l == "…" or l.rstrip(" …") in v2_prompt, l

case = {
    "source": {"benchmark": "AMA-Bench test split (MIT)", "episode": ep["episode_id"], "domain": ep["domain"],
               "task_type": ep["task_type"], "question_uuid": qa["question_uuid"],
               "run": "paired re-run 2026-10-08, rows ep=34 qi=3 (docs/AMA_AGENT_PAIRED.md)",
               "memory": mem["source"], "build_calls": bcalls["source"], "verakv_build": vbuild["how"]},
    "num_steps": ep["num_turns"], "question": qa["question"], "reference": qa["answer"],
    "deciding_steps": [step(5), step(6), step(17)],
    "memory_build": {"ama_agent_llm_calls": len(ranges), "ama_agent_turn_ranges": ranges,
                     "ama_agent_session_chars": mem["session_size"],
                     "ama_agent_wall_seconds": bcalls["build_wall_seconds_logged"],
                     "ama_agent_call_seconds": [c["dt"] for c in calls],      # per turn range, same order
                     "ama_agent_generated_tokens": sum(c["usage"][1] for c in calls),
                     "verakv_model_calls": 0, "verakv_median_ms": vbuild["median_ms"],
                     "verakv_runs": vbuild["repeats"], "verakv_machine": vbuild["machine"]},
    "contexts": {"ama_agent": a_ctx, "verakv": v_ctx},
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
