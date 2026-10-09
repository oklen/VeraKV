# -*- coding: utf-8 -*-
"""Label the disagreements of the paired AMA re-run with a strong model (we used GPT-5.6-Sol; docs/AMA_AGENT_PAIRED.md).

For each item the labeller sees the question, the reference answer, both answers with their verdicts, and both
exact reader prompts (what each answering model saw; cut to fit, the cut is recorded). It does not see which
system produced which side. Fixed categories; JSON reply.

    python3 ama/paired/label.py <items.jsonl> <out.jsonl> [limit]

The model is any OpenAI-compatible chat endpoint, set by environment variables only:
    LABEL_BASE_URL (serves POST <base>/chat/completions), LABEL_API_KEY, LABEL_MODEL
Resumable: skips items already in out. Reasoning effort high (sent as `reasoning_effort`); if a call with high
fails, retries once with medium and records which was used.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import time
import urllib.error
import urllib.request


def chat(messages, max_tokens, effort):
    """One chat completion from LABEL_* (retries rate limits and server errors with backoff)."""
    base, key, model = (os.environ.get("LABEL_BASE_URL", "").rstrip("/"), os.environ.get("LABEL_API_KEY", ""),
                        os.environ.get("LABEL_MODEL", ""))
    if not (base and key and model):
        raise SystemExit("set LABEL_BASE_URL, LABEL_API_KEY and LABEL_MODEL")
    body = json.dumps({"model": model, "messages": messages, "max_completion_tokens": max_tokens,
                       "reasoning_effort": effort}).encode("utf-8")
    err = None
    for attempt in range(8):
        req = urllib.request.Request(base + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            err = "HTTP %s" % e.code
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                raise RuntimeError(err)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            err = repr(e)[:200]
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError("gave up: %s" % err)

MAX_CTX_CHARS = 120000

ERRORS = {
    "missing_info": "says the information is not available / cannot be determined, or stays vague where specifics were needed",
    "wrong_value": "commits to a specific value, step, object or state that is incorrect (e.g. from another step, stale, miscounted)",
    "incomplete": "gets part of what was asked (some of the changes / items / steps) but misses others",
    "wrong_reasoning": "has the relevant facts but draws a wrong conclusion or explanation",
    "judge_noise": "the two answers are essentially equivalent, or the reference / verdict looks inconsistent",
    "other": "none of the above (explain in reason)",
}
NEEDS = {
    "one_step": "a fact at one specific step",
    "several_steps": "facts at two or more specific steps (compare / track between named steps)",
    "whole_trajectory": "a scan over the whole trajectory or a long range (first / last / all changes / counts)",
    "long_observation_detail": "a specific detail inside a long observation (page, table, file, tool output)",
    "reasoning": "mainly reasoning about cause / effect or strategy rather than retrieving facts",
}
EVIDENCE = {
    "yes": "the context contains the facts needed to produce the reference answer",
    "partial": "the context contains some but not all of the needed facts",
    "no": "the needed facts are not in the context",
}

PROMPT = """You are auditing a question-answering experiment on agent trajectories (AMA-Bench). Two memory systems
answered the same question about the same trajectory, each by reading its own context (shown below). One answer
was judged correct, the other wrong.

Question: {q}

Reference answer: {gold}

=== Side R (judged CORRECT) ===
Answer: {r_ans}
Context this side's model read{r_cut}:
<<<
{r_ctx}
>>>

=== Side W (judged WRONG) ===
Answer: {w_ans}
Context this side's model read{w_cut}:
<<<
{w_ctx}
>>>

Answer four things.
1) error: why is side W's answer wrong? Pick exactly one:
{errors}
2) needs: what does the question need from the trajectory? Pick exactly one:
{needs}
3) evidence_w: does side W's context contain what is needed to give the reference answer? Pick one:
{evidence}
4) evidence_r: the same for side R's context.

Reply with JSON only: {{"error": "...", "needs": "...", "evidence_w": "...", "evidence_r": "...", "reason": "<= 30 words"}}"""


def cut(s):
    s = s or ""
    if len(s) <= MAX_CTX_CHARS:
        return s, "", 0
    h = MAX_CTX_CHARS * 2 // 3
    t = MAX_CTX_CHARS - h
    return s[:h] + "\n...[cut for length]...\n" + s[-t:], " (cut for length: %d of %d chars shown)" % (MAX_CTX_CHARS, len(s)), len(s) - MAX_CTX_CHARS


def label(it):
    r_ctx, r_note, r_lost = cut(it["right"]["prompt"])
    w_ctx, w_note, w_lost = cut(it["wrong"]["prompt"])
    prompt = PROMPT.format(q=it["question"], gold=it["gold"], r_ans=it["right"]["answer"][:4000],
                           w_ans=it["wrong"]["answer"][:4000], r_ctx=r_ctx, w_ctx=w_ctx, r_cut=r_note, w_cut=w_note,
                           errors="\n".join("- %s: %s" % kv for kv in ERRORS.items()),
                           needs="\n".join("- %s: %s" % kv for kv in NEEDS.items()),
                           evidence="\n".join("- %s: %s" % kv for kv in EVIDENCE.items()))
    effort, text, err = "high", None, None
    for eff in ("high", "medium"):
        try:
            text = chat([{"role": "user", "content": prompt}], max_tokens=16000, effort=eff)
            effort = eff
            break
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:300]
    out = {}
    if text:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            out = json.loads(m.group(0)) if m else {}
        except ValueError:
            out = {}
    rec = {k: it[k] for k in ("pair", "ep", "qi", "domain", "qtype", "question", "gold")}
    rec.update(right_arm=it["right"]["arm"], wrong_arm=it["wrong"]["arm"],
               right_answer=it["right"]["answer"], wrong_answer=it["wrong"]["answer"],
               wrong_direct=it["wrong"]["direct"], right_direct=it["right"]["direct"],
               cut_chars_right=r_lost, cut_chars_wrong=w_lost, effort=effort if text else None, call_err=err,
               error=out.get("error", "unparsed"), needs=out.get("needs", "unparsed"),
               evidence_w=out.get("evidence_w", "unparsed"), evidence_r=out.get("evidence_r", "unparsed"),
               reason=out.get("reason", (text or "")[:300]))
    return rec


def main():
    src, dst = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    items = [json.loads(l) for l in open(src, encoding="utf-8")][:limit]
    done = set()
    if os.path.exists(dst):
        for l in open(dst, encoding="utf-8"):
            r = json.loads(l)
            if r.get("error") != "unparsed":
                done.add((r["pair"], r["ep"], r["qi"]))
    todo = [it for it in items if (it["pair"], it["ep"], it["qi"]) not in done]
    print("items %d, done %d, todo %d" % (len(items), len(done), len(todo)), flush=True)
    with open(dst, "a", encoding="utf-8") as fh, ThreadPoolExecutor(12) as ex:
        for rec in ex.map(label, todo):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
