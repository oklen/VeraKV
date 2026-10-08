"""Oracle-evidence PAYLOAD ablation (review #6): isolate the effect of payload FORM.

Give every arm the SAME oracle evidence (the exact step a question cites) and vary only how that
evidence is presented to the reader:
  * verbatim : the raw step text (action/observation)               <- kvmemory's substrate
  * summary  : an LLM 1-2 sentence summary of that step             <- "summarize" memory
  * facts    : LLM-extracted atomic facts (bullets) from that step  <- "extract facts" memory
Same reader, same task context, same answer prompt, same question, same judge (all Qwen3-32B). Only
the payload form differs, so a gap is causally the payload -- the reviewer's cleanest test that the
VERBATIM SUBSTRATE, not the router, is what drives the win. Restricted to step-citing questions so the
gold evidence is unambiguous (the cited turn), which also removes retrieval as a confound.

Reader+judge = the live Qwen3-32B vLLM server on localhost:8056 (thinking disabled). No GPU use here:
this is a pure HTTP client, shardable across the two workers' servers.

    python -m kvmemory.kv_payload --data ./data/ama_test.jsonl --port 8056 --shard 0 --nshards 2 \
        --max_ep 60 --max_qa 8 --out ./out/payload_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# served-model-name and port of the local vLLM server hosting the Qwen3-32B reader/judge
MODEL = os.environ.get("VLLM_MODEL_NAME", "Qwen3-32B")
PORT = int(os.environ.get("VLLM_PORT", "8056"))
_THINK = re.compile(r"<think>.*?</think>", re.S)
_CITE = re.compile(r"\b(?:turn|step)s?\s*#?\s*(\d+)", re.I)


def vchat(messages, max_tokens=256, port=PORT, retries=4):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    url = f"http://localhost:{port}/v1/chat/completions"
    last = ""
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            txt = out["choices"][0]["message"]["content"]
            return _THINK.sub("", txt).strip()
        except Exception as e:  # noqa
            last = str(e)
    return f"[vllm-error: {last}]"


def cited_turns(q):
    out = []
    for m in _CITE.finditer(q):
        out.append(int(m.group(1)))
    # ranges "steps 28-30"
    for a, b in re.findall(r"(?:turns?|steps?)\s*#?\s*(\d+)\s*[-–to]+\s*#?\s*(\d+)", q, re.I):
        out.extend(range(int(a), int(b) + 1))
    return sorted(set(out))


def seg_text(t):
    a = (t.get("action") or "").strip()
    o = (t.get("observation") or "").strip()
    return "\n".join(([f"action: {a}"] if a else []) + ([f"observation: {o}"] if o else []))


def load_ama(path, max_tokens=32000):
    eps = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("total_tokens", 0) > max_tokens:
            continue
        by_turn = {}
        for t in r["trajectory"]:
            txt = seg_text(t)
            if txt:
                by_turn[t["turn_idx"]] = txt
        eps.append({"id": r["episode_id"], "domain": r["domain"], "task": r.get("task", ""),
                    "by_turn": by_turn, "qa": r["qa_pairs"]})
    return eps


def summarize(text, cache):
    if ("s", text) not in cache:
        cache[("s", text)] = vchat(
            [{"role": "system", "content": "Summarize the following agent step in 1-2 sentences, "
              "preserving the key entities but written as prose."},
             {"role": "user", "content": text}], max_tokens=160)
    return cache[("s", text)]


def facts(text, cache):
    if ("f", text) not in cache:
        cache[("f", text)] = vchat(
            [{"role": "system", "content": "Extract the atomic facts from the following agent step as "
              "a short bullet list. One fact per line, no commentary."},
             {"role": "user", "content": text}], max_tokens=200)
    return cache[("f", text)]


def answer(task, payload, question):
    sys_p = ("You are answering a question about a completed agent trajectory, using the provided "
             "evidence. Answer concisely and specifically; cite exact values.")
    usr = f"Task: {task}\n\nEvidence:\n{payload}\n\nQuestion: {question}\nAnswer:"
    return vchat([{"role": "system", "content": sys_p}, {"role": "user", "content": usr}], max_tokens=200)


def judge(question, gold, cand):
    usr = (f"Question: {question}\nReference answer: {gold}\nCandidate answer: {cand}\n\n"
           "Is the candidate correct and consistent with the reference? Reply exactly one word: yes or no.")
    out = vchat([{"role": "system", "content": "You grade answers. Reply yes or no."},
                 {"role": "user", "content": usr}], max_tokens=6)
    return out.strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max_ep", type=int, default=60)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="./out/payload.json")
    ap.add_argument("--ans_out", default="", help="if set, save (q,gold,qt,arm,ans) JSONL for cross-judge")
    args = ap.parse_args()

    eps = load_ama(args.data)
    eps.sort(key=lambda e: e["id"])
    eps = [e for i, e in enumerate(eps) if i % args.nshards == args.shard][:args.max_ep]

    # build the work list: step-citing QA whose cited turn is present -> oracle evidence
    items = []
    for e in eps:
        n = 0
        for qa in e["qa"]:
            if n >= args.max_qa:
                break
            turns = [t for t in cited_turns(qa["question"]) if t in e["by_turn"]]
            if not turns:
                continue
            ev = "\n\n".join(f"<step {t}>\n{e['by_turn'][t]}" for t in turns)
            items.append({"task": e["task"], "ev": ev, "q": qa["question"],
                          "gold": qa.get("answer", "") or "", "qt": qa.get("type", "?")})
            n += 1
    print(f"shard {args.shard}/{args.nshards}: {len(eps)} eps -> {len(items)} step-citing "
          f"oracle-evidence QA (port {args.port})", flush=True)

    cache = {}
    arms = ["verbatim", "summary", "facts"]
    acc = {a: 0 for a in arms}
    byqt = {a: defaultdict(lambda: [0, 0]) for a in arms}
    done = [0]

    ans_rows = []

    def work(it):
        payloads = {"verbatim": it["ev"], "summary": summarize(it["ev"], cache),
                    "facts": facts(it["ev"], cache)}
        res, ansd = {}, {}
        for a in arms:
            ans = answer(it["task"], payloads[a], it["q"])
            ansd[a] = ans
            res[a] = judge(it["q"], it["gold"], ans)
        return it, res, ansd

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for it, res, ansd in ex.map(work, items):
            qt = it["qt"]
            for a in arms:
                acc[a] += res[a]
                byqt[a][qt][0] += 1
                byqt[a][qt][1] += res[a]
                if args.ans_out:
                    ans_rows.append({"q": it["q"], "gold": it["gold"], "qt": qt,
                                     "arm": a, "ans": ansd[a]})
            done[0] += 1
            if done[0] % 20 == 0:
                print(f"  ...{done[0]}/{len(items)} | " +
                      " ".join(f"{a} {acc[a]}" for a in arms), flush=True)
            json.dump({"shard": args.shard, "n": done[0], "acc": acc,
                       "byqt": {a: {k: v for k, v in byqt[a].items()} for a in arms}},
                      open(args.out, "w"), indent=2)

    if args.ans_out:
        with open(args.ans_out, "w") as f:
            for r in ans_rows:
                f.write(json.dumps(r) + "\n")

    n = len(items)
    print("\n" + "=" * 64)
    print(f"PAYLOAD ABLATION shard {args.shard} ({n} oracle-evidence QA, Qwen3-32B reader+judge)")
    print("=" * 64)
    for a in arms:
        line = "  ".join(f"{qt}:{byqt[a][qt][1]}/{byqt[a][qt][0]}" for qt in sorted(byqt[a]))
        print(f"  {a:9s}: {acc[a]}/{n} = {100*acc[a]/max(1,n):.1f}%   {line}")
    print(f"PAYLOAD_DONE shard {args.shard}")


if __name__ == "__main__":
    main()
