"""Step-index robustness (review #7): is AMA performance an artifact of step-citing questions?

On the SAME Qwen3-32B reader+judge, compare, for step-citing questions:
  * pin        : give only the exact cited step, verbatim (step-pin-only baseline -- strong?)
  * lex        : lexical top-k routing, question as-is (no pin)             -- can routing find it?
  * lex_nostep : lexical top-k routing on the question with the step NUMBER REMOVED, but answer the
                 original question -- tests reliance on the literal index (the artifact worry)
  * full       : the whole trajectory if it fits the 32k window            -- ceiling
and separately report accuracy on the NON-step questions (lex) vs the step-citing ones, to see whether
the method generalizes beyond questions that hand you the turn index.

Saves every (question, gold, qtype, arm, answer) to a JSONL so a DIFFERENT-family judge can re-score it
(review #5, cross-judge). HTTP client to the local vLLM 32B server; pure-Python routing; shardable.

    python -m kvmemory.kv_stepablate --shard 0 --nshards 2 --k 5 --max_ep 90 --out ./out/step_s0.json \
        --ans_out ./out/step_ans_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.kv_payload import cited_turns, judge, load_ama, vchat  # noqa: E402

_TOK = re.compile(r"[a-z0-9]+")
_STEP = re.compile(r"\b(?:turns?|steps?)\s*#?\s*\d+(?:\s*[-–to]+\s*#?\s*\d+)?", re.I)


def toks(s):
    return set(_TOK.findall(s.lower()))


def lex_topk(question, by_turn, k):
    q = toks(question)
    scored = sorted(by_turn.items(), key=lambda kv: (len(q & toks(kv[1])), kv[0]), reverse=True)
    picked = sorted(t for t, _ in scored[:k])
    return picked


def de_step(question):
    return _STEP.sub("the relevant step", question)


def ctx_from(by_turn, turns):
    return "\n\n".join(f"<step {t}>\n{by_turn[t]}" for t in turns if t in by_turn)


def answer_ctx(task, ctx, question):
    sys_p = ("You are answering a question about a completed agent trajectory, using the provided "
             "trajectory excerpts. Answer concisely and specifically; cite exact values.")
    usr = f"Task: {task}\n\nTrajectory excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    return vchat([{"role": "system", "content": sys_p}, {"role": "user", "content": usr}], max_tokens=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_ep", type=int, default=90)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--max_full_tok", type=int, default=26000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="./out/step.json")
    ap.add_argument("--ans_out", default="./out/step_ans.jsonl")
    args = ap.parse_args()

    eps = load_ama(args.data)
    eps.sort(key=lambda e: e["id"])
    eps = [e for i, e in enumerate(eps) if i % args.nshards == args.shard][:args.max_ep]

    step_items, nonstep_items = [], []
    for e in eps:
        full_ctx = ctx_from(e["by_turn"], sorted(e["by_turn"]))
        full_ok = len(full_ctx) < args.max_full_tok * 4  # ~4 chars/token guard
        ns = nn = 0
        for qa in e["qa"]:
            turns = [t for t in cited_turns(qa["question"]) if t in e["by_turn"]]
            base = {"task": e["task"], "q": qa["question"], "gold": qa.get("answer", "") or "",
                    "qt": qa.get("type", "?"), "by_turn": e["by_turn"], "full_ctx": full_ctx,
                    "full_ok": full_ok, "cited": turns}
            if turns and ns < args.max_qa:
                step_items.append(base); ns += 1
            elif not turns and nn < 3:
                nonstep_items.append(base); nn += 1
    print(f"shard {args.shard}: {len(eps)} eps -> {len(step_items)} step-citing + "
          f"{len(nonstep_items)} non-step QA", flush=True)

    arms_step = ["pin", "lex", "lex_nostep", "full"]
    acc = defaultdict(int); tot = defaultdict(int)
    ns_acc = ns_tot = 0
    ans_rows = []
    done = [0]

    def work_step(it):
        by, k = it["by_turn"], args.k
        ctxs = {
            "pin": ctx_from(by, it["cited"]),
            "lex": ctx_from(by, lex_topk(it["q"], by, k)),
            "lex_nostep": ctx_from(by, lex_topk(de_step(it["q"]), by, k)),
            "full": it["full_ctx"] if it["full_ok"] else None,
        }
        out = {}
        for a in arms_step:
            if ctxs[a] is None:
                out[a] = None; continue
            ans = answer_ctx(it["task"], ctxs[a], it["q"])
            out[a] = (judge(it["q"], it["gold"], ans), ans)
        return it, out

    def work_ns(it):
        ctx = ctx_from(it["by_turn"], lex_topk(it["q"], it["by_turn"], args.k))
        ans = answer_ctx(it["task"], ctx, it["q"])
        return it, judge(it["q"], it["gold"], ans), ans

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for it, out in ex.map(work_step, step_items):
            for a in arms_step:
                if out[a] is None:
                    continue
                ok, ans = out[a]
                acc[a] += ok; tot[a] += 1
                ans_rows.append({"q": it["q"], "gold": it["gold"], "qt": it["qt"], "arm": a, "ans": ans})
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  step {done[0]}/{len(step_items)} | " +
                      " ".join(f"{a} {acc[a]}/{tot[a]}" for a in arms_step), flush=True)
        for it, ok, ans in ex.map(work_ns, nonstep_items):
            ns_acc += ok; ns_tot += 1
            ans_rows.append({"q": it["q"], "gold": it["gold"], "qt": it["qt"], "arm": "nonstep_lex",
                             "ans": ans})

    json.dump({"shard": args.shard, "acc": dict(acc), "tot": dict(tot),
               "nonstep": [ns_acc, ns_tot]}, open(args.out, "w"), indent=2)
    with open(args.ans_out, "w") as f:
        for r in ans_rows:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 64)
    print(f"STEP-INDEX ROBUSTNESS shard {args.shard} (Qwen3-32B reader+judge)")
    print("=" * 64)
    for a in arms_step:
        if tot[a]:
            print(f"  {a:11s}: {acc[a]}/{tot[a]} = {100*acc[a]/tot[a]:.1f}%")
    if ns_tot:
        print(f"  non-step lex: {ns_acc}/{ns_tot} = {100*ns_acc/ns_tot:.1f}%   "
              f"(vs step-citing lex {100*acc['lex']/max(1,tot['lex']):.1f}%)")
    print(f"STEP_DONE shard {args.shard}")


if __name__ == "__main__":
    main()
