"""Cross-judge: re-score already-generated answers with a DIFFERENT-family judge.

Our controlled ablations use Qwen3-32B as both reader and judge; a reviewer worry is that Qwen judges
its own generations leniently. We take the saved (question, gold, arm, answer) rows and re-judge every
answer with a non-Qwen model (Llama-3.1-8B), then check whether the per-arm ACCURACY ORDERING is
preserved. If verbatim>summary>facts (payload) and pin/lex orderings survive a different-family judge,
the conclusions are not a judge artifact.

    SPRAG_MODEL_PATH=/path/to/Llama-3.1-8B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.cross_judge --ans ./out/allans.jsonl --tag llama-3.1-8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink


def judge(llm, head, tail, q, gold, ans):
    body = (f"You are grading an answer to a question about an agent trajectory.\n"
            f"Question: {q}\nReference answer: {gold}\nCandidate answer: {ans}\n\n"
            "Is the candidate correct and consistent with the reference answer? "
            "Reply with exactly one word: yes or no.")
    out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(head + body + tail), 6)
    return out.strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ans", required=True, help="JSONL rows: {q, gold, qt, arm, ans}")
    ap.add_argument("--tag", default=os.environ.get("SPRAG_MODEL_PATH", "judge").split("/")[-1])
    ap.add_argument("--out", default="./out/crossjudge.json")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    rows = [json.loads(l) for l in open(args.ans) if l.strip()]

    acc = defaultdict(int)
    tot = defaultdict(int)
    rowf = open(args.out + ".rows.jsonl", "w")
    byqt = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # arm -> qt -> [tot, hit]
    for i, r in enumerate(rows):
        ok = judge(llm, head, tail, r["q"], r["gold"], r.get("ans", ""))
        rowf.write(json.dumps({"arm": r["arm"], "qt": r.get("qt"),
                               "conv": r.get("conv"), "ok": int(ok)}) + "\n")
        acc[r["arm"]] += ok
        tot[r["arm"]] += 1
        d = byqt[r["arm"]][r.get("qt", "?")]
        d[0] += 1; d[1] += ok
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        json.dump({arm: [acc[arm], tot[arm]] for arm in tot}, open(args.out, "w"), indent=2)

    rowf.close()
    print("\n" + "=" * 64)
    print(f"CROSS-JUDGE by {args.tag} (different-family) on {len(rows)} saved answers")
    print("=" * 64)
    for arm in sorted(tot):
        line = "  ".join(f"{qt}:{byqt[arm][qt][1]}/{byqt[arm][qt][0]}" for qt in sorted(byqt[arm]))
        print(f"  {arm:12s}: {acc[arm]}/{tot[arm]} = {100*acc[arm]/max(1,tot[arm]):.1f}%   {line}")
    print(f"CROSSJUDGE_DONE [{args.tag}]")


if __name__ == "__main__":
    main()
