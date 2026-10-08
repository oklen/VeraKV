"""Cross-family re-judge (paper App. "Judge robustness").

Re-scores the identical predictions of a full official-harness run with an independent
Llama-3.1-70B-Instruct judge through AMA-Bench's own evaluate_batch, keeping the original
Qwen3-32B judge score as `qwen_score`. Reported: Qwen3-32B 0.643, Llama-3.1-70B 0.677
[0.658, 0.695], 84% item-level agreement (n = 2,496).

Re-score (needs a vLLM server for the judge, see ama/configs/judge_llama70b.yaml), run from inside
an AMA-Bench checkout:
    cd $AMA_BENCH && python /path/to/VeraKV/analysis/rejudge_llama70b.py \
        --merged /path/to/run_results.json --config /path/to/VeraKV/ama/configs/judge_llama70b.yaml \
        --out results/judge_llama70b/ama_rejudged.json
Summarize the released output only (no server needed):
    python analysis/rejudge_llama70b.py --summary results/judge_llama70b/ama_rejudged.json
"""
import argparse
import json
import os
import random
import sys


def summarize(rej):
    n = len(rej)
    la = sum(1 for r in rej if r["score"] == 1.0) / n
    qa = sum(1 for r in rej if r["qwen_score"] == 1.0) / n
    ag = sum(1 for r in rej if (r["score"] == 1.0) == (r["qwen_score"] == 1.0)) / n
    random.seed(0)
    bs = sorted(sum(1 for r in (rej[random.randrange(n)] for _ in range(n)) if r["score"] == 1.0) / n
                for _ in range(3000))
    print(f"LLAMA_ACC={la:.4f} CI[{bs[75]:.4f},{bs[2925]:.4f}] QWEN_ACC={qa:.4f} AGREEMENT={ag:.4f} n={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="", help="summarize an existing re-judged file and exit")
    ap.add_argument("--merged", default="", help="per-question results of the run to re-judge")
    ap.add_argument("--config", default="ama/configs/judge_llama70b.yaml")
    ap.add_argument("--out", default="./out/ama_rejudged.json")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.summary:
        summarize(json.load(open(args.summary)))
        return
    sys.path.insert(0, os.environ.get("AMA_BENCH", "."))
    from src.model_client import ModelClient
    from src.evaluate import evaluate_batch
    merged = json.load(open(args.merged))
    for it in merged:
        it["qwen_score"] = it["score"]
    judge = ModelClient(config_path=args.config, server_type="vllm")
    rej = evaluate_batch(qa_results=merged, judge_client=judge, max_workers=args.workers)
    summarize(rej)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rej, open(args.out, "w"))


if __name__ == "__main__":
    main()
