"""br_judge.py -- score the bridge answers with the OFFICIAL AMA-Bench judge.

Run with the official Qwen3-32B judge served by vLLM (AMA-Bench's configs/qwen3-32B.yaml, thinking on),
from inside an AMA-Bench checkout, on the answer files written by kvmemory.kv_ama_bridge:
    cd $AMA_BENCH && AMA_BENCH=$PWD python /path/to/VeraKV/ama/br_judge.py \
        --rows_glob '/path/to/VeraKV/out/br_o/br_ans_*.jsonl' --config configs/qwen3-32B.yaml
Judges each arm (tx / iso / b_hot) with the identical official pipeline (evaluate_batch ->
compute_llm_as_judge), writes per-arm judged files + a paired summary.
"""
import argparse, glob, json, os, sys
from math import comb

sys.path.insert(0, os.environ.get("AMA_BENCH", "."))  # an AMA-Bench checkout (run from inside it)
from src.evaluate import evaluate_batch          # official
from src.model_client import ModelClient          # official

ARMS = ["tx", "iso", "b_hot"]

ap = argparse.ArgumentParser()
ap.add_argument("--rows_glob", default="./out/br_o/br_ans_*.jsonl")
ap.add_argument("--config", default="configs/qwen3-32B.yaml")
ap.add_argument("--out_dir", default="./out/brj_out")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--arms", default="tx,iso,b_hot")
args = ap.parse_args()
ARMS[:] = [a for a in args.arms.split(",") if a]

rows, seen = [], set()
for f in sorted(glob.glob(args.rows_glob)):
    for l in open(f):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        key = (r["episode_id"], r.get("question_uuid") or r["question"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
rows = rows[args.shard::args.nshards]
print("unique QA rows (shard %d/%d):" % (args.shard, args.nshards), len(rows))
os.makedirs(args.out_dir, exist_ok=True)
judge = ModelClient(args.config, server_type="vllm")

scored = {}
for arm in ARMS:
    outf = os.path.join(args.out_dir, "judged_%s_s%d.json" % (arm, args.shard))
    if os.path.exists(outf):
        res = json.load(open(outf))
    else:
        qa = [{"episode_id": r["episode_id"], "question": r["question"],
               "question_uuid": r.get("question_uuid"),
               "predicted_answer": r.get("pred_" + arm) or "",
               "golden_answer": r.get("golden_answer") or "",
               "task_description": r.get("task_description", ""),
               "task_type": r.get("task_type", ""),
               "domain": r.get("domain", "")} for r in rows]
        res = evaluate_batch(qa, judge, max_workers=args.workers)
        json.dump(res, open(outf, "w"))
    scored[arm] = {(x["episode_id"], x.get("question_uuid") or x["question"]): float(x["score"])
                   for x in res}
    print("%-6s official score = %.4f  (n=%d)" % (arm, sum(scored[arm].values()) / len(scored[arm]), len(scored[arm])))

keys = sorted(set.intersection(*(set(scored[a]) for a in ARMS)))
def mc(a, r):
    b = sum(1 for k in keys if scored[a][k] > 0.5 and scored[r][k] <= 0.5)
    c = sum(1 for k in keys if scored[r][k] > 0.5 and scored[a][k] <= 0.5)
    m = b + c
    p = min(1.0, sum(comb(m, i) for i in range(min(b, c) + 1)) * 2 / 2 ** m) if m else 1.0
    return b, c, p
print("\npaired (n=%d):" % len(keys))
for a, r in [p for p in (("b_hot", "tx"), ("b_hot", "iso"), ("tx", "iso"))
             if p[0] in ARMS and p[1] in ARMS]:
    b, c, p = mc(a, r)
    print("%s vs %s: +%d/-%d p=%.5f" % (a, r, b, c, p))
# by domain
doms = {}
for r in rows:
    doms.setdefault(r.get("domain", "?"), []).append((r["episode_id"], r.get("question_uuid") or r["question"]))
print("\nby domain:")
for d, ks in sorted(doms.items()):
    ks = [k for k in ks if k in scored[ARMS[0]]]
    if not ks:
        continue
    print("  %-14s n=%4d " % (d, len(ks)) + " ".join(
        "%s %.3f" % (a, sum(scored[a][k] for k in ks) / len(ks)) for a in ARMS))
