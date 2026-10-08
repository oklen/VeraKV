"""bg_judge.py -- official judging for the deployment-map arms (gsk / g_txt).
br_judge.py variant: per-arm ROW FILTERING (g_txt exists only on harvest episodes; judging
empty predictions would pollute the mean), plus harvest-slice split in the summary.

Run from inside an AMA-Bench checkout with the official Qwen3-32B judge served by vLLM (one shard
per judge replica; configs/qwen3-32B-p0.yaml = AMA-Bench's qwen3-32B.yaml pointed at replica 0):
    cd $AMA_BENCH && AMA_BENCH=$PWD python /path/to/VeraKV/ama/bg_judge.py \
        --rows_glob '/path/to/VeraKV/out/bg_o/*.jsonl' --config configs/qwen3-32B-p0.yaml --shard 0 --nshards 4
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.environ.get("AMA_BENCH", "."))  # an AMA-Bench checkout (run from inside it)
from src.evaluate import evaluate_batch          # official
from src.model_client import ModelClient          # official

ap = argparse.ArgumentParser()
ap.add_argument("--rows_glob", default="./out/bg_o/*.jsonl")
ap.add_argument("--config", default="configs/qwen3-32B.yaml")
ap.add_argument("--out_dir", default="./out/bgj_out")
ap.add_argument("--workers", type=int, default=24)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--arms", default="gsk,g_txt")
args = ap.parse_args()
ARMS = [a for a in args.arms.split(",") if a]

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
print("unique QA rows (shard %d/%d): %d" % (args.shard, args.nshards, len(rows)), flush=True)
os.makedirs(args.out_dir, exist_ok=True)
judge = ModelClient(args.config, server_type="vllm")

for arm in ARMS:
    outf = os.path.join(args.out_dir, "judged_%s_s%d.json" % (arm, args.shard))
    if os.path.exists(outf):
        res = json.load(open(outf))
    else:
        sub = [r for r in rows if r.get("pred_" + arm)]
        qa = [{"episode_id": r["episode_id"], "question": r["question"],
               "question_uuid": r.get("question_uuid"),
               "predicted_answer": r.get("pred_" + arm) or "",
               "golden_answer": r.get("golden_answer") or "",
               "task_description": r.get("task_description", ""),
               "task_type": r.get("task_type", ""),
               "domain": r.get("domain", ""),
               "harvest": r.get("harvest", 0)} for r in sub]
        res = evaluate_batch(qa, judge, max_workers=args.workers)
        # evaluate_batch may drop custom keys; re-attach harvest by uuid
        hv = {r.get("question_uuid") or r["question"]: r.get("harvest", 0) for r in sub}
        for it in res:
            it["harvest"] = hv.get(it.get("question_uuid") or it.get("question"), 0)
        json.dump(res, open(outf, "w"))
    sc = [it.get("score", 0) for it in res]
    hs = [it.get("score", 0) for it in res if it.get("harvest")]
    os_ = [it.get("score", 0) for it in res if not it.get("harvest")]
    print("ARM %s shard %d: n=%d mean=%.4f | harvest n=%d mean=%.4f | over n=%d mean=%.4f"
          % (arm, args.shard, len(sc), sum(sc) / max(1, len(sc)),
             len(hs), sum(hs) / max(1, len(hs)),
             len(os_), sum(os_) / max(1, len(os_))), flush=True)
print("BGJUDGE_DONE shard=%d" % args.shard, flush=True)
