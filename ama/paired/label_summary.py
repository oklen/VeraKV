"""Summarise the model labels of the paired AMA re-run and draw the hand-check sample.

    python ama/paired/label_summary.py <labels.jsonl> [--sample 24 --seed 7 --sample-out sample.jsonl]

Distribution of error / needs / evidence by pair and by which arm was right; the sample is a seeded random
draw stratified by pair, written with everything needed to check each label by hand.
"""
import argparse
import json
import random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--sample", type=int, default=24)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--sample-out", default="")
args = ap.parse_args()

rows = {}
for l in open(args.labels, encoding="utf-8"):
    r = json.loads(l)
    k = (r["pair"], r["ep"], r["qi"])
    if k not in rows or rows[k].get("error") == "unparsed":
        rows[k] = r
rows = list(rows.values())
print("labels", len(rows), "unparsed", sum(1 for r in rows if r.get("error") == "unparsed"),
      "effort", dict(Counter(r.get("effort") for r in rows)))
groups = defaultdict(list)
for r in rows:
    groups["%s | %s right" % (r["pair"], r["right_arm"])].append(r)
    if r["pair"] == "V1_vs_A1":
        groups["%s | %s right | A1 %s" % (r["pair"], r["right_arm"],
                                          "direct" if (r["wrong_direct"] if r["wrong_arm"] == "A1" else r["right_direct"]) else "reader")].append(r)
        if r["qtype"] == "C":
            groups["%s | %s right | state update" % (r["pair"], r["right_arm"])].append(r)
OUT = {}
for g in sorted(groups):
    rs = groups[g]
    n = len(rs)
    d = {}
    print("\n== %s (n=%d)" % (g, n))
    for field in ("error", "needs", "evidence_w", "evidence_r"):
        c = Counter(r.get(field) for r in rs)
        d[field] = {k: v for k, v in c.most_common()}
        print("  %-11s %s" % (field, "  ".join("%s %d%%" % (k, round(100 * v / n)) for k, v in c.most_common())))
    OUT[g] = {"n": n, **d}

if args.sample_out:
    rng = random.Random(args.seed)
    by_pair = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["pair"], r["ep"], r["qi"])):
        by_pair[r["pair"]].append(r)
    total = sum(len(v) for v in by_pair.values())
    pick = []
    for p, rs in sorted(by_pair.items()):
        k = max(2, round(args.sample * len(rs) / total)) if rs else 0
        pick += rng.sample(rs, min(k, len(rs)))
    with open(args.sample_out, "w", encoding="utf-8") as fh:
        for r in pick:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nhand-check sample:", len(pick), {p: sum(1 for r in pick if r["pair"] == p) for p in by_pair})
