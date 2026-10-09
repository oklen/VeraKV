"""Per-question cost of each arm from the logged calls (judge excluded): model calls, prompt and completion
tokens summed over the arm's calls, and wall time of the arm's answer. Medians and means.

    python ama/paired/arm_cost.py <full run dir> <a3 run dir>
"""
import glob
import gzip
import json
import sys
from collections import defaultdict


def load(run, arms):
    out = {}
    for f in sorted(glob.glob(run + "/s*/q.jsonl")) or sorted(glob.glob(run + "/s*/q.jsonl.gz")):
        for l in (gzip.open if f.endswith(".gz") else open)(f, "rt", encoding="utf-8"):
            r = json.loads(l)
            if r["arm"] in arms and not r.get("infra"):
                out[(r["arm"], r["ep"], r["qi"])] = r
    return out


rows = load(sys.argv[1], {"V1", "V2", "A1"})
rows.update(load(sys.argv[2], {"A3"}))
stat = defaultdict(lambda: defaultdict(list))
for (a, e, q), r in rows.items():
    calls = r.get("calls") or []
    stat[a]["calls"].append(len(calls))
    stat[a]["prompt"].append(sum(c[1] or 0 for c in calls))
    stat[a]["completion"].append(sum(c[2] or 0 for c in calls))
    stat[a]["wall_s"].append(r.get("dt") or 0)
    stat[a]["codegen"].append(sum(1 for c in calls if c[0] == "codegen"))
for a in ("V2", "V1", "A1", "A3"):
    s = stat[a]
    if not s:
        continue
    med = lambda xs: sorted(xs)[len(xs) // 2]
    mean = lambda xs: sum(xs) / len(xs)
    print("%-3s n=%d calls mean %.2f | prompt tok median %d mean %d | completion tok median %d mean %d | wall s median %.0f mean %.0f | codegen/q %.2f" % (
        a, len(s["calls"]), mean(s["calls"]), med(s["prompt"]), mean(s["prompt"]), med(s["completion"]),
        mean(s["completion"]), med(s["wall_s"]), mean(s["wall_s"]), mean(s["codegen"])))
