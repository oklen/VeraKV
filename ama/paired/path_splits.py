"""Extra paired splits for the write-up: A2 - A1 where the reader actually read the final context, and
V1 - A1 on state-update questions by A1's answering path. Exact two-sided McNemar on the discordant pairs.

    python ama/paired/path_splits.py <full run dir with s*/q.jsonl[.gz]>
"""
import glob
import gzip
import json
import math
import sys
from collections import defaultdict

rows = {}
for f in sorted(glob.glob(sys.argv[1] + "/s*/q.jsonl")) or sorted(glob.glob(sys.argv[1] + "/s*/q.jsonl.gz")):
    for l in (gzip.open if f.endswith(".gz") else open)(f, "rt", encoding="utf-8"):
        r = json.loads(l)
        rows[(r["arm"], r["ep"], r["qi"])] = r


def mcn(w, l):
    n, m = w + l, min(w, l)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) for i in range(m + 1)) / 2 ** n)


def cmp(a, b, keys, name):
    n = w = l = ca = cb = 0
    for k in keys:
        ra, rb = rows.get((a,) + k), rows.get((b,) + k)
        if not ra or not rb or ra.get("score") is None or rb.get("score") is None:
            continue
        x, y = ra["score"] == 1.0, rb["score"] == 1.0
        n += 1
        ca += x
        cb += y
        w += x and not y
        l += y and not x
    print("%-44s n=%4d %s %.3f %s %.3f diff %+5.1f  %d/%d  p_mcn=%.3g" % (
        name, n, a, ca / max(1, n), b, cb / max(1, n), 100 * (ca - cb) / max(1, n), w, l, mcn(w, l)))


keys = sorted({(e, q) for (_, e, q) in rows})
a1 = {k: rows[("A1",) + k] for k in keys}
direct = {k for k in keys if (a1[k].get("ama") or {}).get("direct")}
cut = {k for k in keys if a1[k].get("ctx_chars") == 23813}
reader = [k for k in keys if k not in direct]
cmp("A2", "A1", reader, "A2-A1 | A1 reader path")
cmp("A2", "A1", [k for k in reader if k in cut], "A2-A1 | reader path, A1 context cut")
cmp("A2", "A1", [k for k in reader if k not in cut], "A2-A1 | reader path, A1 context not cut")
C = [k for k in keys if a1[k]["qtype"] == "C"]
cmp("V1", "A1", [k for k in C if k in direct], "V1-A1 | state update, A1 direct")
cmp("V1", "A1", [k for k in C if k not in direct], "V1-A1 | state update, A1 reader")
cmp("V1", "A1", [k for k in keys if k in direct], "V1-A1 | all, A1 direct")
cmp("V1", "A1", [k for k in keys if k not in direct], "V1-A1 | all, A1 reader")
by = defaultdict(lambda: [0, 0])
for k in keys:
    by[a1[k]["domain"]][0] += 1
    by[a1[k]["domain"]][1] += 1 if k in direct else 0
print("A1 direct share by domain:", {d: "%.0f%%" % (100.0 * v[1] / v[0]) for d, v in by.items()})
