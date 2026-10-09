"""Readout of the A3 test (docs/AMA_AGENT_PAIRED.md, A3).

    python ama/paired/a3_analyze.py <a3 run dir> <full run dir> [--out a3_analysis.json]

A3 and A1r come from the A3 run (same calls up to the point where the released code would answer from the
sufficiency check); A1 and V1 come from the full paired run. Differences use the episode-clustered bootstrap
(4,000 resamples, seed 20261008) and exact McNemar; Holm over the two primaries P3/P4.
"""
import argparse
import glob
import gzip
import json
import math
import random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("a3run")
ap.add_argument("fullrun")
ap.add_argument("--out", default="")
ap.add_argument("--boot", type=int, default=4000)
args = ap.parse_args()
SEED = 20261008
DOMS = ["WEB", "EMBODIED_AI", "Game", "TEXT2SQL", "SOFTWARE"]


def load(run):
    rows = {}
    files = sorted(glob.glob(run + "/s*/q.jsonl")) or sorted(glob.glob(run + "/s*/q.jsonl.gz"))
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                k = (r["arm"], r["ep"], r["qi"])
                if k not in rows or (rows[k].get("infra") and not r.get("infra")):
                    rows[k] = r
    return rows


rows = load(args.fullrun)
rows.update(load(args.a3run))
qs = sorted({(e, q) for (a, e, q) in rows if a == "A3"})
meta = {(e, q): (rows[("A3", e, q)]["domain"], rows[("A3", e, q)]["qtype"]) for e, q in qs}
fast = {k for k in qs if (rows[("A3",) + k].get("ama") or {}).get("a1_fast")}
full_direct = {k for k in qs if ((rows.get(("A1",) + k) or {}).get("ama") or {}).get("direct")}
OUT = {"n": len(qs)}


def ok(r):
    return r is not None and not r.get("infra") and r.get("score") is not None


def cor(r):
    return 1 if r.get("score") == 1.0 else 0


def mcn(w, l):
    n, m = w + l, min(w, l)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) for i in range(m + 1)) / 2 ** n)


def pairs(a, b, keys):
    by_ep = defaultdict(list)
    for k in keys:
        ra, rb = rows.get((a,) + k), rows.get((b,) + k)
        if ok(ra) and ok(rb):
            by_ep[k[0]].append((cor(ra), cor(rb)))
    return by_ep


def paired(a, b, keys):
    by_ep = pairs(a, b, keys)
    eps = sorted(by_ep)
    n = sum(len(v) for v in by_ep.values())
    if not n:
        return None
    w = sum(1 for v in by_ep.values() for x, y in v if x and not y)
    l = sum(1 for v in by_ep.values() for x, y in v if y and not x)
    diff = sum(x - y for v in by_ep.values() for x, y in v) / n
    rng = random.Random(SEED)
    bs = []
    for _ in range(args.boot):
        sx = sy = sn = 0
        for _ in eps:
            v = by_ep[eps[rng.randrange(len(eps))]]
            sx += sum(x for x, _ in v)
            sy += sum(y for _, y in v)
            sn += len(v)
        bs.append((sx - sy) / max(1, sn))
    bs.sort()
    p_boot = min(1.0, 2 * min(sum(1 for v in bs if v <= 0), sum(1 for v in bs if v >= 0)) / len(bs))
    return {"n": n, "diff": diff, "lo": bs[int(.025 * len(bs))], "hi": bs[int(.975 * len(bs)) - 1],
            "wins": w, "losses": l, "p_boot": p_boot, "p_mcnemar": mcn(w, l),
            "acc_a": sum(x for v in by_ep.values() for x, _ in v) / n, "acc_b": sum(y for v in by_ep.values() for _, y in v) / n}


def fmt(r):
    if not r:
        return "n/a"
    return "%+.1f [%+.1f, %+.1f] n=%d (%.3f vs %.3f) %d/%d p_boot=%.3g p_mcn=%.3g" % (
        100 * r["diff"], 100 * r["lo"], 100 * r["hi"], r["n"], r["acc_a"], r["acc_b"], r["wins"], r["losses"],
        r["p_boot"], r["p_mcnemar"])


C = [k for k in qs if meta[k][1] == "C"]
SL = [("all", qs), ("state_update", C)]
SL += [("dom_" + d, [k for k in qs if meta[k][0] == d]) for d in DOMS]
SL += [("type_" + t, [k for k in qs if meta[k][1] == t]) for t in "ABCD"]
SL += [("A1r fast path", [k for k in qs if k in fast]), ("A1r full pipeline", [k for k in qs if k not in fast]),
       ("state update, fast", [k for k in C if k in fast]), ("state update, full pipeline", [k for k in C if k not in fast])]

print("questions %d; A1r fast path %d (%.1f%%); full-run A1 fast path %d" % (len(qs), len(fast), 100 * len(fast) / len(qs), len(full_direct)))
for arm in ("A1", "A1r", "A3", "V1"):
    g = [rows.get((arm,) + k) for k in qs]
    good = [r for r in g if ok(r)]
    print("%-3s n=%d acc=%.4f infra=%d | C %.3f" % (arm, len(good), sum(cor(r) for r in good) / max(1, len(good)),
                                                  sum(1 for r in g if not ok(r)),
                                                  sum(cor(r) for r in good if r["qtype"] == "C") / max(1, sum(1 for r in good if r["qtype"] == "C"))))
OUT["paired"] = {}
for a, b in (("A3", "A1r"), ("V1", "A3"), ("A3", "A1"), ("A1r", "A1"), ("V1", "A1r")):
    print("\n== %s - %s" % (a, b))
    for name, keys in SL:
        r = paired(a, b, keys)
        OUT["paired"]["%s-%s|%s" % (a, b, name)] = r
        print("  %-28s %s" % (name, fmt(r)))

p3, p4 = OUT["paired"]["A3-A1r|all"], OUT["paired"]["A3-A1r|state_update"]
ps = sorted([(p3["p_boot"], "P3 all"), (p4["p_boot"], "P4 state_update")])
holm, run_max = [], 0.0
for i, (p, name) in enumerate(ps):
    run_max = max(run_max, min(1.0, (len(ps) - i) * p))
    holm.append((name, p, run_max))
OUT["holm"] = holm
print("\nHolm:", "; ".join("%s p=%.3g adj=%.3g" % h for h in holm))

# gap closure on state update: (A3 - A1r) / (V1 - A1r), episode bootstrap
by3, byv = pairs("A3", "A1r", C), pairs("V1", "A1r", C)
eps = sorted(set(by3) & set(byv))


def ratio(sample):
    num = sum(x - y for e in sample for x, y in by3[e])
    den = sum(x - y for e in sample for x, y in byv[e])
    return num / den if den else float("nan")


rng = random.Random(SEED)
rs = sorted(r for r in (ratio([eps[rng.randrange(len(eps))] for _ in eps]) for _ in range(args.boot)) if r == r)
close = ratio(eps)
OUT["gap_closed_state_update"] = {"point": close, "lo": rs[int(.025 * len(rs))], "hi": rs[int(.975 * len(rs)) - 1]}
print("share of V1 - A1r on state update closed by A3: %.2f [%.2f, %.2f]" % (close, rs[int(.025 * len(rs))], rs[int(.975 * len(rs)) - 1]))

# A3 path stats
st = Counter()
for k in qs:
    am = rows[("A3",) + k].get("ama") or {}
    st["n"] += 1
    st["fast"] += 1 if am.get("a1_fast") else 0
    st["reader"] += 1 if am.get("a3_reader_called") else 0
    st["suff_rounds"] += am.get("n_suff") or 0
    st["codegen_calls"] += am.get("n_codegen") or 0
    st["cs_" + str(am.get("code_search"))] += 1
    st["cut"] += 1 if am.get("a3_cut") else 0
OUT["a3_paths"] = dict(st)
print("\nA3 paths:", dict(st))
for name, S in (("cut", lambda k: (rows[("A3",) + k].get("ama") or {}).get("a3_cut")),
                ("not cut", lambda k: not (rows[("A3",) + k].get("ama") or {}).get("a3_cut")),
                ("search result", lambda k: (rows[("A3",) + k].get("ama") or {}).get("code_search") == "result"),
                ("search error/timeout/empty", lambda k: (rows[("A3",) + k].get("ama") or {}).get("code_search") in ("error", "timeout", "empty"))):
    keys = [k for k in qs if k in fast and S(k)]
    r = paired("A3", "A1r", keys)
    print("  fast-path questions, A3 %-26s %s" % (name, fmt(r)))
tok = sorted(rows[("A3",) + k].get("ctx_tokens") or 0 for k in qs)
pt = sorted(rows[("A3",) + k].get("answer_prompt_tokens") or 0 for k in qs)
OUT["a3_ctx_tokens"] = {"median": tok[len(tok) // 2], "p90": tok[int(.9 * len(tok))]}
print("A3 final context tokens median %d p90 %d; answer-prompt tokens median %d p90 %d" % (
    tok[len(tok) // 2], tok[int(.9 * len(tok))], pt[len(pt) // 2], pt[int(.9 * len(pt))]))
if args.out:
    json.dump(OUT, open(args.out, "w"), indent=1)
