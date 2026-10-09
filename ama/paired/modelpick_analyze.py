"""Readout of the model-pick fix re-run (docs/MODEL_PICK_FIX.md).

    python ama/paired/modelpick_analyze.py results/modelpick_fix [--paired1008 results/ama_paired/full]
                                           [--out modelpick_analysis.json]

The run dir holds def/ and str/ subfolders of s*/q.jsonl[.gz]. The selection split uses each record's
picked_steps (the released records) or parses the appendix of its context (a fresh run's records).

Arms V1 (deployed config, pick call thinks), V1f (pick call without thinking), V2 (lexical + step pin); readers
def (harness default instruction) and str (structured instruction). Differences: episode-clustered bootstrap
(4,000 resamples, seed 20261009) and exact McNemar; Holm over the two primaries P1 (def) and P2 (str), both V1f - V1.
"""
import argparse
import glob
import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--paired1008", default="")
ap.add_argument("--out", default="")
ap.add_argument("--boot", type=int, default=4000)
args = ap.parse_args()
SEED = 20261009
DOMS = ["WEB", "EMBODIED_AI", "Game", "TEXT2SQL", "SOFTWARE", "OPENWORLD_QA"]
TYPES = {"A": "recall", "B": "causal", "C": "state update", "D": "state abstraction"}
MARK = "Full text of the most relevant earlier steps:"


def load(pattern_dir):
    rows = {}
    files = sorted(glob.glob(pattern_dir + "/s*/q.jsonl")) or sorted(glob.glob(pattern_dir + "/s*/q.jsonl.gz"))
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                k = (r["arm"], r["ep"], r["qi"])
                if k not in rows or (rows[k].get("infra") and not r.get("infra")):
                    rows[k] = r
    return rows


def ok(r):
    return r is not None and not r.get("infra") and r.get("score") is not None


def cor(r):
    return 1 if r.get("score") == 1.0 else 0


def mcn(w, l):
    n, m = w + l, min(w, l)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i))
                            for i in range(m + 1)) / 2 ** n)  # no math.comb on master's python3


def paired(rows, a, b, keys):
    by_ep = defaultdict(list)
    for k in keys:
        ra, rb = rows.get((a,) + k), rows.get((b,) + k)
        if ok(ra) and ok(rb):
            by_ep[k[0]].append((cor(ra), cor(rb)))
    eps = sorted(by_ep)
    n = sum(len(v) for v in by_ep.values())
    if not n:
        return None
    w = sum(1 for v in by_ep.values() for x, y in v if x and not y)
    l = sum(1 for v in by_ep.values() for x, y in v if y and not x)
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
    return {"n": n, "diff": (w - l) / n, "lo": bs[int(.025 * len(bs))], "hi": bs[int(.975 * len(bs)) - 1],
            "wins": w, "losses": l, "p_boot": p_boot, "p_mcnemar": mcn(w, l),
            "acc_a": sum(x for v in by_ep.values() for x, _ in v) / n, "acc_b": sum(y for v in by_ep.values() for _, y in v) / n}


def fmt(r):
    if not r:
        return "n/a"
    return "%+.1f [%+.1f, %+.1f] n=%d (%.4f vs %.4f) %d/%d p_boot=%.3g p_mcn=%.3g" % (
        100 * r["diff"], 100 * r["lo"], 100 * r["hi"], r["n"], r["acc_a"], r["acc_b"], r["wins"], r["losses"],
        r["p_boot"], r["p_mcnemar"])


def picked_steps(ctx):
    i = (ctx or "").find(MARK)
    return tuple(int(x) for x in re.findall(r"<step (\d+)>", ctx[i:])) if i >= 0 else ()


def steps_of(r):
    """Step numbers in the record's appendix: stored in the released records, parsed from a fresh run's context."""
    if "picked_steps" in r:
        return tuple(r["picked_steps"] or ())
    return picked_steps(r.get("context"))


OUT = {"readers": {}}
runs = {rd: load("%s/%s" % (args.run, rd)) for rd in ("def", "str")}
for rd, rows in runs.items():
    O = OUT["readers"][rd] = {}
    qs = sorted({(e, q) for (a, e, q) in rows if a == "V1f"})
    meta = {}
    for (a, e, q), r in rows.items():
        meta.setdefault((e, q), (r["domain"], r["qtype"], r.get("question", "")))
    print("\n######## reader %s: %d questions" % (rd, len(qs)))
    O["arms"] = {}
    for arm in ("V1", "V1f", "V2"):
        g = [rows.get((arm,) + k) for k in qs]
        good = [r for r in g if ok(r)]
        d = {"n": len(good), "infra": sum(1 for r in g if not ok(r)), "acc": sum(cor(r) for r in good) / max(1, len(good))}
        for dom in DOMS:
            gd = [r for r in good if r["domain"] == dom]
            d["dom_" + dom] = [len(gd), sum(cor(r) for r in gd) / max(1, len(gd))]
        for t in TYPES:
            gt = [r for r in good if r["qtype"] == t]
            d["type_" + t] = [len(gt), sum(cor(r) for r in gt) / max(1, len(gt))]
        O["arms"][arm] = d
        print("%-4s n=%d infra=%d acc=%.4f | %s" % (arm, d["n"], d["infra"], d["acc"], " ".join(
            "%s %.3f" % (dom[:4], d["dom_" + dom][1]) for dom in DOMS)))
    # which questions got a different appendix (selected steps) under V1f than under V1
    changed = set()
    sel_stats = Counter()
    for k in qs:
        r1, rf = rows.get(("V1",) + k), rows.get(("V1f",) + k)
        if not (r1 and rf):
            continue
        s1, sf = steps_of(r1), steps_of(rf)
        sel_stats["compared"] += 1
        if s1 != sf:
            changed.add(k)
            sel_stats["changed"] += 1
        named = {int(x) for x in re.findall(r"[Ss]tep\s+(\d+)", meta[k][2])}
        sel_stats["v1f_steps"] += len(sf)
        sel_stats["v1f_steps_named"] += sum(1 for s in sf if s in named)
        sel_stats["v1_steps"] += len(s1)
        sel_stats["v1_steps_named"] += sum(1 for s in s1 if s in named)
    O["selection"] = dict(sel_stats)
    print("selection: %s (changed share %.1f%%)" % (dict(sel_stats), 100 * sel_stats["changed"] / max(1, sel_stats["compared"])))
    # pick-call replies
    pk = Counter()
    for (a, e, q), r in rows.items():
        for p in r.get("pick") or []:
            txt = p[0] or ""
            pk[a + "_calls"] += 1
            pk[a + "_finish_" + str(p[1])] += 1
            pk[a + "_has_number"] += 1 if re.search(r"\d", txt.split("</think>")[-1] if a == "V1" else txt) else 0
            pk[a + "_open_think"] += 1 if txt.lstrip().startswith("<think>") and "</think>" not in txt else 0
            pk[a + "_nothink_param"] += 1 if p[2] == {"chat_template_kwargs": {"enable_thinking": False}} else 0
    O["pick_calls"] = dict(pk)
    print("pick calls:", dict(pk))
    # paired contrasts
    O["paired"] = {}
    slices = [("all", qs)]
    slices += [("dom_" + d, [k for k in qs if meta[k][0] == d]) for d in DOMS]
    slices += [("type_" + t, [k for k in qs if meta[k][1] == t]) for t in TYPES]
    slices += [("five domains (10-08 set)", [k for k in qs if meta[k][0] != "OPENWORLD_QA"]),
               ("selection changed", [k for k in qs if k in changed]),
               ("selection unchanged", [k for k in qs if k not in changed])]
    for a, b in (("V1f", "V1"), ("V1f", "V2"), ("V1", "V2")):
        print("\n== %s - %s (%s reader)" % (a, b, rd))
        for name, keys in slices:
            res = paired(rows, a, b, keys)
            O["paired"]["%s-%s|%s" % (a, b, name)] = res
            print("  %-26s %s" % (name, fmt(res)))

p1 = OUT["readers"]["def"]["paired"].get("V1f-V1|all")
p2 = OUT["readers"]["str"]["paired"].get("V1f-V1|all")
if p1 and p2:
    ps = sorted([(p1["p_boot"], "P1 def"), (p2["p_boot"], "P2 str")])
    holm, run_max = [], 0.0
    for i, (p, name) in enumerate(ps):
        run_max = max(run_max, min(1.0, (len(ps) - i) * p))
        holm.append((name, p, run_max))
    OUT["holm"] = holm
    print("\nHolm:", "; ".join("%s p=%.3g adj=%.3g" % h for h in holm))

if args.paired1008:
    old = load(args.paired1008)
    rows = runs["def"]
    rep = {}
    for arm in ("V1", "V2"):
        n = agree = c_now = c_old = 0
        for (a, e, q), r in rows.items():
            if a != arm or not ok(r):
                continue
            o = old.get((arm, e, q))
            if not ok(o):
                continue
            n += 1
            agree += 1 if cor(r) == cor(o) else 0
            c_now += cor(r)
            c_old += cor(o)
        rep[arm] = {"n": n, "agree": agree / max(1, n), "acc_now": c_now / max(1, n), "acc_1008": c_old / max(1, n)}
        print("replication %s vs 10-08 (default reader, 5 domains): n=%d agreement %.3f acc now %.4f vs 10-08 %.4f" % (
            arm, n, agree / max(1, n), c_now / max(1, n), c_old / max(1, n)))
    OUT["replication_1008"] = rep
if args.out:
    json.dump(OUT, open(args.out, "w"), indent=1)
