"""Paired contrasts on the per-question run outputs in results/oow/ (paper Sec. 3.6, Sec. 4, App. ledger).

Every arm in a run file is a 0/1 per-question score on the identical question set, so a contrast
A - B reports: accuracy difference, discordant counts (+A-only / -B-only), the exact two-sided
McNemar (binomial) p, and a 95% episode-clustered bootstrap CI (episodes, or LOCOMO conversations,
resampled as clusters; 10,000 draws, seed 0). The paper's per-experiment scripts used the same
estimators with their own seeds and draw counts, so a CI can differ from the paper in the last digit;
accuracies, discordant counts and p-values reproduce exactly.

    python analysis/oow_paired.py results/oow/gh.jsonl.gz glob_hotR-bh glob_R-glob
    python analysis/oow_paired.py results/oow/ev.jsonl.gz tx_graph-tx_base bh_graph-bh_base --nonempty bridges
    python analysis/oow_paired.py results/oow/official_deploymap.jsonl.gz gsk-b_hot gsk-tx \
        --join results/oow/official_k12.jsonl.gz --where harvest=1
    python analysis/oow_paired.py results/oow/official_k12.jsonl.gz tx-tx@k5 b_hot-b_hot@k5 \
        --join results/oow/official_k5.jsonl.gz@k5
    python analysis/oow_paired.py results/oow/rv.jsonl.gz --arms        # list arms and accuracies

`--join FILE[@suffix]` pairs rows on (episode_id, question_uuid or question); the joined file's arms
are renamed `<arm>@suffix` when a suffix is given. A DiD contrast `(a-b)-(c-d)` is accepted too.
"""
import argparse
import gzip
import json
import random
from math import comb


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cid(r):
    for k in ("episode_id", "conv_id", "conv"):
        if r.get(k) is not None:
            return str(r[k])
    return ""


def qkey(r):
    ep = cid(r)
    return (str(ep), str(r.get("question_uuid") or r.get("q") or r.get("question")))


def cluster_of(r):
    return cid(r)


def binary_fields(rows):
    out = []
    for k, v in rows[0].items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or k in ("episode_id", "conv", "conv_id", "cat", "category"):
            continue
        vals = {r.get(k) for r in rows}
        if vals <= {0, 1, None} and len(vals - {None}) >= 1:
            out.append(k)
    return out


def mcnemar(b, c):
    m = b + c
    return min(1.0, sum(comb(m, i) for i in range(min(b, c) + 1)) * 2 / 2 ** m) if m else 1.0


def cluster_ci(rows, diff, reps=10000, seed=0):
    eps = {}
    for r in rows:
        eps.setdefault(cluster_of(r), []).append(diff(r))
    keys = sorted(eps)
    rng = random.Random(seed)
    ds = []
    for _ in range(reps):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        ds.append(sum(sum(eps[k]) for k in pick) / sum(len(eps[k]) for k in pick))
    ds.sort()
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def contrast(rows, spec):
    if spec.startswith("(") and ")-(" in spec:          # difference-in-differences
        l, r = spec[1:-1].split(")-(")
        a, b = l.split("-", 1)
        c, d = r.split("-", 1)
        rr = [x for x in rows if all(x.get(k) is not None for k in (a, b, c, d))]
        diff = lambda x: (x[a] - x[b]) - (x[c] - x[d])
        est = sum(diff(x) for x in rr) / len(rr)
        lo, hi = cluster_ci(rr, diff)
        print("  %-34s n=%5d  DiD=%+.2fpp  95%% cCI [%+.2f, %+.2f]" % (spec, len(rr), 100 * est, 100 * lo, 100 * hi))
        return
    a, b = spec.split("-", 1)
    rr = [x for x in rows if x.get(a) is not None and x.get(b) is not None]
    n = len(rr)
    pa = sum(x[a] for x in rr) / n
    pb = sum(x[b] for x in rr) / n
    up = sum(1 for x in rr if x[a] == 1 and x[b] == 0)
    dn = sum(1 for x in rr if x[a] == 0 and x[b] == 1)
    lo, hi = cluster_ci(rr, lambda x: x[a] - x[b])
    print("  %-34s n=%5d  %.4f vs %.4f  d=%+.2fpp  +%d/-%d  p=%.2g  95%% cCI [%+.2f, %+.2f]"
          % (spec, n, pa, pb, 100 * (pa - pb), up, dn, mcnemar(up, dn), 100 * lo, 100 * hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("contrasts", nargs="*", help="A-B, or (A-B)-(C-D)")
    ap.add_argument("--join", action="append", default=[], help="FILE[@suffix]: pair with another run")
    ap.add_argument("--where", action="append", default=[], help="field=value row filter")
    ap.add_argument("--nonempty", action="append", default=[], help="keep rows whose field is non-empty")
    ap.add_argument("--arms", action="store_true", help="list arms with accuracy")
    args = ap.parse_args()

    rows = load(args.run)
    for j in args.join:
        path, _, suf = j.partition("@")
        other = {qkey(r): r for r in load(path)}
        merged = []
        for r in rows:
            o = other.get(qkey(r))
            if o is None:
                continue
            r = dict(r)
            for k in binary_fields([o]):
                r[k + ("@" + suf if suf else "")] = o[k]
            merged.append(r)
        rows = merged
    for w in args.where:
        k, v = w.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    for k in args.nonempty:
        rows = [r for r in rows if r.get(k)]
    print("%s  rows=%d  clusters=%d" % (args.run, len(rows), len({cluster_of(r) for r in rows})))
    if args.arms or not args.contrasts:
        for k in binary_fields(rows):
            vals = [r[k] for r in rows if r.get(k) is not None]
            print("  %-16s %.4f  (%d/%d)" % (k, sum(vals) / len(vals), sum(vals), len(vals)))
    for c in args.contrasts:
        contrast(rows, c)


if __name__ == "__main__":
    main()
