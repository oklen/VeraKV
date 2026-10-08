#!/usr/bin/env python3
"""Analyze the 4-arm E4 decomposition (kv_vardecomp). Adjudicates: routing/denoising vs
pre-digestion vs joint-conditioning, on the SAME far-chain samples."""
import glob
import json
import math
import os
import random
import sys

ARMS = ("full_txt", "sel_txt", "sel_ikv", "harv_kv")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def mcnemar(rows, a, b):
    bw = sum(1 for r in rows if r[a] == 1 and r[b] == 0)
    cw = sum(1 for r in rows if r[a] == 0 and r[b] == 1)
    n = bw + cw
    if n == 0:
        return bw, cw, 1.0
    k = min(bw, cw)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return bw, cw, min(1.0, p)


def boot(rows, a, b, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(B):
        na = nb = 0
        for _ in range(n):
            r = rows[rng.randrange(n)]
            na += r[a]
            nb += r[b]
        diffs.append((na - nb) / n)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def contrast(name, rows, a, b, sfx=""):
    ma = sum(r[a + sfx] for r in rows) / len(rows)
    mb = sum(r[b + sfx] for r in rows) / len(rows)
    rr = [{"x": r[a + sfx], "y": r[b + sfx]} for r in rows]
    bw, cw, p = mcnemar(rr, "x", "y")
    lo, hi = boot(rr, "x", "y")
    print(f"    {name:22s}: {a}={ma:.3f} {b}={mb:.3f}  d={ma-mb:+.3f} "
          f"CI[{lo:+.3f},{hi:+.3f}] McNemar +{bw}/-{cw} p={p:.2g}")


def main():
    D = sys.argv[1]
    rows = []
    maxseq = 0
    for f in sorted(glob.glob(os.path.join(D, "vd_s*.json"))):
        d = json.load(open(f))
        rows += d["rows"]
        maxseq = max(maxseq, d.get("maxseq", 0))
    print(f"### n={len(rows)} rows ; MAXSEQ={maxseq} (H≈60, Qwen3-8B ctx 32k+ => no truncation)\n")

    print("=== PER-CELL means (exact) full_txt / sel_txt / sel_ikv / harv_kv ===")
    for hops in (1, 2, 4):
        for far in (0, 1):
            g = [r for r in rows if r["hops"] == hops and r["far"] == far]
            s = "  ".join(f"{a.split('_')[0]}{'_'+a.split('_')[1] if '_' in a else ''}="
                          f"{sum(r[a] for r in g)/len(g):.2f}" for a in ARMS)
            print(f"  h{hops} far{far} n={len(g)}: {s}")

    for sfx, lab in (("", "EXACT substring"), ("_rec", "NUMBER-set recall")):
        for far, flab in ((1, "FAR (decisive)"), (0, "NEAR (reference)")):
            g = [r for r in rows if r["far"] == far]
            print(f"\n=== {flab} pooled, {lab}  (n={len(g)}) ===")
            for a in ARMS:
                p, lo, hi = wilson(sum(r[a + sfx] for r in g), len(g))
                print(f"    {a:9s}: {p:.3f} [{lo:.3f},{hi:.3f}]")
            print("  -- decomposition --")
            contrast("routing/denoising", g, "sel_txt", "full_txt", sfx)
            contrast("pre-digestion", g, "harv_kv", "sel_txt", sfx)
            contrast("joint-conditioning", g, "harv_kv", "sel_ikv", sfx)
            contrast("harvest vs full(total)", g, "harv_kv", "full_txt", sfx)


if __name__ == "__main__":
    main()
