#!/usr/bin/env python3
"""Analyze the pre-digestion test (kv_predigest). Decisive: on drop_mid (gold source
removed from the served set), does harvested KV beat fresh/independent re-encoding?"""
import glob
import json
import math
import os
import random
import sys


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
    return bw, cw, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def boot(rows, a, b, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(rows)
    d = []
    for _ in range(B):
        na = nb = 0
        for _ in range(n):
            r = rows[rng.randrange(n)]
            na += r[a]
            nb += r[b]
        d.append((na - nb) / n)
    d.sort()
    return d[int(0.025 * B)], d[int(0.975 * B)]


def fmt(rows, k):
    p, lo, hi = wilson(sum(r[k] for r in rows), len(rows))
    return f"{p:.3f} [{lo:.3f},{hi:.3f}]"


def contrast(rows, a, b):
    ma = sum(r[a] for r in rows) / len(rows)
    mb = sum(r[b] for r in rows) / len(rows)
    bw, cw, p = mcnemar(rows, a, b)
    lo, hi = boot(rows, a, b)
    print(f"    {a} - {b}: d={ma-mb:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"McNemar +{bw}/-{cw} p={p:.3g}")


def main():
    D = sys.argv[1]
    rows = []
    maxseq = 0
    for f in sorted(glob.glob(os.path.join(D, "pd_s*.json"))):
        d = json.load(open(f))
        rows += d["rows"]
        maxseq = max(maxseq, d.get("maxseq", 0))
    n = len(rows)
    print(f"### n={n} ; MAXSEQ={maxseq} (no truncation)\n")
    print(f"full_txt (all events, distracted): {fmt(rows, 'full_txt')}\n")

    for sfx, lab in (("", "EXACT"), ("_rec", "NUMBER-recall")):
        print(f"===== {lab} =====")
        print("  COMPLETE (control -- source served, want all high):")
        for a in ("sel_txt", "sel_ikv", "harv_kv"):
            print(f"    {a}_complete: {fmt(rows, a + '_complete' + sfx)}")
        print("  DROP_MID (decisive -- source removed; gold in NO served token):")
        for a in ("sel_txt", "sel_ikv", "harv_kv"):
            print(f"    {a}_drop_mid: {fmt(rows, a + '_drop_mid' + sfx)}")
        print("  -- pre-digestion contrasts (drop_mid) --")
        contrast(rows, "harv_kv_drop_mid" + sfx, "sel_txt_drop_mid" + sfx)
        contrast(rows, "harv_kv_drop_mid" + sfx, "sel_ikv_drop_mid" + sfx)
        print()

    print("== decoy-echo on drop_mid (did the arm wrongly emit the decoy number?) ==")
    for a in ("sel_txt", "sel_ikv", "harv_kv"):
        print(f"    {a}_drop_mid decoy-echo: {fmt(rows, a + '_drop_mid_dec')}")


if __name__ == "__main__":
    main()
