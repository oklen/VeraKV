#!/usr/bin/env python3
"""Analyze the pre-digestion SWEEP (kv_predigest_sweep): where does harvest beat sel?"""
import glob
import json
import math
import os
import random
import sys

SETTINGS = ["bin2", "dir4", "name8", "num", "verdict"]
KCARD = {"bin2": 2, "dir4": 4, "name8": 8, "num": 900, "verdict": 2}


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


def main():
    D = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "pds_s*.json"))):
        rows += json.load(open(f))["rows"]
    print(f"### total n={len(rows)}\n")
    print(f"{'setting':8s} {'K':>4s} {'chance':>6s} {'n':>4s} | {'full':>5s} | "
          f"{'C:sel':>6s} {'C:harv':>7s} | {'D:sel_t':>8s} {'D:sel_k':>8s} {'D:harv':>7s} "
          f"{'D:h-dec':>8s} | {'harv-sel(drop)':>22s}")
    for s in SETTINGS:
        g = [r for r in rows if r["setting"] == s]
        if not g:
            continue
        n = len(g)
        def m(k):
            return sum(r[k] for r in g) / n
        ha, hl, hh = wilson(sum(r["harv_kv_drop_mid"] for r in g), n)
        sa = m("sel_txt_drop_mid")
        bw, cw, p = mcnemar(g, "harv_kv_drop_mid", "sel_txt_drop_mid")
        lo, hi = boot(g, "harv_kv_drop_mid", "sel_txt_drop_mid")
        print(f"{s:8s} {KCARD[s]:>4d} {1/KCARD[s]:>6.2f} {n:>4d} | {m('full_txt'):>5.2f} | "
              f"{m('sel_txt_complete'):>6.2f} {m('harv_kv_complete'):>7.2f} | "
              f"{sa:>8.2f} {m('sel_ikv_drop_mid'):>8.2f} {ha:>7.2f} "
              f"{m('harv_kv_drop_mid_dec'):>8.2f} | "
              f"d={ha-sa:+.2f} CI[{lo:+.2f},{hi:+.2f}] p={p:.1g} (+{bw}/-{cw})")
    print("\nharv above CHANCE? (drop_mid, one-sample vs 1/K):")
    for s in SETTINGS:
        g = [r for r in rows if r["setting"] == s]
        if not g:
            continue
        a, lo, hi = wilson(sum(r["harv_kv_drop_mid"] for r in g), len(g))
        ch = 1 / KCARD[s]
        verdict = "ABOVE" if lo > ch else ("at/below" if hi <= ch else "spans")
        print(f"  {s:8s}: harv={a:.2f} [{lo:.2f},{hi:.2f}] vs chance {ch:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
