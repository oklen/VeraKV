#!/usr/bin/env python3
"""Analyze the PAIRED note-knob retest (kv_noteknob2). Everything is within-item now."""
import glob
import json
import math
import os
import random
import sys

SETTINGS = ["bin2", "num", "verdict"]
KCARD = {"bin2": 2, "num": 900, "verdict": 2}


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


def paired(g, a, b, label):
    ma = sum(r[a] for r in g) / len(g)
    mb = sum(r[b] for r in g) / len(g)
    bw, cw, p = mcnemar(g, a, b)
    lo, hi = boot(g, a, b)
    print(f"    {label:26s}: fill={mb:.3f} note={ma:.3f} d={ma-mb:+.3f} "
          f"CI[{lo:+.3f},{hi:+.3f}] McNemar +{bw}/-{cw} p={p:.3g}")


def main():
    D = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "nk2_s*.json"))):
        rows += json.load(open(f))["rows"]
    print(f"### total n={len(rows)} (paired: each item run with note AND filler)\n")
    for s in SETTINGS:
        g = [r for r in rows if r["setting"] == s]
        if not g:
            continue
        n = len(g)
        sl_n = sum(r["slotlen_note"] for r in g) / n
        sl_f = sum(r["slotlen_fill"] for r in g) / n
        ha, hl, hh = wilson(sum(r["harv_drop_note"] for r in g), n)
        print(f"== {s} (n={n}, chance={1/KCARD[s]:.2f}, slot tokens note/fill "
              f"{sl_n:.1f}/{sl_f:.1f}) ==")
        paired(g, "full_note", "full_fill", "NO-HARM full_txt")
        paired(g, "sel_drop_note", "sel_drop_fill", "LEAK sel_txt(drop)")
        paired(g, "harv_complete_note", "harv_complete_fill", "harv(complete)")
        paired(g, "harv_drop_note", "harv_drop_fill", "** KNOB harv(drop) **")
        print(f"    harv(drop,note)={ha:.3f} [{hl:.3f},{hh:.3f}] vs chance "
              f"{1/KCARD[s]:.2f} -> "
              f"{'ABOVE' if hl > 1/KCARD[s] else 'not above'}")
        print(f"    decoy-echo harv(drop): fill="
              f"{sum(r['harv_drop_fill_dec'] for r in g)/n:.2f} note="
              f"{sum(r['harv_drop_note_dec'] for r in g)/n:.2f}\n")


if __name__ == "__main__":
    main()
