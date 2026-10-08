#!/usr/bin/env python3
"""Merge + analyze the 8-way high-N phantom (E1-E3) + vartrack (E4) shards.

Usage: python3 phvt_analyze.py <dir_with_phhi_s*.json_and_vthi_s*.json>
Reports Wilson 95% CIs for every rate and exact-McNemar for the paired contrasts
that carry the paper's claims.
"""
import glob
import json
import math
import os
import sys

# ---- polarity lexicon for E1 semantic-separation (curated antonym PAIRS) ----
POS = {"online", "active", "enabled", "up", "on", "open", "valid", "pass", "passed",
       "green", "complete", "completed", "healthy", "ready", "live", "running", "ok",
       "good", "current", "fresh", "warm", "hot", "mounted", "authed", "authenticated",
       "trusted", "paired", "synced", "in sync", "held", "secure", "attached",
       "granted", "success", "successful", "alive", "reachable", "clean", "normal"}
NEG = {"offline", "inactive", "disabled", "down", "off", "closed", "shut", "locked",
       "invalid", "expired", "fail", "failed", "red", "aborted", "degraded", "stale",
       "cold", "dead", "revoked", "lost", "orphan", "orphaned", "detached", "guest",
       "guested", "exposed", "released", "denied", "suspended", "error", "unreachable",
       "broken", "unhealthy", "not ready", "abnormal"}


def pole(text):
    t = text.lower()
    hp = any(w in t for w in POS)
    hn = any(w in t for w in NEG)
    if hp and not hn:
        return 1
    if hn and not hp:
        return -1
    return 0  # ambiguous / neither


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def mcnemar(pairs):
    """pairs: list of (a, b) 0/1. Returns (b_wins, c_wins, two-sided exact p)."""
    b = sum(1 for a, x in pairs if a == 1 and x == 0)  # a=1,b=0
    c = sum(1 for a, x in pairs if a == 0 and x == 1)  # a=0,b=1
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, p)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.3f} [{lo:.3f},{hi:.3f}] ({k}/{n})"


def load_rows(pattern, key=None):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        rows += (d[key]["rows"] if key else d["rows"])
    return rows


def main():
    D = sys.argv[1]
    print(f"### dir={D}")

    # ---------------- E1 SWAP ----------------
    sw = load_rows(os.path.join(D, "phhi_s*.json"), "swap")
    n = len(sw)
    fa = sum(r["A"]["follows_donor"] for r in sw)
    fb = sum(r["B"]["follows_donor"] for r in sw)
    lit_k, lit_n = fa + fb, 2 * n
    iso_k = sum(r["iso_names_any"] for r in sw)
    # semantic separation: A-out pole matches donorA pole AND B-out matches donorB pole
    sep = 0
    a_follow = b_follow = 0
    for r in sw:
        pa = 1 if r["A"]["donor_val"].lower() in POS else -1
        pb = 1 if r["B"]["donor_val"].lower() in POS else -1
        oa, ob = pole(r["A"]["out"]), pole(r["B"]["out"])
        if oa == pa:
            a_follow += 1
        if ob == pb:
            b_follow += 1
        if oa == pa and ob == pb and pa != pb:
            sep += 1
    print("\n=== E1 SWAP (does the spliced event carry the invisible donor?) ===")
    print(f"  n_pairs={n}  (donor-trials={lit_n})")
    print(f"  LITERAL follow-donor rate : {fmt(lit_k, lit_n)}")
    print(f"  SEMANTIC A-out in donorA pole: {fmt(a_follow, n)}")
    print(f"  SEMANTIC B-out in donorB pole: {fmt(b_follow, n)}")
    print(f"  SEMANTIC both-separated rate : {fmt(sep, n)}")
    print(f"  iso-names-any (control, want ~0): {fmt(iso_k, n)}")

    # ---------------- E2 ERASE ----------------
    er = load_rows(os.path.join(D, "phhi_s*.json"), "erase")
    nc = [r for r in er if not r["conflict"]]
    cf = [r for r in er if r["conflict"]]
    print("\n=== E2 ERASE (delete a state event's rows; does it leak?) ===")
    print(f"  n={len(er)}  (noconflict={len(nc)} conflict={len(cf)})")
    print(f"  no-conflict erased leaks-old : {fmt(sum(r['erased_leaks_old'] for r in nc), len(nc))}")
    print(f"  no-conflict ORACLE  leaks-old: {fmt(sum(r['oracle_leaks_old'] for r in nc), len(nc))}")
    print(f"  conflict   erased leaks-OLD  : {fmt(sum(r['erased_leaks_old'] for r in cf), len(cf))}")
    print(f"  conflict   erased has-NEW    : {fmt(sum(r['erased_has_new'] for r in cf), len(cf))}")

    # ---------------- E3 SINK 2x2 ----------------
    sk = load_rows(os.path.join(D, "phhi_s*.json"), "sink")
    N = len(sk)
    print("\n=== E3 SINK 2x2 (shared head vs none; retrieval vs multi-hop) ===")
    print(f"  n={N}")
    for k in ("full_retr", "head_retr", "nohead_retr", "full_hop", "head_hop", "nohead_hop"):
        print(f"    {k:12s}: {fmt(sum(r[k] for r in sk), N)}")
    for a, b in (("head_hop", "nohead_hop"), ("head_retr", "nohead_retr"),
                 ("full_hop", "head_hop"), ("full_retr", "head_retr")):
        bw, cw, p = mcnemar([(r[a], r[b]) for r in sk])
        print(f"  McNemar {a} vs {b}: {a}-only={bw} {b}-only={cw} p={p:.2e}")

    # ---------------- E4 VARTRACK ----------------
    vt = load_rows(os.path.join(D, "vthi_s*.json"))
    print("\n=== E4 VARTRACK (sparse chain reuse; full/harvest/iso/anch) ===")
    cells = {}
    for r in vt:
        cells.setdefault((r["hops"], r["far"]), []).append(r)
    for (h, far) in sorted(cells):
        g = cells[(h, far)]
        s = "  ".join(f"{a}={sum(x[a] for x in g)/len(g):.2f}"
                      for a in ("full", "harvest", "iso", "anch"))
        print(f"  h{h} far{far} n={len(g)}: {s}")
    # key paired contrast: harvest vs full on FAR cells (harvest strips distractor filler)
    far_rows = [r for r in vt if r["far"] == 1]
    bw, cw, p = mcnemar([(r["harvest"], r["full"]) for r in far_rows])
    print(f"  [FAR] harvest vs full: harvest-only={bw} full-only={cw} p={p:.2e} "
          f"(harvest {sum(r['harvest'] for r in far_rows)}/{len(far_rows)}, "
          f"full {sum(r['full'] for r in far_rows)}/{len(far_rows)})")
    bw, cw, p = mcnemar([(r["harvest"], r["iso"]) for r in vt])
    print(f"  [ALL] harvest vs iso : harvest-only={bw} iso-only={cw} p={p:.2e}")
    bw, cw, p = mcnemar([(r["harvest"], r["anch"]) for r in vt])
    print(f"  [ALL] harvest vs anch: harvest-only={bw} anch-only={cw} p={p:.2e}")


if __name__ == "__main__":
    main()
