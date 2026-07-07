"""Consume j2x2 verdicts -> exact upstream x final 2x2 for HOFTCL + attribution report."""
import json, re, glob, os
from collections import Counter, defaultdict

RES = "/Users/bytedance/Downloads/kvmemory-release/results"
rows = json.load(open(f"{RES}/j2x2_rows.json"))
if os.path.exists(f"{RES}/j2x4_rows.json"):
    full = {r["i"]: r for r in json.load(open(f"{RES}/j2x4_rows.json"))}
    mism = sum(1 for r in rows if r["i"] in full and full[r["i"]]["q"] != r["q"])
    print("j2x4 full-up overlay; q mismatches:", mism)
    for r in rows:
        if r["i"] in full:
            r["up"] = full[r["i"]]["up"]
verd = {}
SRC = os.environ.get("J2SRC", "j2x2")
if SRC == "j2x4":
    # official judge on FULL upstream answers; final = official score field
    for s in range(3):
        for l in open(f"{RES}/j2x4_out_{s}.jsonl"):
            o = json.loads(l)
            o["up_ok"] = None if o["up_ok"] is None else o["up_ok"] >= 1
            o["fin_ok"] = None
            verd[o["i"]] = o
elif SRC == "j2x3":
    for s in range(3):
        for l in open(f"{RES}/j2x3_out_{s}.jsonl"):
            o = json.loads(l)
            o["up_ok"] = None if o["up_ok"] is None else o["up_ok"] >= 1
            o["fin_ok"] = None if o["fin_ok"] is None else o["fin_ok"] >= 1
            verd[o["i"]] = o
else:
    for l in open(f"{RES}/j2x2_out.jsonl"):
        o = json.loads(l)
        verd[o["i"]] = o

# consistency: judge-of-final vs official score
ag = [(r, verd[r["i"]]) for r in rows if r["i"] in verd and verd[r["i"]]["fin_ok"] is not None]
if ag:
    agree = sum(1 for r, o in ag if o["fin_ok"] == (r["score"] >= 1))
    print(f"n={len(rows)}  fin-judge vs official agreement: {agree}/{len(ag)} = {agree/len(ag):.3f}")
else:
    print(f"n={len(rows)}  (fin = official score field)")

# the 2x2 under ONE consistent judge (up_ok x fin_ok, same prompt/model/day)
USE_CONSISTENT = SRC != "j2x4"   # j2x4: official-prompt up vs official score fin (same judge family)
def finv(r):
    o = verd.get(r["i"], {})
    return o.get("fin_ok") if USE_CONSISTENT else (r["score"] >= 1)

cells = Counter()
for r in rows:
    o = verd.get(r["i"], {})
    up = o.get("up_ok")
    fin = finv(r)
    if up is None or fin is None:
        cells[("?", fin)] += 1
    else:
        cells[(up, fin)] += 1
tot = sum(cells.values())
print("\n== 2x2 (both cells judged by the same binary judge) ==")
for (up, fin), c in sorted(cells.items(), key=lambda x: -x[1]):
    lab = {(True, True): "KEPT      up✓ fin✓", (True, False): "DESTROYED up✓ fin✗",
           (False, True): "REPAIRED  up✗ fin✓", (False, False): "BOTHWRONG up✗ fin✗"}.get(
        (up, fin), f"up={up} fin={fin}")
    print(f"  {lab:22s} {c:5d}  ({c/tot:.1%})")
upacc = sum(c for (u, f), c in cells.items() if u is True) / tot
finacc = sum(c for (u, f), c in cells.items() if f) / tot
print(f"upstream acc={upacc:.4f}  final acc={finacc:.4f}  relay delta={finacc-upacc:+.4f}")

# domain breakdown of the two error cells
print("\n== error cells by domain ==")
dom_d = defaultdict(Counter)
for r in rows:
    o = verd.get(r["i"], {})
    up, fin = o.get("up_ok"), finv(r)
    if fin is None:
        continue
    if up is True and not fin:
        dom_d[r["dom"]]["DESTROYED"] += 1
    elif up is False and not fin:
        dom_d[r["dom"]]["BOTHWRONG"] += 1
    elif up is False and fin:
        dom_d[r["dom"]]["REPAIRED"] += 1
for d in sorted(dom_d):
    n_d = sum(1 for r in rows if r["dom"] == d)
    c = dom_d[d]
    print(f"  {d:14s} destroyed={c['DESTROYED']:3d}  repaired={c['REPAIRED']:3d}  "
          f"net={c['REPAIRED']-c['DESTROYED']:+4d}  bothwrong={c['BOTHWRONG']:3d}  (n={n_d})")

# qa_type
print("\n== error cells by qa_type ==")
qt_d = defaultdict(Counter)
for r in rows:
    o = verd.get(r["i"], {})
    up, fin = o.get("up_ok"), finv(r)
    if fin is None:
        continue
    key = ("DESTROYED" if up is True and not fin else
           "REPAIRED" if up is False and fin else
           "BOTHWRONG" if up is False and not fin else None)
    if key:
        qt_d[r["qt"]][key] += 1
for qt in sorted(qt_d):
    c = qt_d[qt]
    print(f"  {qt}: destroyed={c['DESTROYED']:3d} repaired={c['REPAIRED']:3d} "
          f"net={c['REPAIRED']-c['DESTROYED']:+4d} bothwrong={c['BOTHWRONG']:3d}")

# ever-solved overlay on BOTHWRONG (is the both-wrong pool reachable at all?)
others = {}
for p in sorted(glob.glob(f"{RES}/mu_merged_*.json")):
    tag = os.path.basename(p)[10:-5]
    if tag == "HOFTCL":
        continue
    d = {}
    for r in json.load(open(p)):
        d[(r["domain"], str(r["episode_id"]), r["question"])] = float(r["score"])
    others[tag] = d
solved = Counter()
seen = Counter()
for tag, d in others.items():
    for r in rows:
        k = (r["dom"], r["ep"], r["q"])
        if k in d:
            seen[r["i"]] += 1
            if d[k] >= 1.0:
                solved[r["i"]] += 1
bw = [r for r in rows if verd.get(r["i"], {}).get("up_ok") is False and finv(r) is False]
de = [r for r in rows if verd.get(r["i"], {}).get("up_ok") is True and finv(r) is False]
bw_ever = sum(1 for r in bw if solved[r["i"]] > 0)
de_ever = sum(1 for r in de if solved[r["i"]] > 0)
print(f"\nBOTHWRONG ever-solved-by-any-arm: {bw_ever}/{len(bw)}; DESTROYED ever-solved: {de_ever}/{len(de)}")

json.dump({"destroyed": [r["i"] for r in de], "bothwrong": [r["i"] for r in bw]},
          open(f"{RES}/j2x2_cells.json", "w"))

# samples of DESTROYED (the relay's own losses: upstream right, final wrong)
import random
random.seed(2)
print("\n#################### DESTROYED samples (up✓ fin✗) ####################")
for r in random.sample(de, min(6, len(de))):
    print("-" * 100)
    print(f"[{r['dom']} ep{r['ep']}] {r['qt']}  everSolved={solved[r['i']]}/{seen[r['i']]}")
    print("Q:", r["q"][:200])
    print("GOLD:", str(r["gold"])[:260])
    print("UP  :", re.sub(r"\s+", " ", r["up"])[:260])
    print("FIN :", re.sub(r"\s+", " ", str(r["fin"]))[:260])
