"""vote_analysis.py -- answer-level self-consistency over the 12-arm conditioner matrix (kvmemory/kv_matrix.py
outputs, dev + held-out episodes, n=824; offline, no GPU). One of the fallback-trigger signal families
the paper reports as failing to beat chance (Sec. 4, "Matched evidence, and the ceiling").

    python analysis/vote_analysis.py [results/oow/mx.jsonl.gz]

Signal being tested: wrong answers scatter (each view hallucinates differently), correct answers
coincide -> (1) majority vote across cached views, (2) peer-agreement as the gold-free wrongness
detector the verifier failed to be: accept b_anch iff >=m peer views agree with it, else tx.
Judgments reuse the stored per-arm oks (the returned answer is always one of the stored strings).
"""
import glob, json, re, sys
from collections import defaultdict
from math import comb

import gzip, os
_P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "oow", "mx.jsonl.gz")
rows = [json.loads(l) for l in gzip.open(_P, "rt", encoding="utf-8") if l.strip()]
print("rows:", len(rows))

STOP = set("the a an of to in is are was were be been and or for with that this it as on at by from into over after before during their its his her they he she we you i not no yes what which when where how why".split())

def toks(s, qset=None):
    t = set(re.findall(r"[a-z0-9]+", s.lower())) - STOP
    if qset:
        t = t - qset
    return t

def jac(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def clusters(items, th):
    """items: list of (arm, tokset). greedy connected components by jaccard>=th."""
    n = len(items)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if jac(items[i][1], items[j][1]) >= th:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    comp = defaultdict(list)
    for i in range(n):
        comp[find(i)].append(i)
    return list(comp.values())

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(min(b, c) + 1)) * 2 / 2 ** n)

ENSEMBLES = {
    "E4cache": ["b_anch", "roll", "sem", "randf"],
    "E6cache": ["b_anch", "anch", "c_anch", "roll", "sem", "randf"],
    "E3cache": ["b_anch", "roll", "sem"],
    "E12all":  ["tx", "iso", "dum_s", "dum_l", "fixnat", "ext", "randf", "sem", "roll", "anch", "b_anch", "c_anch"],
}
PRIORITY = {a: i for i, a in enumerate(["b_anch", "roll", "sem", "tx", "randf", "anch", "c_anch", "iso", "dum_l", "dum_s", "fixnat", "ext"])}

base = {a: sum(r[a] for r in rows) / len(rows) for a in ENSEMBLES["E12all"]}
print("base acc:", " ".join("%s %.3f" % (a, base[a]) for a in ["b_anch", "roll", "sem", "randf", "tx"]))

for th in (0.5, 0.6):
    print("\n########## jaccard threshold %.1f (question tokens removed) ##########" % th)
    for ename, arms in ENSEMBLES.items():
        acc_vote = 0
        ora = 0
        votes_when_right = []
        for r in rows:
            qset = toks(r["q"])
            items = [(a, toks(r.get("ans_" + a) or "", qset)) for a in arms]
            cl = clusters(items, th)
            cl.sort(key=lambda c: (-len(c), min(PRIORITY[items[i][0]] for i in c)))
            top = cl[0]
            rep = min(top, key=lambda i: PRIORITY[items[i][0]])
            acc_vote += r[items[rep][0]]
            ora += max(r[a] for a in arms)
        n = len(rows)
        print("%-8s vote=%.3f  (oracle %.3f, best-member %.3f)" %
              (ename, acc_vote / n, ora / n, max(base[a] for a in arms)))

    # peer-agreement cascade: accept b_anch iff >=m peers (within E) agree with its answer, else tx
    print("--- peer-agreement cascade (primary b_anch, fallback tx) ---")
    for ename, arms in ENSEMBLES.items():
        if "b_anch" not in arms:
            continue
        peers = [a for a in arms if a != "b_anch"]
        for m in range(1, min(4, len(peers)) + 1):
            acc = esc = 0
            tp = fp = tn = fn = 0
            b = c = 0
            for r in rows:
                qset = toks(r["q"])
                bset = toks(r.get("ans_b_anch") or "", qset)
                nag = sum(1 for a in peers if jac(bset, toks(r.get("ans_" + a) or "", qset)) >= th)
                take_b = nag >= m
                ok = r["b_anch"] if take_b else r["tx"]
                acc += ok
                esc += 0 if take_b else 1
                wrong_b = 1 - r["b_anch"]
                fire = 0 if take_b else 1
                if wrong_b and fire: tp += 1
                elif wrong_b: fn += 1
                elif fire: fp += 1
                else: tn += 1
                if ok and not r["tx"]: b += 1
                if r["tx"] and not ok: c += 1
            n = len(rows)
            print("%-8s m>=%d: acc=%.3f esc=%.2f  TPR=%.3f FPR=%.3f  vs tx p=%.4f" %
                  (ename, m, acc / n, esc / n, tp / max(1, tp + fn), fp / max(1, fp + tn), mcnemar(b, c)))
