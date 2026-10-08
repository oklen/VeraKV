"""Failure-fallback (cascade) ceiling analysis from the fully-paired matrix.

1) PERFECT-detector ceilings: cascade with oracle wrongness detection == Oracle of the arm set,
   with expected escalation cost.
2) Detector-quality frontier: cascade acc as a function of (TPR=P(flag|wrong), FPR=P(flag|correct))
   using TRUE joint outcome distributions per chain.
3) REALIZABLE detectors evaluated now: degen/abstain/short regexes; cross-arm agreement as a
   CONFIDENCE gate (agree->accept, disagree->escalate).

Input: the 12-arm conditioner matrix (kvmemory/kv_matrix.py, dev + held-out episodes, n=824).
    python analysis/cascade_ceiling.py [results/oow/mx.jsonl.gz]
"""
import glob
import json
import re

import sys
import gzip, os
_P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "oow", "mx.jsonl.gz")
rows = [json.loads(l) for l in gzip.open(_P, "rt", encoding="utf-8") if l.strip()]
n = len(rows)
print("pooled n=%d | refs: b_anch 18.4  roll 16.7  sem 16.6  tx 17.7" % n)

# ---- 1) perfect-detector ceilings ----
print("\n=== 1) PERFECT-detector cascade ceilings (== Oracle of the set) ===")
chains = [["b_anch", "tx"], ["b_anch", "roll", "tx"], ["b_anch", "roll", "sem", "tx"],
          ["b_anch", "roll", "sem", "randf", "iso", "tx"],
          ["b_anch", "roll", "sem", "randf", "anch", "iso", "ext", "fixnat", "dum_l", "tx"]]
for ch in chains:
    orc = sum(1 for r in rows if any(r[a] for a in ch)) / n
    # expected stages consumed under perfect detection (stop at first correct; if none, run all)
    es = 0.0
    for r in rows:
        k = next((i for i, a in enumerate(ch) if r[a]), len(ch) - 1)
        es += k
    print("  %-46s ceiling %5.1f%%  mean extra stages %.2f"
          % ("->".join(ch), 100.0 * orc, es / n))

# ---- 2) detector-quality frontier ----
print("\n=== 2) frontier: acc of chain given detector (TPR, FPR), applied at every stage ===")
def chain_acc(ch, tpr, fpr):
    acc = 0.0
    for r in rows:
        p_reach = 1.0
        for i, a in enumerate(ch):
            ok = r[a]
            last = (i == len(ch) - 1)
            if last:
                acc += p_reach * ok
                break
            stay = (1 - fpr) if ok else (1 - tpr)   # prob we ACCEPT this stage's answer
            acc += p_reach * stay * ok
            p_reach *= (1 - stay)
    return 100.0 * acc / n
for ch in (["b_anch", "tx"], ["b_anch", "roll", "tx"]):
    print("  chain %s" % "->".join(ch))
    hdr = "    TPR\\FPR " + "".join("%8.2f" % f for f in (0.0, 0.05, 0.1, 0.2))
    print(hdr)
    for tpr in (0.5, 0.7, 0.85, 0.95, 1.0):
        line = "    %7.2f " % tpr
        for fpr in (0.0, 0.05, 0.1, 0.2):
            line += "%8.1f" % chain_acc(ch, tpr, fpr)
        print(line)

# ---- 3) realizable detectors ----
_DEG = re.compile(r"^\s*`{0,3}\s*(action|click|scroll|press|type|hover|goto|go to|stop)\b|"
                  r"action \[arg\]|^\s*```", re.I)
_NA = re.compile(r"cannot be answered|no information|not provided|not mentioned|not specified", re.I)
_T = re.compile(r"[a-z0-9]+")
def bad_regex(ans):
    a = (ans or "").strip()
    return len(a) < 12 or bool(_DEG.search(a)) or bool(_NA.search(a))
def jac(x, y):
    X, Y = set(_T.findall((x or "").lower())), set(_T.findall((y or "").lower()))
    return len(X & Y) / max(1, len(X | Y))

print("\n=== 3) realizable detectors ===")
# 3a regex gate on b_anch -> tx / -> roll -> tx
for chain_desc, seq in (("b_anch -regex-> tx", ["b_anch", "tx"]),
                        ("b_anch -regex-> roll -regex-> tx", ["b_anch", "roll", "tx"])):
    acc = esc = 0
    for r in rows:
        done = False
        for i, a in enumerate(seq):
            if i == len(seq) - 1 or not bad_regex(r.get("ans_" + a)):
                acc += r[a]
                esc += i
                done = True
                break
        if not done:
            acc += r[seq[-1]]
    print("  %-38s acc %5.1f%%  mean escalations %.3f" % (chain_desc, 100.0 * acc / n, esc / n))
# 3b agreement gate: run b_anch AND roll; agree->accept b_anch; disagree->policy
print("  -- agreement gate (b_anch vs roll), sweep threshold --")
for th in (0.25, 0.35, 0.5):
    for pol in ("tx", "sem"):
        acc = dis = 0
        for r in rows:
            if jac(r.get("ans_b_anch"), r.get("ans_roll")) >= th:
                acc += r["b_anch"]
            else:
                dis += 1
                acc += r[pol]
        print("    th=%.2f disagree->%-4s acc %5.1f%%  (disagree rate %4.1f%%)"
              % (th, pol, 100.0 * acc / n, 100.0 * dis / n))
    # diagnosticity
    agree_ok = [r["b_anch"] for r in rows if jac(r.get("ans_b_anch"), r.get("ans_roll")) >= th]
    dis_ok = [r["b_anch"] for r in rows if jac(r.get("ans_b_anch"), r.get("ans_roll")) < th]
    print("    th=%.2f  P(b_anch ok | agree)=%.1f%% (n=%d)   P(ok | disagree)=%.1f%% (n=%d)"
          % (th, 100.0 * sum(agree_ok) / max(1, len(agree_ok)), len(agree_ok),
             100.0 * sum(dis_ok) / max(1, len(dis_ok)), len(dis_ok)))
