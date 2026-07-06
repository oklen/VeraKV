"""Audit the -11.8pp export tax for mundane causes (run on master).

Checks: (1) extraction failures — SXPH predicted_answer that is a raw Evidence/Derivation
dump or empty; (2) answer-length shift RESTR4 vs SXPH (did the terse few-shot shorten
final answers?); (3) score-vs-length relation within SXPH; (4) side-by-side cases
SXPH-wrong & RESTR4-right: gold vs both predictions, to classify terse-but-right-ish vs
genuinely wrong.
"""
import json, re, collections

BASE = "/home/tiger/"

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def load(tag):
    return {(str(r["episode_id"]), norm(r["question"])): r
            for r in json.load(open(BASE + "mu_merged_%s.json" % tag))}

def ok(r):
    return r["score"] >= 0.5

A, X = load("RESTR4"), load("SXPH")
ks = sorted(set(A) & set(X))

# 1. extraction failures
dump_leak = [k for k in ks if "Evidence:" in (X[k]["predicted_answer"] or "")
             or "Derivation:" in (X[k]["predicted_answer"] or "")]
empty = [k for k in ks if not (X[k]["predicted_answer"] or "").strip()]
print("SXPH preds containing Evidence:/Derivation: (extraction leak): %d/%d" % (len(dump_leak), len(ks)))
print("  of which judged wrong: %d" % sum(1 for k in dump_leak if not ok(X[k])))
print("SXPH empty preds: %d" % len(empty))

# 2. length shift
la = sorted(len(A[k]["predicted_answer"] or "") for k in ks)
lx = sorted(len(X[k]["predicted_answer"] or "") for k in ks)
def q(v, p):
    return v[int(p * (len(v) - 1))]
print("\npred-answer chars  RESTR4: p25=%d med=%d p75=%d   SXPH: p25=%d med=%d p75=%d"
      % (q(la, .25), q(la, .5), q(la, .75), q(lx, .25), q(lx, .5), q(lx, .75)))
for t in "ABCD":
    kk = [k for k in ks if (A[k].get("qa_type") or "").startswith(t) or
          re.match(r"Type %s" % t, A[k]["question"])]
    if len(kk) < 30:
        continue
    ma = q(sorted(len(A[k]["predicted_answer"] or "") for k in kk), .5)
    mx = q(sorted(len(X[k]["predicted_answer"] or "") for k in kk), .5)
    print("  type %s n=%3d  med chars RESTR4=%4d SXPH=%4d" % (t, len(kk), ma, mx))

# 3. within-SXPH: acc by answer-length quartile
byq = collections.defaultdict(list)
cuts = [q(lx, .25), q(lx, .5), q(lx, .75)]
for k in ks:
    L = len(X[k]["predicted_answer"] or "")
    b = sum(L > c for c in cuts)
    byq[b].append(ok(X[k]))
print("\nSXPH acc by own answer-length quartile:",
      {b: "%.3f(n=%d)" % (sum(v) / len(v), len(v)) for b, v in sorted(byq.items())})

# same split for RESTR4 as baseline
byqa = collections.defaultdict(list)
cutsa = [q(la, .25), q(la, .5), q(la, .75)]
for k in ks:
    L = len(A[k]["predicted_answer"] or "")
    b = sum(L > c for c in cutsa)
    byqa[b].append(ok(A[k]))
print("RESTR4 acc by own answer-length quartile:",
      {b: "%.3f(n=%d)" % (sum(v) / len(v), len(v)) for b, v in sorted(byqa.items())})

# 4. side-by-side broke cases
broke = [k for k in ks if ok(A[k]) and not ok(X[k])]
print("\nbroke (RESTR4-right, SXPH-wrong) n=%d; showing 8:" % len(broke))
for i, k in enumerate(broke[:8]):
    print("\n--- CASE %d [%s | %s] ---" % (i + 1, A[k].get("domain"), A[k].get("qa_type")))
    print("Q:", k[1][:160])
    print("GOLD:", norm(str(X[k].get("golden_answer")))[:220])
    print("RESTR4 (right):", norm(A[k]["predicted_answer"])[:260])
    print("SXPH   (wrong):", norm(X[k]["predicted_answer"])[:260])
