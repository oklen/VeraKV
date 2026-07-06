"""Tax decomposition verdict: SBRF (brevity-only) and SXPF (export+full answer) vs RESTR4.

Verifies the manipulation (answer lengths), then paired deltas w/ bootstrap CIs, type/F+
splits, and SXPF quote fidelity — the inputs for the corrected paper passage.
"""
import json, re, random, collections

BASE = "/home/tiger/"

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def nkey(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def load(tag):
    return {(str(r["episode_id"]), norm(r["question"])): r
            for r in json.load(open(BASE + "mu_merged_%s.json" % tag))}

def ok(r):
    return r["score"] >= 0.5

A = load("RESTR4"); SX = load("SXPH"); SB = load("SBRF"); SF = load("SXPF")
S2, D2 = load("RESTR2"), load("REDEF2")
ks = sorted(set(A) & set(SX) & set(SB) & set(SF))
print("common n =", len(ks))

def acc(M, kk):
    return sum(ok(M[k]) for k in kk) / len(kk)

def med(M, kk):
    v = sorted(len(M[k]["predicted_answer"] or "") for k in kk)
    return v[len(v) // 2]

random.seed(0)
def ci(M, kk):
    d = [ok(M[k]) - ok(A[k]) for k in kk]
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
    return sum(d) / len(d), bs[100], bs[3900]

print("%-6s acc=%.4f  med-chars=%d  (anchor)" % ("RESTR4", acc(A, ks), med(A, ks)))
for tag, M in (("SXPH", SX), ("SBRF", SB), ("SXPF", SF)):
    m, lo, hi = ci(M, ks)
    print("%-6s acc=%.4f  med-chars=%d  d=%+.4f CI[%+.4f,%+.4f]" % (tag, acc(M, ks), med(M, ks), m, lo, hi))

# type split
print("\nby qa_type (RESTR4 / SBRF / SXPF / SXPH):")
byt = collections.defaultdict(list)
for k in ks:
    m = re.match(r"Type ([A-D])", A[k].get("qa_type") or A[k]["question"])
    byt[m.group(1) if m else "?"].append(k)
for t, kk in sorted(byt.items()):
    if len(kk) < 30:
        continue
    print("  %s n=%3d  %.3f / %.3f / %.3f / %.3f   med-chars %d/%d/%d/%d"
          % (t, len(kk), acc(A, kk), acc(SB, kk), acc(SF, kk), acc(SX, kk),
             med(A, kk), med(SB, kk), med(SF, kk), med(SX, kk)))

# F+ pool
fplus = [k for k in ks if k in S2 and k in D2 and ok(S2[k]) and not ok(D2[k])]
print("\nF+ (n=%d): RESTR4 %.3f  SBRF %.3f  SXPF %.3f  SXPH %.3f"
      % (len(fplus), acc(A, fplus), acc(SB, fplus), acc(SF, fplus), acc(SX, fplus)))

# SXPF quote fidelity + compliance
sel = {}
try:
    for l in open(BASE + "sel_SXPF_full.jsonl"):
        r = json.loads(l)
        sel[nkey(r["q"])[:120]] = r
    rows = list(sel.values())
    comp = sum(1 for r in rows if "Evidence:" in r["art"] and "Derivation:" in r["art"] and "Answer[1]:" in r["art"])
    qoks = [r["quotes_ok"] for r in rows if r.get("quotes_ok") is not None]
    print("\nSXPF dump n=%d compliant=%.0f%% all-verbatim=%.0f%%"
          % (len(rows), 100 * comp / len(rows), 100 * sum(1 for q in qoks if q == 1.0) / max(1, len(qoks))))
except Exception as e:
    print("no SXPF dump:", e)
