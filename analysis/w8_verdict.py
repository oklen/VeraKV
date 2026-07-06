"""Full-scale export-tax verdict (run on master after RESTR4/SXPH/HOFX land).

1. The tax: SXPH vs RESTR4, paired delta with question-level bootstrap CI.
2. Relay faithfulness: HOFX vs SXPH; HOFX vs HOFT (cross-day, band caveat).
3. Mechanism at scale: compliance / quotes_ok / adoption for SXPH & HOFX dumps.
4. Gap recovery on F+ (REDEF2-wrong & RESTR2-right) — what the small set could not measure.
5. Tax by domain and qa_type (soft-type concentration check).
"""
import json, re, random, os, collections

BASE = "/home/tiger/"

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def nkey(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def load(tag):
    f = BASE + "mu_merged_%s.json" % tag
    if not os.path.exists(f):
        return None
    return {(str(r["episode_id"]), norm(r["question"])): r for r in json.load(open(f))}

def ok(r):
    return r["score"] >= 0.5

def load_sel(tag):
    f = BASE + "sel_%s_full.jsonl" % tag
    out = {}
    if os.path.exists(f):
        for l in open(f):
            try:
                r = json.loads(l)
                out[nkey(r["q"])[:120]] = r
            except Exception:
                pass
    return out

def adopted(up, ans):
    u, a = nkey(up), nkey(ans)
    if not u or not a:
        return None
    if u in a or a in u:
        return True
    su, sa = set(u.split()), set(a.split())
    return len(su & sa) / max(1, len(su | sa)) >= 0.6

A4, SX, HX = load("RESTR4"), load("SXPH"), load("HOFX")
S2, D2, HT = load("RESTR2"), load("REDEF2"), load("HOFT")
ks = sorted(set(A4) & set(SX) & set(HX))
print("common n =", len(ks))

def acc(M, kk):
    return sum(ok(M[k]) for k in kk) / len(kk)

print("RESTR4 %.4f | SXPH %.4f | HOFX %.4f" % (acc(A4, ks), acc(SX, ks), acc(HX, ks)))

# 1. tax with paired bootstrap CI
random.seed(0)
d = [ok(SX[k]) - ok(A4[k]) for k in ks]
bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(4000))
print("TAX SXPH-RESTR4 = %+.4f  CI[%+.4f,%+.4f]  fixed=%d broke=%d"
      % (sum(d) / len(d), bs[100], bs[3900],
         sum(1 for k in ks if not ok(A4[k]) and ok(SX[k])),
         sum(1 for k in ks if ok(A4[k]) and not ok(SX[k]))))
d2 = [ok(HX[k]) - ok(SX[k]) for k in ks]
bs2 = sorted(sum(random.choice(d2) for _ in d2) / len(d2) for _ in range(4000))
print("RELAY HOFX-SXPH = %+.4f  CI[%+.4f,%+.4f]" % (sum(d2) / len(d2), bs2[100], bs2[3900]))
if HT:
    kk = sorted(set(HX) & set(HT))
    print("HOFX %.4f vs HOFT %.4f on joint n=%d (cross-day band ±2pp)"
          % (acc(HX, kk), acc(HT, kk), len(kk)))

# 2. mechanism at scale
for t in ("SXPH", "HOFX"):
    sel = load_sel(t)
    rows = list(sel.values())
    if not rows:
        print(t, "no dump"); continue
    comp = sum(1 for r in rows if "Evidence:" in r["art"] and "Derivation:" in r["art"]
               and "Answer[1]:" in r["art"])
    qoks = [r["quotes_ok"] for r in rows if r.get("quotes_ok") is not None]
    ads = [a for a in (adopted(r.get("up_ans"), r.get("ans")) for r in rows) if a is not None]
    print("%s dump n=%d compliant=%.0f%% all-verbatim=%s adoption=%s"
          % (t, len(rows), 100 * comp / len(rows),
             ("%.0f%%" % (100 * sum(1 for q in qoks if q == 1.0) / len(qoks))) if qoks else "n/a",
             ("%.0f%%" % (100 * sum(ads) / len(ads))) if ads else "n/a"))

# 3. F+ gap recovery (REDEF2-wrong & RESTR2-right), vs HOFT on same pool
if S2 and D2:
    fplus = [k for k in ks if k in S2 and k in D2 and ok(S2[k]) and not ok(D2[k])]
    fminus = [k for k in ks if k in S2 and k in D2 and not ok(S2[k]) and ok(D2[k])]
    line = "F+ n=%d: RESTR4 %.3f SXPH %.3f HOFX %.3f" % (
        len(fplus), acc(A4, fplus), acc(SX, fplus), acc(HX, fplus))
    if HT:
        fp2 = [k for k in fplus if k in HT]
        line += " HOFT %.3f (n=%d)" % (acc(HT, fp2), len(fp2))
    print(line)
    print("F- n=%d: RESTR4 %.3f SXPH %.3f HOFX %.3f" % (
        len(fminus), acc(A4, fminus), acc(SX, fminus), acc(HX, fminus)))

# 4. tax by domain / qa_type
bydom = collections.defaultdict(list)
bytyp = collections.defaultdict(list)
for k in ks:
    r = A4[k]
    bydom[r.get("domain", "?")].append(k)
    m = re.match(r"Type ([A-D])", r.get("qa_type", "") or r["question"])
    bytyp[(m.group(1) if m else (r.get("qa_type") or "?")[:12])].append(k)
print("\ntax by domain (RESTR4 -> SXPH):")
for dm, kk in sorted(bydom.items()):
    print("  %-14s n=%3d  %.3f -> %.3f  (%+.3f)" % (dm, len(kk), acc(A4, kk), acc(SX, kk),
                                                    acc(SX, kk) - acc(A4, kk)))
print("tax by qa_type:")
for tp, kk in sorted(bytyp.items()):
    if len(kk) >= 30:
        print("  %-14s n=%3d  %.3f -> %.3f  (%+.3f)" % (tp, len(kk), acc(A4, kk), acc(SX, kk),
                                                        acc(SX, kk) - acc(A4, kk)))
