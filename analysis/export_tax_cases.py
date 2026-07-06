"""S1 case study: per-domain PRSTR/PRDEF split + export-tax victim cases (run on master).

Victims = PRX1 adopted-but-wrong where PRSTR (structured, no export demand) got it right:
the questions where forcing a visible derivation plausibly corrupted the upstream pass.
"""
import json, re, os

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

S, D, X = load("PRSTR"), load("PRDEF"), load("PRX1")
ks = sorted(set(S) & set(D) & set(X))
print("row keys:", sorted(next(iter(S.values())).keys()))

# domain per episode from the probe dataset
dom = {}
for l in open(BASE + "AMA-Bench/dataset/test_prx/open_end_qa_set.jsonl", encoding="utf-8"):
    r = json.loads(l)
    dom[str(r["episode_id"])] = r.get("domain", "?")

per = {}
for k in ks:
    d = dom.get(k[0], "?")
    a = per.setdefault(d, [0, 0, 0, 0])  # nSTR_right, nDEF_right, nX_right, n
    a[0] += ok(S[k]); a[1] += ok(D[k]); a[2] += ok(X[k]); a[3] += 1
print("\nper-domain acc (n): STR / DEF / PRX1")
for d, (s, f, x, n) in sorted(per.items()):
    print("  %-12s n=%2d  STR=%.2f DEF=%.2f PRX1=%.2f" % (d, n, s / n, f / n, x / n))

# export-tax victims: PRX1 wrong & adopted, PRSTR right
sel = {}
for l in open(BASE + "sel_PRX1_full.jsonl"):
    r = json.loads(l)
    sel[nkey(r["q"])[:120]] = r

def adopted(up, ans):
    u, a = nkey(up), nkey(ans)
    if not u or not a:
        return None
    if u in a or a in u:
        return True
    su, sa = set(u.split()), set(a.split())
    return len(su & sa) / max(1, len(su | sa)) >= 0.6

shown = 0
for k in ks:
    if ok(X[k]) or not ok(S[k]):
        continue
    r = sel.get(nkey(k[1])[:120])
    if not r or not adopted(r.get("up_ans"), r.get("ans")):
        continue
    shown += 1
    row = X[k]
    gold = row.get("ground_truth") or row.get("answer") or row.get("gold") or "?"
    sp = S[k].get("prediction") or S[k].get("response") or S[k].get("final_answer") or "?"
    print("\n" + "=" * 24, "VICTIM %d" % shown, "(domain %s)" % dom.get(k[0], "?"), "=" * 24)
    print("Q:", k[1][:180])
    print("GOLD:", str(gold)[:160])
    print("PRSTR-right pred:", str(sp)[:160])
    print("quotes_ok:", r.get("quotes_ok"), "nq:", r.get("nq"))
    print("PRX1 ART:", r["art"][:1500])
    print("READER:", (r.get("ans") or "")[:200])
    if shown >= 8:
        break
print("\ntotal victims (PRX1 adopt-wrong & PRSTR-right): counted above, shown", shown)
