"""S1 small-set verdict: joint analysis of the test_prx variant matrix (run on master).

Arms: PRSTR (structured in-place, ceiling) PRDEF (default floor) PRTRU (answer+trust relay)
PRXD (export+trust) PRX1 (export+verify). All n~144 on identical questions -> paired deltas.
For relay arms, joins the sel_*_full.jsonl dump for compliance/adoption/quote metrics and
the adoption x correctness quadrant.
"""
import json, re, glob, os

BASE = "/home/tiger/"

def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def nkey(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def load_merged(tag):
    f = BASE + "mu_merged_%s.json" % tag
    if not os.path.exists(f):
        return None
    return {(str(r["episode_id"]), norm(r["question"])): r for r in json.load(open(f))}

def ok(r):
    return r["score"] >= 0.5

def load_sel(tag):
    f = BASE + "sel_%s_full.jsonl" % tag
    if not os.path.exists(f):
        return {}
    out = {}
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        out[nkey(r["q"])[:120]] = r
    return out

def adopted(up, ans):
    u, a = nkey(up), nkey(ans)
    if not u or not a:
        return None
    if u in a or a in u:
        return True
    su, sa = set(u.split()), set(a.split())
    return len(su & sa) / max(1, len(su | sa)) >= 0.6

def main():
    import sys
    tags = sys.argv[1:] or ["PRSTR", "PRDEF", "PRTRU", "PRXD", "PRX1"]
    M = {t: load_merged(t) for t in tags}
    have = [t for t in tags if M[t]]
    S = M.get("PRSTR"); D = M.get("PRDEF")
    common = set.intersection(*[set(M[t]) for t in have]) if have else set()
    print("arms present:", have, " common questions:", len(common))
    ks = sorted(common)
    for t in have:
        acc = sum(ok(M[t][k]) for k in ks) / len(ks)
        line = "%-6s acc=%.3f" % (t, acc)
        if S and t != "PRSTR":
            dS = acc - sum(ok(S[k]) for k in ks) / len(ks)
            fixed = sum(1 for k in ks if not ok(S[k]) and ok(M[t][k]))
            broke = sum(1 for k in ks if ok(S[k]) and not ok(M[t][k]))
            ret = (sum(1 for k in ks if ok(S[k]) and ok(M[t][k]))
                   / max(1, sum(1 for k in ks if ok(S[k]))))
            line += "  dSTR=%+.3f fixed=%d broke=%d retention=%.3f" % (dS, fixed, broke, ret)
        if D and t != "PRDEF":
            dD = acc - sum(ok(D[k]) for k in ks) / len(ks)
            line += "  dDEF=%+.3f" % dD
        print(line)
    # relay-arm mechanism metrics
    for t in [t for t in tags if t not in ("PRSTR", "PRDEF")]:
        if not M.get(t):
            continue
        sel = load_sel(t)
        if not sel:
            print(t, ": no sel dump"); continue
        rows = list(sel.values())
        comp = sum(1 for r in rows if "Evidence:" in r["art"] and "Derivation:" in r["art"]
                   and "Answer[1]:" in r["art"])
        qoks = [r["quotes_ok"] for r in rows if r.get("quotes_ok") is not None]
        ads = [adopted(r.get("up_ans"), r.get("ans")) for r in rows]
        ads = [a for a in ads if a is not None]
        print("%-6s dump n=%d  compliant=%.0f%%  quotes(all-verbatim)=%s  adoption=%.0f%%"
              % (t, len(rows), 100 * comp / max(1, len(rows)),
                 ("%.0f%%" % (100 * sum(1 for q in qoks if q == 1.0) / len(qoks))) if qoks else "n/a",
                 100 * sum(ads) / max(1, len(ads))))
        # adoption x correctness quadrant (join dump -> merged by question key)
        byq = {}
        for k in ks:
            byq[nkey(k[1])[:120]] = ok(M[t][k])
        quad = {"adopt_right": 0, "adopt_wrong": 0, "reject_right": 0, "reject_wrong": 0}
        for r in rows:
            key = nkey(r["q"])[:120]
            if key not in byq:
                continue
            a = adopted(r.get("up_ans"), r.get("ans"))
            if a is None:
                continue
            quad[("adopt_" if a else "reject_") + ("right" if byq[key] else "wrong")] += 1
        print("       quadrant:", quad)

if __name__ == "__main__":
    main()
