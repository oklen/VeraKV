"""Paired analysis of the AMA-Agent vs VeraKV re-run (docs/AMA_AGENT_PAIRED.md).

    python ama/paired/analyze.py <run dir with s*/q.jsonl[.gz]> [--july <dir with mu_merged_FPA/FPB.json>] [--out x.json]

Prints, and writes as JSON:
  - per arm: n, accuracy, infra failures; by domain and question type
  - paired differences with episode-clustered bootstrap CIs (4,000 resamples, seed 20261008), bootstrap p,
    exact McNemar p, wins/losses; Holm over the two primary endpoints
  - context length per arm (tokens) and accuracy by whether A1's final context hit its 23,808-char cap
  - AMA-Agent path stats (direct answers, sufficiency rounds, code search, 60-second deadline)
  - reproduction checks: V1 against the July FPA+FPB verdicts; A1 against the published per-domain numbers
"""
import argparse
import glob
import gzip
import json
import math
import random
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--july", default="")
ap.add_argument("--data", default="", help="AMA open-ended jsonl, for trajectory lengths")
ap.add_argument("--out", default="")
ap.add_argument("--boot", type=int, default=4000)
args = ap.parse_args()

SEED = 20261008
CAP_CUT_CHARS = int(23808 * 0.7) + 5 + (23808 - int(23808 * 0.7))  # 23,813: head + "\n...\n" + tail
DOMS = ["WEB", "EMBODIED_AI", "Game", "TEXT2SQL", "SOFTWARE"]
TYPES = {"A": "recall", "B": "causal", "C": "state_update", "D": "state_abstraction"}
PUB = {"WEB": .517, "EMBODIED_AI": .499, "Game": .644, "TEXT2SQL": .609, "SOFTWARE": .636}


def load(run):
    rows = {}
    files = sorted(glob.glob(run + "/s*/q.jsonl")) or sorted(glob.glob(run + "/s*/q.jsonl.gz"))
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", encoding="utf-8") as fh:
            for l in fh:
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                k = (r["arm"], r["ep"], r["qi"])
                # keep the last non-infra record for a key (a resumed run may append a retry)
                if k not in rows or (rows[k].get("infra") and not r.get("infra")):
                    rows[k] = r
    return rows


rows = load(args.run)
arms = sorted({k[0] for k in rows})
qs = sorted({(k[1], k[2]) for k in rows})
meta = {}
for (a, e, q), r in rows.items():
    meta[(e, q)] = (r["domain"], r["qtype"])
OUT = {"n_questions": len(qs), "arms": {}}


def correct(r):
    return 1 if (r.get("score") == 1.0) else 0


def ok(r):
    return r is not None and not r.get("infra") and r.get("score") is not None


print("questions:", len(qs), "episodes:", len({e for e, _ in qs}))
for a in arms:
    rs = [rows.get((a, e, q)) for e, q in qs]
    good = [r for r in rs if ok(r)]
    inf = sum(1 for r in rs if r is None or not ok(r))
    acc = sum(correct(r) for r in good) / max(1, len(good))
    d = {"n": len(good), "acc": acc, "infra": inf, "infra_rate": inf / max(1, len(rs))}
    for dom in DOMS:
        g = [r for r in good if r["domain"] == dom]
        d["dom_" + dom] = [len(g), sum(correct(r) for r in g) / max(1, len(g))]
    for t in TYPES:
        g = [r for r in good if r["qtype"] == t]
        d["type_" + t] = [len(g), sum(correct(r) for r in g) / max(1, len(g))]
    OUT["arms"][a] = d
    print("%-3s n=%d acc=%.4f infra=%d (%.1f%%) | %s | %s" % (
        a, len(good), acc, inf, 100 * d["infra_rate"],
        " ".join("%s %.3f" % (dom[:4], d["dom_" + dom][1]) for dom in DOMS),
        " ".join("%s %.3f" % (t, d["type_" + t][1]) for t in TYPES)))


def binom_two_sided(k, n):
    """Exact two-sided McNemar p: P(X <= min(k, n-k)) * 2 under Binomial(n, 1/2), capped at 1."""
    if n == 0:
        return 1.0
    m = min(k, n - k)
    s = sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i)) for i in range(0, m + 1)) / (2 ** n)
    return min(1.0, 2 * s)


def paired(a, b, sel=lambda dom, t: True):
    """a - b over questions where both arms have a verdict and sel(domain, qtype)."""
    by_ep = defaultdict(list)
    w = l = 0
    for e, q in qs:
        dom, t = meta[(e, q)]
        if not sel(dom, t):
            continue
        ra, rb = rows.get((a, e, q)), rows.get((b, e, q))
        if not (ok(ra) and ok(rb)):
            continue
        x, y = correct(ra), correct(rb)
        by_ep[e].append((x, y))
        w += 1 if (x and not y) else 0
        l += 1 if (y and not x) else 0
    eps = sorted(by_ep)
    n = sum(len(v) for v in by_ep.values())
    if n == 0:
        return None
    diff = sum(x - y for v in by_ep.values() for x, y in v) / n
    rng = random.Random(SEED)
    bs = []
    for _ in range(args.boot):
        sx = sy = sn = 0
        for _ in eps:
            v = by_ep[eps[rng.randrange(len(eps))]]
            sx += sum(x for x, _ in v)
            sy += sum(y for _, y in v)
            sn += len(v)
        bs.append((sx - sy) / max(1, sn))
    bs.sort()
    lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
    p_boot = min(1.0, 2 * min(sum(1 for v in bs if v <= 0), sum(1 for v in bs if v >= 0)) / len(bs))
    return {"n": n, "eps": len(eps), "diff": diff, "lo": lo, "hi": hi, "p_boot": p_boot,
            "wins": w, "losses": l, "p_mcnemar": binom_two_sided(w, w + l)}


def fmt(r):
    if r is None:
        return "n/a"
    return "%+.1f [%+.1f, %+.1f] n=%d %d/%d p_boot=%.3g p_mcn=%.3g" % (
        100 * r["diff"], 100 * r["lo"], 100 * r["hi"], r["n"], r["wins"], r["losses"], r["p_boot"], r["p_mcnemar"])


SLICES = [("all", lambda d, t: True), ("state_update", lambda d, t: t == "C")]
SLICES += [("dom_" + dom, (lambda dd: (lambda d, t: d == dd))(dom)) for dom in DOMS]
SLICES += [("type_" + t, (lambda tt: (lambda d, t: t == tt))(t)) for t in TYPES]
PAIRS = [("V1", "A1"), ("A2", "A1"), ("V1", "A2"), ("V2", "V1"), ("V2", "A1")]
OUT["paired"] = {}
for a, b in PAIRS:
    if a not in arms or b not in arms:
        continue
    print("\n== %s - %s" % (a, b))
    for name, sel in SLICES:
        r = paired(a, b, sel)
        OUT["paired"]["%s-%s|%s" % (a, b, name)] = r
        print("  %-22s %s" % (name, fmt(r)))

# Holm over the two primary endpoints (bootstrap p)
prim = [OUT["paired"].get("V1-A1|all"), OUT["paired"].get("V1-A1|state_update")]
if all(prim):
    ps = sorted([(prim[0]["p_boot"], "P1 all"), (prim[1]["p_boot"], "P2 state_update")])
    holm = []
    run_max = 0.0
    for i, (p, name) in enumerate(ps):
        adj = min(1.0, max(run_max, (len(ps) - i) * p))
        run_max = adj
        holm.append((name, p, adj))
    OUT["holm_primary"] = holm
    print("\nHolm (primary):", "; ".join("%s p=%.3g adj=%.3g" % h for h in holm))


# ---- context lengths and the A1 cap
def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


print("\n== context length (tokens): median [p10, p90]; answer prompt tokens (server) median")
OUT["ctx"] = {}
for a in arms:
    ct = [r["ctx_tokens"] for (aa, e, q), r in rows.items() if aa == a and ok(r) and r.get("ctx_tokens")]
    pt = [r["answer_prompt_tokens"] for (aa, e, q), r in rows.items() if aa == a and ok(r) and r.get("answer_prompt_tokens")]
    OUT["ctx"][a] = {"ctx_med": pct(ct, .5), "ctx_p10": pct(ct, .1), "ctx_p90": pct(ct, .9),
                     "prompt_med": pct(pt, .5), "prompt_p90": pct(pt, .9)}
    print("  %-3s ctx %s [%s, %s]  answer-prompt %s (p90 %s)" % (
        a, pct(ct, .5), pct(ct, .1), pct(ct, .9), pct(pt, .5), pct(pt, .9)))

if "A1" in arms:
    cut = {(e, q) for (a, e, q), r in rows.items() if a == "A1" and r.get("ctx_chars") == CAP_CUT_CHARS}
    direct = {(e, q) for (a, e, q), r in rows.items() if a == "A1" and (r.get("ama") or {}).get("direct")}
    nA1 = sum(1 for (a, e, q) in rows if a == "A1")
    OUT["a1_cut_share"] = len(cut) / max(1, nA1)
    OUT["a1_direct_share"] = len(direct) / max(1, nA1)
    print("\nA1 final context cut at the char cap: %d/%d (%.1f%%); direct-path answers: %d (%.1f%%)" % (
        len(cut), nA1, 100 * OUT["a1_cut_share"], len(direct), 100 * OUT["a1_direct_share"]))
    for name, S in (("cut", lambda k: k in cut), ("not cut", lambda k: k not in cut),
                    ("reader path & cut", lambda k: k in cut and k not in direct),
                    ("reader path & not cut", lambda k: k not in cut and k not in direct),
                    ("direct path", lambda k: k in direct)):
        line = []
        for a in arms:
            g = [rows.get((a,) + k) for k in qs if S(k)]
            g = [r for r in g if ok(r)]
            line.append("%s %.3f (n=%d)" % (a, sum(correct(r) for r in g) / max(1, len(g)), len(g)))
        print("  %-22s %s" % (name, "  ".join(line)))
        sc = [k for k in qs if S(k) and meta[k][1] == "C"]
        line = []
        for a in arms:
            g = [r for r in (rows.get((a,) + k) for k in sc) if ok(r)]
            line.append("%s %.3f (n=%d)" % (a, sum(correct(r) for r in g) / max(1, len(g)), len(g)))
        print("  %-22s %s" % ("  ...state update", "  ".join(line)))
    # A1 path stats
    st = Counter()
    for (a, e, q), r in rows.items():
        if a != "A1":
            continue
        am = r.get("ama") or {}
        stages = am.get("stages") or []
        st["n"] += 1
        st["direct"] += 1 if am.get("direct") else 0
        st["suff_calls"] += stages.count("suff")
        st["code_search"] += 1 if "codegen" in stages else 0
        st["codegen_calls"] += stages.count("codegen")
        suff_dt = sum(c[3] or 0 for c in r.get("calls") or [] if c[0] == "suff")
        st["suff_over_60s"] += 1 if suff_dt >= 60 else 0
        code_dt = sum(c[3] or 0 for c in r.get("calls") or [] if c[0] == "codegen")
        st["codegen_over_60s"] += 1 if code_dt >= 60 else 0
    OUT["a1_paths"] = dict(st)
    print("\nA1 paths:", dict(st))

# ---- V1 - A1 by trajectory length and by how much A1's answering call read
def bin_table(title, keyfn, edges, a="V1", b="A1"):
    print("\n== %s - %s by %s" % (a, b, title))
    res = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel_keys = {k for k in qs if keyfn(k) is not None and lo <= keyfn(k) < hi}
        n = w = l = ca = cb = 0
        for k in sel_keys:
            ra, rb = rows.get((a,) + k), rows.get((b,) + k)
            if not (ok(ra) and ok(rb)):
                continue
            x, y = correct(ra), correct(rb)
            n += 1
            ca += x
            cb += y
            w += 1 if (x and not y) else 0
            l += 1 if (y and not x) else 0
        if n:
            res.append({"lo": lo, "hi": hi, "n": n, "acc_a": ca / n, "acc_b": cb / n, "diff": (ca - cb) / n,
                        "wins": w, "losses": l, "p_mcnemar": binom_two_sided(w, w + l)})
            print("  [%6s, %6s) n=%4d %s %.3f %s %.3f diff %+.1f  %d/%d p_mcn=%.3g" % (
                lo, hi, n, a, ca / n, b, cb / n, 100 * (ca - cb) / n, w, l, binom_two_sided(w, w + l)))
    return res


if args.data and "V1" in arms and "A1" in arms:
    ttok = {}
    for l in open(args.data, encoding="utf-8"):
        if l.strip():
            e = json.loads(l)
            ttok[e["episode_id"]] = e.get("total_tokens")
    OUT["by_traj_len"] = bin_table("trajectory length (tokens)", lambda k: ttok.get(k[0]),
                                   [0, 10000, 20000, 40000, 80000, 10 ** 9])
    a1p = {(e, q): r.get("answer_prompt_tokens") for (a, e, q), r in rows.items() if a == "A1"}
    OUT["by_a1_read"] = bin_table("A1 answer-prompt tokens", lambda k: a1p.get(k),
                                  [0, 2000, 4000, 8000, 16000, 10 ** 9])
    v1p = {(e, q): r.get("answer_prompt_tokens") for (a, e, q), r in rows.items() if a == "V1"}
    OUT["by_v1_read"] = bin_table("V1 answer-prompt tokens", lambda k: v1p.get(k),
                                  [0, 6000, 10000, 14000, 18000, 10 ** 9])

# ---- reproduction checks
if args.july and "V1" in arms:
    jr = []
    for t in ("FPA", "FPB"):
        jr += json.load(open("%s/mu_merged_%s.json" % (args.july, t)))
    jk = {}
    for r in jr:
        jk.setdefault((r["episode_id"], r["question"]), r)
    agree = n = jc = vc = 0
    for (a, e, q), r in rows.items():
        if a != "V1" or not ok(r):
            continue
        j = jk.get((e, r["question"]))
        if j is None:
            continue
        n += 1
        x, y = correct(r), (1 if j["score"] == 1.0 else 0)
        agree += 1 if x == y else 0
        vc += x
        jc += y
    OUT["v1_vs_july"] = {"n": n, "agree": agree / max(1, n), "acc_now": vc / max(1, n), "acc_july": jc / max(1, n)}
    print("\nV1 vs July FPA+FPB: n=%d agreement %.3f acc now %.4f vs July %.4f" % (
        n, agree / max(1, n), vc / max(1, n), jc / max(1, n)))
if "A1" in arms:
    d = OUT["arms"]["A1"]
    w = sum(d["dom_" + dom][0] for dom in DOMS)
    pubw = sum(PUB[dom] * d["dom_" + dom][0] for dom in DOMS) / max(1, w)
    OUT["a1_vs_published"] = {"a1": d["acc"], "published_weighted": pubw,
                              "by_dom": {dom: [d["dom_" + dom][1], PUB[dom]] for dom in DOMS}}
    print("A1 vs published (leaderboard, weighted by these counts): %.4f vs %.4f | %s" % (
        d["acc"], pubw, " ".join("%s %.3f/%.3f" % (dom[:4], d["dom_" + dom][1], PUB[dom]) for dom in DOMS)))

if args.out:
    json.dump(OUT, open(args.out, "w"), indent=1)
