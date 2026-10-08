"""Memory-assembly 2x2 (paper App. D, Tables 'assembly' and 'assemblydom').

Same-batch official-harness runs (full 2,496 QA, Qwen3-32B reader + judge, structured answer
instruction, lexical router, 22k budget) crossing routed verbatim SELECTION x a one-line gist
OVERVIEW of the non-selected turns:

  ASM_RECENT     neither (hot window only)          ama/configs/cfg_arm_recent.json
  ASM_GIST       overview only                      ama/configs/cfg_arm_gist.json
  ASM_RETRIEVAL  selection only                     ama/configs/cfg_arm_retrieval.json
  ASM_KVLEX      selection + overview (deployed)    ama/configs/cfg_kvmem_lex.json

    python analysis/assembly_2x2.py            # reads results/mu_merged_ASM_*.json
"""
import json
import math
import os

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
TAGS = {"recent": "ASM_RECENT", "gist": "ASM_GIST", "retrieval": "ASM_RETRIEVAL", "kvlex": "ASM_KVLEX"}


def rows(tag):
    return json.load(open(os.path.join(RES, "mu_merged_%s.json" % tag)))


def load(tag):
    """(episode_id, question) -> 0/1; the 20 repeated questions collapse (as in the paper's pairing)."""
    m, dom = {}, {}
    for r in rows(tag):
        k = (r["episode_id"], r["question"])
        m[k] = 1 if float(r["score"]) >= 0.5 else 0
        dom[k] = r["domain"]
    return m, dom


ARMS = {a: load(t)[0] for a, t in TAGS.items()}
DOM = load(TAGS["recent"])[1]

print("Table 'assembly' (all rows):  arm       overall   SOFTWARE")
for a, t in TAGS.items():
    rr = rows(t)
    sw = [r for r in rr if r["domain"] == "SOFTWARE"]
    print("  %-10s n=%d  %.4f   %.4f" % (a, len(rr), sum(r["score"] == 1.0 for r in rr) / len(rr),
                                       sum(r["score"] == 1.0 for r in sw) / len(sw)))

print("\nUnique (episode, question) keys used for pairing: n=%d" % len(ARMS["recent"]))
for a, m in ARMS.items():
    print("  %-10s %.4f" % (a, sum(m.values()) / len(m)))


def paired(a, b):
    """delta = acc(b) - acc(a) on common keys; McNemar (normal approximation)."""
    keys = set(ARMS[a]) & set(ARMS[b])
    n = len(keys)
    ma, mb = ARMS[a], ARMS[b]
    b_wins = sum(1 for k in keys if ma[k] == 0 and mb[k] == 1)
    a_wins = sum(1 for k in keys if ma[k] == 1 and mb[k] == 0)
    disc = b_wins + a_wins
    delta = (b_wins - a_wins) / n
    se = math.sqrt(disc) / n if disc else 0.0
    z = (b_wins - a_wins) / math.sqrt(disc) if disc else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return n, delta, delta - 1.96 * se, delta + 1.96 * se, b_wins, a_wins, p


print("\nPaired contrasts  (delta = acc(second) - acc(first)):")
for a, b in [("recent", "gist"), ("recent", "retrieval"), ("gist", "retrieval"),
             ("retrieval", "kvlex"), ("gist", "kvlex"), ("recent", "kvlex")]:
    n, delta, lo, hi, bw, aw, p = paired(a, b)
    print("  %9s - %-9s  d=%+.2fpp  CI[%+.2f,%+.2f]  discord %s+=%d/%s+=%d  p=%.3f%s"
          % (b, a, delta * 100, lo * 100, hi * 100, b, bw, a, aw, p, "*" if p < 0.05 else " "))

print("\nPer-domain accuracy:")
print("  %-14s %7s %7s %10s %7s" % ("domain", "recent", "gist", "retrieval", "kvlex"))
for d in sorted(set(DOM.values())):
    keys = [k for k in ARMS["recent"] if DOM[k] == d]
    print("  %-14s " % d + " ".join("%7.3f" % (sum(ARMS[a][k] for k in keys) / len(keys))
                                   for a in ("recent", "gist", "retrieval", "kvlex")))

print("\nPer-domain delta (retrieval - gist)  [selection vs overview]:")
for d in sorted(set(DOM.values())):
    keys = [k for k in ARMS["gist"] if DOM[k] == d]
    dg = sum(ARMS["retrieval"][k] - ARMS["gist"][k] for k in keys) / len(keys)
    print("  %-14s d=%+.2fpp  (n=%d, gist=%.3f -> retr=%.3f)"
          % (d, dg * 100, len(keys), sum(ARMS["gist"][k] for k in keys) / len(keys),
             sum(ARMS["retrieval"][k] for k in keys) / len(keys)))
