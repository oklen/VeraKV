"""Compare kvmemory router variants (lexical / hybrid / model) on the AMA-Bench 40-ep subset, and
do the head-to-head vs ama_agent — testing the two case-study findings (FINDINGS §10.1):
  * does the RRF HybridRouter lift the headline over the free LexicalRouter? (router-is-the-headroom)
  * does it claw back the B-Causal gap where kvmemory trailed ama_agent's causal graph?

Loads the per-QA judge results, restricts to the shared key set, and reports overall + per-qtype
accuracy for every method plus paired win/loss (hybrid vs lexical = the improvement; hybrid vs ama =
the head-to-head). Pure stdlib, no GPU.

  python kvmemory/router_compare.py --lexical result_kvmem_lexical.json \
      --hybrid result_kvmem_hybrid.json --model result_kvmem_model.json --ama <ama_result>.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def load(path):
    d = json.load(open(path))
    return {(r["episode_id"], r["question"]): r for r in d["results"]}


def acc_by_qtype(res, keys):
    tot, hit = defaultdict(int), defaultdict(int)
    for k in keys:
        r = res[k]
        qt = r.get("qa_type", "?")
        tot[qt] += 1; tot["ALL"] += 1
        if r["score"] >= 1:
            hit[qt] += 1; hit["ALL"] += 1
    return {qt: (hit[qt], tot[qt]) for qt in tot}


def paired(a, b, keys):
    cell = [0, 0, 0, 0]                       # a✓b✗ / b✓a✗ / both✓ / both✗
    byqt = defaultdict(lambda: [0, 0, 0, 0])
    for k in keys:
        x, y = a[k]["score"] >= 1, b[k]["score"] >= 1
        idx = 0 if (x and not y) else 1 if (y and not x) else 2 if x else 3
        cell[idx] += 1
        byqt[a[k].get("qa_type", "?")][idx] += 1
    return cell, byqt


def main():
    ap = argparse.ArgumentParser()
    for fl in ["lexical", "hybrid", "model", "embed", "ama"]:
        ap.add_argument("--" + fl)
    a = ap.parse_args()
    R = {}
    for name in ["lexical", "hybrid", "model", "embed", "ama"]:
        p = getattr(a, name)
        if p:
            R[name] = load(p)
    if not R:
        ap.error("pass at least one result file")

    # NB: --samples picks a RANDOM (unseeded) subset per run, so each method covers a different 40-ep
    # sample. Report each method's own-sample accuracy (directional), then PAIRED comparisons on the QA
    # each pair happens to SHARE (same episode+question, different router → a clean within-pair test).
    print("=== own-sample accuracy (each method on its own ~480 QA) ===")
    allq = set()
    for name in R:
        allq |= {R[name][k].get("qa_type", "?") for k in R[name]}
    qts = sorted(allq | {"ALL"})
    print("%-10s%6s  " % ("method", "n") + "".join("%9s" % q for q in qts))
    for name in R:
        ks = set(R[name]); ab = acc_by_qtype(R[name], ks)
        print("%-10s%6d  " % (name, len(ks)) + "".join(
            "%9s" % ("%.3f" % (ab[q][0] / ab[q][1]) if q in ab and ab[q][1] else "-") for q in qts))

    for base in ["lexical", "embed", "model", "ama"]:
        if "hybrid" in R and base in R:
            shared = set(R["hybrid"]) & set(R[base])
            if not shared:
                continue
            ha = sum(R["hybrid"][k]["score"] >= 1 for k in shared) / len(shared)
            ba = sum(R[base][k]["score"] >= 1 for k in shared) / len(shared)
            cell, byqt = paired(R["hybrid"], R[base], shared)
            print("\n=== hybrid vs %s — PAIRED on %d shared QA   [hyb✓%s✗ / %s✓hyb✗ / both✓ / both✗] ==="
                  % (base, len(shared), base, base))
            print("  acc on shared:  hybrid %.3f  vs  %s %.3f   (Δ%+.3f)" % (ha, base, ba, ha - ba))
            print("  overall cells  %s   net=%+d" % (cell, cell[0] - cell[1]))
            for qt in sorted(byqt):
                v = byqt[qt]
                print("  %-12s %s   net=%+d" % (qt, v, v[0] - v[1]))


if __name__ == "__main__":
    main()
