"""Goal-experiment verdict: can answer-accountable memory-side reasoning reach the structured score?

Usage:
    python analysis/handoff_verdict.py results/mu_merged_HOFA.json [results/mu_merged_HOFS.json ...]

Compares each arm against the anchors (RESTR2 structured 0.6418, REDEF2 default) on h0, overall and
on the 88-question plan-poisoned pool (default-right & structured-right & PLDEF-wrong) — the pool
the diagnosis says answer-accountability should rescue.
"""
import json, sys, collections

def load(f):
    return {(str(r["episode_id"]), " ".join(r["question"].split())): r for r in json.load(open(f))}

def ok(r): return r["score"] >= 0.5

def main(arms):
    base = "results/"
    P = load(base + "mu_merged_PLDEF.json"); D0 = load(base + "mu_merged_REDEF.json")
    S0 = load(base + "mu_merged_RESTR.json"); S2 = load(base + "mu_merged_RESTR2.json")
    anchors = {"RESTR2(structured)": S2}
    try:
        anchors["REDEF2(default)"] = load(base + "mu_merged_REDEF2.json")
    except Exception:
        pass
    poisoned = [k for k in set(P) & set(D0) & set(S0) if ok(D0[k]) and ok(S0[k]) and not ok(P[k])]
    for f in arms:
        A = load(f); name = f.split("mu_merged_")[-1].split(".")[0]
        ks = sorted(set(A) & set(S2))
        acc = sum(ok(A[k]) for k in ks) / len(ks)
        line = f"{name:8s} overall {acc:.4f} (n={len(ks)})"
        for an, R in anchors.items():
            kk = sorted(set(A) & set(R))
            d = sum(ok(A[k]) for k in kk)/len(kk) - sum(ok(R[k]) for k in kk)/len(kk)
            line += f"  vs {an} {d:+.4f}"
        pk = [k for k in poisoned if k in A]
        line += f"  | poisoned-pool recovery {sum(ok(A[k]) for k in pk)}/{len(pk)}"
        print(line)
    # anchor sanity
    for an, R in anchors.items():
        kk = sorted(set(R) & set(S2))
        print(f"anchor {an}: {sum(ok(R[k]) for k in kk)/len(kk):.4f}")
    print(f"(plan-poisoned pool n={len(poisoned)}; PLDEF got 0 on all by construction)")

if __name__ == "__main__":
    main(sys.argv[1:])
