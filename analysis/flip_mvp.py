"""RAMP MVP verdict: packet-vs-raw on the prompt-flip set.

Joins mu_merged_PKTD.json (packet appendix + DEFAULT reader) and mu_merged_RFLIP.json
(raw appendix + DEFAULT reader, same batch) against the flip-set membership:
  F+ = default-wrong & structured-right on the original full runs (the questions the
       structured prompt "fixes"); S = 300-question both-right stability sample.
Signal = recovery on F+ net of the raw arm's spontaneous (regression-to-mean) recovery,
plus breakage on S. Paired per-question, McNemar counts included.

    python analysis/flip_mvp.py results/mu_merged_PKTD.json results/mu_merged_RFLIP.json /tmp/flipsets.json
"""
import json, sys, collections

def load(f):
    out = {}
    for r in json.load(open(f)):
        out[(str(r["episode_id"]), " ".join(r["question"].split()))] = r
    return out

def acc(rows, keys):
    ks = [k for k in keys if k in rows]
    return sum(rows[k]["score"] >= 0.5 for k in ks), len(ks)

def main(pkt_f, raw_f, flips_f):
    P, R = load(pkt_f), load(raw_f)
    fs = json.load(open(flips_f))
    Fp = [tuple(k) for k in fs["Fplus"]]
    S = [tuple(k) for k in fs["both_right_sample"]]
    for name, keys in (("F+ (default-wrong, structured-right)", Fp), ("S (both-right stability)", S)):
        pk, pn = acc(P, keys); rk, rn = acc(R, keys)
        print(f"\n== {name}: n(pkt)={pn} n(raw)={rn}")
        print(f"  packet+default : {pk}/{pn} = {pk/max(pn,1):.3f}")
        print(f"  raw+default    : {rk}/{rn} = {rk/max(rn,1):.3f}   (spontaneous)")
        print(f"  delta          : {pk/max(pn,1)-rk/max(rn,1):+.3f}")
        both = [k for k in keys if k in P and k in R]
        mc = collections.Counter((P[k]["score"] >= 0.5, R[k]["score"] >= 0.5) for k in both)
        print(f"  paired n={len(both)}  pkt-only-right={mc[(True,False)]}  raw-only-right={mc[(False,True)]}  "
              f"both={mc[(True,True)]}  neither={mc[(False,False)]}")
        bydom = collections.defaultdict(lambda: [0, 0, 0])
        for k in both:
            d = bydom[P[k]["domain"]]
            d[0] += 1; d[1] += P[k]["score"] >= 0.5; d[2] += R[k]["score"] >= 0.5
        for dom, (n, p, r) in sorted(bydom.items()):
            print(f"    {dom:14s} n={n:3d}  pkt {p/n:.2f}  raw {r/n:.2f}  d={p/n-r/n:+.2f}")

if __name__ == "__main__":
    main(*sys.argv[1:4])
