#!/usr/bin/env python3
"""KL-probe analysis (paper Sec. 4, "Matched evidence, and the ceiling"): is the cache path's residual
a distribution shift or a marginal argmax flip? Reads the kv_klprobe output (8B mechanism subset,
n=824): median first-token KL(full || b_hot) 0.005 vs KL(full || tx) 0.031 nats.

    python analysis/kl_probe.py [results/oow/kl.jsonl.gz]
"""
import gzip
import json
import os
import statistics as st
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "oow", "kl.jsonl.gz")
rows = [json.loads(ln) for ln in gzip.open(PATH, "rt", encoding="utf-8") if ln.strip()]
print(f"n={len(rows)}  eps={len({r['episode_id'] for r in rows})}")
N = len(rows)
for a in ("b_hot", "tx", "full"):
    print(f"  {a:6s} acc {sum(r[a] for r in rows)/N:.4f}")


def med(v):
    v = [x for x in v if x is not None]
    return (st.median(v), st.quantiles(v, n=4)[0], st.quantiles(v, n=4)[2]) if v else (0, 0, 0)


def cellrows(cond):
    return [r for r in rows if cond(r)]


def stats(rs, tag):
    if not rs:
        print(f"  {tag}: empty")
        return
    dlp = [r["lp_gold_bh"] - r["lp_gold_full"] for r in rs]
    dlt = [r["lp_gold_tx"] - r["lp_gold_full"] for r in rs]
    klb = [r["kl_full_bh"] for r in rs]
    klt = [r["kl_full_tx"] for r in rs]
    eb = [r["ent_bh"] for r in rs]
    ef = [r["ent_full"] for r in rs]
    ag_b = sum(r["agree_bh"] for r in rs) / len(rs)
    ag_t = sum(r["agree_tx"] for r in rs) / len(rs)
    end = [r["lp_bhans_under_full"] for r in rs if r.get("lp_bhans_under_full") is not None]
    own = [r["lp_bhans_under_bh"] for r in rs if r.get("lp_bhans_under_bh") is not None]
    print(f"  {tag}  n={len(rs)}")
    print(f"    dLP_gold(bh-full) mean {st.mean(dlp):+.3f} med {st.median(dlp):+.3f} | "
          f"(tx-full) mean {st.mean(dlt):+.3f} med {st.median(dlt):+.3f}")
    print(f"    KL(full||bh) med {med(klb)[0]:.3f} [q1 {med(klb)[1]:.3f} q3 {med(klb)[2]:.3f}] | "
          f"KL(full||tx) med {med(klt)[0]:.3f} [q1 {med(klt)[1]:.3f} q3 {med(klt)[2]:.3f}]")
    print(f"    ent: full med {st.median(ef):.3f}  bh med {st.median(eb):.3f} | "
          f"argmax-agree bh {ag_b:.2f} tx {ag_t:.2f}")
    if end:
        print(f"    bh's own answer: lp under bh {st.mean(own):+.3f}  under full "
              f"{st.mean(end):+.3f}  (endorsement gap {st.mean(own)-st.mean(end):+.3f})")


print("\n== overall ==")
stats(rows, "ALL")
print("\n== outcome cells (tx vs b_hot, in-run 8B judge) ==")
stats(cellrows(lambda r: r["tx"] == 1 and r["b_hot"] == 0), "tx WIN / bh LOSS (residual)")
stats(cellrows(lambda r: r["tx"] == 1 and r["b_hot"] == 1), "both RIGHT")
stats(cellrows(lambda r: r["tx"] == 0 and r["b_hot"] == 1), "bh WIN / tx LOSS")
stats(cellrows(lambda r: r["tx"] == 0 and r["b_hot"] == 0), "both WRONG")
print("\n== cells vs full ==")
stats(cellrows(lambda r: r["full"] == 1 and r["b_hot"] == 0), "full WIN / bh LOSS")
stats(cellrows(lambda r: r["full"] == 1 and r["b_hot"] == 1), "full+bh both RIGHT")

print("\n== distribution-shift verdict inputs ==")
kl_loss = [r["kl_full_bh"] for r in rows if r["tx"] == 1 and r["b_hot"] == 0]
kl_win = [r["kl_full_bh"] for r in rows if r["b_hot"] == 1]
print(f"  KL(full||bh) median: loss-cell {st.median(kl_loss):.3f} vs bh-right {st.median(kl_win):.3f}")
klb_all = [r["kl_full_bh"] for r in rows]
klt_all = [r["kl_full_tx"] for r in rows]
paired = sum(1 for b, t in zip(klb_all, klt_all) if b < t)
print(f"  QAs where KL(full||bh) < KL(full||tx): {paired}/{N} ({paired/N*100:.1f}%)")
big_b = sum(1 for x in klb_all if x > 5)
big_t = sum(1 for x in klt_all if x > 5)
print(f"  KL>5 nats (large shift): bh {big_b} ({big_b/N*100:.1f}%)  tx {big_t} ({big_t/N*100:.1f}%)")
