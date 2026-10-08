"""pfx_ttft.py -- summarize the prefix-cache fairness microbench (kvmemory/kv_pfx.py rows; paper
Sec. 3.7 fairness note: sub-selected question 40-48 ms vs 79 ms over the resident full cache).

Per k: median TTFT for tx (fresh text re-prefill) / sub (gather+question over E rows) /
pfx (question over ALL N resident rows), median decode s/tok for sub vs pfx, and the
gather-only overhead. Speedups quoted as tx/sub and pfx/sub.
"""
import json, os, statistics as st, sys

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else
                      os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "results", "kv_cost", "pfx_8b.json")))
print(f"rows: {len(rows)}  episodes: {len(set(r['episode_id'] for r in rows))} "
      f"N median {st.median(r['total'] for r in rows)}")
by = {}
for r in rows:
    by.setdefault(r["k"], []).append(r)
print(f"\n{'k':>3} {'E_tok':>6} | {'ttft_tx':>8} {'ttft_sub':>8} {'ttft_pfx':>8} | "
      f"{'tx/sub':>6} {'pfx/sub':>7} | {'dec_sub':>8} {'dec_pfx':>8} {'dec_tx':>7} | {'gather':>7}")
for k in sorted(by):
    g = by[k]
    med = lambda f: st.median(f(r) for r in g)
    print(f"{k:>3} {med(lambda r: r['etok']):>6.0f} | "
          f"{med(lambda r: r['ttft_tx'])*1e3:>7.0f}m {med(lambda r: r['ttft_sub'])*1e3:>7.0f}m "
          f"{med(lambda r: r['ttft_pfx'])*1e3:>7.0f}m | "
          f"{med(lambda r: r['ttft_tx']/r['ttft_sub']):>5.1f}x "
          f"{med(lambda r: r['ttft_pfx']/r['ttft_sub']):>6.1f}x | "
          f"{med(lambda r: r['dec_sub'])*1e3:>6.1f}ms {med(lambda r: r['dec_pfx'])*1e3:>6.1f}ms "
          f"{med(lambda r: r['dec_tx'])*1e3:>5.1f}ms | {med(lambda r: r['gather'])*1e3:>5.0f}ms")
allr = rows
print("\npooled medians: ttft sub %.0fms vs pfx %.0fms vs tx %.0fms | dec sub %.1f vs pfx %.1f ms/tok"
      % (st.median(r["ttft_sub"] for r in allr) * 1e3,
         st.median(r["ttft_pfx"] for r in allr) * 1e3,
         st.median(r["ttft_tx"] for r in allr) * 1e3,
         st.median(r["dec_sub"] for r in allr) * 1e3,
         st.median(r["dec_pfx"] for r in allr) * 1e3))
