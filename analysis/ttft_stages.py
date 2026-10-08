"""Stage-resolved query cost of the serving paths (kvmemory/kv_ttft.py rows, 12 within-window episodes
x 4 QA, hot=2, K=5; paper Sec. 4 "Failure, repair, and the budget": in-window TTFT 1.4-2.3x faster than
text re-prefill, decode ~29% slower under the scattered layout, store writes ~5x one episode prefill).

TTFT per arm = the stages before the first decoded token:
  tx     prefill(view) + ingest(question)
  glob   gather(from a per-episode full prefill) + ingest     [full prefill itself amortized separately]
  b_anch assemble(cached blocks) + ingest
  b_hot  assemble + fresh(hot tail) + ingest
Rows are split into terciles by episode size (total tokens); means per tercile. Each arm then decodes
the same fixed budget of --dec_tokens (32) tokens; the per-token decode penalty below assumes all 32
were decoded.

    python analysis/ttft_stages.py [results/kv_cost/ttft_8b.jsonl results/kv_cost/ttft_32b.jsonl]
"""
import json
import os
import statistics as st
import sys

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "kv_cost")
paths = sys.argv[1:] or [os.path.join(RES, "ttft_8b.jsonl"), os.path.join(RES, "ttft_32b.jsonl")]
NDEC = 32


def ttft(r, arm):
    return {"tx": r["tx_prefill"] + r["tx_ingest"],
            "glob": r["glob_gather"] + r["glob_ingest"],
            "b_anch": r["banch_assemble"] + r["banch_ingest"],
            "b_hot": r["bhot_assemble"] + r["bhot_fresh"] + r["bhot_ingest"]}[arm]


for p in paths:
    rows = sorted((json.loads(l) for l in open(p) if l.strip()), key=lambda r: r["total_tok"])
    k = len(rows) // 3
    bands = [("small", rows[:k]), ("mid", rows[k:2 * k]), ("large", rows[2 * k:])]
    print("== %s  (n=%d QA, %d episodes)" % (os.path.basename(p), len(rows), len({r["episode_id"] for r in rows})))
    print("  %-6s %8s | %7s %7s %7s %7s | %s" % ("band", "ep_tok", "tx", "glob", "b_anch", "b_hot", "tx/b_hot"))
    for name, rs in bands:
        m = {a: st.mean(ttft(r, a) for r in rs) for a in ("tx", "glob", "b_anch", "b_hot")}
        print("  %-6s %8.0f | %6.3fs %6.3fs %6.3fs %6.3fs | %.1fx"
              % (name, st.mean(r["total_tok"] for r in rs), m["tx"], m["glob"], m["b_anch"], m["b_hot"],
                 m["tx"] / m["b_hot"]))
    dtx = st.mean(r["tx_decode"] for r in rows)
    dbh = st.mean(r["bhot_decode"] for r in rows)
    pen = (dbh - dtx) / NDEC
    large = bands[-1][1]
    save = st.mean(ttft(r, "tx") for r in large) - st.mean(ttft(r, "b_hot") for r in large)
    print("  decode %d tok: tx %.2fs  b_hot %.2fs  (b_hot %+.0f%%, %+.1f ms/token)"
          % (NDEC, dtx, dbh, 100 * (dbh / dtx - 1), 1e3 * pen))
    if pen > 0:
        print("  large-band TTFT saving %.2fs / decode penalty %.1f ms/token -> end-to-end break-even ~%.0f answer tokens"
              % (save, 1e3 * pen, save / pen))
    print("  store write / one full prefill (wall time): %.1fx"
          % st.mean(r["t_write"] / r["t_fullprefill"] for r in rows if r.get("t_fullprefill")))
