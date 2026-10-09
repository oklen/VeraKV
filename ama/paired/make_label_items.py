"""Build the labelling items for the paired AMA re-run (docs/AMA_AGENT_PAIRED.md):
  - every question where V1 and A1 disagree (one judged correct, the other wrong)
  - state-update and WEB questions where A2 and A1 disagree
Each item carries both sides' exact reader prompts (what the answering model saw).

    python make_label_items.py <run dir with s*/q.jsonl[.gz]> <items.jsonl>
"""
import glob
import gzip
import json
import sys

run, dst = sys.argv[1], sys.argv[2]
rows = {}
for f in sorted(glob.glob(run + "/s*/q.jsonl")) or sorted(glob.glob(run + "/s*/q.jsonl.gz")):
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt", encoding="utf-8") as fh:
        for l in fh:
            try:
                r = json.loads(l)
            except ValueError:
                continue
            k = (r["arm"], r["ep"], r["qi"])
            if k not in rows or (rows[k].get("infra") and not r.get("infra")):
                rows[k] = r


def ok(r):
    return r is not None and not r.get("infra") and r.get("score") is not None


def side(r):
    return {"arm": r["arm"], "answer": r.get("pred") or "", "prompt": r.get("answer_prompt") or "",
            "answer_stage": r.get("answer_stage"), "direct": bool((r.get("ama") or {}).get("direct")),
            "ctx_tokens": r.get("ctx_tokens"), "prompt_tokens": r.get("answer_prompt_tokens")}


items = []
keys = sorted({(e, q) for (_, e, q) in rows})
for e, q in keys:
    for pair, a, b, sel in (("V1_vs_A1", "V1", "A1", lambda r: True),
                            ("A2_vs_A1", "A2", "A1", lambda r: r["qtype"] == "C" or r["domain"] == "WEB")):
        ra, rb = rows.get((a, e, q)), rows.get((b, e, q))
        if not (ok(ra) and ok(rb)) or not sel(ra):
            continue
        ca, cb = ra["score"] == 1.0, rb["score"] == 1.0
        if ca == cb:
            continue
        right, wrong = (ra, rb) if ca else (rb, ra)
        items.append({"pair": pair, "ep": e, "qi": q, "domain": ra["domain"], "qtype": ra["qtype"],
                      "question": ra["question"], "gold": ra["gold"],
                      "right": side(right), "wrong": side(wrong)})
with open(dst, "w", encoding="utf-8") as fh:
    for it in items:
        fh.write(json.dumps(it, ensure_ascii=False) + "\n")
print("items", len(items), {p: sum(1 for i in items if i["pair"] == p) for p in ("V1_vs_A1", "A2_vs_A1")})
