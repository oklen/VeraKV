"""Explicit-address coverage in AMA-Bench questions, per domain (reviewer #8: is structural
addressing a step-index benchmark artifact, or a general property of agent-trajectory queries?)."""
import json, re, collections

PATS = {
    "step_ref":   re.compile(r"\b(?:step|turn)s?\s+\d+", re.I),
    "file_path":  re.compile(r"(?:[\w.-]+/){1,}[\w.-]+|\b[\w-]+\.(?:py|js|ts|java|cpp|json|yaml|yml|txt|md|csv|sh|html|css|sql|log|cfg|xml)\b"),
    "url":        re.compile(r"https?://|www\.", re.I),
    "exc_name":   re.compile(r"\b[A-Z][a-zA-Z]*(?:Error|Exception|Warning)\b"),
    "quoted_lit": re.compile(r"['\"`][^'\"`]{3,60}['\"`]"),
    "identifier": re.compile(r"\b[a-z]+_[a-z_]+\b|\b[a-z]+[A-Z][a-zA-Z]+\b"),  # snake/camel identifiers
}

rows = [json.loads(l) for l in open("/home/tiger/AMA-Bench/dataset/test/open_end_qa_set.jsonl", encoding="utf-8") if l.strip()]
dom_of = lambda eid: str(eid).split("_")[0].upper() if not str(eid)[0].isdigit() else "?"
# episode_id may not carry domain; fall back to a 'domain'/'source' field
per = collections.defaultdict(lambda: collections.defaultdict(int))
tot = collections.Counter()
sample = {}
for ep in rows:
    dom = (ep.get("domain") or ep.get("source") or ep.get("task_type") or dom_of(ep.get("episode_id",""))).upper()
    for qa in ep.get("qa_pairs", []):
        q = qa.get("question","")
        tot[dom] += 1
        hit_any = False
        for name, pat in PATS.items():
            if pat.search(q):
                per[dom][name] += 1
                hit_any = True
        if hit_any:
            per[dom]["ANY"] += 1
        elif dom not in sample:
            sample[dom] = q[:110]
print("domain      n     ANY  step  path   url   exc  qlit  ident")
for dom in sorted(tot):
    n = tot[dom]; p = per[dom]
    print("%-10s %4d  %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f" % (dom, n,
        100*p["ANY"]/n, 100*p["step_ref"]/n, 100*p["file_path"]/n, 100*p["url"]/n,
        100*p["exc_name"]/n, 100*p["quoted_lit"]/n, 100*p["identifier"]/n))
n_all = sum(tot.values())
print("%-10s %4d  %5.1f" % ("ALL", n_all, 100*sum(p["ANY"] for p in per.values())/n_all))
for d,s in sample.items(): print("no-addr sample", d, ":", s)
