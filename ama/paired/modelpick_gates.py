"""Gates of PREREG_modelpick_fix_20261009.md, checked before reading any difference between arms.

    python ama/paired/modelpick_gates.py <run dir with def/ and str/ subfolders of s*/q.jsonl[.gz]> [--full]
    MP_DEC_INSTR  the structured instruction file (default ama/dec_instr.txt)

G1  V1f: every model-pick call carried enable_thinking=false, no reply contains a think block, and >= 80% of
    the questions that made a pick call got a reply with at least one number.
G2  V1: the model-pick replies are still unfinished <think> blocks (the bug is reproduced, not fixed by accident).
G3  str: the reader prompt holds the structured instruction; def: it holds the harness default instruction.
G4  infra failures <= 2% per arm and reader.
G5  (--full) V1 and V2 calls never carry extra_body; V2 makes no pick call.
Prints the counts behind each gate and PASS/FAIL; exits 1 if any gate fails.
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

root = sys.argv[1]
DEFAULT_INSTR = "Provide a direct and concise answer."
STRUCT_MARK = None
for _p in (os.environ.get("MP_DEC_INSTR", ""), "ama/dec_instr.txt"):
    try:
        STRUCT_MARK = open(_p, encoding="utf-8").read().strip()[:60]
        break
    except OSError:
        pass

rows = defaultdict(dict)   # reader -> {(arm, ep, qi): rec}
for rd in ("def", "str"):
    for f in sorted(glob.glob("%s/%s/s*/q.jsonl" % (root, rd))) or sorted(glob.glob("%s/%s/s*/q.jsonl.gz" % (root, rd))):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows[rd][(r["arm"], r["ep"], r["qi"])] = r

ok = True
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
for rd in ("def", "str"):
    R = rows[rd]
    print("== reader %s: %d records %s" % (rd, len(R), dict(Counter(a for a, _, _ in R))))
    # G1
    c = Counter()
    for (a, e, q), r in R.items():
        if a != "V1f":
            continue
        picks = r.get("pick") or []
        c["questions"] += 1
        if not picks:
            continue
        c["with_pick"] += 1
        c["calls"] += len(picks)
        c["extra_ok"] += sum(1 for p in picks if p[2] == NOTHINK)
        c["think_in_reply"] += sum(1 for p in picks if "<think>" in (p[0] or ""))
        c["has_number"] += 1 if any(re.search(r"\d", p[0] or "") for p in picks) else 0
        c["finish_" + str(picks[-1][1])] += 1
    g1 = c["calls"] > 0 and c["extra_ok"] == c["calls"] and c["think_in_reply"] == 0 and \
        c["has_number"] >= 0.8 * max(1, c["with_pick"])
    print("  G1 V1f picks %s -> %s" % (dict(c), "PASS" if g1 else "FAIL"))
    # G2
    c = Counter()
    for (a, e, q), r in R.items():
        if a != "V1":
            continue
        for p in r.get("pick") or []:
            c["calls"] += 1
            txt = p[0] or ""
            c["open_think"] += 1 if txt.lstrip().startswith("<think>") and "</think>" not in txt else 0
            c["extra_none"] += 1 if p[2] is None else 0
    g2 = c["calls"] > 0 and c["open_think"] >= 0.95 * c["calls"] and c["extra_none"] == c["calls"]
    print("  G2 V1 picks %s -> %s" % (dict(c), "PASS" if g2 else "FAIL"))
    # G3
    c = Counter()
    for (a, e, q), r in R.items():
        ap = r.get("answer_prompt") or ""
        if not ap:
            continue
        c["prompts"] += 1
        c["default_instr"] += 1 if DEFAULT_INSTR in ap else 0
        c["struct_instr"] += 1 if (STRUCT_MARK and STRUCT_MARK in ap) else 0
        c["instr_" + str(r.get("reader_instr"))] += 1
    if rd == "def":
        g3 = c["prompts"] > 0 and c["default_instr"] == c["prompts"] and c["struct_instr"] == 0
    else:
        g3 = c["prompts"] > 0 and STRUCT_MARK is not None and c["struct_instr"] == c["prompts"]
    print("  G3 reader prompts %s -> %s" % (dict(c), "PASS" if g3 else "FAIL"))
    # G4
    g4 = True
    for arm in ("V1", "V1f", "V2"):
        rs = [r for (a, _, _), r in R.items() if a == arm]
        bad = sum(1 for r in rs if r.get("infra"))
        share = bad / max(1, len(rs))
        g4 &= len(rs) > 0 and share <= 0.02
        print("  G4 %s n=%d infra=%d (%.1f%%)" % (arm, len(rs), bad, 100 * share))
    print("  G4 -> %s" % ("PASS" if g4 else "FAIL"))
    ok &= g1 and g2 and g3 and g4
    if "--full" in sys.argv:
        c = Counter()
        for (a, e, q), r in R.items():
            for p in r.get("pick") or []:
                if a in ("V1", "V2") and p[2] is not None:
                    c[a + "_extra"] += 1
            if a == "V2" and r.get("pick"):
                c["V2_pick"] += 1
        g5 = not c
        print("  G5 %s -> %s" % (dict(c), "PASS" if g5 else "FAIL"))
        ok &= g5
print("ALL GATES PASS" if ok else "GATE FAILURE")
sys.exit(0 if ok else 1)
