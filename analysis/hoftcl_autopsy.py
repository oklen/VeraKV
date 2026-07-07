"""Autopsy of HOFTCL (gate relay, lexical base, full 2496): decompose the 822 misses.

Buckets: adopted-upstream-wrong (memory-side answering error), bad override (reader
departed from a plausibly-right upstream), both-wrong-but-ever-solved (recoverable),
never-solved-by-any-arm (ceiling / gold-judge candidates).
"""
import json, re, glob, os
from collections import Counter, defaultdict

RES = "/Users/bytedance/Downloads/kvmemory-release/results"
TARGET = "HOFTCL"

def nkey(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def adopted(up, ans):
    u, a = nkey(up), nkey(ans)
    if not u or not a:
        return None
    if u in a or a in u:
        return True
    su, sa = set(u.split()), set(a.split())
    return len(su & sa) / max(1, len(su | sa)) >= 0.5

def f1(a, b):
    ta, tb = nkey(a).split(), nkey(b).split()
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    ov = sum((ca & cb).values())
    if ov == 0:
        return 0.0
    p, r = ov / len(ta), ov / len(tb)
    return 2 * p * r / (p + r)

def load(path):
    rows = json.load(open(path))
    d, dup = {}, set()
    for r in rows:
        k = (r["domain"], str(r["episode_id"]), r["question"])
        if k in d:
            dup.add(k)
        d[k] = r
    for k in dup:
        d.pop(k, None)
    return d

tgt = load(f"{RES}/mu_merged_{TARGET}.json")

# ever-solved across all other arms
others = {}
for p in sorted(glob.glob(f"{RES}/mu_merged_*.json")):
    tag = os.path.basename(p)[10:-5]
    if tag == TARGET:
        continue
    others[tag] = load(p)

seen = defaultdict(int); solved = defaultdict(int); solvers = defaultdict(list)
for tag, d in others.items():
    for k, r in d.items():
        if k in tgt:
            seen[k] += 1
            if float(r["score"]) >= 1.0:
                solved[k] += 1
                solvers[k].append(tag)

# join sel dump by question prefix
dump = [json.loads(l) for l in open(f"{RES}/sel_{TARGET}_full.jsonl")]
byq = defaultdict(list)
for k in tgt:
    byq[k[2]].append(k)
pref2q = {}
amb = 0
qtexts = list(byq.keys())
for rec in dump:
    q = rec["q"]
    hits = [qt for qt in qtexts if qt.startswith(q)] if q not in byq else [q]
    if len(hits) == 1 and len(byq[hits[0]]) == 1:
        pref2q.setdefault(byq[hits[0]][0], rec)
    else:
        amb += 1
print(f"target n={len(tgt)}  dump joined={len(pref2q)}  ambiguous/dup dropped={amb}")

anchA = others.get("RESTRLEXF", {}); anchB = others.get("RESTRF2", {})

miss = [k for k, r in tgt.items() if float(r["score"]) < 1.0]
right = [k for k, r in tgt.items() if float(r["score"]) >= 1.0]
print(f"acc={len(right)/len(tgt):.4f}  miss={len(miss)}")

def anchor_right(k):
    a = k in anchA and float(anchA[k]["score"]) >= 1.0
    b = k in anchB and float(anchB[k]["score"]) >= 1.0
    return a, b

buckets = defaultdict(list)
for k in miss:
    r = tgt[k]
    rec = pref2q.get(k)
    gold, pred = r["golden_answer"], r["predicted_answer"]
    up = rec["up_ans"] if rec else None
    ad = adopted(up, pred) if rec else None
    upf1 = f1(up, gold) if up else 0.0
    aA, aB = anchor_right(k)
    ever = solved[k] > 0
    if ad is True:
        b = "ADOPT_ever" if ever else "ADOPT_never"
    elif ad is False:
        if aA or aB or upf1 >= 0.5:
            b = "OVERRIDE_plausible"   # upstream/in-place had it; relay departed & lost
        else:
            b = "OVERRIDE_bothwrong_ever" if ever else "OVERRIDE_bothwrong_never"
    else:
        b = "nojoin"
    buckets[b].append(k)

print("\n== miss decomposition ==")
for b in sorted(buckets, key=lambda x: -len(buckets[x])):
    ks = buckets[b]
    doms = Counter(k[0] for k in ks).most_common(3)
    print(f"{b:26s} {len(ks):4d}  top-domains: {doms}")

nev = [k for k in miss if solved[k] == 0]
print(f"\nnever-solved among misses: {len(nev)} "
      f"(arm-coverage of these: {Counter(seen[k] for k in nev).most_common(5)})")
ever_m = [k for k in miss if solved[k] > 0]
print(f"ever-solved among misses (recoverable ceiling): {len(ever_m)} "
      f"→ potential acc if all recovered: {(len(right)+len(ever_m))/len(tgt):.3f}")

# adoption stats on rights for contrast
ad_r = Counter(adopted(pref2q[k]["up_ans"], tgt[k]["predicted_answer"])
               for k in right if k in pref2q)
ad_m = Counter(adopted(pref2q[k]["up_ans"], tgt[k]["predicted_answer"])
               for k in miss if k in pref2q)
print(f"\nadoption on RIGHT: {dict(ad_r)}   on MISS: {dict(ad_m)}")

print("\n== domain x bucket (miss) ==")
doms = sorted(set(k[0] for k in tgt))
bs = sorted(buckets)
print("%-14s" % "domain" + "".join("%-24s" % b[:23] for b in bs) + "miss_total")
for d in doms:
    row = "%-14s" % d[:13]
    for b in bs:
        row += "%-24d" % sum(1 for k in buckets[b] if k[0] == d)
    row += str(sum(1 for k in miss if k[0] == d))
    print(row)

print("\n== qa_type x key buckets ==")
qt_all = Counter(tgt[k].get("qa_type", "?") for k in miss)
for b in ["OVERRIDE_plausible", "ADOPT_ever", "ADOPT_never"]:
    c = Counter(tgt[k].get("qa_type", "?") for k in buckets[b])
    print(b, dict(c.most_common(8)))
print("all-miss qa_type:", dict(qt_all.most_common(8)))

# samples
import random
random.seed(1)
def show(k, rec):
    r = tgt[k]
    print("-" * 100)
    print(f"[{k[0]} ep{k[1]}] {r.get('qa_type','?')}  everSolved={solved[k]}/{seen[k]} "
          f"anchors(LEX,F2)={anchor_right(k)}  solvers={solvers[k][:6]}")
    print("Q:", k[2][:220])
    print("GOLD:", str(r['golden_answer'])[:300])
    print("UPSTREAM:", re.sub(r'\s+', ' ', rec['up_ans'] or '')[:300] if rec else "(nojoin)")
    print("FINAL:", re.sub(r'\s+', ' ', str(r['predicted_answer']))[:300])

for b in ["OVERRIDE_plausible", "ADOPT_ever", "ADOPT_never"]:
    ks = [k for k in buckets[b] if k in pref2q]
    print(f"\n#################### samples: {b} (n={len(ks)}) ####################")
    for k in random.sample(ks, min(5, len(ks))):
        show(k, pref2q.get(k))
