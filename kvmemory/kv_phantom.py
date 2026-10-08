"""kv_phantom.py -- three judge-free causal diagnostics of the "phantom context" in
event KV (user-designed, 2026-07-16). All synthetic, exact-scored, Qwen3-8B.

E1 SWAP (the core): two equal-length prefixes P_A/P_B carrying OPPOSITE states; the
    SAME event text E teacher-forced after each. Serve only [H; E] rows (+question).
    If the answer follows the invisible donor prefix, the event KV demonstrably
    carries conclusions about text the reader never sees.
E2 ERASE: an early event writes a state value; suffix events (computed with it in
    context) are served after DELETING the state event's rows. Compare with a fresh
    no-state oracle. Conflict variant: old value 30, later update 60, erase the old
    event --- does the survivor win, or does the deleted value leak back?
E3 SINK-vs-DEPENDENCY 2x2: independently encoded events served with one shared head
    vs no head, on single-retrieval vs multi-hop questions. If the head fixes
    retrieval but not multi-hop, sink stabilization and missing cross-event
    conditioning are separated mechanisms.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_phantom --exp all --out ./out/ph.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble

HDR = ("You are reviewing a completed agent trajectory. Use it to answer the "
       "question precisely.\n\nTask: operate the deployment tools and report system "
       "state faithfully.\n\nTrajectory:\n")

PAIRS = [
    ("server", "status", "ONLINE", "OFFLINE"),
    ("database", "state", "LOCKED", "OPEN"),
    ("firewall", "mode", "ENABLED", "DISABLED"),
    ("healthcheck", "result", "PASSED", "FAILED"),
    ("api_gateway", "state", "UP", "DOWN"),
    ("license", "check", "VALID", "EXPIRED"),
    ("backup_job", "status", "COMPLETE", "ABORTED"),
    ("build", "verdict", "GREEN", "RED"),
    ("sandbox", "flag", "ACTIVE", "SUSPENDED"),
    ("replica", "sync", "CURRENT", "STALE"),
    ("quota", "check", "GRANTED", "DENIED"),
    ("cluster", "reach", "HEALTHY", "DEGRADED"),
    ("cache", "state", "WARM", "COLD"),
    ("tunnel", "link", "OPEN", "SHUT"),
    ("node", "health", "READY", "FAULT"),
    ("token", "check", "FRESH", "STALE"),
    ("disk", "state", "MOUNTED", "DETACHED"),
    ("session", "state", "AUTHED", "GUESTED"),
    ("pipeline", "verdict", "PASS", "FAIL"),
    ("cert", "check", "TRUSTED", "REVOKED"),
    ("route", "state", "PAIRED", "ORPHAN"),
    ("daemon", "status", "LIVE", "DEAD"),
    ("index", "state", "SYNCED", "FAILED"),
    ("lease", "check", "HELD", "LOST"),
    ("channel", "mode", "SECURE", "EXPOSED"),
    ("volume", "state", "ATTACHED", "RELEASED"),
]

# step-2 filler notes (rep-varied); the phantom prefix differs only in the invisible
# donor value, so filler length cancels in the equal-length check.
FILLERS = [
    "routine sweep of the {obj} subsystem finished",
    "periodic audit of the {obj} module completed without incident",
    "the on-call engineer acknowledged the {obj} page and moved on",
    "a scheduled healthcheck touched the {obj} path and returned",
    "the {obj} watchdog logged a heartbeat and reset its timer",
    "maintenance window for the {obj} component opened briefly",
]


def ids_of(llm, text):
    return list(llm.tok(text, add_special_tokens=False).input_ids)


def serve_and_answer(llm, blocks, qtext, tail, ntok=16):
    c, p = assemble(llm, blocks)
    qids = llm._ids(qtext + tail)
    out, _, _ = llm._greedy_pos(c, p, qids, ntok)
    del c
    return out


# ---------------- E1: donor-prefix swap ----------------

def exp_swap(llm, head, tail, shard=0, nshards=1, reps=1, seed_base=0):
    head_full = head + HDR
    H_ids = ids_of(llm, head_full)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    mine = [t for i, t in enumerate(PAIRS) if i % nshards == shard]
    rows = []
    for rep in range(reps):
        filler = FILLERS[(rep + shard) % len(FILLERS)]
        for obj, field, vA, vB in mine:
            note = filler.format(obj=obj)
            for flip in (0, 1):
                valA, valB = (vA, vB) if flip == 0 else (vB, vA)
                def prefix(v):
                    return (f"<step 1>\naction: check_{obj}()\nobservation: {obj} {field} "
                            f"= {v}\n<step 2>\naction: log_note('{note}')\n"
                            f"observation: ok\n")
                pA, pB = ids_of(llm, prefix(valA)), ids_of(llm, prefix(valB))
                if len(pA) != len(pB):
                    continue
                P = len(pA)
                E_txt = (f"<step 3>\naction: record_to_report()\nobservation: the {obj} "
                         f"{field} from the step 1 check was copied into the incident "
                         f"report for the auditor.\n")
                E_ids = ids_of(llm, E_txt)
                est, een = H + P, H + P + len(E_ids)
                q = (f"\n\nQuestion: According to the step 1 check, what was the {obj} "
                     f"{field}? Answer with the single status word only:")
                arms = {}
                for donor, dv in (("A", valA), ("B", valB)):
                    pref = pA if donor == "A" else pB
                    E_kv = encode_block(llm, H_ids + pref + E_ids,
                                        list(range(H + P)) + list(range(est, een)),
                                        keep_a=H + P)
                    out = serve_and_answer(llm, [(H_kv, list(range(H))),
                                                 (E_kv, list(range(est, een)))], q, tail)
                    arms[donor] = {"donor_val": dv, "out": out.strip(),
                                   "follows_donor": int(dv.lower() in out.lower()
                                                        and (valB if donor == "A" else valA)
                                                        .lower() not in out.lower())}
                iso_kv = encode_block(llm, H_ids + E_ids,
                                      list(range(H)) + list(range(est, een)), keep_a=H)
                out_iso = serve_and_answer(llm, [(H_kv, list(range(H))),
                                                 (iso_kv, list(range(est, een)))], q, tail)
                rows.append({"obj": obj, "flip": flip, "rep": rep,
                             "A": arms["A"], "B": arms["B"],
                             "iso_out": out_iso.strip(),
                             "iso_names_any": int(valA.lower() in out_iso.lower()
                                                  or valB.lower() in out_iso.lower())})
                print(f"[swap] {obj} flip{flip} rep{rep} A->{arms['A']['out'][:16]!r} "
                      f"B->{arms['B']['out'][:16]!r} iso->{out_iso.strip()[:16]!r}",
                      flush=True)
    fa = sum(r["A"]["follows_donor"] for r in rows)
    fb = sum(r["B"]["follows_donor"] for r in rows)
    return {"n_pairs": len(rows), "follow_A": fa, "follow_B": fb,
            "follow_rate": (fa + fb) / max(1, 2 * len(rows)),
            "iso_names_any_rate": sum(r["iso_names_any"] for r in rows) / max(1, len(rows)),
            "rows": rows}


# ---------------- E2: erase / contaminated suffix ----------------

def exp_erase(llm, head, tail, n_items=24, seed=1):
    rng = random.Random(seed)
    head_full = head + HDR
    H_ids = ids_of(llm, head_full)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rows = []
    for it in range(n_items):
        old = rng.choice([15, 20, 30, 45, 90, 120])
        new = old * 2
        conflict = it % 2 == 1
        evs = [f"<step 1>\naction: set_config('timeout', {old})\nobservation: timeout "
               f"configured to {old} seconds\n",
               "<step 2>\naction: restart_worker()\nobservation: worker restarted "
               "cleanly\n",
               "<step 3>\naction: verify_config()\nobservation: the configured timeout "
               "value was checked and accepted by the validator\n",
               "<step 4>\naction: run_smoke()\nobservation: smoke suite finished, "
               "3 cases, all green\n"]
        if conflict:
            evs.append(f"<step 5>\naction: set_config('timeout', {new})\nobservation: "
                       f"timeout reconfigured to {new} seconds\n")
        ids_all = H_ids[:]
        spans = []
        cur = H
        for t in evs:
            e = ids_of(llm, t)
            spans.append((cur, cur + len(e)))
            ids_all += e
            cur += len(e)
        total = cur
        # one joint prefill; serve all rows except step-1's (the erased state event)
        full_kv = encode_block(llm, ids_all, list(range(total)), keep_a=H)
        keep_rows = [p for p in range(H, total) if not (spans[0][0] <= p < spans[0][1])]
        rt = torch.tensor([p - H for p in keep_rows], dtype=torch.long)
        sub = [(K.index_select(2, rt), V.index_select(2, rt)) for K, V in full_kv]
        q = ("\n\nQuestion: What is the configured timeout value in seconds, based "
             "only on the trajectory above? If it is not specified, answer "
             "'not specified'. Answer:")
        out_er = serve_and_answer(llm, [(H_kv, list(range(H))), (sub, keep_rows)],
                                  q, tail, ntok=24)
        # oracle: fresh text without step 1
        oracle_txt = head_full + "".join(evs[1:])
        oids = llm._ids(oracle_txt + q + tail)
        out_or, _, _ = llm._greedy(DynamicCache(), 0, oids, 24)
        rows.append({"old": old, "new": new, "conflict": conflict,
                     "erased_out": out_er.strip()[:60], "oracle_out": out_or.strip()[:60],
                     "erased_leaks_old": int(str(old) in out_er),
                     "erased_has_new": int(conflict and str(new) in out_er),
                     "oracle_leaks_old": int(str(old) in out_or)})
        print(f"[erase] old={old} conflict={conflict} erased={out_er.strip()[:28]!r} "
              f"oracle={out_or.strip()[:28]!r}", flush=True)
    nc = [r for r in rows if not r["conflict"]]
    cf = [r for r in rows if r["conflict"]]
    return {"n": len(rows),
            "noconflict_leak_rate": sum(r["erased_leaks_old"] for r in nc) / max(1, len(nc)),
            "noconflict_oracle_leak": sum(r["oracle_leaks_old"] for r in nc) / max(1, len(nc)),
            "conflict_leak_old": sum(r["erased_leaks_old"] for r in cf) / max(1, len(cf)),
            "conflict_has_new": sum(r["erased_has_new"] for r in cf) / max(1, len(cf)),
            "rows": rows}


# ---------------- E3: sink vs dependency 2x2 ----------------

def exp_sink(llm, head, tail, n_items=20, seed=2):
    rng = random.Random(seed)
    head_full = head + HDR
    H_ids = ids_of(llm, head_full)
    H = len(H_ids)
    H_kv = encode_block(llm, H_ids, list(range(H)))
    rows = []
    for it in range(n_items):
        nev = 12
        sensors = [f"{rng.choice(['temp','psi','volt','flow','rpm','lux'])}_{i}"
                   for i in range(nev)]
        vals = [rng.randrange(100, 999) for _ in range(nev)]
        # hop chain over three events: VAR_a = <val>, VAR_b = VAR_a, VAR_c = VAR_b
        ch = sorted(rng.sample(range(nev), 3))
        names = ["ALPHA", "BETA", "GAMMA"]
        ev_txt = []
        for i in range(nev):
            if i == ch[0]:
                body = f"register {names[0]} initialized to {vals[i]}"
            elif i == ch[1]:
                body = f"register {names[1]} copied from register {names[0]}"
            elif i == ch[2]:
                body = f"register {names[2]} copied from register {names[1]}"
            else:
                body = f"sensor {sensors[i]} measured {vals[i]}"
            ev_txt.append(f"<step {i+1}>\naction: probe()\nobservation: {body}\n")
        # positions from a virtual contiguous layout
        spans = []
        cur = H
        eids = []
        for t in ev_txt:
            e = ids_of(llm, t)
            eids.append(e)
            spans.append((cur, cur + len(e)))
            cur += len(e)
        rq = rng.choice([i for i in range(nev) if i not in ch])
        q_retr = (f"\n\nQuestion: What value did sensor {sensors[rq]} measure? "
                  "Answer with the number only:")
        gold_retr = str(vals[rq])
        q_hop = (f"\n\nQuestion: What is the value of register {names[2]}? "
                 "Answer with the number only:")
        gold_hop = str(vals[ch[0]])
        arms = {}
        for headmode in ("head", "nohead"):
            blocks = [(H_kv, list(range(H)))]
            for i in range(nev):
                if headmode == "head":
                    kv = encode_block(llm, H_ids + eids[i],
                                      list(range(H)) + list(range(*spans[i])),
                                      keep_a=H)
                else:
                    kv = encode_block(llm, eids[i], list(range(*spans[i])))
                blocks.append((kv, list(range(*spans[i]))))
            for task, q, gold in (("retr", q_retr, gold_retr), ("hop", q_hop, gold_hop)):
                out = serve_and_answer(llm, blocks, q, tail, ntok=12)
                arms[f"{headmode}_{task}"] = int(gold in out)
        # full-prefill controls
        full_txt = head_full + "".join(ev_txt)
        for task, q, gold in (("retr", q_retr, gold_retr), ("hop", q_hop, gold_hop)):
            out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(full_txt + q + tail), 12)
            arms[f"full_{task}"] = int(gold in out)
        rows.append(arms)
        print(f"[sink] it{it} " + " ".join(f"{k}={v}" for k, v in arms.items()),
              flush=True)
    agg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    return {"n": len(rows), "acc": agg, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "swap", "erase", "sink"])
    ap.add_argument("--out", default="./out/ph.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1, help="E1 filler reps")
    ap.add_argument("--n2", type=int, default=24, help="E2 items this shard")
    ap.add_argument("--n3", type=int, default=20, help="E3 items this shard")
    ap.add_argument("--seed_base", type=int, default=0)
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    res = {}
    if args.exp in ("all", "swap"):
        res["swap"] = exp_swap(llm, head, tail, args.shard, args.nshards,
                               args.reps, args.seed_base)
    if args.exp in ("all", "erase"):
        res["erase"] = exp_erase(llm, head, tail, args.n2,
                                 args.seed_base + args.shard * 99991 + 1)
    if args.exp in ("all", "sink"):
        res["sink"] = exp_sink(llm, head, tail, args.n3,
                               args.seed_base + args.shard * 99991 + 2)
    json.dump(res, open(args.out, "w"), indent=1)
    print("\n===== SUMMARY =====")
    if "swap" in res:
        print(f"E1 swap: follow-donor {res['swap']['follow_rate']:.2%} "
              f"(n={res['swap']['n_pairs']}x2), iso-names-any "
              f"{res['swap']['iso_names_any_rate']:.2%}")
    if "erase" in res:
        e = res["erase"]
        print(f"E2 erase: no-conflict leak {e['noconflict_leak_rate']:.2%} "
              f"(oracle {e['noconflict_oracle_leak']:.2%}); conflict leak-old "
              f"{e['conflict_leak_old']:.2%}, has-new {e['conflict_has_new']:.2%}")
    if "sink" in res:
        print("E3 sink2x2:", {k: f"{v:.2f}" for k, v in res["sink"]["acc"].items()})
    print("PH_DONE", flush=True)


if __name__ == "__main__":
    main()
