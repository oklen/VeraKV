"""kv_skillib.py -- the QUERY-ROUTED SKILL LIBRARY experiment: N answering protocols
encoded once as KV blocks, routed per question by auditable rules, RoPE-rotated to sit
before the question at zero forward passes.

Two routing designs, each in both carriers (6 arms, all on the identical anchored
b_hot serving base, n=824 mechanism subset):

  it_dec  / ik_dec    the universal DEC protocol, text / rotated-KV      (sk3 anchors)
  it_adapt/ ik_adapt  REPLACEMENT routing (R1-R5 class protocols from the 26-case
                      study; known to lose as text: fragments ablate the DEC engine)
  it_hint / ik_hint   ADDITIVE routing: the intact DEC protocol + a class-specific
                      exception clause (the design the old null never tested; cannot
                      ablate the engine, so it bounds regression structurally)

Pre-registered readings:
  carrier:  ik_* == it_* per arm and per routed class -> whatever the router picks,
            the KV library transmits it losslessly (the capability claim; 11 blocks
            pre-encoded once, zero-forward hot-swap)
  routing:  it_hint vs it_dec  -> does additive specialization pay at all?
            it_adapt vs it_dec -> replacement regression replicates (reference)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_skillib --shard 0 --queue_dir ./out/sl_q
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_distanchor import make_rotator, rotation_gate

ARMS = ["it_dec", "ik_dec", "it_adapt", "ik_adapt", "it_hint", "ik_hint"]
VPOS = 26000

DEC = ("Answer using ONLY the context above. Break the request into sub-questions if "
       "needed, answer each citing exact evidence (values, step indices, outcomes) "
       "VERBATIM, then give the final answer.")

_ONLY = "Answer using ONLY the context above. "
# Replacement protocols + routing regexes: verbatim from the deployed adapt mode
# (rules derived from the annotated 26-case study; auditable).
ADAPT_RULES = [
    ("R1", r"\bif\b.{0,80}\bhad\b|\bwould (it|the|have|this|that)\b|what should|should the agent|most reasonable|why did no|why (was|were) [^?]{0,40} not\b|why didn'?t|counterfactual",
     _ONLY + "Give the single most direct answer, directly and concisely; do not enumerate sub-questions; "
             "prefer the simplest explanation consistent with the trajectory; do not invent specifics."),
    ("R3", r"(what|which) (types|kinds)\b|how frequent|\bfrequency\b|how often",
     _ONLY + "Tally by scanning the whole range and report the final counts only; do not list every step "
             "individually; do not invent counts."),
    ("R2", r"how many times|how many [^?]{0,60}\b(before|between|until|after|at step|first|last|prior)\b",
     _ONLY + "First list every matching occurrence with its step number (cite only steps you can quote), "
             "then give the count of that list as the answer."),
    ("R4", r"how did the state|location histor|state change[sd]?\b|throughout the trajectory|state of [^?]{0,40} change",
     _ONLY + "Reconstruct the state chronologically: one line per step where it changes, citing each step "
             "verbatim; add nothing beyond the cited steps."),
    ("R5", r"\bwhat exact|\bexactly what|\bthe exact\b",
     _ONLY + "Locate the exact item asked for and copy it VERBATIM from the context, citing its step; do "
             "not generalize or abstract."),
]

# Additive exception clauses: the intact DEC engine stays; each class appends one rule.
HINTS = {
    "R1": ("Exception: if the question is counterfactual, asks what should have been "
           "done, or why something was NOT done, skip the sub-question enumeration and "
           "give the single most direct answer; prefer the simplest explanation and do "
           "not invent specifics."),
    "R3": ("For frequency/type-tally questions, scan the whole range and report the "
           "final counts only, not a step-by-step list."),
    "R2": ("For counting questions, first list every matching occurrence with its step "
           "number, then give the count of exactly that list."),
    "R4": ("For state-history questions, reconstruct the state chronologically, one "
           "line per step where it changes, citing each step."),
    "R5": ("When asked for an exact value or string, copy it VERBATIM from the context "
           "and cite its step; do not paraphrase it."),
}

_RX = [(n, re.compile(rx, re.I)) for n, rx, _ in ADAPT_RULES]


def route(question):
    for name, rx in _RX:
        if rx.search(question):
            return name
    return "DEC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--instr_tokens", type=int, default=320)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/sl.json")
    ap.add_argument("--ans_out", default="./out/sl_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # ---- the skill LIBRARY: 11 blocks, each encoded once ----
    head_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    LIB_TEXT = {"DEC": DEC}
    for name, _, repl in ADAPT_RULES:
        LIB_TEXT["A_" + name] = repl                      # replacement protocols
    for name, hint in HINTS.items():
        LIB_TEXT["H_" + name] = DEC + " " + hint          # additive protocols
    LIB = {}
    for key, text in LIB_TEXT.items():
        ids = list(llm.tok("\n\n" + text, add_special_tokens=False).input_ids)
        kv = encode_block(llm, head_ids + ids,
                          list(range(len(head_ids))) + list(range(VPOS, VPOS + len(ids))),
                          keep_a=len(head_ids))
        LIB[key] = (kv, len(ids))
    rot_keys = make_rotator(llm)
    rotation_gate(llm, rot_keys, delta=8192, p0=9000)
    print(f"[w{args.shard}] skill library encoded: {len(LIB)} blocks "
          f"({sum(l for _, l in LIB.values())} rows total), rotation gate passed",
          flush=True)

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    target = args.ep_offset + args.max_ep
    eps = []
    while len(eps) < target and any(queues):
        for qd in queues:
            if qd and len(eps) < target:
                eps.append(qd.popleft())
    eps = eps[args.ep_offset:]

    wid = args.shard
    acc = {a: 0 for a in RUN}
    n = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        all_seg_ids, spans, cur = [], [], H
        for txt in segment_texts:
            sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
            all_seg_ids.append(sids)
            spans.append((cur, cur + len(sids)))
            cur += len(sids)
        total = cur
        if total + 600 >= VPOS:
            print(f"[w{wid}] ep {ep.episode_id} skipped (total={total} too close to VPOS)",
                  flush=True)
            return
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k)
                      if p in id2idx}
            k5 = sorted(hot_idx | picked)
            routed.append(k5)
            need |= set(k5)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st_anch = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            st_anch[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                      pos, keep_a=H + alen)
        print(f"[w{wid}] ep {ep.episode_id} total={total} need={len(need)} qa={len(qas)}",
              flush=True)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                s0, e0 = spans[i]
                for pth in range(max(H, s0), min(H + w_anch, e0)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def serve(kept, qids_, skill_key, ntok):
            keep = [i for i in kept if i not in hot_idx]
            rbl = rblk_for(kept)
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            pos_list = p.tolist() + hot_pos
            if skill_key is not None:
                kv, Ls = LIB[skill_key]
                start = max(pos_list) + 1
                rkv = rot_keys(kv, start - VPOS)
                for li, (K, V) in enumerate(rkv):
                    c.update(K.to(llm.device, non_blocking=True),
                             V.to(llm.device, non_blocking=True), li)
                pos_list += list(range(start, start + Ls))
            out, _, _ = llm._greedy_pos(
                c, torch.tensor(pos_list, dtype=torch.long, device=llm.device),
                qids_, ntok)
            del c
            return out

        for qi, (qa, k5) in enumerate(zip(qas, routed)):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            cls = route(q)
            key_adapt = "DEC" if cls == "DEC" else "A_" + cls
            key_hint = "DEC" if cls == "DEC" else "H_" + cls
            q_bare = f"\n\nQuestion: {q}\nAnswer:" + tail
            qids_bare = llm._ids(q_bare)
            ans = {}
            if "it_dec" in RUN:
                ans["it_dec"] = serve(k5, llm._ids(
                    "\n\n" + DEC + f"\n\nQuestion: {q}\nAnswer:" + tail), None,
                    args.instr_tokens)
            if "ik_dec" in RUN:
                ans["ik_dec"] = serve(k5, qids_bare, "DEC", args.instr_tokens)
            if "it_adapt" in RUN:
                ans["it_adapt"] = serve(k5, llm._ids(
                    "\n\n" + LIB_TEXT[key_adapt] + f"\n\nQuestion: {q}\nAnswer:" + tail),
                    None, args.instr_tokens)
            if "ik_adapt" in RUN:
                ans["ik_adapt"] = serve(k5, qids_bare, key_adapt, args.instr_tokens)
            if "it_hint" in RUN:
                ans["it_hint"] = serve(k5, llm._ids(
                    "\n\n" + LIB_TEXT[key_hint] + f"\n\nQuestion: {q}\nAnswer:" + tail),
                    None, args.instr_tokens)
            if "ik_hint" in RUN:
                ans["ik_hint"] = serve(k5, qids_bare, key_hint, args.instr_tokens)

            row = {"episode_id": ep.episode_id, "qi": qi, "q": q, "gold": gold,
                   "domain": ep.domain, "cls": cls}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} done | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] SKILLIB QUEUE over {len(eps)} episodes", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM ep idx {i} skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 60)
    print(f"SKILLIB w{wid}: {n} QA")
    for a in RUN:
        print(f"  {a:9s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%")
    print(f"SL_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
