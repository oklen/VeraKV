"""kv_skill.py -- close the paper's structural gap (user, 2026-07-14): Part I's biggest
lever is the reader-side answer instruction (+5pp officially) -- pure prompt text, decoupled
from the KV substrate. Repackage it as a RETRIEVABLE SKILL KV BLOCK: encoded once globally
(chat-head-conditioned, baked at a fixed virtual position beyond any trajectory), spliced
into the served cache with ZERO computation, upgrading the answering protocol from prompt
engineering to a first-class memory object. If KV-served skill == text-served instruction,
the reader lever rejoins the KV story and the paper unifies.

Arms (n=824 paired, 8B; the champion "structured" instruction verbatim from the paper's
appendix):
  bh      b_hot + bare question                          gate (.2209)
  bh_it   b_hot + instruction as TEXT before question    the "plain" target to tie
  bh_ik   b_hot + instruction as KV block @pos 26000     skill store (zero-compute splice)
  gr      glob_R + bare question                         gate (.2100)
  gr_it   glob_R + instruction text
  gr_ik   glob_R + instruction KV                        zero-write memory + zero-compute skill

Skill block details: encoded ONCE per shard as [chat-head; instr @ 26000..26000+L], keeping
the instr rows -- episode-independent (the whole point of a skill library). At serve time it
is appended AFTER the fresh tail (frozen rows, no forward pass), so the tail's computation
stays causal; the question then starts at max(pos)+1 past the skill. Law-II note: the
skill's write-time context (bare chat head) mismatches the served reality (evidence-filled
positions 0..24k) -- evidence blocks pay -3.4pp for such mismatch (fx); if the skill is
immune, mode-setting blocks and fact blocks obey different conditioning laws.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_skill --shard 0 --queue_dir ./out/sk_q
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_globhot import gather_rows
from kvmemory.kv_distanchor import make_rotator

ARMS = ["bh", "bh_it", "bh_ik", "gr", "gr_it", "gr_ik"]

INSTR = ("Answer using ONLY the context above. Break the request into sub-questions if "
         "needed, answer each citing exact evidence (values, step indices, outcomes) "
         "VERBATIM, then give the final answer.")
VPOS = 26000


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
    ap.add_argument("--dep_budget", type=int, default=4096)  # runner flag-sanity
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--instr_tokens", type=int, default=320)
    ap.add_argument("--place", default="vpos", choices=["vpos", "rot"],
                    help="skill placement: frozen at VPOS, or K-rotated per query to sit "
                         "contiguously before the question (zero forward passes)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/sk.json")
    ap.add_argument("--ans_out", default="./out/sk_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # the global skill store: encoded ONCE, reused across all episodes/queries
    head_ids = list(llm.tok(head, add_special_tokens=False).input_ids)
    instr_ids = list(llm.tok("\n\n" + INSTR, add_special_tokens=False).input_ids)
    Ls = len(instr_ids)
    skill_kv = encode_block(llm, head_ids + instr_ids,
                            list(range(len(head_ids))) + list(range(VPOS, VPOS + Ls)),
                            keep_a=len(head_ids))
    skill_pos = list(range(VPOS, VPOS + Ls))
    rot_keys = make_rotator(llm) if args.place == "rot" else None
    print(f"[w{args.shard}] skill block encoded: {Ls} rows @ {VPOS} place={args.place}",
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
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)
        if total + 600 >= VPOS:
            print(f"[w{wid}] ep {ep.episode_id} too long for VPOS ({total}) -- skipped",
                  flush=True)
            return

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k)
                      if p in id2idx}
            k5 = sorted(hot_idx | picked)
            routed.append(k5)
            need |= set(k5)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv, st_anch = None, {}
        if any(a.startswith("bh") for a in RUN):
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
            for i in sorted(need):
                s0, e0 = spans[i]
                alen = min(w_anch, s0 - H)
                st_anch[i] = encode_block(
                    llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                    list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0)),
                    keep_a=H + alen)
        full_cache = None
        if any(a.startswith("gr") for a in RUN):
            full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts, chunk=4096)
            assert ftot == total

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

        def append_skill(c, start):
            """Splice the skill rows; place=rot rotates K to sit at [start, start+Ls)
            (elementwise, no forward pass). Returns the positions used."""
            if rot_keys is not None:
                kv = rot_keys(skill_kv, start - VPOS)
                pos = list(range(start, start + Ls))
            else:
                kv, pos = skill_kv, skill_pos
            for li, (K, V) in enumerate(kv):
                c.update(K.to(llm.device, non_blocking=True),
                         V.to(llm.device, non_blocking=True), li)
            return pos

        def serve_bh(kept, qids_, with_skill, ntok):
            keep = [i for i in kept if i not in hot_idx]
            rbl = rblk_for(kept)
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st_anch[i], list(range(spans[i][0], spans[i][1])))
                            for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            pos_list = p.tolist() + hot_pos
            if with_skill:
                pos_list += append_skill(c, max(pos_list) + 1)
            out, _, _ = llm._greedy_pos(
                c, torch.tensor(pos_list, dtype=torch.long, device=llm.device),
                qids_, ntok)
            del c
            return out

        def serve_gr(kept, qids_, with_skill, ntok):
            idx = list(range(H))
            for p in range(H, H + w_anch):
                if not any(spans[i][0] <= p < spans[i][1] for i in kept):
                    idx.append(p)
            for i in sorted(kept):
                idx += list(range(spans[i][0], spans[i][1]))
            idx = sorted(set(idx))
            c = gather_rows(llm, full_cache, idx)
            pos_list = list(idx)
            if with_skill:
                pos_list += append_skill(c, max(pos_list) + 1)
            out, _, _ = llm._greedy_pos(
                c, torch.tensor(pos_list, dtype=torch.long, device=llm.device),
                qids_, ntok)
            del c
            return out

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            q_bare = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            q_it = "\n\n" + INSTR + f"\n\nQuestion: {q}\nAnswer:" + tail
            q_kv = f"\n\nQuestion: {q}\nAnswer:" + tail
            ans = {}
            try:
                if "bh" in RUN:
                    ans["bh"] = serve_bh(k5, llm._ids(q_bare), False, args.ans_tokens)
                if "bh_it" in RUN:
                    ans["bh_it"] = serve_bh(k5, llm._ids(q_it), False, args.instr_tokens)
                if "bh_ik" in RUN:
                    ans["bh_ik"] = serve_bh(k5, llm._ids(q_kv), True, args.instr_tokens)
                if "gr" in RUN:
                    ans["gr"] = serve_gr(k5, llm._ids(q_bare), False, args.ans_tokens)
                if "gr_it" in RUN:
                    ans["gr_it"] = serve_gr(k5, llm._ids(q_it), False, args.instr_tokens)
                if "gr_ik" in RUN:
                    ans["gr_ik"] = serve_gr(k5, llm._ids(q_kv), True, args.instr_tokens)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] SKILL QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
        for i in range(len(eps)):
            cdir = os.path.join(args.queue_dir, f"c{i}")
            try:
                os.mkdir(cdir)
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM on episode idx {i} -- skipped", flush=True)
                torch.cuda.empty_cache()
            open(os.path.join(cdir, "done"), "w").close()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()
    print(f"SKILL_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
