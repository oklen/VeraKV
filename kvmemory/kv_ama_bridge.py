"""kv_ama_bridge.py -- b_hot on the FULL official AMA-Bench (208 eps / 2,496 QAs), Track-B aligned.

Bridges the KV-level design into the official protocol: answers are generated here (HF backend,
Qwen3-32B reader = Track-B backbone) with the champion #1 configuration wherever it is
cache-compatible -- pin router (hybrid lexical+embedding RRF + name_steps) at K=12, hot window,
DECOMPOSE answer instruction -- then scored OFFLINE by the OFFICIAL judge (src/evaluate.py against
a Qwen3-32B vLLM server), giving leaderboard-comparable numbers for:

  tx     routed evidence re-prefilled as text  (the text memory system's serving path)
  iso    same evidence as BARE per-event KV, header-conditioned only (naive PIC-style caching --
         measures the isolation drop on the official set)
  b_hot  same evidence served as anchored per-event KV + fresh hot tail (our serving path)

Positions: episodes <=125k keep ORIGINAL span positions (YaRN 4x covers 131k); the 19 longer
episodes (up to 1M tok) use the PACKED-UNION scheme validated by the iso_compact null: the union
of all routed spans is laid out chronologically after [header; anchor], stores encoded and served
at those packed positions (gaps where unrouted spans would sit are fine -- position-gap null).

    SPRAG_MODEL_PATH=/path/to/Qwen3-32B SPRAG_EMBED_PATH=/path/to/Qwen3-Embedding-0.6B \
        SPRAG_ATTN_IMPL=sdpa SPRAG_ROPE_FACTOR=4.0 SPRAG_MAX_CTX=131072 PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_ama_bridge --queue_dir ./out/br_q \
        --out ./out/br_o/br_s0.json --ans_out ./out/br_o/br_ans_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter, EmbeddingRouter, HybridRouter, MultiHopRouter
from kvmemory.embed import QwenEmbedder
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_globhot import gather_rows
from kvmemory.kv_distanchor import make_rotator, rotation_gate

ARMS = ["tx", "iso", "b_hot"]
POS_NATIVE_MAX = 125000  # original positions if total fits YaRN-131k with headroom
# deployment-map arms (pre-registered 2026-07-15): gsk = harvest(glob_R)+rotated-skill-KV
# in-window, b_hot+rotated-skill-KV over-window; g_txt = harvest+DEC-as-text (in-window only;
# over-window it would equal the official b_hot arm row-for-row, so those rows are reused at
# analysis instead of regenerated).
HARVEST_MAX = 20000   # 32B + embedder + full-cache + gather copy must co-reside on 80GB
VPOS = 50000          # skill block's write-time virtual position (rotated per query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--instr", default="./ama/dec_instr.txt")
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=320)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--arms", default="tx,iso,b_hot")
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/br_o/br.json")
    ap.add_argument("--ans_out", default="./out/br_o/br_ans.jsonl")
    ap.add_argument("--only", default="", help="json list of episode_ids to run (patch mode)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ARMS[:] = [a for a in args.arms.split(",") if a]
    DEC = open(args.instr, encoding="utf-8").read().strip()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    skill_kv, skill_len, rotator = None, 0, None
    if "gsk" in ARMS:
        head_ids0 = list(llm.tok(head, add_special_tokens=False).input_ids)
        instr_ids = list(llm.tok("\n\n" + DEC, add_special_tokens=False).input_ids)
        skill_len = len(instr_ids)
        skill_kv = encode_block(
            llm, head_ids0 + instr_ids,
            list(range(len(head_ids0))) + list(range(VPOS, VPOS + skill_len)),
            keep_a=len(head_ids0))
        rotator = make_rotator(llm)
        rotation_gate(llm, rotator, delta=40000, p0=9000)
        print(f"[w{args.shard}] skill block: {skill_len} rows @ {VPOS} (gate passed)",
              flush=True)
    embedder = QwenEmbedder()
    router = MultiHopRouter(HybridRouter([LexicalRouter(), EmbeddingRouter(embedder)]),
                            name_steps=True)
    eps = load_episodes(args.data)  # ALL episodes -- the official full set
    only = set(json.load(open(args.only))) if args.only else None

    wid = args.shard
    n = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        all_seg_ids, lens = [], []
        for txt in segment_texts:
            sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
            all_seg_ids.append(sids)
            lens.append(len(sids))
        # single-span cap (mega-episode segments): truncation applied to KV AND tx text alike
        SPAN_CAP = 12000
        for i in range(n_seg):
            if lens[i] > SPAN_CAP:
                all_seg_ids[i] = all_seg_ids[i][:SPAN_CAP]
                lens[i] = SPAN_CAP
                segment_texts[i] = llm.tok.decode(all_seg_ids[i], skip_special_tokens=True)
        total = H + sum(lens)
        native = total <= POS_NATIVE_MAX
        # original-position map (used when native)
        orig_spans, cur = [], H
        for L in lens:
            orig_spans.append((cur, cur + L))
            cur += L
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa

        t0 = time.time()
        # per-QA serve budget (the text champion was likewise forced to a ~22k cap) and a
        # union/store budget so packed positions stay within YaRN validity on mega-episodes
        SERVE_BUDGET, UNION_BUDGET = 20000, 110000
        hot_cost = sum(lens[i] for i in hot_idx)
        routed, need, rank_of = [], set(), {i: -1 for i in hot_idx}
        for qa in qas:
            ranked = [id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx]
            kept, cost = list(hot_idx), hot_cost
            for r_, j in enumerate(ranked):
                if j in kept:
                    continue
                if cost + lens[j] > SERVE_BUDGET:
                    continue
                kept.append(j)
                cost += lens[j]
                rank_of[j] = min(rank_of.get(j, 999), r_)
            kept = sorted(set(kept))
            routed.append(kept)
            need |= set(kept)
        keep_union, cum = set(), 0
        for j in sorted(need, key=lambda x: (rank_of.get(x, 999), x)):
            if j not in hot_idx and cum + lens[j] > UNION_BUDGET:
                continue
            keep_union.add(j)
            cum += lens[j]
        need = keep_union
        routed = [[j for j in kept if j in need] for kept in routed]
        t_route = time.time() - t0

        w_anch = min(args.w, total - H)
        # position layout for stores
        spans = {}
        if native:
            for i in sorted(need):
                spans[i] = orig_spans[i]
        else:
            cur = H + w_anch  # packed-union: chronological spans right after the anchor
            for i in sorted(need):
                spans[i] = (cur, cur + lens[i])
                cur += lens[i]
            if cur > 130000:
                print(f"[w{wid}] ep {ep.episode_id} WARN packed len {cur} > 130k", flush=True)

        harvest_ok = native and total <= HARVEST_MAX \
            and any(a in ARMS for a in ("gsk", "g_txt"))

        t0 = time.time()
        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv, st_anch, st_iso = None, {}, {}
        if ("b_hot" in ARMS) or (("gsk" in ARMS) and not harvest_ok):
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
            for i in sorted(need):
                st, en = spans[i]
                alen = min(w_anch, st - H)
                st_anch[i] = encode_block(
                    llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                    list(range(H)) + list(range(H, H + alen)) + list(range(st, en)),
                    keep_a=H + alen)
        if "iso" in ARMS:
            for i in sorted(need):
                st, en = spans[i]
                st_iso[i] = encode_block(
                    llm, header_ids + all_seg_ids[i],
                    list(range(H)) + list(range(st, en)), keep_a=H)
        full_cache = None
        if harvest_ok:
            full_cache = DynamicCache()
            ids_all = header_ids + traj_flat
            dev = llm.device
            core = getattr(llm.model, "model", llm.model)  # skip lm_head: no vocab logits
            with torch.no_grad():  # NO autograd graph: 64 layers of activations otherwise
                for s0_ in range(0, len(ids_all), 4096):
                    seg = torch.tensor([ids_all[s0_: s0_ + 4096]], dtype=torch.long,
                                       device=dev)
                    L = seg.shape[1]
                    core(input_ids=seg, past_key_values=full_cache, use_cache=True,
                         position_ids=torch.arange(s0_, s0_ + L, device=dev).unsqueeze(0),
                         cache_position=torch.arange(s0_, s0_ + L, device=dev),
                         attention_mask=torch.ones(1, s0_ + L, dtype=torch.long,
                                                   device=dev))
        t_write = time.time() - t0
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} total={total} native={native} "
              f"harvest={int(harvest_ok)} need={len(need)} qa={len(qas)} "
              f"route {t_route:.0f}s stores {t_write:.0f}s", flush=True)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                st, en = spans[i]
                for pth in range(max(H, st), min(H + w_anch, en)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
            return assemble(llm, bs)

        def gather_idx(kept):
            idx = list(range(H))
            for pth in range(H, H + w_anch):
                if not any(spans[i][0] <= pth < spans[i][1] for i in kept):
                    idx.append(pth)
            for i in sorted(kept):
                idx += list(range(spans[i][0], spans[i][1]))
            return sorted(set(idx))

        def append_skill(c, start):
            kv = rotator(skill_kv, start - VPOS)
            for li, (K, V) in enumerate(kv):
                c.update(K.to(llm.device, non_blocking=True),
                         V.to(llm.device, non_blocking=True), li)
            return list(range(start, start + skill_len))

        for qa, kept in zip(qas, routed):
            q = qa["question"]
            qtext = f"\n\n{DEC}\n\nQuestion: {q}\nAnswer:" + tail
            q_bare = f"\n\nQuestion: {q}\nAnswer:" + tail
            qids = llm._ids(qtext)
            preds = {}

            try:
                if "tx" in ARMS:
                    # chunked no-logit prefill: big routed views OOM'd the one-shot path (fp32 logits)
                    c_tx, tl, _ = prefill_chunked(llm, header, [segment_texts[i] for i in kept])
                    preds["tx"], _, _ = llm._greedy(c_tx, tl, llm._ids(qtext), args.ans_tokens)
                    del c_tx

                if "iso" in ARMS:
                    iso_blocks = [(st_iso[i], list(range(spans[i][0], spans[i][1]))) for i in kept]
                    c, p = build([hkb] + iso_blocks)
                    preds["iso"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                    del c

                if "b_hot" in ARMS:
                    r5 = rblk_for(kept)
                    kept_old = [i for i in kept if i not in hot_idx]
                    abo = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept_old]
                    c, p = build([hkb] + ([r5] if r5 else []) + abo)
                    c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                    p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
                    preds["b_hot"], _, _ = llm._greedy_pos(c, p2, qids, args.ans_tokens)
                    del c

                if "gsk" in ARMS:
                    if harvest_ok:
                        idx = gather_idx(kept)
                        c = gather_rows(llm, full_cache, idx)
                        pos_list = list(idx)
                    else:
                        r5 = rblk_for(kept)
                        kept_old = [i for i in kept if i not in hot_idx]
                        abo = [(st_anch[i], list(range(spans[i][0], spans[i][1])))
                               for i in kept_old]
                        c, p = build([hkb] + ([r5] if r5 else []) + abo)
                        c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                        pos_list = p.tolist() + hot_pos
                    pos_list += append_skill(c, max(pos_list) + 1)
                    preds["gsk"], _, _ = llm._greedy_pos(
                        c, torch.tensor(pos_list, dtype=torch.long, device=llm.device),
                        llm._ids(q_bare), args.ans_tokens)
                    del c

                if "g_txt" in ARMS and harvest_ok:
                    idx = gather_idx(kept)
                    c = gather_rows(llm, full_cache, idx)
                    preds["g_txt"], _, _ = llm._greedy_pos(
                        c, torch.tensor(idx, dtype=torch.long, device=llm.device),
                        qids, args.ans_tokens)
                    del c
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "task_type": ep.task_type, "task_description": ep.task,
                   "question_uuid": qa.get("question_uuid"), "question": q,
                   "golden_answer": qa.get("answer", ""), "qtype": qa.get("type", "?"),
                   "native_pos": int(native), "harvest": int(harvest_ok)}
            for a in ARMS:
                if a in preds:
                    row["pred_" + a] = preds[a]
            if not preds:
                continue
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, st_iso, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "n": n}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} done | rows={n}", flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] BRIDGE QUEUE over {len(eps)} episodes (official full)", flush=True)
        for i in range(len(eps)):
            if only is not None and eps[i].episode_id not in only:
                continue
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(eps[i])
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM ep idx {i} skipped", flush=True)
                torch.cuda.empty_cache()
            # completion marker inside the claim dir: lets a supervisor reclaim
            # stale (claimed-but-dead) episodes without double-running finished ones
            open(os.path.join(args.queue_dir, f"c{i}", "done"), "w").close()
    else:
        for ep in eps[args.shard::args.nshards]:
            run_episode(ep)
    ansf.close()
    print(f"BRIDGE_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
