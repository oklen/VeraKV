"""kv_floor.py -- FLOOR-RAISING shots: can a better conditioner/serving design push the cached
view ABOVE text re-prefill (not just parity)? Router is closed (5 signal families dead); the value
now has to come from raising b_anch itself.

Arms (all share the b_anch skeleton: spans encoded independently against a served conditioner):
  tx        text re-prefill of the routed evidence (reference)
  b_anch    anchored spans + served anchor (the validated design, reference)
  b_hot     b_anch stores, but the hot tail (last `hot` segments) is NOT served from cache --
            its TOKENS are freshly recomputed over the view at read time (sp_hot's integrator
            mechanism on top of write-side conditioning; tests additivity, and whether
            isolation-minus-interference + fresh integration can EXCEED joint prefill)
  b_dig     conditioner = a generated global episode digest (map-reduce, question-blind, greedy,
            cacheable) instead of the raw first-w tokens; digest KV served. Targets the hop-0 /
            aggregation weakness: digest carries episode-GLOBAL information the opening lacks.
            Write-legality: batch/archival setting (episode complete at write time).
  b_dig_hot b_dig + fresh hot tail (combo)
  b_k8      b_anch with router K=8 instead of 5: the equal-LATENCY comparison -- cached views
            make serving more evidence nearly free, text re-prefill cannot afford it
  b_qsel    b_anch view + query-selected verbatim sentences (from the WHOLE episode) freshly
            rolled in with the question (user's "query picks the roll-in" instinct, cheapest form)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_floor --shard 0 --queue_dir ./out/fl_q
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
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble, toks
from kvmemory.kv_ow import prefill_chunked

ARMS = ["tx", "glob", "b_anch", "b_hot", "b_dig", "b_dig_hot", "b_k8", "b_qsel"]


@torch.no_grad()
def prefill_fresh(llm, cache, cache_len, ids, positions):
    """Feed `ids` (original `positions`) on top of an assembled view; KV appended in place."""
    dev = llm.device
    core = getattr(llm.model, "model", llm.model)
    core(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
         past_key_values=cache, use_cache=True,
         position_ids=torch.tensor(positions, dtype=torch.long, device=dev).unsqueeze(0),
         cache_position=torch.arange(cache_len, cache_len + len(ids), device=dev),
         attention_mask=torch.ones(1, cache_len + len(ids), dtype=torch.long, device=dev))
    return cache


def make_digest(llm, head, tail, task, segment_texts):
    """Question-blind map-reduce digest of the whole episode; greedy, deterministic."""
    chunks, cur, curlen = [], [], 0
    for t in segment_texts:
        n = len(llm.tok(t, add_special_tokens=False).input_ids)
        if curlen + n > 6000 and cur:
            chunks.append("".join(cur))
            cur, curlen = [], 0
        cur.append(t)
        curlen += n
    if cur:
        chunks.append("".join(cur))
    notes = []
    for ch in chunks:
        p = (head + "You are indexing part of an agent trajectory for later review.\n\nTask: %s\n\n"
             "Trajectory part:\n%s\n\nWrite dense factual notes on this part: entities, objects, "
             "locations, numeric values, actions and their outcomes, state changes. Use verbatim "
             "names and values. Do not speculate. At most 150 words." % (task, ch) + tail)
        t, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(p), 320)
        notes.append(t.strip())
    if len(notes) == 1:
        digest = notes[0]
    else:
        p = (head + "Merge these notes about one agent trajectory into a single factual digest: "
             "goal, phases in order, key entities/values/locations, final state. Keep names and "
             "values verbatim. Do not speculate. At most 350 words.\n\nNotes:\n%s"
             % "\n---\n".join(notes) + tail)
        digest, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(p), 768)
    return "[Episode digest]\n" + digest.strip() + "\n\n"


def qsel_text(q, segment_texts, budget_tok=400):
    """Query-selected verbatim sentences from the whole episode (question-only scoring)."""
    qs = toks(q)
    sents = []
    for i, t in enumerate(segment_texts):
        for s in re.split(r"(?<=[.!?\n])\s+", t):
            s = s.strip()
            st = toks(s)
            if len(s) > 20 and st:
                sents.append((len(qs & st) / (len(st) ** 0.5 + 1e-6), i, s))
    sents.sort(key=lambda x: -x[0])
    out, used = [], 0
    for sc, i, s in sents:
        if sc <= 0 or used >= budget_tok:
            break
        n = len(s) // 4 + 1
        if used + n > budget_tok + 80:
            continue
        out.append("<step %d> %s" % (i, s))
        used += n
    return ("\n\nKey excerpts (verbatim):\n" + "\n".join(out) + "\n") if out else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/fl.json")
    ap.add_argument("--ans_out", default="./out/fl_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list; skips unneeded machinery")
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]
    need_dig = any(a in RUN for a in ("b_dig", "b_dig_hot"))

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

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
    byhop = {}
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

        routed5, routed8, need_all, need5 = [], [], set(), set()
        for qa in qas:
            p5 = {id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx}
            k5 = sorted(hot_idx | p5)
            routed5.append(k5)
            need5 |= set(k5)
            need_all |= set(k5)
            if "b_k8" in RUN:
                p8 = {id2idx[p] for p in router.select(qa["question"], old, args.k2) if p in id2idx}
                k8 = sorted(hot_idx | p8)
                routed8.append(k8)
                need_all |= set(k8)
            else:
                routed8.append(None)

        # digest (question-blind, cached per episode on shared FS for replicability)
        dig_ids, dlen, DIG_kv = [], 0, None
        if need_dig:
            digf = "./out/fl_digests/ep%d.txt" % ep.episode_id
            os.makedirs("./out/fl_digests", exist_ok=True)
            if os.path.exists(digf):
                digest = open(digf, encoding="utf-8").read()
            else:
                digest = make_digest(llm, head, tail, ep.task, segment_texts)
                open(digf, "w", encoding="utf-8").write(digest)
            dig_ids = list(llm.tok(digest, add_special_tokens=False).input_ids)
            dlen = len(dig_ids)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        if need_dig:
            DIG_kv = encode_block(llm, header_ids + dig_ids,
                                  list(range(H)) + list(range(H, H + dlen)), keep_a=H)

        st_anch, st_dig, pos_dig = {}, {}, {}
        nxt_dig = H + dlen  # early spans shift cumulatively past the digest (compact-null backed)
        for i in sorted(need_all):
            st, en = spans[i]
            span_ids = all_seg_ids[i]
            alen = min(w_anch, st - H)
            st_anch[i] = encode_block(
                llm, header_ids + traj_flat[:alen] + span_ids,
                list(range(H)) + list(range(H, H + alen)) + list(range(st, en)),
                keep_a=H + alen)
            if need_dig and i in need5:
                sp = max(st, nxt_dig)
                pos_dig[i] = list(range(sp, sp + len(span_ids)))
                nxt_dig = sp + len(span_ids)
                st_dig[i] = encode_block(
                    llm, header_ids + dig_ids + span_ids,
                    list(range(H)) + list(range(H, H + dlen)) + pos_dig[i],
                    keep_a=H + dlen)

        hkb = (header_kv, list(range(H)))
        digb = (DIG_kv, list(range(H, H + dlen))) if need_dig else None

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

        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]
        full_cache = None
        if "glob" in RUN:
            full_cache, _tl, _tp = prefill_chunked(llm, header, segment_texts)

        for qa, k5, k8 in zip(qas, routed5, routed8):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans = {}
            k5_old = [i for i in k5 if i not in hot_idx]
            r5 = rblk_for(k5)
            ab5 = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in k5]

            if "tx" in RUN:
                text = header + "".join(segment_texts[i] for i in k5) + qtext
                ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)

            if "glob" in RUN:
                sub_c, sub_pos = llm.subselect_cache(full_cache, spans, H, k5)
                ans["glob"], _, _ = llm._greedy_pos(sub_c, sub_pos, qids, args.ans_tokens)
                del sub_c

            if "b_anch" in RUN:
                c, p = build([hkb] + ([r5] if r5 else []) + ab5)
                ans["b_anch"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c

            if "b_qsel" in RUN:
                c, p = build([hkb] + ([r5] if r5 else []) + ab5)
                qx = qsel_text(q, segment_texts)
                ans["b_qsel"], _, _ = llm._greedy_pos(c, p, llm._ids(qx + qtext), args.ans_tokens)
                del c

            if "b_hot" in RUN:
                abo = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in k5_old]
                c, p = build([hkb] + ([r5] if r5 else []) + abo)
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
                ans["b_hot"], _, _ = llm._greedy_pos(c, p2, qids, args.ans_tokens)
                del c

            if "b_dig" in RUN:
                db5 = [(st_dig[i], pos_dig[i]) for i in k5]
                c, p = build([hkb, digb] + db5)
                ans["b_dig"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c
            if "b_dig_hot" in RUN:
                dbo = [(st_dig[i], pos_dig[i]) for i in k5_old]
                c, p = build([hkb, digb] + dbo)
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
                ans["b_dig_hot"], _, _ = llm._greedy_pos(c, p2, qids, args.ans_tokens)
                del c

            if "b_k8" in RUN:
                r8 = rblk_for(k8)
                ab8 = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in k8]
                c, p = build([hkb] + ([r8] if r8 else []) + ab8)
                ans["b_k8"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                del c

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            d = byhop.setdefault(row["hop"], {a: 0 for a in RUN})
            d.setdefault("_n", 0)
            d["_n"] += 1
            for a in RUN:
                d[a] += row[a]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st_anch, st_dig, header_kv, R_kv, full_cache, DIG_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc, "byhop": byhop},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} dlen={dlen} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] FLOOR QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM on episode idx {i} -- skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 72)
    print(f"FLOOR w{wid}: {n} QA")
    for a in RUN:
        print(f"  {a:10s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%")
    print(f"FLOOR_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
