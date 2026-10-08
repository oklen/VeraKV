"""kv_matrix.py -- the Late-Bound KV matrix: curated answer-leakage-controlled roll-in set S(q,m)
plus the conditioning-vs-prompting ABCD square, in ONE paired run.

Roll-in variants (mode A unless stated: filler conditions the span at write, filler KV DISCARDED;
span KV served at ORIGINAL positions):
  dum_s   short neutral dummy (~64 tok) abutting the span
  dum_l   long neutral dummy tiled to w
  fixnat  fixed natural foreign text (one fixed other episode, same slice for everyone)
  ext     cross-episode random segments (per-span shuffle)
  randf   same-episode random segments, ANSWER-FILTERED (segments lexically covering any of the
          episode's gold answers are excluded) -- randf vs the earlier unfiltered rand (joinable,
          pipeline is deterministic) IS the leakage audit
  sem     semantically-similar non-temporal neighbours (|j-i|>4), answer-filtered
  roll    true temporal predecessors (banded sliding write, w)
  anch    episode anchor: the trajectory's own FIRST w tokens at their NATURAL positions
          (gap to span tolerated per iso_compact result; anchor KV shareable across spans)
ABCD (R = the episode anchor block):
  anch    == A  (conditioned on R, R dropped at read)
  b_anch  == B  (conditioned on R, R's KV also served in the view)
  c_anch  == C  (spans ISOLATED, R's KV pasted into the view at read time -- readout repair)
  iso     == D  (nothing)
Anchors: tx (text re-prefill), iso.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_matrix --shard 0 --queue_dir ./out/mx_q --max_ep 64 --max_qa 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend, _iter_cache_kv
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_write import banded_prefill, gather_replay_big

ARMS = ["tx", "iso", "dum_s", "dum_l", "fixnat", "ext", "randf", "sem", "roll", "anch",
        "b_anch", "c_anch"]
STORE_MODES = ["iso", "dum_s", "dum_l", "fixnat", "ext", "randf", "sem", "anch"]

_TOKR = re.compile(r"[a-z0-9]+")
_STOP = frozenset(("the a an to of and in is are was were on at for with as by it this that be "
                   "been you your step".split()))
DUMMY = ("The following section contains routine procedural notes recorded during standard "
         "operation. All recorded values were verified and archived according to protocol. No "
         "anomalies were reported during this period, and routine checks continued as scheduled. ")


def toks(s):
    return {t for t in _TOKR.findall(s.lower()) if t not in _STOP}


@torch.no_grad()
def encode_block(llm, ids, positions, keep_a=None, keep_b=None):
    """One plain-causal forward over ascending-position ids; returns per-layer (K,V) for the
    [keep_a, keep_b) slice, moved to CPU. Uses the transformer CORE (skips lm_head): we only need
    KV, and materializing 152k-vocab logits per encode was the run's dominant waste."""
    dev = llm.device
    idt = torch.tensor([ids], dtype=torch.long, device=dev)
    post = torch.tensor(positions, dtype=torch.long, device=dev)
    cache = DynamicCache()
    core = getattr(llm.model, "model", llm.model)
    # 2026 unified multimodal wrappers: the text decoder lives at .language_model inside
    # the core; identical to `core` for classic causal LMs (attribute absent -> unchanged).
    core = getattr(core, "language_model", None) or core
    # SPRAG_ENCODE_CHUNK: feed the sequence in causal chunks through the same cache --
    # mathematically identical, peak attention memory chunk*L instead of L^2 (needed where
    # fused kernels reject the model's custom mask and fall back to materializing scores,
    # e.g. Gemma4Unified long-context under tf5 SDPA).
    _chunk = int(os.environ.get("SPRAG_ENCODE_CHUNK", "0"))
    if _chunk and len(ids) > _chunk:
        for a0 in range(0, len(ids), _chunk):
            b0 = min(a0 + _chunk, len(ids))
            core(input_ids=idt[:, a0:b0], past_key_values=cache, use_cache=True,
                 position_ids=post[a0:b0].unsqueeze(0),
                 cache_position=torch.arange(a0, b0, device=dev),
                 attention_mask=torch.ones(1, b0, dtype=torch.long, device=dev))
    else:
        core(input_ids=idt, past_key_values=cache, use_cache=True,
             position_ids=post.unsqueeze(0), cache_position=torch.arange(len(ids), device=dev),
             attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=dev))
    a = 0 if keep_a is None else keep_a
    b = len(ids) if keep_b is None else keep_b
    # slice on GPU FIRST, then move: whole-sequence CPU copies were the other dominant waste
    return [(K[:, :, a:b].contiguous().cpu(), V[:, :, a:b].contiguous().cpu())
            for _, K, V in _iter_cache_kv(cache)]


def slice_kv(kv, a, b):
    return [(K[:, :, a:b], V[:, :, a:b]) for K, V in kv]


@torch.no_grad()
def assemble(llm, blocks):
    """blocks: list of (kv_slices, positions_list) in ascending-position order -> (cache, pos)."""
    dev = llm.device
    new = DynamicCache()
    nl = len(blocks[0][0])
    for li in range(nl):
        Ks = [b[0][li][0].to(dev, non_blocking=True) for b in blocks]
        Vs = [b[0][li][1].to(dev, non_blocking=True) for b in blocks]
        new.update(torch.cat(Ks, dim=2), torch.cat(Vs, dim=2), li)
    pos = []
    for b in blocks:
        pos.extend(b[1])
    return new, torch.tensor(pos, dtype=torch.long, device=llm.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=64)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/mx.json")
    ap.add_argument("--ans_out", default="./out/mx_ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    dummy_ids = llm.tok(DUMMY, add_special_tokens=False).input_ids

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
    acc = {a: 0 for a in ARMS}
    n = 0
    byhop = {}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def seg_texts_of(ep):
        return [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        segment_texts = seg_texts_of(ep)
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
        traj_flat = [t for sids in all_seg_ids for t in sids]  # positions H..total
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]

        routed, need = [], set()
        for qa in qas:
            picked = router.select(qa["question"], old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            routed.append(kept)
            need |= set(kept)

        # answer-bearing filter: segments lexically covering any evaluated gold
        golds = [toks(qa.get("answer", "") or "") for qa in qas]
        seg_tok = [toks(t) for t in segment_texts]
        banned = {j for j in range(n_seg)
                  if any(g and len(seg_tok[j] & g) >= max(6, int(0.3 * len(g))) for g in golds)}
        # semantic ranking (per needed span): similar, non-temporal, not banned
        ext_ep = eps[(ei + 1) % len(eps)]
        ext_pool = [list(llm.tok(t, add_special_tokens=False).input_ids)
                    for t in seg_texts_of(ext_ep)]
        fix_ep = eps[(ei + 2) % len(eps)]
        fix_flat = [t for txt in seg_texts_of(fix_ep)
                    for t in llm.tok(txt, add_special_tokens=False).input_ids]
        w_anch = min(args.w, total - H)

        def filler_ids(mode, i, w_avail, rng):
            if w_avail <= 0:
                return []
            if mode == "iso":
                return []
            if mode == "dum_s":
                return dummy_ids[: min(64, w_avail)]
            if mode == "dum_l":
                out = []
                while len(out) < w_avail:
                    out.extend(dummy_ids)
                return out[:w_avail]
            if mode == "fixnat":
                return fix_flat[:w_avail]
            if mode == "anch":
                return None  # handled specially (natural positions)
            if mode == "ext":
                order = list(range(len(ext_pool)))
                rng.shuffle(order)
                out = []
                for j in order:
                    if len(out) >= w_avail:
                        break
                    out.extend(ext_pool[j])
                return out[:w_avail]
            if mode == "randf":
                cand = [j for j in range(n_seg) if j != i and j not in banned]
                rng.shuffle(cand)
                out = []
                for j in cand:
                    if len(out) >= w_avail:
                        break
                    out.extend(all_seg_ids[j])
                return out[:w_avail]
            if mode == "sem":
                ti = seg_tok[i]
                cand = [j for j in range(n_seg)
                        if abs(j - i) > 4 and j not in banned and j != i]
                cand.sort(key=lambda j: -len(ti & seg_tok[j]) / (len(seg_tok[j]) ** 0.5 + 1e-6))
                out = []
                for j in cand:
                    if len(out) >= w_avail:
                        break
                    out.extend(all_seg_ids[j])
                return out[:w_avail]
            raise ValueError(mode)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        # R = episode anchor block at natural positions [H, H+w_anch), conditioned on header
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)

        stores = {m: {} for m in STORE_MODES}
        for i in sorted(need):
            st, en = spans[i]
            span_ids = all_seg_ids[i]
            rng = random.Random(ep.episode_id * 1000 + i)
            for m in STORE_MODES:
                if m == "anch":
                    alen = min(w_anch, st - H)
                    ids = header_ids + traj_flat[:alen] + span_ids
                    pos = list(range(H)) + list(range(H, H + alen)) + list(range(st, en))
                else:
                    fids = filler_ids(m, i, min(args.w, st - H), rng)
                    ids = header_ids + fids + span_ids
                    pos = (list(range(H)) + list(range(st - len(fids), st)) +
                           list(range(st, en)))
                stores[m][i] = encode_block(llm, ids, pos, keep_a=len(ids) - len(span_ids))

        band_cache, spans_b, total_b, ids_t = banded_prefill(llm, header, segment_texts, args.w)
        assert spans_b == spans

        hkb = (header_kv, list(range(H)))
        for qa, kept in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            hb = hop_bucket(q)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            ans = {}
            text = header + "".join(segment_texts[i] for i in kept) + qtext
            ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            rc, rp, _ = gather_replay_big(llm, band_cache, ids_t, spans, H, kept, {})
            ans["roll"], _, _ = llm._greedy_pos(rc, rp, qids, args.ans_tokens)

            def span_blocks(mode):
                return [(stores[mode][i], list(range(spans[i][0], spans[i][1])))
                        for i in sorted(kept)]

            for m in ("iso", "dum_s", "dum_l", "fixnat", "ext", "randf", "sem", "anch"):
                c, p = assemble(llm, [hkb] + span_blocks(m))
                ans[m], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
            # B/C: serve R too (drop R tokens whose positions fall inside a selected span)
            keep_mask = [True] * w_anch
            for i in kept:
                st, en = spans[i]
                for pth in range(max(H, st), min(H + w_anch, en)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if ridx:
                rt = torch.tensor(ridx, dtype=torch.long)  # CPU: R_kv slices live on CPU until assemble
                Rblk = ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                        [H + j for j in ridx])
                # R precedes all served spans positionally only if spans start after H; sort blocks
                for m, name in (("anch", "b_anch"), ("iso", "c_anch")):
                    blocks = [hkb, Rblk] + span_blocks(m)
                    blocks_sorted = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
                    c, p = assemble(llm, blocks_sorted)
                    ans[name], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
            else:
                ans["b_anch"], ans["c_anch"] = ans["anch"], ans["iso"]

            row = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype, "hop": hb,
                   "q": q, "gold": gold}
            for a in ARMS:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            n += 1
            d = byhop.setdefault(hb, {a: 0 for a in ARMS})
            d.setdefault("_n", 0)
            d["_n"] += 1
            for a in ARMS:
                d[a] += row[a]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del stores, header_kv, R_kv, band_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": ARMS, "n": n, "acc": acc, "w": args.w,
                   "n_banned_mean": len(banned), "byhop": byhop}, open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} banned={len(banned)} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] MATRIX QUEUE over {len(eps)} episodes, w={args.w}, arms={ARMS}", flush=True)
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
    print("\n" + "=" * 76)
    print(f"MATRIX w{wid}: {n} QA, w={args.w}")
    for a in ARMS:
        print(f"  {a:8s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%")
    print(f"MATRIX_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
