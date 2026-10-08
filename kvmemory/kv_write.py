"""kv_write.py -- the write-policy x query-repair GRID (closes the sliding-window objection).

Reviewer-shaped question: "over-window people just slide the window -- each new event's KV sees the
previous x tokens. Isn't that the natural cached-write policy?" This run measures the FULL grid on
the standard within-window sample (where the isolation gap exists and every policy is constructible):

  write-time context for cached span KV:   none (iso) | sliding-window w=4k (roll) | full (glob)
  query-time repair:                       none       | hot-tail replay

Arms (tx = text re-prefill anchor):
  tx        per-query re-prefill of the routed view                    (100% recompute)
  iso       block-diagonal write (header+self only), gather            (write ctx: none)
  roll      BANDED full-trajectory prefill (each token attends header + previous w tokens ONLY --
            exact streaming sliding-window write, every token computed once), gather selected spans
  glob      plain full-trajectory prefill, gather                      (write ctx: full == A2)
  iso_hot   iso + fully replay the hot tail over the view              (the validated design)
  roll_hot  roll + fully replay the hot tail over the view             (sliding write + repair)

Prediction from the Sprint-0 decomposition: what matters is seeing the CO-SELECTED evidence, which is
scattered across the trajectory and unknowable at write time -> roll lands near iso overall, gaining
only where the selected evidence is locally clustered (extent <= w). Each row records the selected
old spans' token extent so the report can split roll-iso by near/far evidence.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_write --shard 0 --queue_dir ./out/wr_q --max_ep 64 --max_qa 8
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
from kvmemory.llm_hf import HFBackend, _iter_cache_kv
from kvmemory.kv_equiv import judge, norm
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_replay import iso_prefill, replayed_cache, build_alloc

ARMS = ["tx", "iso", "roll", "glob", "iso_hot", "roll_hot"]


@torch.no_grad()
def banded_prefill(llm, header_text, segment_texts, w, chunk=2048):
    """Full-trajectory prefill where token t attends ONLY to the header (shared sink, matching every
    other arm's always-visible header) and to tokens within the last `w` positions -- the exact KV a
    streaming sliding-window writer would have cached, each token computed once. Chunked so the
    per-chunk score tensor stays small. Returns (cache, spans, total, ids_t)."""
    ids: list[int] = list(llm.tok(header_text, add_special_tokens=False).input_ids)
    header_len = len(ids)
    spans: list[tuple[int, int]] = []
    for txt in segment_texts:
        sids = llm.tok(txt, add_special_tokens=False).input_ids
        spans.append((len(ids), len(ids) + len(sids)))
        ids.extend(sids)
    total = len(ids)
    dev = llm.device
    ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
    cache = DynamicCache()
    minv = torch.finfo(llm.model.dtype).min
    for s in range(0, total, chunk):
        L = min(chunk, total - s)
        tpos = torch.arange(s, s + L, device=dev)
        kpos = torch.arange(0, s + L, device=dev)
        allow = (kpos[None, :] <= tpos[:, None]) & \
                ((kpos[None, :] < header_len) | (tpos[:, None] - kpos[None, :] <= w))
        m4 = torch.where(allow, torch.zeros((), device=dev),
                         torch.full((), minv, device=dev)).to(llm.model.dtype)[None, None]
        llm.model(input_ids=ids_t[:, s:s + L], past_key_values=cache, use_cache=True,
                  position_ids=tpos.unsqueeze(0), cache_position=tpos, attention_mask=m4)
    return cache, spans, total, ids_t


@torch.no_grad()
def gather_replay_big(llm, big_cache, ids_t, spans, header_len, kept, alloc):
    """Gather header + kept-span prefixes from a FULL-TRAJECTORY cache (contiguous positions) and
    recompute each kept span's suffix (alloc[i] tokens) attending causally-by-position over the
    gathered view. Same math as kv_replay.replayed_cache, but the base KV comes from a trajectory-
    level cache (banded or plain) instead of a per-view isolated prefill."""
    dev = llm.device
    keep_idx = list(range(header_len))
    rep_idx: list[int] = []
    for i in sorted(kept):
        st, en = spans[i]
        r = min(alloc.get(i, 0), en - st)
        keep_idx.extend(range(st, en - r))
        rep_idx.extend(range(en - r, en))
    keep_t = torch.tensor(keep_idx, dtype=torch.long, device=dev)
    new_cache = DynamicCache()
    for li, K0, V0 in _iter_cache_kv(big_cache):
        new_cache.update(K0.index_select(2, keep_t).contiguous(),
                         V0.index_select(2, keep_t).contiguous(), li)
    if not rep_idx:
        return new_cache, keep_t, 0
    rep_t = torch.tensor(rep_idx, dtype=torch.long, device=dev)
    rep_ids = ids_t.index_select(1, rep_t)
    K, Lr = keep_t.shape[0], rep_t.shape[0]
    allpos = torch.cat([keep_t, rep_t])
    allow = allpos[None, :] <= rep_t[:, None]
    minv = torch.finfo(llm.model.dtype).min
    m4 = torch.where(allow, torch.zeros((), device=dev),
                     torch.full((), minv, device=dev)).to(llm.model.dtype)[None, None]
    llm.model(input_ids=rep_ids, past_key_values=new_cache, use_cache=True,
              position_ids=rep_t.unsqueeze(0), cache_position=torch.arange(K, K + Lr, device=dev),
              attention_mask=m4)
    return new_cache, torch.cat([keep_t, rep_t]), Lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=64)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096, help="sliding write window (tokens)")
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/wr.json")
    ap.add_argument("--ans_out", default="./out/wr_ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())

    wid = args.shard
    acc = {a: 0 for a in ARMS}
    n = 0
    byhop = {}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_len = len(llm.tok(header, add_special_tokens=False).input_ids)
        # the two trajectory-level write policies, built once per episode
        band_cache, spans, total, ids_t = banded_prefill(llm, header, segment_texts, args.w)
        full_cache, spans2, total2 = llm.prefill_full(segment_texts, header)
        assert spans == spans2 and total == total2, "tokenization drift between prefills"
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}

        for qa in ep.qa[: args.max_qa]:
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            hb = hop_bucket(q)
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            old_kept = [i for i in kept if i not in hot_idx]
            extent = (spans[max(old_kept)][1] - spans[min(old_kept)][0]) if len(old_kept) >= 2 else 0
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            hot_alloc = build_alloc("sp_hot", sorted(kept), spans, segment_texts, hot_idx, q)
            zero_alloc = {}
            ans = {}

            text = header + "".join(segment_texts[i] for i in kept) + qtext
            ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            # write ctx = sliding / full, no repair (fresh gathered copies; big caches stay pristine)
            rc, rp, _ = gather_replay_big(llm, band_cache, ids_t, spans, header_len, kept, zero_alloc)
            ans["roll"], _, _ = llm._greedy_pos(rc, rp, qids, args.ans_tokens)
            gc_, gp, _ = gather_replay_big(llm, full_cache, ids_t, spans, header_len, kept, zero_alloc)
            ans["glob"], _, _ = llm._greedy_pos(gc_, gp, qids, args.ans_tokens)
            # write ctx = sliding, + hot-tail repair
            rc, rp, _ = gather_replay_big(llm, band_cache, ids_t, spans, header_len, kept, hot_alloc)
            ans["roll_hot"], _, _ = llm._greedy_pos(rc, rp, qids, args.ans_tokens)
            # write ctx = none (view-level isolated prefill), +/- repair
            iso_c, ipos, iids, blocks = iso_prefill(llm, header, segment_texts, kept, spans, header_len)
            hc, hp, _, _ = replayed_cache(llm, iso_c, ipos, iids, blocks, header_len, hot_alloc)
            ans["iso_hot"], _, _ = llm._greedy_pos(hc, hp, qids, args.ans_tokens)
            ans["iso"], _, _ = llm._greedy_pos(iso_c, ipos, qids, args.ans_tokens)  # mutates iso_c, last

            row = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype, "hop": hb,
                   "q": q, "gold": gold, "extent": extent, "near": int(0 < extent <= args.w)}
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
        del band_cache, full_cache
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": ARMS, "n": n, "acc": acc, "w": args.w, "byhop": byhop},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} ({total} tok) | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] WRITE-GRID QUEUE over {len(eps)} episodes, w={args.w}, arms={ARMS}", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            run_episode(eps[i])
    else:
        for ep in eps[args.shard::args.nshards]:
            run_episode(ep)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 72)
    print(f"WRITE-GRID w{wid}: {n} QA, w={args.w}")
    for a in ARMS:
        print(f"  {a:9s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%")
    print(f"WRITE_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
