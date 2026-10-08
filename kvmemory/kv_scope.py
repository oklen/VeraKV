"""kv_scope.py -- does a selected span's KV need the context it was computed in?

On within-window AMA episodes we route the SAME spans for a query, then answer THREE ways under the
SAME Qwen3-8B judge. The three arms differ ONLY in how much cross-event context each selected span's
KV was allowed to absorb -- tokens, positions, and the decode path are identical:

  A1 text      : re-prefill [header + selected spans (verbatim) + q] from scratch.
                 Selected spans attend to EACH OTHER (joint prefill) but never to the dropped steps.
                 (== the deployed text pipeline; compact positions.)
  A2 kv_global : prefill the FULL trajectory once, gather the selected spans' KV at their ORIGINAL
                 positions, decode q. Each span's KV absorbed EVERYTHING before it -- including the
                 dropped steps (this is where the 66% upstream-perturbation sensitivity lives).
  A3 kv_iso    : prefill each selected span in ISOLATION (block-diagonal mask: a span attends only to
                 the shared header + itself), at ORIGINAL positions, gather, decode q. No cross-event
                 context at all == "independent chunk-local KV" (the PIC/MiniPIC assembly).

Why these three: A2 vs A1 is the parity kv_equiv tests (Delta ~0.1pp) => the dropped-context leakage is
already accuracy-neutral. The UNDECIDED number is A3 vs {A1,A2}: does removing even the inter-selected
contextualization cost accuracy? Read it BY HOP (cited-turn count / qtype).
  * A1 ~= A3 flat across hops => isolation is accuracy-safe.
  * A1  >> A3 concentrated in 2+ hop => cross-span context is load-bearing.

Two drivers:
  --queue_dir DIR : work-queue mode (high GPU util). 8 processes (one per GPU) each load the model once
                    and pull episodes from a shared atomic claim dir until empty -- no tail imbalance,
                    no redundant model reloads. --shard is reused as the worker id (0..7).
  (default)       : static strided shard (--shard/--nshards) -- used for the 1-episode smoke gate.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_scope --shard 0 --queue_dir ./out/scope_q --max_ep 64 --max_qa 8 \
        --max_tokens 24000 --out ./out/scope_s0.json --ans_out ./out/scope_ans_s0.jsonl
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
from kvmemory.kv_equiv import judge, norm
from kvmemory.kv_select_smoke import split_wrap_nothink

_CIT = re.compile(r"\b(?:turns?|steps?)\s*#?\s*(\d+)", re.I)


def hop_bucket(q: str) -> str:
    """Cheap hop proxy: how many distinct step/turn indices the question names."""
    nc = len(set(_CIT.findall(q)))
    return "0" if nc == 0 else ("1" if nc == 1 else "2+")


@torch.no_grad()
def isolated_cache(llm, header_text, segment_texts, kept, spans, header_len):
    """A3 assembly: prefill header + kept spans in ONE forward, but with a block-diagonal attention
    mask so each span attends ONLY to the shared header and itself -- never to other spans. Tokens are
    tokenized exactly as prefill_full does and placed at their ORIGINAL positions, so the resulting
    (cache, kept_positions) plug into _greedy_pos identically to A2's subselect_cache output. The ONLY
    difference from A2 is that the stored span KV never absorbed cross-event context.
    """
    ids: list[int] = list(llm.tok(header_text, add_special_tokens=False).input_ids)
    positions: list[int] = list(range(header_len))
    block: list[int] = [-1] * header_len  # -1 == shared header (visible to all spans)
    for b, i in enumerate(sorted(kept)):
        sids = list(llm.tok(segment_texts[i], add_special_tokens=False).input_ids)
        st, en = spans[i]
        if len(sids) != en - st:  # tokenization must match prefill_full's boundaries
            sids = (sids + [llm.tok.pad_token_id or 0] * (en - st))[: en - st]
        ids.extend(sids)
        positions.extend(range(st, en))
        block.extend([b] * (en - st))
    L = len(ids)
    dev = llm.device
    idt = torch.tensor([ids], dtype=torch.long, device=dev)
    pos = torch.tensor(positions, dtype=torch.long, device=dev)
    blk = torch.tensor(block, dtype=torch.long, device=dev)
    idx = torch.arange(L, device=dev)
    causal = idx[None, :] <= idx[:, None]
    is_hdr = blk[None, :] == -1
    same = blk[None, :] == blk[:, None]
    allow = causal & (is_hdr | same)
    minv = torch.finfo(llm.model.dtype).min
    m4 = torch.where(allow, torch.zeros((), device=dev), torch.full((), minv, device=dev))
    m4 = m4.to(llm.model.dtype)[None, None]  # [1,1,L,L] -> broadcasts over heads
    cache = DynamicCache()
    cpos = torch.arange(L, device=dev)
    llm.model(input_ids=idt, past_key_values=cache, use_cache=True,
              position_ids=pos.unsqueeze(0), cache_position=cpos, attention_mask=m4)
    return cache, pos


@torch.no_grad()
def process_episode(llm, ep, head, tail, router, args):
    """Run all three arms (judged) for every QA in one episode. Returns (qa_rows, total_tok, n_seg)."""
    segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
    n_seg = len(segment_texts)
    header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
              f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
    full_cache, spans, total = llm.prefill_full(segment_texts, header)
    header_len = spans[0][0]
    hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
    old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
    id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}

    rows = []
    for qa in ep.qa[:args.max_qa]:
        q = qa["question"]
        gold = qa.get("answer", "") or ""
        qtype = qa.get("type", "?")
        hb = hop_bucket(q)
        picked = router.select(q, old, args.k)
        kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
        qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
        qids = llm._ids(qtext)

        text = header + "".join(segment_texts[i] for i in kept) + qtext
        ans_tx, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
        sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
        ans_kv, _, _ = llm._greedy_pos(sub_cache, kpos, qids, args.ans_tokens)
        iso_c, ipos = isolated_cache(llm, header, segment_texts, kept, spans, header_len)
        ans_iso, _, _ = llm._greedy_pos(iso_c, ipos, qids, args.ans_tokens)

        tg = int(judge(llm, head, tail, q, gold, ans_tx))
        kg = int(judge(llm, head, tail, q, gold, ans_kv))
        ig = int(judge(llm, head, tail, q, gold, ans_iso))
        rows.append({"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype, "hop": hb,
                     "q": q, "gold": gold, "ans_tx": ans_tx, "ans_kv": ans_kv, "ans_iso": ans_iso,
                     "tx8b": tg, "kv8b": kg, "iso8b": ig,
                     "am_kv_iso": int(norm(ans_kv) == norm(ans_iso)),
                     "am_kv_tx": int(norm(ans_kv) == norm(ans_tx))})
    return rows, total, n_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=64)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="", help="if set: work-queue mode (atomic per-episode claim)")
    ap.add_argument("--out", default="./out/scope.json")
    ap.add_argument("--ans_out", default="./out/scope_ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # deterministic domain round-robin (identical in every worker), then optionally strided
    from collections import defaultdict, deque
    alleps = load_episodes(args.data, max_tokens=args.max_tokens)
    bydom = defaultdict(list)
    for e in alleps:
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())

    wid = args.shard
    agg = {"n": 0, "tx_ok": 0, "kv_ok": 0, "iso_ok": 0, "am_kv_iso": 0, "am_kv_tx": 0}
    byhop, byqt, bydom_t = {}, {}, {}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def fold_and_write(ep, rows, total, n_seg):
        for r in rows:
            agg["n"] += 1
            agg["tx_ok"] += r["tx8b"]; agg["kv_ok"] += r["kv8b"]; agg["iso_ok"] += r["iso8b"]
            agg["am_kv_iso"] += r["am_kv_iso"]; agg["am_kv_tx"] += r["am_kv_tx"]
            for tbl, key in ((byhop, r["hop"]), (byqt, r["qtype"]), (bydom_t, r["domain"])):
                d = tbl.setdefault(key, [0, 0, 0, 0])
                d[0] += 1; d[1] += r["tx8b"]; d[2] += r["kv8b"]; d[3] += r["iso8b"]
            ansf.write(json.dumps(r, ensure_ascii=False) + "\n")
        ansf.flush()
        json.dump({"shard": wid, **agg, "byhop": byhop, "byqt": byqt, "bydom": bydom_t},
                  open(args.out, "w"), indent=2)
        n = agg["n"]
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} ({n_seg} seg/{total} tok) | "
              f"tx {agg['tx_ok']}/{n} kv {agg['kv_ok']}/{n} iso {agg['iso_ok']}/{n} | "
              f"am(kv=iso) {agg['am_kv_iso']}/{n}", flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] QUEUE mode over {len(eps)} within-window episodes "
              f"(<= {args.max_tokens} tok, {len(bydom)} domains) k={args.k} hot={args.hot}", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))  # atomic claim (single node, local fs)
            except (FileExistsError, OSError):
                continue
            rows, total, n_seg = process_episode(llm, eps[i], head, tail, router, args)
            fold_and_write(eps[i], rows, total, n_seg)
    else:
        mine = eps[args.shard::args.nshards]
        print(f"[w{wid}] STATIC {len(mine)}/{len(eps)} episodes k={args.k} hot={args.hot}", flush=True)
        for ep in mine:
            rows, total, n_seg = process_episode(llm, ep, head, tail, router, args)
            fold_and_write(ep, rows, total, n_seg)
    ansf.close()

    n = max(1, agg["n"])
    print("\n" + "=" * 72)
    print(f"SCOPE w{wid}: {agg['n']} within-window AMA QA (Qwen3-8B reader+judge)")
    print(f"  A1 text      {agg['tx_ok']}/{agg['n']} = {100*agg['tx_ok']/n:.1f}%")
    print(f"  A2 kv_global {agg['kv_ok']}/{agg['n']} = {100*agg['kv_ok']/n:.1f}%  (Δ text {100*(agg['kv_ok']-agg['tx_ok'])/n:+.1f}pp)")
    print(f"  A3 kv_iso    {agg['iso_ok']}/{agg['n']} = {100*agg['iso_ok']/n:.1f}%  (Δ text {100*(agg['iso_ok']-agg['tx_ok'])/n:+.1f}pp, Δ kv {100*(agg['iso_ok']-agg['kv_ok'])/n:+.1f}pp)")
    print(f"  GATE acc(A2)-acc(A3) = {100*(agg['kv_ok']-agg['iso_ok'])/n:+.1f}pp")
    print(f"SCOPE_DONE shard={wid} n={agg['n']}", flush=True)


if __name__ == "__main__":
    main()
