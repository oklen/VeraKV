"""kv_replay.py -- Sprint-1 Pareto probe: buy back contextualization with partial suffix REPLAY.

Sprint-0 (kv_scope, n=512) measured the trilemma corners: isolated per-span KV (A3) loses 4.7pp to
text re-prefill (A1) and 5.9pp to global gather (A2); the load-bearing part is INTER-SELECTED-SPAN
context (A1-A3), not dropped-step context (A2-A1 = +1.2 n.s.). This probe asks: how much of the
A1-A3 gap does query-time SELECTIVE REPLAY recover, as a function of replay budget?

Key causal-attention fact that makes replay well-posed: a span's PREFIX KV (computed seeing only
header+own-prefix) is valid regardless of what precedes it; only recomputed tokens change. So we
keep each selected span's first (1-f) tokens from the cached isolated prefill and RECOMPUTE the last
f tokens attending causally (by original position) over the whole assembled view. f=0 == A3 (iso);
f=1 == joint view prefill == A1-at-original-positions (sanity anchor, should ~match tx accuracy).

Arms (same routed spans, same judge, per QA):
  tx      : A1 text re-prefill of selected spans (anchor; the accuracy target)
  iso     : A3 isolated per-span KV at original positions (anchor; f=0)
  rep05/15/30/100 : uniform suffix replay, f = .05/.15/.30/1.0 of each span (first-in-view span skipped
            -- replaying it is a no-op: nothing before it but the header it already saw)
  dep15/30: dependency-directed replay at the SAME total budget as rep15/30 -- budget allocated
            proportional to each span's token-overlap with EARLIER selected spans (spans that
            reference prior objects/steps need contextualization; standalone spans don't). Falls
            back to uniform when no overlap signal exists.

Metrics: 8B-judge accuracy per arm (ordering is the signal), answer-match to tx, replay tokens
(realized budget), replay+ingest time. Gate: >= half of (tx-iso) recovered at <=20-30% replay,
and dep > rep at equal budget.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_replay --selftest            # correctness gate (mask/plumbing)
        python -m kvmemory.kv_replay --shard 0 --queue_dir ./out/replay_q --max_ep 64 ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend, _iter_cache_kv
from kvmemory.kv_equiv import judge, norm
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket

# Two arm sets, one mask framework. "suffix" = Sprint-1a token-granular suffix replay (FALSIFIED:
# convex Pareto, 26% replay buys 17-25% recovery -- the query reads span BODY tokens, so
# contextualizing only the tail is useless). "span" = Sprint-1b span-granular all-or-nothing replay:
# fully recompute the m most question-relevant old spans (and/or the hot tail) over the view --
# the deploy-shaped "query view = cached base KV + fully replayed reading set".
ARMSETS = {
    "suffix": ["tx", "iso", "rep05", "rep15", "rep30", "rep100", "dep15", "dep30"],
    "span": ["tx", "iso", "sp_q1", "sp_q2", "sp_q3", "sp_hot", "sp_q1h"],
    # held-out policy validation: anchors + the three arms the PRE-REGISTERED hop-conditional
    # policies combine (P1: hop0->rep100, hop1->iso, hop2+->sp_q1; P2: hop2+->sp_hot).
    "policy": ["tx", "iso", "sp_q1", "sp_hot", "rep100"],
    # tail-size knob: how small can the replayed recency tail get? sp_hotN = replay only the LAST N
    # spans of the view (view content identical everywhere -- ONLY the replay set shrinks).
    "knob": ["tx", "iso", "sp_hot1", "sp_hot2", "sp_hot", "rep100"],
    # 32B scale-transfer spot check: do the 8B arm orderings survive at deployment scale?
    "scale": ["tx", "iso", "sp_hot"],
}
ARMS = ARMSETS["suffix"]
REPLAY_ARMS = [a for a in ARMS if a not in ("tx", "iso")]
_FRAC = {"rep05": 0.05, "rep15": 0.15, "rep30": 0.30, "rep100": 1.0, "dep15": 0.15, "dep30": 0.30}

_TOKR = re.compile(r"[a-z0-9]+")
_STOP = frozenset(("the a an to of and in is are was were on at for with as by it this that be been "
                   "you your step").split())


@torch.no_grad()
def iso_prefill(llm, header_text, segment_texts, kept, spans, header_len):
    """Isolated (block-diagonal) prefill of header + kept spans at ORIGINAL positions.

    Same computation as kv_scope.isolated_cache, but also returns the token ids and per-span cache
    blocks so suffixes can be replayed. Returns (cache, pos, ids, blocks) with blocks[j] =
    (span_idx, cache_start, cache_end) in cache coordinates.
    """
    ids: list[int] = list(llm.tok(header_text, add_special_tokens=False).input_ids)
    positions: list[int] = list(range(header_len))
    block: list[int] = [-1] * header_len
    blocks: list[tuple[int, int, int]] = []
    for b, i in enumerate(sorted(kept)):
        sids = list(llm.tok(segment_texts[i], add_special_tokens=False).input_ids)
        st, en = spans[i]
        if len(sids) != en - st:  # keep boundaries identical to prefill_full's tokenization
            sids = (sids + [llm.tok.pad_token_id or 0] * (en - st))[: en - st]
        blocks.append((i, len(ids), len(ids) + len(sids)))
        ids.extend(sids)
        positions.extend(range(st, en))
        block.extend([b] * (en - st))
    L = len(ids)
    dev = llm.device
    idt = torch.tensor([ids], dtype=torch.long, device=dev)
    pos = torch.tensor(positions, dtype=torch.long, device=dev)
    blk = torch.tensor(block, dtype=torch.long, device=dev)
    idx = torch.arange(L, device=dev)
    allow = (idx[None, :] <= idx[:, None]) & ((blk[None, :] == -1) | (blk[None, :] == blk[:, None]))
    minv = torch.finfo(llm.model.dtype).min
    m4 = torch.where(allow, torch.zeros((), device=dev),
                     torch.full((), minv, device=dev)).to(llm.model.dtype)[None, None]
    cache = DynamicCache()
    llm.model(input_ids=idt, past_key_values=cache, use_cache=True,
              position_ids=pos.unsqueeze(0), cache_position=torch.arange(L, device=dev),
              attention_mask=m4)
    return cache, pos, idt, blocks


def alloc_uniform(kept_sorted, spans, f, skip_first=True):
    """Suffix-replay allocation: fraction f of each span's tokens (first-in-view span skipped)."""
    alloc = {}
    for j, i in enumerate(kept_sorted):
        L = spans[i][1] - spans[i][0]
        alloc[i] = 0 if (f <= 0 or (skip_first and j == 0)) else min(L, max(1, int(round(f * L))))
    return alloc


def alloc_dep(kept_sorted, spans, segment_texts, f):
    """Dependency-directed allocation at the SAME total budget as alloc_uniform(f): budget goes to
    spans proportional to their informative-token overlap with EARLIER selected spans (the causal
    direction replay can exploit). Zero-signal fallback = uniform."""
    lens = {i: spans[i][1] - spans[i][0] for i in kept_sorted}
    budget = sum(alloc_uniform(kept_sorted, spans, f).values())
    toks = {i: {t for t in _TOKR.findall(segment_texts[i].lower()) if t not in _STOP}
            for i in kept_sorted}
    score, seen = {}, set()
    for j, i in enumerate(kept_sorted):
        score[i] = 0.0 if j == 0 else len(toks[i] & seen) / (len(toks[i]) ** 0.5 + 1e-6)
        seen |= toks[i]
    if budget <= 0 or sum(score.values()) <= 0:
        return alloc_uniform(kept_sorted, spans, f)
    alloc = {i: 0 for i in kept_sorted}
    rem = budget
    for _ in range(4):
        cand = [i for i in kept_sorted if score[i] > 0 and alloc[i] < lens[i]]
        tot = sum(score[i] for i in cand)
        if rem <= 0 or not cand or tot <= 0:
            break
        for i in cand:
            g = min(lens[i] - alloc[i], int(round(rem * score[i] / tot)))
            alloc[i] += g
        rem = budget - sum(alloc.values())
    if rem > 0:  # dump remainder into highest-score non-full spans
        for i in sorted(kept_sorted, key=lambda x: -score[x]):
            g = min(lens[i] - alloc[i], rem)
            alloc[i] += g
            rem -= g
            if rem <= 0:
                break
    return alloc


def qrank_old(q, kept_sorted, hot_idx, segment_texts):
    """Kept OLD spans ranked by question-relevance (informative-token overlap, length-normalized)."""
    qt = {t for t in _TOKR.findall(q.lower()) if t not in _STOP}
    olds = [i for i in kept_sorted if i not in hot_idx]
    return sorted(olds, key=lambda i: -len(qt & {t for t in _TOKR.findall(segment_texts[i].lower())
                                                 if t not in _STOP}) /
                                       (len(_TOKR.findall(segment_texts[i].lower())) ** 0.5 + 1e-6))


def alloc_span(kept_sorted, spans, chosen):
    """Span-granular replay: chosen spans are recomputed IN FULL over the view, others stay isolated."""
    return {i: (spans[i][1] - spans[i][0] if i in chosen else 0) for i in kept_sorted}


def build_alloc(arm, kept_sorted, spans, segment_texts, hot_idx, q):
    if arm.startswith("rep"):
        return alloc_uniform(kept_sorted, spans, _FRAC[arm])
    if arm.startswith("dep"):
        return alloc_dep(kept_sorted, spans, segment_texts, _FRAC[arm])
    if arm == "sp_hot":
        return alloc_span(kept_sorted, spans, set(hot_idx) & set(kept_sorted))
    if arm.startswith("sp_hot") and arm[6:].isdigit():  # sp_hot1 / sp_hot2: last N view spans only
        return alloc_span(kept_sorted, spans, set(kept_sorted[-int(arm[6:]):]))
    ranked = qrank_old(q, kept_sorted, hot_idx, segment_texts)
    if arm == "sp_q1h":
        return alloc_span(kept_sorted, spans, set(ranked[:1]) | (set(hot_idx) & set(kept_sorted)))
    m = int(arm[4:])  # sp_q1 / sp_q2 / sp_q3
    return alloc_span(kept_sorted, spans, set(ranked[:m]))


@torch.no_grad()
def replayed_cache(llm, iso_cache, iso_pos, iso_ids, blocks, header_len, alloc):
    """Build the replay-f view: keep header + span prefixes from the isolated cache, RECOMPUTE each
    span's suffix (alloc[i] tokens) in one forward attending causally-by-original-position over the
    whole view (earlier spans' prefixes = cached isolated KV, earlier suffixes = freshly replayed).

    Returns (cache, positions, n_replay, t_replay). Cache slot order = [header+prefixes][suffixes];
    attention is mask-addressed so slot order need not be position order.
    """
    dev = llm.device
    keep_idx = list(range(header_len))
    rep_idx: list[int] = []
    for (sidx, cs, ce) in blocks:
        r = min(alloc.get(sidx, 0), ce - cs)
        keep_idx.extend(range(cs, ce - r))
        rep_idx.extend(range(ce - r, ce))
    keep_t = torch.tensor(keep_idx, dtype=torch.long, device=dev)
    new_cache = DynamicCache()
    for li, K0, V0 in _iter_cache_kv(iso_cache):
        new_cache.update(K0.index_select(2, keep_t).contiguous(),
                         V0.index_select(2, keep_t).contiguous(), li)
    keep_pos = iso_pos.index_select(0, keep_t)
    if not rep_idx:  # nothing to replay -> identical to iso (fresh copy so iso_cache stays clean)
        return new_cache, keep_pos, 0, 0.0
    rep_t = torch.tensor(rep_idx, dtype=torch.long, device=dev)
    rep_pos = iso_pos.index_select(0, rep_t)
    rep_ids = iso_ids.index_select(1, rep_t)
    K, Lr = keep_t.shape[0], rep_t.shape[0]
    allpos = torch.cat([keep_pos, rep_pos])
    allow = allpos[None, :] <= rep_pos[:, None]  # causal by ORIGINAL position (self included: unique pos)
    minv = torch.finfo(llm.model.dtype).min
    m4 = torch.where(allow, torch.zeros((), device=dev),
                     torch.full((), minv, device=dev)).to(llm.model.dtype)[None, None]
    torch.cuda.synchronize(); t0 = time.time()
    llm.model(input_ids=rep_ids, past_key_values=new_cache, use_cache=True,
              position_ids=rep_pos.unsqueeze(0), cache_position=torch.arange(K, K + Lr, device=dev),
              attention_mask=m4)
    torch.cuda.synchronize(); t_replay = time.time() - t0
    return new_cache, torch.cat([keep_pos, rep_pos]), Lr, t_replay


@torch.no_grad()
def _first_logits_pos(llm, cache, positions, qids):
    """First-token logits decoding qids on a (possibly slot-permuted) positional cache."""
    K, L = positions.shape[0], qids.shape[1]
    nxt = int(positions.max().item()) + 1
    pos = torch.arange(nxt, nxt + L, device=llm.device)
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=pos.unsqueeze(0), cache_position=torch.arange(K, K + L, device=llm.device),
                    attention_mask=torch.ones(1, K + L, dtype=torch.long, device=llm.device))
    return out.logits[0, -1].float()


@torch.no_grad()
def selftest(llm, eps, head, tail, router, args):
    """Correctness gate: full replay (f=1, INCLUDING the first span => keep = header only) must match
    a plain joint prefill of the same tokens at the same original positions at first-token argmax."""
    n = ok = 0
    for ep in eps[: args.selftest_ep]:
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_len = len(llm.tok(header, add_special_tokens=False).input_ids)
        spans, cur = [], header_len
        for txt in segment_texts:
            tl = len(llm.tok(txt, add_special_tokens=False).input_ids)
            spans.append((cur, cur + tl))
            cur += tl
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        for qa in ep.qa[: args.selftest_qa]:
            q = qa["question"]
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            iso_c, ipos, iids, blocks = iso_prefill(llm, header, segment_texts, kept, spans, header_len)
            full_alloc = {i: spans[i][1] - spans[i][0] for i in kept}  # replay EVERYTHING
            rc, rpos, _, _ = replayed_cache(llm, iso_c, ipos, iids, blocks, header_len, full_alloc)
            lg_rep = _first_logits_pos(llm, rc, rpos, qids)
            # oracle: single joint prefill of the same ids at the same positions, plain causal
            cache = DynamicCache()
            L = iids.shape[1]
            llm.model(input_ids=iids, past_key_values=cache, use_cache=True,
                      position_ids=ipos.unsqueeze(0), cache_position=torch.arange(L, device=llm.device),
                      attention_mask=torch.ones(1, L, dtype=torch.long, device=llm.device))
            lg_joint = _first_logits_pos(llm, cache, ipos, qids)
            n += 1
            ok += int(lg_rep.argmax()) == int(lg_joint.argmax())
    rate = ok / max(1, n)
    print(f"SELFTEST full-replay vs joint-prefill argmax: {ok}/{n} = {100*rate:.1f}%")
    print(f"SELFTEST {'PASS' if rate >= 0.9 else 'FAIL'}")
    return rate >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=64)
    ap.add_argument("--ep_offset", type=int, default=0,
                    help="skip the first N round-robin episodes (held-out trajectories)")
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--qa_offset", type=int, default=0,
                    help="skip the first N QAs per episode (held-out questions)")
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="", help="if set: work-queue mode (atomic per-episode claim)")
    ap.add_argument("--armset", choices=list(ARMSETS), default="suffix")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest_ep", type=int, default=2)
    ap.add_argument("--selftest_qa", type=int, default=4)
    ap.add_argument("--out", default="./out/replay.json")
    ap.add_argument("--ans_out", default="./out/replay_ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    alleps = load_episodes(args.data, max_tokens=args.max_tokens)
    bydom = defaultdict(list)
    for e in alleps:
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    target = args.ep_offset + args.max_ep
    eps = []
    while len(eps) < target and any(queues):
        for qd in queues:
            if qd and len(eps) < target:
                eps.append(qd.popleft())
    eps = eps[args.ep_offset:]

    if args.selftest:
        sys.exit(0 if selftest(llm, eps, head, tail, router, args) else 1)

    global ARMS, REPLAY_ARMS
    ARMS = ARMSETS[args.armset]
    REPLAY_ARMS = [a for a in ARMS if a not in ("tx", "iso")]
    wid = args.shard
    acc = {a: 0 for a in ARMS}
    am_tx = {a: 0 for a in REPLAY_ARMS + ["iso"]}
    nrep = {a: 0 for a in REPLAY_ARMS}
    t_rep = {a: 0.0 for a in REPLAY_ARMS}
    byhop = {}
    n = 0
    sel_tok_total = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n, sel_tok_total
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        hids = llm.tok(header, add_special_tokens=False).input_ids
        header_len = len(hids)
        spans, cur = [], header_len
        for txt in segment_texts:
            tl = len(llm.tok(txt, add_special_tokens=False).input_ids)
            spans.append((cur, cur + tl))
            cur += tl
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}

        for qa in ep.qa[args.qa_offset: args.qa_offset + args.max_qa]:
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            hb = hop_bucket(q)
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            sel_tok = sum(spans[i][1] - spans[i][0] for i in kept)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)

            ans = {}
            # tx (A1): compact re-prefill
            text = header + "".join(segment_texts[i] for i in kept) + qtext
            ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            # shared isolated prefill (the per-event cacheable base)
            iso_c, ipos, iids, blocks = iso_prefill(llm, header, segment_texts, kept, spans, header_len)
            # replay arms (each gathers a fresh copy from iso_c; iso_c itself stays pristine)
            qrep = {}
            for arm in REPLAY_ARMS:
                alloc = build_alloc(arm, sorted(kept), spans, segment_texts, hot_idx, q)
                rc, rpos, lr, tr = replayed_cache(llm, iso_c, ipos, iids, blocks, header_len, alloc)
                a, ti, _ = llm._greedy_pos(rc, rpos, qids, args.ans_tokens)
                ans[arm] = a
                qrep[arm] = lr
                nrep[arm] += lr
                t_rep[arm] += tr + ti
            # iso (A3) LAST -- _greedy_pos mutates iso_c and nothing needs it afterwards
            ans["iso"], _, _ = llm._greedy_pos(iso_c, ipos, qids, args.ans_tokens)

            row = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype, "hop": hb,
                   "q": q, "gold": gold, "sel_tok": sel_tok}
            for arm, lr in qrep.items():
                row["nrep_" + arm] = lr
            for a in ARMS:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
                if a != "tx":
                    m = int(norm(ans[a]) == norm(ans["tx"]))
                    am_tx[a] += m
                    row["am_" + a] = m
            n += 1
            sel_tok_total += sel_tok
            d = byhop.setdefault(hb, {a: 0 for a in ARMS})
            d.setdefault("_n", 0)
            d["_n"] += 1
            for a in ARMS:
                d[a] += row[a]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()

        json.dump({"shard": wid, "arms": ARMS, "n": n, "acc": acc, "am_tx": am_tx, "nrep": nrep,
                   "t_rep": t_rep, "sel_tok_total": sel_tok_total, "byhop": byhop},
                  open(args.out, "w"), indent=2)
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] QUEUE mode over {len(eps)} episodes ({len(bydom)} domains) "
              f"k={args.k} hot={args.hot} arms={ARMS}", flush=True)
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
    print("\n" + "=" * 76)
    print(f"REPLAY w{wid}: {n} QA | sel_tok/QA {sel_tok_total//nn}")
    for a in ARMS:
        extra = ""
        if a in REPLAY_ARMS:
            extra = f"  replay/QA {nrep[a]//nn} ({100*nrep[a]/max(1,sel_tok_total):.0f}% of sel)"
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%{extra}")
    print(f"REPLAY_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
