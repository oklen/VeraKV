"""Equivalence check on REAL AMA-Bench episodes: is KV sub-selection answer-preserving vs the
text re-prefill pipeline the headline used?

For each QA on real trajectories we route the SAME spans, then answer two ways:
  * text      : re-prefill header + selected spans (verbatim text) + question  (the deployed pipeline)
  * kv_select : prefill the trajectory once, gather the selected spans' KV at original positions,
                decode only the question                                        (Variant B)
Position-preserving KV reuse is a DIFFERENT computation from a compact text re-prefill (its
accuracy is often >=), so we do not expect bit-identity; we test that it is ANSWER-preserving: high first-token
agreement, high answer-match, and accuracy PARITY (contains-gold) between the two paths.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        /path/to/python -m kvmemory.kv_equiv --data ./data/ama_test.jsonl \
        --max_tokens 16000 --max_ep 30 --max_qa 8 --k 5
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
from kvmemory.kv_select_smoke import (split_wrap_nothink, build_full_ids, first_logits_on_ids,
                                      first_logits_masked_full)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def judge(llm, head, tail, q, gold, ans) -> bool:
    """Same-model (Qwen3-8B) judge scoring one answer against the gold. Same judge scores BOTH arms,
    so systematic judge bias cancels and the arm-vs-arm PARITY is what matters (not the absolute)."""
    body = ("You are grading an answer to a question about an agent trajectory.\n"
            f"Question: {q}\nReference answer: {gold}\nCandidate answer: {ans}\n\n"
            "Is the candidate answer correct and consistent with the reference answer? "
            "Reply with exactly one word: yes or no.")
    out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(head + body + tail), 4)
    return out.strip().lower().startswith("y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_ep", type=int, default=30)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--gate_ep", type=int, default=4,
                    help="run the GATE-B masked-oracle faithfulness check on the first N episodes")
    ap.add_argument("--out", default="./out/kv_equiv.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # domain-diverse selection (round-robin) so the sample is not one domain
    from collections import defaultdict, deque
    alleps = load_episodes(args.data, max_tokens=args.max_tokens)
    bydom = defaultdict(list)
    for e in alleps:
        bydom[e.domain].append(e)
    queues = [deque(v) for v in bydom.values()]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())
    # strided episode sharding (eps is domain-interleaved, so striding keeps the mix balanced)
    if args.nshards > 1:
        eps = eps[args.shard::args.nshards]
    print(f"[shard {args.shard}/{args.nshards}] equivalence on {len(eps)} real AMA episodes (<= {args.max_tokens} tok) "
          f"across {len(bydom)} domains, k={args.k} hot={args.hot}", flush=True)

    n = argmax_agree = answer_match = kv_ok = tx_ok = 0
    gateb_n = gateb_agree = 0
    by_type = {}       # qtype -> [n, argmax_agree, kv_ok, tx_ok]
    rows = []
    for ei, ep in enumerate(eps):
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        full_cache, spans, total = llm.prefill_full(segment_texts, header)
        header_len = spans[0][0]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        do_gate = ei < args.gate_ep

        for qa in ep.qa[:args.max_qa]:
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail

            # kv_select arm (spans served from cache)
            sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
            ans_kv, _, _ = llm._greedy_pos(sub_cache, kpos, llm._ids(qtext), args.ans_tokens)
            # text arm (deployed pipeline: re-prefill selected spans as text)
            text = header + "".join(segment_texts[i] for i in kept) + qtext
            ans_tx, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            # same-judge accuracy for BOTH arms (parity is the claim)
            kg = judge(llm, head, tail, q, gold, ans_kv)
            tg = judge(llm, head, tail, q, gold, ans_tx)
            # first-token agreement (kv position-preserving vs text compact)
            lk = llm.first_logits_subselect(full_cache, spans, header_len, kept, qtext)
            lt = first_logits_on_ids(llm, build_full_ids(llm, header, segment_texts, kept, qtext))
            a_ok = int(lk.argmax()) == int(lt.argmax())
            m_ok = norm(ans_kv) == norm(ans_tx)
            # GATE-B faithfulness spot-check: kv vs full-prefill-then-mask-dropped oracle
            if do_gate:
                masked = first_logits_masked_full(llm, header, segment_texts, kept, qtext)
                gateb_agree += int(lk.argmax()) == int(masked.argmax())
                gateb_n += 1

            n += 1
            argmax_agree += a_ok
            answer_match += m_ok
            kv_ok += kg
            tx_ok += tg
            d = by_type.setdefault(qtype, [0, 0, 0, 0])
            d[0] += 1; d[1] += a_ok; d[2] += kg; d[3] += tg
        rows.append({"episode_id": ep.episode_id, "domain": ep.domain, "n_seg": n_seg,
                     "total_tok": total})
        json.dump({"rows": rows, "n": n, "argmax_agree": argmax_agree, "answer_match": answer_match,
                   "kv_ok": kv_ok, "tx_ok": tx_ok, "gateb_agree": gateb_agree, "gateb_n": gateb_n,
                   "by_type": by_type},
                  open(args.out, "w"), indent=2)
        print(f"  ep {ep.episode_id} {ep.domain} ({n_seg} seg/{total} tok) | "
              f"argmax {argmax_agree}/{n} | judge-acc kv {kv_ok}/{n} tx {tx_ok}/{n} | "
              f"gateB {gateb_agree}/{gateb_n}", flush=True)

    print("\n" + "=" * 72)
    print(f"EQUIVALENCE ON REAL AMA-Bench ({n} QA, {len(eps)} episodes, Qwen3-8B reader+judge)")
    print("=" * 72)
    print(f"  GATE-B faithfulness (kv vs full-prefill+mask oracle): {gateb_agree}/{gateb_n} "
          f"({100*gateb_agree/max(1,gateb_n):.1f}%)  [must be ~100%: kv correctly keeps selected spans]")
    print(f"  ACCURACY PARITY (same 8B judge):  kv_select {kv_ok}/{n} "
          f"({100*kv_ok/max(1,n):.1f}%)  vs  text {tx_ok}/{n} ({100*tx_ok/max(1,n):.1f}%)  "
          f"[Δ = {100*(kv_ok-tx_ok)/max(1,n):+.1f} pp; >=0 = KV selection does not cost accuracy]")
    print(f"  first-token agreement (kv vs text): {argmax_agree}/{n} ({100*argmax_agree/max(1,n):.1f}%)"
          f"  [<100% expected: position-preserving != compact re-prefill]")
    print(f"  full-answer exact match (kv vs text): {answer_match}/{n} ({100*answer_match/max(1,n):.1f}%)")
    print("  by qtype (n / kv-judge-acc / tx-judge-acc / argmax-agree):")
    for t, d in sorted(by_type.items()):
        print(f"    {t}: {d[0]} / {d[2]} / {d[3]} / {d[1]}")
    print("EQUIV_DONE")


if __name__ == "__main__":
    main()
