"""kv_equiv500 -- scaled, SHARDED text-vs-KV equivalence on real AMA episodes (review #10).

Same two arms as kv_equiv (text re-prefill vs position-preserving KV gather), same 8B same-judge
parity + first-token agreement + GATE-B mask-oracle spot-check, but:
  * --shard/--nshards: deterministic episode split so 8 GPUs run in parallel
  * --ans_out: per-QA JSONL dump (q, gold, both answers, domain, qtype) for a 32B post-judge pass
  * per-domain tallies in the shard summary

    SPRAG_MODEL_PATH=/tmp/Qwen3-8B PYTHONPATH=/home/tiger CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_equiv500 --shard 0 --nshards 8 --max_ep 72 --max_qa 8 \
        --max_tokens 26000 --out /home/tiger/w4_eq_s0.json --ans_out /home/tiger/w4_ans_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import (split_wrap_nothink, build_full_ids, first_logits_on_ids,
                                      first_logits_masked_full)
from kvmemory.kv_equiv import judge, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/tiger/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=26000)
    ap.add_argument("--max_ep", type=int, default=72)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--gate_ep", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="/home/tiger/w4_eq.json")
    ap.add_argument("--ans_out", default="/home/tiger/w4_ans.jsonl")
    ap.add_argument("--layout", choices=["compact", "deployed"], default="compact",
                    help="compact = [selected||q]; deployed = overview(hot verbatim+old gists) present in "
                         "BOTH arms: kv gathers {selected spans + overview segment} (overview prefilled once "
                         "after the trajectory, fixed positions => per-episode cacheable), tx re-prefills the "
                         "deployed order [overview||selected||q]")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # identical deterministic round-robin in every shard, then strided split
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
    mine = eps[args.shard::args.nshards]
    print(f"[s{args.shard}] {len(mine)}/{len(eps)} episodes (<= {args.max_tokens} tok, "
          f"{len(bydom)} domains) k={args.k} hot={args.hot}", flush=True)

    n = argmax_agree = answer_match = kv_ok = tx_ok = 0
    gateb_n = gateb_agree = 0
    bydom_t = {}
    ansf = open(args.ans_out, "w", encoding="utf-8")
    for ei, ep in enumerate(mine):
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        overview = None
        if args.layout == "deployed":
            ov_lines = []
            for i, sseg in enumerate(ep.segments):
                if i in hot_idx:
                    ov_lines.append(f"<step {sseg.turn}>\n{sseg.text}")
                else:
                    gist = " ".join(sseg.text.split())[:220]
                    ov_lines.append(f"<step {sseg.turn}> [summary] {gist}")
            overview = ("\n\nRecency overview of the whole trajectory (recent steps verbatim, "
                        "older steps one-line summaries):\n" + "\n\n".join(ov_lines) + "\n")
            prefill_texts = segment_texts + [overview]
        else:
            prefill_texts = segment_texts
        full_cache, spans, total = llm.prefill_full(prefill_texts, header)
        header_len = spans[0][0]
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        do_gate = args.shard == 0 and ei < args.gate_ep

        for qa in ep.qa[:args.max_qa]:
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtype = qa.get("type", "?")
            picked = router.select(q, old, args.k)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            if args.layout == "deployed":
                sel_old = sorted({id2idx[p] for p in picked if p in id2idx} - hot_idx)
                kept = sel_old + [n_seg]          # selected old spans + the overview segment
                text = (header + overview + "\n\nFull text of the most relevant earlier steps:\n"
                        + "".join(segment_texts[i] for i in sel_old) + qtext)
            else:
                kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
                text = header + "".join(segment_texts[i] for i in kept) + qtext

            sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
            ans_kv, _, _ = llm._greedy_pos(sub_cache, kpos, llm._ids(qtext), args.ans_tokens)
            ans_tx, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
            kg = judge(llm, head, tail, q, gold, ans_kv)
            tg = judge(llm, head, tail, q, gold, ans_tx)
            lk = llm.first_logits_subselect(full_cache, spans, header_len, kept, qtext)
            lt = first_logits_on_ids(llm, build_full_ids(llm, header, prefill_texts, kept, qtext))
            a_ok = int(lk.argmax()) == int(lt.argmax())
            if do_gate:
                masked = first_logits_masked_full(llm, header, prefill_texts, kept, qtext)
                gateb_agree += int(lk.argmax()) == int(masked.argmax())
                gateb_n += 1

            n += 1
            argmax_agree += a_ok
            answer_match += norm(ans_kv) == norm(ans_tx)
            kv_ok += kg
            tx_ok += tg
            d = bydom_t.setdefault(ep.domain, [0, 0, 0, 0])
            d[0] += 1; d[1] += kg; d[2] += tg; d[3] += a_ok
            ansf.write(json.dumps({"episode_id": ep.episode_id, "domain": ep.domain,
                                   "qtype": qtype, "q": q, "gold": gold, "ans_kv": ans_kv,
                                   "ans_tx": ans_tx, "kv8b": int(kg), "tx8b": int(tg),
                                   "argmax": int(a_ok)}, ensure_ascii=False) + "\n")
            ansf.flush()
        json.dump({"shard": args.shard, "n": n, "argmax_agree": argmax_agree,
                   "answer_match": answer_match, "kv_ok": kv_ok, "tx_ok": tx_ok,
                   "gateb_agree": gateb_agree, "gateb_n": gateb_n, "bydom": bydom_t},
                  open(args.out, "w"), indent=2)
        print(f"[s{args.shard}] ep {ep.episode_id} {ep.domain} ({n_seg} seg/{total} tok) | "
              f"argmax {argmax_agree}/{n} | kv {kv_ok}/{n} tx {tx_ok}/{n} | "
              f"gateB {gateb_agree}/{gateb_n}", flush=True)
    ansf.close()
    print(f"EQ500_DONE shard={args.shard} n={n}", flush=True)


if __name__ == "__main__":
    main()
