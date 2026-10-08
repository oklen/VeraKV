"""Replay AMA-Bench trajectories through KVMemory arms and score recall QA (text-level first signal).

Arms:
  full      — all turns verbatim (oracle upper bound; may exceed context on big episodes)
  recent    — only the hot window verbatim, drop the rest (lower bound)
  gist      — hot verbatim + every old turn replaced by its gist (does summary suffice?)
  retrieval — hot verbatim + router-rehydrated old turns verbatim, rest dropped (no recency floor)
  kvmemory  — hot verbatim + router-rehydrated old turns verbatim + gist floor for the rest (ours)

One trajectory is compressed once and answers all its QA (compress-once-answer-many). Sharded across
GPUs (SHARD_ID/NUM_SHARDS, one model copy per CUDA_VISIBLE_DEVICES). Incremental JSON write.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import AutoGistSummarizer, ModelPickRouter
from kvmemory.core import KVMemory, KVMemoryConfig
from kvmemory.llm_hf import HFBackend

ARMS = ["full", "recent", "gist", "retrieval", "kvmemory"]


def answer_prompt_parts(task: str, context: str, question: str) -> tuple[str, str]:
    """Split the answer prompt at the question boundary. The prefix (task + trajectory) is identical
    across an episode's query-independent QA, so it can be prefilled once and the KV reused; only the
    short suffix (the question) is re-encoded per query. See HFBackend.answer_many."""
    prefix = (
        "You answer questions about a completed agent trajectory. Some past steps are shown in full "
        "and others only as short summaries.\n"
        f"Task: {task}\n\nTrajectory:\n{context}\n\nQuestion: "
    )
    suffix = f"{question}\nAnswer concisely and specifically:"
    return prefix, suffix


def answer_prompt(task: str, context: str, question: str) -> str:
    prefix, suffix = answer_prompt_parts(task, context, question)
    return prefix + suffix


def judge_prompt(question: str, gold: str, pred: str) -> str:
    return (
        "Judge whether a candidate answer is correct.\n"
        f"Question: {question}\nReference answer: {gold}\nCandidate answer: {pred}\n\n"
        "Does the candidate match the reference's key facts? Reply with exactly 'yes' or 'no'."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--hot_turns", type=int, default=4)
    ap.add_argument("--k_rehydrate", type=int, default=5)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SHARD_ID", 0)))
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    llm = HFBackend()
    summ = AutoGistSummarizer()
    router = ModelPickRouter(llm)
    domains = set(args.domains.split(",")) if args.domains else None
    eps = load_episodes(args.data, max_tokens=args.max_tokens, domains=domains)
    eps = [e for i, e in enumerate(eps) if i % args.num_shards == args.shard]
    if args.limit:
        eps = eps[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(eps)} episodes", flush=True)

    def make_mem(ep, arm):
        m = KVMemory(
            KVMemoryConfig(arm=arm, hot_turns=args.hot_turns, k_rehydrate=args.k_rehydrate),
            summarizer=summ, router=router, count_tok=llm.count_tok,
        )
        for s in ep.segments:
            m.append(s)
        return m

    results = []
    for ei, ep in enumerate(eps):
        for s in ep.segments:
            s.n_tok = llm.count_tok(s.text)
            s.gist = summ.gist(s)
        mems = {a: make_mem(ep, a) for a in ARMS}
        for qa in ep.qa:
            q, gold, qtype = qa["question"], qa["answer"], qa.get("type", "?")
            rehy = {a: mems[a].stats.get("rehydrated", 0) for a in ARMS}
            ctxs = {a: mems[a].assemble(q) for a in ARMS}
            ans = llm.generate([answer_prompt(ep.task, ctxs[a], q) for a in ARMS],
                               max_tokens=200, batch_size=1)
            verdicts = llm.generate([judge_prompt(q, gold, p) for p in ans],
                                    max_tokens=4, batch_size=8)
            rec = {"episode_id": ep.episode_id, "domain": ep.domain, "qtype": qtype,
                   "total_tokens": ep.total_tokens, "arms": {}}
            for i, a in enumerate(ARMS):
                rec["arms"][a] = {
                    "acc": 1 if verdicts[i].strip().lower().startswith("y") else 0,
                    "ctx_tok": llm.count_tok(ctxs[a]),
                    "rehydrated": mems[a].stats.get("rehydrated", 0) - rehy[a],
                    "pred": ans[i][:160],
                }
            results.append(rec)
        json.dump(results, open(args.out, "w"))
        print(f"[shard {args.shard}] ep {ei+1}/{len(eps)} id={ep.episode_id} {ep.domain} "
              f"{len(ep.qa)}qa done", flush=True)
    print(f"[shard {args.shard}] DONE {len(results)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
