"""kv_verify.py -- gold-free self-verification as the cascade's wrongness detector.

For every QA (pooled matrix sample) and each chain arm's stored answer {b_anch, roll, tx}, ask the
verifier model -- WITHOUT the reference answer -- whether the candidate is supported by the routed
evidence text. Output yes/no per (QA, arm). Offline we compute the detector's TPR/FPR against the
true judgments and the REALIZED cascade accuracy (accept on yes, escalate on no).

Runs per model (SPRAG_MODEL_PATH decides 8B vs 32B).

    SPRAG_MODEL_PATH=/path/to/Qwen3-32B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_verify --queue_dir ./out/v32_q --out ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink

HOTN, K = 4, 5
ARMS = ["b_anch", "roll", "tx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/vf.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    rows = []
    for pfx in ("mx", "mxh"):
        for f in sorted(glob.glob("./out/%s_ans_s*.jsonl" % pfx)):
            for l in open(f):
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    byep = {}
    for r in rows:
        byep.setdefault(r["episode_id"], []).append(r)
    ep_ids = sorted(byep)
    eps_all = load_episodes(args.data, max_tokens=args.max_tokens)
    byid = {e.episode_id: e for e in eps_all}
    outf = open(args.out, "w", encoding="utf-8")

    @torch.no_grad()
    def ask(prompt):
        out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(prompt), 4)
        return out.strip().lower()

    def run_episode(eid):
        ep = byid[eid]
        seg_texts = ["<step %d>\n%s\n" % (s.turn, s.text) for s in ep.segments]
        hot = set(range(max(0, len(ep.segments) - HOTN), len(ep.segments)))
        old = [s for i, s in enumerate(ep.segments) if i not in hot]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        for r in byep[eid]:
            q = r["q"]
            kept = sorted(hot | {id2idx[p] for p in router.select(q, old, K) if p in id2idx})
            ev = "".join(seg_texts[i] for i in kept)
            rec = {"episode_id": eid, "q": q}
            for a in ARMS:
                ans = r.get("ans_" + a) or ""
                prompt = (head +
                          "You are verifying an answer about a completed agent trajectory, using "
                          "ONLY the evidence excerpts below.\n\nTask: %s\n\nEvidence excerpts:\n%s\n\n"
                          "Question: %s\nCandidate answer: %s\n\nBased only on the evidence, is the "
                          "candidate answer correct and well-supported? Reply with exactly one word: "
                          "yes or no." % (ep.task, ev, q, ans) + tail)
                v = ask(prompt)
                rec["v_" + a] = int(v.startswith("y"))
                rec["ok_" + a] = r[a]
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outf.flush()
        print("[s%d] ep %d done" % (args.shard, eid), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        for i, eid in enumerate(ep_ids):
            try:
                os.mkdir(os.path.join(args.queue_dir, "c%d" % i))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(eid)
            except torch.OutOfMemoryError:
                print("[s%d] OOM ep %d skipped" % (args.shard, eid), flush=True)
                torch.cuda.empty_cache()
    else:
        for eid in ep_ids[args.shard::args.nshards]:
            run_episode(eid)
    outf.close()
    print("VERIFY_DONE shard=%d" % args.shard, flush=True)


if __name__ == "__main__":
    main()
