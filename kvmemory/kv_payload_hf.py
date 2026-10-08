"""Payload ablation (#6) via HFBackend -- regenerate 32B answers WITHOUT a vLLM server.

The two vLLM workers that hosted Qwen3-32B were recycled during a network outage, so we re-run the
oracle-evidence payload ablation by loading Qwen3-32B directly through transformers (single GPU, bf16,
thinking disabled) on a fresh 8xA100 worker. Same three arms -- verbatim / summary / facts -- same
reader=judge, restricted to step-citing questions (unambiguous gold evidence). Phase-batched generation
(summaries, then facts, then answers, then judgments) for throughput; shard by episode across the 8 GPUs.
Saves (q, gold, qt, arm, ans) so the different-family Llama cross-judge (#5) can re-score these too.

  SPRAG_MODEL_PATH=/path/to/Qwen3-32B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    python -m kvmemory.kv_payload_hf --shard 0 --nshards 8 --max_ep 300 \
      --out ./out/payhf_s0.json --ans_out ./out/payhf_ans_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend  # noqa: E402
from kvmemory.kv_payload import cited_turns, load_ama  # noqa: E402

_THINK = re.compile(r"<think>.*?</think>", re.S)

SUM_P = ("Summarize the following agent step in 1-2 sentences, preserving the key entities but "
         "written as prose.\n\n")
FACT_P = ("Extract the atomic facts from the following agent step as a short bullet list. One fact "
          "per line, no commentary.\n\n")


def ans_prompt(task, payload, q):
    return ("You are answering a question about a completed agent trajectory, using the provided "
            "evidence. Answer concisely and specifically; cite exact values.\n\n"
            f"Task: {task}\n\nEvidence:\n{payload}\n\nQuestion: {q}\nAnswer:")


def judge_prompt(q, gold, cand):
    return (f"You grade answers. Question: {q}\nReference answer: {gold}\nCandidate answer: {cand}\n\n"
            "Is the candidate correct and consistent with the reference? Reply exactly one word: yes or no.")


class Gen:
    def __init__(self, llm, bs=8):
        self.llm = llm
        self.tok = llm.tok
        self.model = llm.model
        self.bs = bs
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @torch.no_grad()
    def batch(self, prompts, max_tokens=160):
        outs = []
        for i in range(0, len(prompts), self.bs):
            b = prompts[i:i + self.bs]
            texts = [self.tok.apply_chat_template(
                        [{"role": "user", "content": p}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False) for p in b]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.llm.max_ctx, add_special_tokens=False).to(self.model.device)
            g = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            for j in range(len(b)):
                new = g[j][enc.input_ids.shape[1]:]
                outs.append(_THINK.sub("", self.tok.decode(new, skip_special_tokens=True)).strip())
            print(f"    gen {min(i + self.bs, len(prompts))}/{len(prompts)}", flush=True)
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--max_ep", type=int, default=300)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only this many items (self-test)")
    ap.add_argument("--out", default="./out/payhf.json")
    ap.add_argument("--ans_out", default="./out/payhf_ans.jsonl")
    args = ap.parse_args()

    eps = load_ama(args.data)
    eps.sort(key=lambda e: e["id"])
    eps = [e for i, e in enumerate(eps) if i % args.nshards == args.shard][:args.max_ep]

    items = []
    for e in eps:
        n = 0
        for qa in e["qa"]:
            if n >= args.max_qa:
                break
            turns = [t for t in cited_turns(qa["question"]) if t in e["by_turn"]]
            if not turns:
                continue
            ev = "\n\n".join(f"<step {t}>\n{e['by_turn'][t]}" for t in turns)
            items.append({"task": e["task"], "ev": ev, "q": qa["question"],
                          "gold": qa.get("answer", "") or "", "qt": qa.get("type", "?")})
            n += 1
    if args.smoke:
        items = items[:args.smoke]
    print(f"shard {args.shard}/{args.nshards}: {len(eps)} eps -> {len(items)} step-citing oracle QA",
          flush=True)

    llm = HFBackend(max_ctx=8192)
    llm.warmup()
    gen = Gen(llm, bs=args.bs)

    # phase 1+2: summaries and facts over UNIQUE evidences
    evs = sorted({it["ev"] for it in items})
    print(f"  [summaries] {len(evs)} unique evidences", flush=True)
    summ = dict(zip(evs, gen.batch([SUM_P + e for e in evs], max_tokens=160)))
    print(f"  [facts] {len(evs)} unique evidences", flush=True)
    fact = dict(zip(evs, gen.batch([FACT_P + e for e in evs], max_tokens=200)))

    arms = ["verbatim", "summary", "facts"]
    payloads = {"verbatim": lambda it: it["ev"], "summary": lambda it: summ[it["ev"]],
                "facts": lambda it: fact[it["ev"]]}

    # phase 3: answers (arm-major so we can report progress per arm)
    flat = [(a, it) for a in arms for it in items]
    print(f"  [answers] {len(flat)} (arms x items)", flush=True)
    answers = gen.batch([ans_prompt(it["task"], payloads[a](it), it["q"]) for a, it in flat],
                        max_tokens=160)

    # phase 4: judge
    print(f"  [judge] {len(flat)}", flush=True)
    verds = gen.batch([judge_prompt(it["q"], it["gold"], ans)
                       for (a, it), ans in zip(flat, answers)], max_tokens=6)

    acc = {a: 0 for a in arms}
    tot = {a: 0 for a in arms}
    byqt = {a: defaultdict(lambda: [0, 0]) for a in arms}
    ans_rows = []
    for (a, it), ans, v in zip(flat, answers, verds):
        ok = v.strip().lower().startswith("y")
        acc[a] += ok
        tot[a] += 1
        byqt[a][it["qt"]][0] += 1
        byqt[a][it["qt"]][1] += ok
        ans_rows.append({"q": it["q"], "gold": it["gold"], "qt": it["qt"], "arm": a, "ans": ans})

    with open(args.ans_out, "w") as f:
        for r in ans_rows:
            f.write(json.dumps(r) + "\n")
    json.dump({"shard": args.shard, "acc": acc, "tot": tot,
               "byqt": {a: {k: v for k, v in byqt[a].items()} for a in arms}},
              open(args.out, "w"), indent=2)

    print("\n" + "=" * 64)
    print(f"PAYLOAD(HF) shard {args.shard} ({tot[arms[0]]} oracle QA, Qwen3-32B HF reader+judge)")
    print("=" * 64)
    for a in arms:
        line = "  ".join(f"{qt}:{byqt[a][qt][1]}/{byqt[a][qt][0]}" for qt in sorted(byqt[a]))
        print(f"  {a:9s}: {acc[a]}/{tot[a]} = {100*acc[a]/max(1,tot[a]):.1f}%   {line}")
    print(f"PAYHF_DONE shard {args.shard}")


if __name__ == "__main__":
    main()
