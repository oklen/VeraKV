"""Evaluate kvmemory on LOCOMO under the Mem0-"J" protocol (lenient LLM-judge binary CORRECT/WRONG over
cats 1-4), judge = a Qwen3-32B vLLM server (the same backbone the AMA-Bench harness used). Reports J per
category and overall (1-4); scores cat 5 (adversarial / unanswerable) SEPARATELY as an abstention rate
(the kvmemory abstention showcase — verbatim memory *knows* whether evidence exists, so it can correctly
say "Not mentioned"); and — using the gold `evidence` dia_ids — splits kvmemory's misses into
SELECTION-fail (evidence not in the rehydrated context → fixable by a better router) vs REASONING-fail
(evidence present, model still wrong). That clean split is what AMA-Bench's synthesized golds blocked.

Arms: `full` (every turn verbatim — LOCOMO convs ~26k tok fit the 32k window, the oracle) and `kvmemory`
(recency hot window + router-rehydrated verbatim + gist floor), with --router lexical|embed|hybrid.
Thinking is disabled on the server side (recall task; matches the no-think gpt-4o-mini judge Mem0 used).

  PYTHONPATH=. python kvmemory/locomo_eval.py --arm kvmemory --router hybrid \
    --data ./data/locomo10.json --base-url http://localhost:8056/v1 --model /path/to/Qwen3-32B
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from kvmemory.locomo import load_locomo
from kvmemory.core import KVMemory, KVMemoryConfig
from kvmemory.components import AutoGistSummarizer, LexicalRouter, EmbeddingRouter, HybridRouter, MultiHopRouter

ANSWER_SYS = ("You answer questions about a long, multi-session conversation between two friends, using "
              "ONLY the recalled dialogue provided. Each line is timestamped [date]. Answer in as few "
              "words as possible (a name, a date, a short phrase). For a date, give the exact date. If the "
              "recalled dialogue does not contain the answer, reply exactly: Not mentioned.")
ANSWER_USER = "Recalled conversation:\n{ctx}\n\nQuestion: {q}\nShort answer:"
JUDGE_USER = ("Judge whether the PREDICTED answer is correct for the question, given the GOLD answer. Be "
              "lenient: accept paraphrases, synonyms, and any answer that contains the key fact; accept a "
              "date within ~2 weeks of the gold date. Reply with exactly ONE word: CORRECT or WRONG.\n\n"
              "Question: {q}\nGOLD answer: {gold}\nPREDICTED answer: {pred}\nVerdict:")

ABSTAIN = ("notmentioned", "nomention", "notinthe", "notintheconversation", "cannotdetermine", "dontknow",
           "donotknow", "noinformation", "notavailable", "notstated", "notdiscussed", "noanswer", "unknown")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _strip_think(t):
    return (t.split("</think>")[-1] if "</think>" in t else t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--arm", default="kvmemory")           # full | kvmemory | recent | gist | retrieval
    ap.add_argument("--router", default="lexical")          # lexical | embed | hybrid
    ap.add_argument("--base-url", default="http://localhost:8056/v1")
    ap.add_argument("--model", default="/path/to/Qwen3-32B")
    ap.add_argument("--hot-turns", type=int, default=4)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--max-ctx", type=int, default=28000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)         # first N conversations (smoke)
    ap.add_argument("--out", default="./out/locomo_result.json")
    a = ap.parse_args()

    for k in [k for k in os.environ if k.lower().endswith("_proxy")]:
        os.environ.pop(k, None)
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)
    NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

    def chat(prompt, system=None, max_tokens=128):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        r = client.chat.completions.create(model=a.model, messages=msgs, temperature=0.0,
                                           max_tokens=max_tokens, extra_body=NOTHINK)
        return _strip_think(r.choices[0].message.content or "")

    def make_router():
        if a.router == "lexical":
            return LexicalRouter()
        from kvmemory.embed import QwenEmbedder
        emb = EmbeddingRouter(QwenEmbedder(os.environ.get("SPRAG_EMBED_PATH")))
        if a.router == "embed":
            return emb
        hyb = HybridRouter([LexicalRouter(), emb])
        if a.router == "causal":      # multi-hop expansion over the hybrid seeds (entity bridges carry dialogue)
            return MultiHopRouter(hyb, temporal=1, entity_bridge=True, name_steps=True)
        return hyb

    summ = AutoGistSummarizer()
    router = make_router()
    eps = load_locomo(a.data)
    if a.limit:
        eps = eps[:a.limit]

    def cap(ctx):
        ids = tok(ctx, add_special_tokens=False).input_ids
        return tok.decode(ids[-a.max_ctx:]) if len(ids) > a.max_ctx else ctx

    def verbatim_ids(mem, q):
        mem._retier()
        hot = {s.seg_id for s in mem.segments if s.tier == "hot"}
        if mem.cfg.arm in ("full",):
            return {s.seg_id for s in mem.segments}
        if mem.cfg.arm == "recent" or not mem.router:
            return hot
        old = [s for s in mem.segments if s.tier != "hot"]
        return hot | set(mem.router.select(q, old, mem.cfg.k_rehydrate))

    def answer_and_judge(ep, qa, ctx, vids):
        q = qa["question"]
        cat = qa.get("category")
        pred = chat(ANSWER_USER.format(ctx=ctx, q=q), system=ANSWER_SYS, max_tokens=96).strip()
        abst = any(x in _norm(pred) for x in ABSTAIN)
        rec = {"conv": ep.conv_id, "cat": cat, "q": q[:140], "pred": pred[:160], "abstained": abst,
               "evidence": qa.get("evidence") or []}
        if cat == 5:                                   # adversarial / unanswerable → correct iff it abstains
            rec["gold"] = qa.get("adversarial_answer")
            rec["correct"] = abst
        else:
            gold = qa.get("answer")
            rec["gold"] = gold
            if abst:
                rec["correct"] = False
            else:
                rec["correct"] = chat(JUDGE_USER.format(q=q, gold=gold, pred=pred), max_tokens=4).strip().lower().startswith("c")
            ev = rec["evidence"]
            rec["evidence_in_ctx"] = bool(ev) and all(e in vids for e in ev)
        return rec

    rows = []
    for ep in eps:
        mem = KVMemory(KVMemoryConfig(arm=a.arm, hot_turns=a.hot_turns, k_rehydrate=a.k),
                       summarizer=summ, router=router, count_tok=lambda s: len(tok(s, add_special_tokens=False).input_ids))
        for s in ep.segments:
            mem.append(s)
        # PASS 1 (sequential): route + assemble (GPU-safe), capture verbatim ids for attribution
        prepared = []
        for qa in ep.qa:
            ctx = cap(mem.assemble(qa["question"]))
            prepared.append((qa, ctx, verbatim_ids(mem, qa["question"])))
        # PASS 2 (concurrent): answer + judge (server HTTP only)
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(answer_and_judge, ep, qa, ctx, vids) for qa, ctx, vids in prepared]
            for f in as_completed(futs):
                rows.append(f.result())
        done = sum(1 for r in rows)
        print(f"conv {ep.conv_id}: {len(ep.qa)} QA done (total {done})", flush=True)
        json.dump({"arm": a.arm, "router": a.router, "rows": rows}, open(a.out, "w"))

    # ---- report ----
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["cat"] == 5:
            by["cat5_abstain"][0] += r["correct"]; by["cat5_abstain"][1] += 1
        else:
            by[f"cat{r['cat']}"][0] += r["correct"]; by[f"cat{r['cat']}"][1] += 1
            by["J(1-4)"][0] += r["correct"]; by["J(1-4)"][1] += 1
    print(f"\n=== LOCOMO  arm={a.arm} router={a.router}  ({len(eps)} convs) ===")
    for k in ["cat1", "cat2", "cat3", "cat4", "J(1-4)", "cat5_abstain"]:
        h, n = by[k]
        if n:
            print(f"  {k:14s} {h/n:.4f}  ({h}/{n})")
    miss = [r for r in rows if r["cat"] != 5 and not r["correct"]]
    sel = sum(1 for r in miss if not r.get("evidence_in_ctx"))
    print(f"  -- miss attribution (cats1-4, n={len(miss)}): selection-fail {sel} ({100*sel/max(1,len(miss)):.0f}%) "
          f"| reasoning-fail {len(miss)-sel} ({100*(len(miss)-sel)/max(1,len(miss)):.0f}%)")
    print("LOCOMO_DONE", flush=True)


if __name__ == "__main__":
    main()
