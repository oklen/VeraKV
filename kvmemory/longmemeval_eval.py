"""Evaluate kvmemory on LongMemEval (Wu et al., ICLR'25) — the THIRD benchmark family (long-context
chat-assistant memory). Each of the 500 questions carries its OWN ~115k-token haystack of ~48 chat
sessions; the answer is pinned to ~2 gold sessions (`answer_session_ids`). The haystack dwarfs the 32k
window, so the `full` arm must TRUNCATE — the budget under which routed selection should win (cf. §10.3).

Mirrors locomo_eval.py: arms `full` (all sessions verbatim, truncated to the window — the oracle) and
`kvmemory` / `retrieval` (recency hot window + router-rehydrated sessions), router lexical|embed|hybrid|
causal. Metric = a lenient Qwen3-32B judge (CORRECT/WRONG) — OUR judge, not the official GPT-4o, so the
apples-to-apples comparison is against our own `full` arm (as for LOCOMO). Abstention questions (id ends
"_abs", ~30) are scored SEPARATELY as an abstention rate (correct = the model says "I don't know"), the
recall-grounded-memory showcase. Using the gold `answer_session_ids`, misses split into SELECTION-fail
(gold session not rehydrated → better router fixes it) vs REASONING-fail (present, model still wrong).

  PYTHONPATH=. python kvmemory/longmemeval_eval.py --arm retrieval \
    --router hybrid --data ./data/longmemeval_s_cleaned.json --base-url http://localhost:8056/v1
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from kvmemory.longmemeval import load_longmemeval
from kvmemory.core import KVMemory, KVMemoryConfig
from kvmemory.components import AutoGistSummarizer, LexicalRouter, EmbeddingRouter, HybridRouter, MultiHopRouter

ANSWER_SYS = ("You are a personal assistant answering a question about the user, using ONLY the recalled "
              "chat sessions provided. Each session is timestamped [date]. Answer in as few words as "
              "possible (a name, a date, a number, a short phrase); for a date give the exact date. If the "
              "recalled sessions do not contain the answer, reply exactly: I don't know.")
ANSWER_USER = "Recalled chat sessions:\n{ctx}\n\nCurrent date: {qdate}\nQuestion: {q}\nShort answer:"
JUDGE_USER = ("Judge whether the PREDICTED answer is correct for the question, given the GOLD answer. Be "
              "lenient: accept paraphrases, synonyms, and any answer containing the key fact; for dates accept "
              "within ~3 days. Reply with exactly ONE word: CORRECT or WRONG.\n\n"
              "Question: {q}\nGOLD answer: {gold}\nPREDICTED answer: {pred}\nVerdict:")

ABSTAIN = ("idontknow", "donotknow", "dontknow", "notmentioned", "nomention", "notintheconversation",
           "cannotdetermine", "noinformation", "notavailable", "notstated", "notdiscussed", "noanswer",
           "unknown", "insufficientinformation", "notenoughinformation", "norelevant")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _strip_think(t):
    return (t.split("</think>")[-1] if "</think>" in t else t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/longmemeval_s_cleaned.json")
    ap.add_argument("--arm", default="retrieval")           # full | kvmemory | retrieval | recent | gist
    ap.add_argument("--router", default="hybrid")           # lexical | embed | hybrid | causal
    ap.add_argument("--base-url", default="http://localhost:8056/v1")
    ap.add_argument("--model", default="/path/to/Qwen3-32B")
    ap.add_argument("--hot-turns", type=int, default=4)     # 4 most-recent sessions kept verbatim
    ap.add_argument("--k", type=int, default=10)            # router-rehydrated old sessions
    ap.add_argument("--max-ctx", type=int, default=28000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)         # first N questions (smoke)
    ap.add_argument("--out", default="./out/lme_result.json")
    a = ap.parse_args()

    for k in [k for k in os.environ if k.lower().endswith("_proxy")]:
        os.environ.pop(k, None)
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800)
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
        if a.router == "causal":
            return MultiHopRouter(hyb, temporal=1, entity_bridge=True, name_steps=True)
        return hyb

    summ = AutoGistSummarizer()
    router = make_router()
    eps = load_longmemeval(a.data)
    if a.limit:
        eps = eps[:a.limit]

    def cap(ctx):
        ids = tok(ctx, add_special_tokens=False).input_ids
        return tok.decode(ids[-a.max_ctx:]) if len(ids) > a.max_ctx else ctx

    def rehydrated_ids(mem, q):
        mem._retier()
        if mem.cfg.arm == "full":
            # `full` = every session concatenated then truncated to the last max_ctx tokens (recency
            # suffix). Honest attribution counts only the sessions that SURVIVE that truncation — so a
            # dropped older evidence session reads as selection-fail (truncation), not reasoning-fail.
            surviving, used = set(), 0
            for s in reversed(mem.segments):
                used += mem.count_tok(s.text)
                if used > a.max_ctx and surviving:
                    break
                surviving.add(s.seg_id)
            return surviving
        hot = {s.seg_id for s in mem.segments if s.tier == "hot"}
        if mem.cfg.arm == "recent" or not mem.router:
            return hot
        old = [s for s in mem.segments if s.tier != "hot"]
        return hot | set(mem.router.select(q, old, mem.cfg.k_rehydrate))

    def answer_and_judge(ep, ctx, rids):
        q = ep.question
        pred = chat(ANSWER_USER.format(ctx=ctx, q=q, qdate=ep.question_date), system=ANSWER_SYS, max_tokens=96).strip()
        abst = any(x in _norm(pred) for x in ABSTAIN)
        rec = {"qid": ep.qid, "qtype": ep.qtype, "abstention_q": ep.is_abstention,
               "q": q[:160], "pred": pred[:160], "abstained": abst, "gold": ep.answer[:160]}
        if ep.is_abstention:                              # unanswerable → correct iff it abstains
            rec["correct"] = abst
        elif abst:
            rec["correct"] = False
            rec["evidence_in_ctx"] = bool(ep.evidence_ids) and all(e in rids for e in ep.evidence_ids)
        else:
            rec["correct"] = chat(JUDGE_USER.format(q=q, gold=ep.answer, pred=pred),
                                  max_tokens=4).strip().lower().startswith("c")
            rec["evidence_in_ctx"] = bool(ep.evidence_ids) and all(e in rids for e in ep.evidence_ids)
        return rec

    # PASS 1 (sequential): build memory + route + assemble (GPU-safe embed), capture rehydrated session ids
    prepared = []
    for ep in eps:
        mem = KVMemory(KVMemoryConfig(arm=a.arm, hot_turns=a.hot_turns, k_rehydrate=a.k),
                       summarizer=summ, router=router,
                       count_tok=lambda s: len(tok(s, add_special_tokens=False).input_ids))
        for s in ep.segments:
            mem.append(s)
        ctx = cap(mem.assemble(ep.question))
        prepared.append((ep, ctx, rehydrated_ids(mem, ep.question)))
        if len(prepared) % 50 == 0:
            print(f"prepared {len(prepared)}/{len(eps)}", flush=True)

    # PASS 2 (concurrent): answer + judge (server HTTP only)
    rows = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(answer_and_judge, ep, ctx, rids) for ep, ctx, rids in prepared]
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if n % 50 == 0:
                print(f"answered {n}/{len(futs)}", flush=True)
                json.dump({"arm": a.arm, "router": a.router, "rows": rows}, open(a.out, "w"))
    json.dump({"arm": a.arm, "router": a.router, "rows": rows}, open(a.out, "w"))

    # ---- report ----
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["abstention_q"]:
            by["abstention"][0] += r["correct"]; by["abstention"][1] += 1
        else:
            by[r["qtype"]][0] += r["correct"]; by[r["qtype"]][1] += 1
            by["ANSWERABLE"][0] += r["correct"]; by["ANSWERABLE"][1] += 1
    print(f"\n=== LongMemEval  arm={a.arm} router={a.router} k={a.k}  ({len(rows)} Qs) ===")
    order = ["single-session-user", "single-session-assistant", "single-session-preference",
             "multi-session", "temporal-reasoning", "knowledge-update", "ANSWERABLE", "abstention"]
    for key in order:
        h, n = by[key]
        if n:
            print(f"  {key:28s} {h/n:.4f}  ({h}/{n})")
    miss = [r for r in rows if not r["abstention_q"] and not r["correct"]]
    sel = sum(1 for r in miss if not r.get("evidence_in_ctx"))
    print(f"  -- miss attribution (answerable, n={len(miss)}): selection-fail {sel} "
          f"({100*sel/max(1,len(miss)):.0f}%) | reasoning-fail {len(miss)-sel} "
          f"({100*(len(miss)-sel)/max(1,len(miss)):.0f}%)")
    print("LME_DONE", flush=True)


if __name__ == "__main__":
    main()
