"""LOCOMO under the public leaderboard protocol: gpt-4o-mini answer + gpt-4o-mini judge, kvmemory
retrieval. Mirrors kvmemory/locomo_eval.py but swaps the local vLLM server for gpt-4o-mini so the
number is comparable to the published Mem0/Zep LOCOMO table (paper Table "LOCOMO", left).

The paper's runs reached gpt-4o-mini through an OpenAI-compatible router. Configure any such
endpoint via the environment (no defaults):
  LOCOMO_API_BASE_URL   OpenAI-compatible base URL
  LOCOMO_API_KEY        its API key
  LOCOMO_API_MODEL      model id at that endpoint (default: gpt-4o-mini)

  PYTHONPATH=. python kvmemory/locomo_gpt4omini.py --arm retrieval --router lexical --limit 1   # smoke
  PYTHONPATH=. python kvmemory/locomo_gpt4omini.py --arm retrieval --router hybrid --force-answer  # J=0.704
  PYTHONPATH=. python kvmemory/locomo_gpt4omini.py --arm full --force-answer                      # J=0.756
  (hybrid needs SPRAG_EMBED_PATH -> a local Qwen3-Embedding-0.6B)
"""
import os, sys, json, re, argparse, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.locomo import load_locomo
from kvmemory.core import KVMemory, KVMemoryConfig
from kvmemory.components import AutoGistSummarizer, LexicalRouter, EmbeddingRouter, HybridRouter
from openai import OpenAI

# ---- gpt-4o-mini through an OpenAI-compatible endpoint (configured via the environment) ----
MODEL = os.environ.get("LOCOMO_API_MODEL", "gpt-4o-mini")
_client = None


def _api():
    global _client
    if _client is None:
        base, key = os.environ.get("LOCOMO_API_BASE_URL"), os.environ.get("LOCOMO_API_KEY")
        if not base or not key:
            raise RuntimeError("set LOCOMO_API_BASE_URL and LOCOMO_API_KEY (an OpenAI-compatible endpoint)")
        _client = OpenAI(base_url=base, api_key=key, timeout=120)
    return _client

# Mem0-style lenient LLM-judge protocol (same as kvmemory/locomo_eval.py)
ANSWER_SYS = ("You answer questions about a long, multi-session conversation between two friends, using "
              "ONLY the recalled dialogue provided. Each line is timestamped [date]. Answer in as few "
              "words as possible (a name, a date, a short phrase). For a date, give the exact date. If the "
              "recalled dialogue does not contain the answer, reply exactly: Not mentioned.")
# leaderboard-faithful: force a best-guess on answerable Qs (no abstention escape) — matches the Mem0/LOCOMO
# convention where cat1-4 are always attempted and cat5 adversarial is scored separately.
FORCE_ANSWER_SYS = ("You answer questions about a long, multi-session conversation between two friends, using "
              "the recalled dialogue provided. Each line is timestamped [date]. Always give your single best "
              "answer in as few words as possible (a name, a date, or a short phrase). For a date, give the "
              "exact date. Even if you are unsure, make your most likely guess from the dialogue.")
ANSWER_USER = "Recalled conversation:\n{ctx}\n\nQuestion: {q}\nShort answer:"
JUDGE_USER = ("Judge whether the PREDICTED answer is correct for the question, given the GOLD answer. Be "
              "lenient: accept paraphrases, synonyms, and any answer that contains the key fact; accept a "
              "date within ~2 weeks of the gold date. Reply with exactly ONE word: CORRECT or WRONG.\n\n"
              "Question: {q}\nGOLD answer: {gold}\nPREDICTED answer: {pred}\nVerdict:")
ABSTAIN = ("notmentioned","nomention","notinthe","notintheconversation","cannotdetermine","dontknow",
           "donotknow","noinformation","notavailable","notstated","notdiscussed","noanswer","unknown")

def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def chat(prompt, system=None, max_tokens=128, _tries=4):
    msgs = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":prompt}]
    for i in range(_tries):
        try:
            r = _api().chat.completions.create(model=MODEL, messages=msgs, temperature=0.0, max_tokens=max_tokens)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if i == _tries-1: raise
            time.sleep(1.5*(i+1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--arm", default="retrieval")
    ap.add_argument("--router", default="lexical")
    ap.add_argument("--hot-turns", type=int, default=4)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max-ctx", type=int, default=120000)     # gpt-4o-mini 128k → never trim LOCOMO
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force-answer", action="store_true", help="leaderboard-faithful: no abstention escape")
    ap.add_argument("--out", default="./out/locomo_4omini_result.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    SYS = FORCE_ANSWER_SYS if a.force_answer else ANSWER_SYS
    ct = lambda s: max(1, len(s)//4)                            # char-based token proxy (LOCOMO is small)

    def make_router():
        if a.router == "lexical": return LexicalRouter()
        import torch
        from kvmemory.embed import QwenEmbedder
        dev = os.environ.get("SPRAG_EMBED_DEVICE", "cpu")   # MPS gives NaN for this model; CPU-fp32 works
        emb = EmbeddingRouter(QwenEmbedder(os.environ.get("SPRAG_EMBED_PATH"), device=dev, dtype=torch.float32))
        if a.router == "embed": return emb
        return HybridRouter([LexicalRouter(), emb])

    summ, router = AutoGistSummarizer(), make_router()
    eps = load_locomo(a.data)
    if a.limit: eps = eps[:a.limit]

    def cap(ctx):
        return ctx if len(ctx) <= a.max_ctx*4 else ctx[-a.max_ctx*4:]

    def verbatim_ids(mem, q):
        mem._retier()
        hot = {s.seg_id for s in mem.segments if s.tier == "hot"}
        if mem.cfg.arm == "full": return {s.seg_id for s in mem.segments}
        if mem.cfg.arm == "recent" or not mem.router: return hot
        old = [s for s in mem.segments if s.tier != "hot"]
        return hot | set(mem.router.select(q, old, mem.cfg.k_rehydrate))

    def answer_and_judge(ep, qa, ctx, vids):
        q = qa["question"]; cat = qa.get("category")
        pred = chat(ANSWER_USER.format(ctx=ctx, q=q), system=SYS, max_tokens=96).strip()
        abst = any(x in _norm(pred) for x in ABSTAIN)
        rec = {"conv": ep.conv_id, "cat": cat, "q": q[:140], "pred": pred[:160], "abstained": abst,
               "evidence": qa.get("evidence") or []}
        if cat == 5:
            rec["gold"] = qa.get("adversarial_answer"); rec["correct"] = abst
        else:
            gold = qa.get("answer"); rec["gold"] = gold
            rec["correct"] = False if abst else \
                chat(JUDGE_USER.format(q=q, gold=gold, pred=pred), max_tokens=4).strip().lower().startswith("c")
            rec["evidence_in_ctx"] = bool(rec["evidence"]) and all(e in vids for e in rec["evidence"])
        return rec

    rows = []
    for ep in eps:
        mem = KVMemory(KVMemoryConfig(arm=a.arm, hot_turns=a.hot_turns, k_rehydrate=a.k),
                       summarizer=summ, router=router, count_tok=ct)
        for s in ep.segments: mem.append(s)
        prepared = [(qa, cap(mem.assemble(qa["question"])), verbatim_ids(mem, qa["question"])) for qa in ep.qa]
        with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
            futs = [ex.submit(answer_and_judge, ep, qa, ctx, vids) for qa, ctx, vids in prepared]
            for f in as_completed(futs): rows.append(f.result())
        print(f"conv {ep.conv_id}: {len(ep.qa)} QA done (total {len(rows)})", flush=True)
        json.dump({"arm": a.arm, "router": a.router, "rows": rows}, open(a.out, "w"))

    by = defaultdict(lambda: [0,0])
    for r in rows:
        if r["cat"] == 5: by["cat5_abstain"][0]+=r["correct"]; by["cat5_abstain"][1]+=1
        else:
            by[f"cat{r['cat']}"][0]+=r["correct"]; by[f"cat{r['cat']}"][1]+=1
            by["J(1-4)"][0]+=r["correct"]; by["J(1-4)"][1]+=1
    print(f"\n=== LOCOMO (gpt-4o-mini)  arm={a.arm} router={a.router} k={a.k}  ({len(eps)} convs) ===")
    for k in ["cat1","cat2","cat3","cat4","J(1-4)","cat5_abstain"]:
        h,n = by[k]
        if n: print(f"  {k:14s} {h/n:.4f}  ({h}/{n})")
    miss = [r for r in rows if r["cat"]!=5 and not r["correct"]]
    sel = sum(1 for r in miss if not r.get("evidence_in_ctx"))
    print(f"  -- miss attribution (n={len(miss)}): selection-fail {sel} ({100*sel/max(1,len(miss)):.0f}%) | "
          f"reasoning-fail {len(miss)-sel} ({100*(len(miss)-sel)/max(1,len(miss)):.0f}%)")
    print("LOCOMO_DONE", flush=True)

if __name__ == "__main__":
    main()
