"""Isolate the CAUSE of VeraKV's Software temporal-scan failure (acc ~0.15).

Hypothesis space:
  (R) retrieval/coverage : the answer-bearing step is simply not surfaced to the reader.
  (S) scan/verification  : even given the step, the model can't confirm "first/last" without a
                           complete view of the trajectory.
  (X) reasoning/judge    : the model can't answer even with the gold step in hand.

We hold the reader (Qwen3-8B) fixed and vary ONLY the evidence, on the 74 temporal-scan Software
questions whose gold answer names a clean step number:
  1. selection        : deployed context = recent(hot=4) + lexical top-k(5) verbatim.
  2. sel+goldstep     : selection PLUS the gold step (+-1 neighbor) verbatim.        (isolates R)
  3. fullindex+gold   : EVERY step as a 1-line extractive gist (complete index) + gold verbatim. (isolates S)

  sel+goldstep >> selection  => R (retrieval is the bottleneck; ensure the addressed step is fetched)
  fullindex+gold >> sel+gold => S (needs a complete index to verify first/last)
  all three low              => X (reasoning/judge limit; a 32B re-check is warranted)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.sw_mechanism --data ./data/ama_test.jsonl --out ./out/sw_mechanism_8b.json
    (32B reader: SPRAG_MODEL_PATH=/path/to/Qwen3-32B SPRAG_DEVICE_MAP=auto, several GPUs visible.)
"""
from __future__ import annotations
import argparse, json, os, re, sys
import torch
from transformers import DynamicCache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink


def norm(s): return re.sub(r"\s+", " ", s.lower().strip())


def judge(llm, head, tail, q, gold, ans):
    body = ("You are grading an answer to a question about an agent trajectory.\n"
            f"Question: {q}\nReference answer: {gold}\nCandidate answer: {ans}\n\n"
            "Is the candidate answer correct and consistent with the reference answer? "
            "Reply with exactly one word: yes or no.")
    out, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(head + body + tail), 4)
    return out.strip().lower().startswith("y")


def is_ts(q):
    q = q.lower()
    return bool(re.search(r"\bfirst\b|\blast\b|earliest|latest|which step (caused|had|showed|made|resulted)", q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_q", type=int, default=74)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ans_tokens", type=int, default=48)
    ap.add_argument("--out", default="./out/sw_mechanism.json")
    args = ap.parse_args()

    llm = HFBackend(); llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    eps = [e for e in load_episodes(args.data, max_tokens=200000) if e.domain == "SOFTWARE"]

    items = []
    for e in eps:
        for qa in e.qa:
            if not is_ts(qa["question"]):
                continue
            m = re.search(r"step\s+(\d+)", str(qa.get("answer", "")).lower())
            if m:
                items.append((e, qa, int(m.group(1))))
    items = items[:args.max_q]
    print(f"Software temporal-scan mechanism test: {len(items)} questions, Qwen3-8B reader+judge", flush=True)

    cond = {"selection": [0, 0], "sel+goldstep": [0, 0], "fullindex+gold": [0, 0]}
    gold_in_selection = 0
    rows = []
    for qi, (ep, qa, G) in enumerate(items):
        segs = ep.segments
        n_seg = len(segs)
        q = qa["question"]; gold = str(qa.get("answer", ""))
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(segs) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(segs)}
        turn2idx = {s.turn: i for i, s in enumerate(segs)}
        picked = router.select(q, old, args.k)
        sel = set(hot_idx) | {id2idx[p] for p in picked if p in id2idx}
        gidx = turn2idx.get(G)
        if gidx is None:
            continue
        gold_here = int(gidx in sel)
        gold_in_selection += gold_here

        def verb(idxs):
            return "".join(f"<step {segs[i].turn}>\n{segs[i].text}\n" for i in sorted(idxs))

        # 1. selection
        c1 = header + verb(sel) + qtext
        # 2. selection + gold step (+-1)
        goldset = {gidx} | {gidx - 1, gidx + 1} & set(range(n_seg))
        c2 = header + verb(sel | goldset) + qtext
        # 3. complete 1-line index + gold verbatim
        idx_lines = [f"<step {s.turn}> {re.sub(chr(10), ' ', s.text[:140])}" for s in segs]
        idx_lines.append(f"\n<step {segs[gidx].turn} FULL>\n{segs[gidx].text}")
        c3 = header + "\n".join(idx_lines) + "\n" + qtext

        res = {}
        for name, ctx in [("selection", c1), ("sel+goldstep", c2), ("fullindex+gold", c3)]:
            ans, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(ctx), args.ans_tokens)
            ok = int(judge(llm, head, tail, q, gold, ans))
            cond[name][0] += 1; cond[name][1] += ok
            res[name] = (ok, norm(ans)[:60])
        rows.append({"ep": ep.episode_id, "G": G, "gold_in_sel": gold_here, "res": res})
        json.dump({"cond": cond, "gold_in_selection": gold_in_selection, "n": len(rows),
                   "rows": rows}, open(args.out, "w"), indent=2)
        if qi % 5 == 0:
            def a(c): return cond[c][1] / max(1, cond[c][0])
            print(f"  [{qi+1}/{len(items)}] gold_in_sel {gold_in_selection}/{len(rows)} | "
                  f"sel {a('selection'):.3f} +gold {a('sel+goldstep'):.3f} "
                  f"fullidx {a('fullindex+gold'):.3f}", flush=True)

    print("\n" + "=" * 70)
    print(f"SOFTWARE TEMPORAL-SCAN MECHANISM ({len(rows)} Qs, Qwen3-8B)")
    print("=" * 70)
    print(f"  gold step already in deployed selection: {gold_in_selection}/{len(rows)} "
          f"({100*gold_in_selection/max(1,len(rows)):.1f}%)")
    for name in ["selection", "sel+goldstep", "fullindex+gold"]:
        n_, ok_ = cond[name]
        print(f"  {name:<16} acc = {ok_}/{n_} = {ok_/max(1,n_):.3f}")
    print("SWMECH_DONE")


if __name__ == "__main__":
    main()
