"""KV side-channel probe on REAL AMA-Bench episodes.

Claim under test (paper: gathered KV is a *richer* realization of verbatim than a text re-encode):
a selected span's KV was computed IN SITU, attending to everything before it, so it carries
trajectory-conditioned context from turns that are NOT themselves selected. A compact text
re-prefill of the same selected spans cannot: it never sees the unselected turns.

Test: hold the SELECTED spans' text fixed; perturb only UNSELECTED upstream turns
(length-matched replacement, so span positions are preserved); re-gather the same selected
spans' KV and re-answer. Measure whether the gathered-KV output moves.

  * kv arm   : prefill trajectory, gather selected spans' KV at original positions, decode.
               -> perturbing unselected upstream CHANGES its inputs (attention saw those turns).
  * text arm : re-prefill header + selected spans (verbatim text) + question.
               -> INVARIANT to the perturbation by construction (it never included those turns).

Positive result = kv arm's first-token argmax / answer moves under upstream perturbation while
the text arm is provably unchanged: the KV carried unselected-context signal that text drops.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_sidechannel --data ./data/ama_test.jsonl \
        --max_tokens 16000 --max_ep 30 --max_qa 6 --k 5 --n_perturb 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


def length_matched_replacement(orig: str, donor: str) -> str:
    """A length-matched (char-for-char) replacement of `orig` built by tiling `donor`, so the
    perturbed segment keeps ~the same token count and downstream span positions are preserved.
    Content is destroyed; length is held."""
    if not donor:
        donor = "the agent then observed a different unrelated state and continued. "
    rep = (donor * (len(orig) // max(1, len(donor)) + 1))[:len(orig)]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_ep", type=int, default=30)
    ap.add_argument("--max_qa", type=int, default=6)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_perturb", type=int, default=3,
                    help="how many unselected upstream turns to corrupt per query")
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--gate_ep", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./out/kv_sidechannel.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

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
    print(f"KV side-channel on {len(eps)} real AMA episodes (<= {args.max_tokens} tok) "
          f"across {len(bydom)} domains, k={args.k} hot={args.hot} n_perturb={args.n_perturb}",
          flush=True)

    # counters
    n = 0
    kv_flip = 0            # gathered-KV first-token argmax changed under upstream perturbation
    kv_ans_change = 0      # gathered-KV full answer text changed
    tx_flip = 0            # text-arm first-token argmax changed (MUST be 0: control)
    tx_ans_change = 0      # text-arm answer changed (MUST be 0: control)
    gate_agree = gate_n = 0
    shifts_top = []        # |logit(orig-top) unperturbed - perturbed|
    shifts_linf = []       # ||l0 - l1||_inf
    by_dom = {}            # domain -> [n, kv_flip, kv_ans_change]
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
            qtype = qa.get("type", "?")
            picked = router.select(q, old, args.k)
            kept = sorted(hot_idx | {id2idx[p] for p in picked if p in id2idx})
            kept_set = set(kept)
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail

            # unselected turns strictly upstream of the last selected span (they influence a kept KV)
            last_kept = max(kept)
            upstream = [i for i in range(n_seg) if i not in kept_set and i < last_kept]
            if not upstream:
                continue
            targets = rng.sample(upstream, min(args.n_perturb, len(upstream)))

            # ---- unperturbed ----
            l0 = llm.first_logits_subselect(full_cache, spans, header_len, kept, qtext)
            sub0, kpos0 = llm.subselect_cache(full_cache, spans, header_len, kept)
            a0, _, _ = llm._greedy_pos(sub0, kpos0, llm._ids(qtext), args.ans_tokens)
            text0 = header + "".join(segment_texts[i] for i in kept) + qtext
            lt0 = first_logits_on_ids(llm, llm._ids(text0))
            at0, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text0), args.ans_tokens)

            # GATE: gathered KV must equal full-prefill-then-mask oracle (faithfulness), unperturbed
            if do_gate:
                masked = first_logits_masked_full(llm, header, segment_texts, kept, qtext)
                gate_agree += int(l0.argmax()) == int(masked.argmax())
                gate_n += 1

            # ---- perturbed: corrupt only unselected upstream turns, length-matched ----
            ptexts = list(segment_texts)
            for i in targets:
                donor = segment_texts[(i + n_seg // 2 + 1) % n_seg]
                ptexts[i] = length_matched_replacement(segment_texts[i], donor)
            pcache, pspans, _ = llm.prefill_full(ptexts, header)
            phlen = pspans[0][0]
            l1 = llm.first_logits_subselect(pcache, pspans, phlen, kept, qtext)
            psub, pkpos = llm.subselect_cache(pcache, pspans, phlen, kept)
            a1, _, _ = llm._greedy_pos(psub, pkpos, llm._ids(qtext), args.ans_tokens)
            # text arm under the SAME perturbation (kept spans unchanged -> must be identical)
            text1 = header + "".join(ptexts[i] for i in kept) + qtext
            lt1 = first_logits_on_ids(llm, llm._ids(text1))
            at1, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text1), args.ans_tokens)

            # ---- metrics ----
            top0 = int(l0.argmax())
            kflip = int(top0 != int(l1.argmax()))
            kans = int(norm(a0) != norm(a1))
            tflip = int(int(lt0.argmax()) != int(lt1.argmax()))
            tans = int(norm(at0) != norm(at1))
            shift_top = float((l0[top0] - l1[top0]).abs())
            linf = float((l0 - l1).abs().max())

            n += 1
            kv_flip += kflip; kv_ans_change += kans
            tx_flip += tflip; tx_ans_change += tans
            shifts_top.append(shift_top); shifts_linf.append(linf)
            d = by_dom.setdefault(ep.domain, [0, 0, 0])
            d[0] += 1; d[1] += kflip; d[2] += kans
            del pcache, psub, sub0
            torch.cuda.empty_cache()

        del full_cache
        torch.cuda.empty_cache()
        rows.append({"episode_id": ep.episode_id, "domain": ep.domain, "n_seg": n_seg})
        sm = sum(shifts_top) / max(1, len(shifts_top))
        json.dump({"n": n, "kv_flip": kv_flip, "kv_ans_change": kv_ans_change,
                   "tx_flip": tx_flip, "tx_ans_change": tx_ans_change,
                   "gate_agree": gate_agree, "gate_n": gate_n,
                   "mean_shift_top": sm, "by_dom": by_dom, "rows": rows}, open(args.out, "w"), indent=2)
        print(f"  ep {ep.episode_id} {ep.domain} | n={n} | KV flip {kv_flip}/{n} "
              f"ansΔ {kv_ans_change}/{n} | TEXT flip {tx_flip}/{n} ansΔ {tx_ans_change}/{n} "
              f"| mean|Δlogit_top|={sm:.2f} | gate {gate_agree}/{gate_n}", flush=True)

    def pct(a, b):
        return 100.0 * a / max(1, b)
    med_top = sorted(shifts_top)[len(shifts_top) // 2] if shifts_top else 0.0
    med_linf = sorted(shifts_linf)[len(shifts_linf) // 2] if shifts_linf else 0.0
    print("\n" + "=" * 74)
    print(f"KV SIDE-CHANNEL ON REAL AMA-Bench ({n} QA, {len(eps)} episodes, Qwen3-8B)")
    print("=" * 74)
    print(f"  GATE (gathered KV == mask oracle, unperturbed): {gate_agree}/{gate_n} "
          f"({pct(gate_agree, gate_n):.1f}%)  [must be ~100%]")
    print(f"  KV arm  : first-token argmax FLIP under upstream perturbation: {kv_flip}/{n} "
          f"({pct(kv_flip, n):.1f}%);  answer changed: {kv_ans_change}/{n} ({pct(kv_ans_change, n):.1f}%)")
    print(f"  TEXT arm: first-token argmax FLIP (CONTROL, must be 0):        {tx_flip}/{n} "
          f"({pct(tx_flip, n):.1f}%);  answer changed: {tx_ans_change}/{n} ({pct(tx_ans_change, n):.1f}%)")
    print(f"  gathered-KV logit sensitivity: mean|Δlogit_top|={sum(shifts_top)/max(1,len(shifts_top)):.2f} "
          f"median={med_top:.2f} ; median ||Δlogits||_inf={med_linf:.2f}")
    print("  by domain (n / kv-flip / kv-ansΔ):")
    for dm, d in sorted(by_dom.items()):
        print(f"    {dm:<14} {d[0]} / {d[1]} / {d[2]}")
    print("SIDECHANNEL_DONE")


if __name__ == "__main__":
    main()
