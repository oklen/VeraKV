"""Self-contained validation of the position-preserving KV sub-selection path.

Every experiment in this release serves a chosen subset of a trajectory's cache rows, so the
sub-selection path itself has to be shown faithful before any result built on it means anything.
This runs on a real HF model over a controlled synthetic trajectory and exercises the HFBackend
methods the experiments use (prefill_full / subselect_cache / answer_from_cache / _greedy_pos).
It also exports `split_wrap_nothink`, the chat-template wrapper the other modules import.

Correctness is tested against the RIGHT oracles (comparing against a *compact* re-prefill would
be wrong, since position-preserving sub-selection legitimately differs from it):

  GATE-A (plumbing identity, confound-free): subselect(keep=ALL) vs a fresh full prefill.
    Same tokens, same positions, no gaps -> must match at first-token argmax (~bf16 noise only).
  GATE-B (faithful drop oracle): subselect(keep=subset) vs FULL-prefill + attention-mask the
    dropped tokens (same KV provenance, same original positions). Must match ~99%.
  INFO  (behavioral delta): subselect vs compact re-prefill of the kept text. Expected to differ
    (position-preserving vs compact + the kept tokens saw the dropped ones) — this is the paper's
    'downstream attention trace' signal, reported, NOT a pass/fail.

Then efficiency (kv_select vs re-prefill) and accuracy (do the answers contain the gold fact).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B PYTHONPATH=. \
        /path/to/python -m kvmemory.kv_select_smoke
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.llm_hf import HFBackend

# ---- synthetic trajectory: buried load-bearing facts among distractors ----
FACTS = {
    3:  ("open('config/region')",      "The deployment region is ap-southeast-1.", "deployment region", "ap-southeast-1"),
    7:  ("open('runbook/rollback')",   "The rollback token is RB-49217.",          "rollback token",     "RB-49217"),
    12: ("open('slo/canary')",         "The canary error budget is 0.8 percent.",  "canary error budget","0.8"),
    18: ("open('oncall/schedule')",    "The primary oncall is Priya Nair.",        "primary oncall",     "Priya Nair"),
    23: ("open('db/topology')",        "The database shard count is 64.",          "database shard count","64"),
    29: ("open('incident/1042')",      "The incident severity is SEV-2.",          "incident severity",  "SEV-2"),
}
DISTRACTORS = [
    ("search('logs')", "Scanned 12043 log lines; nothing conclusive."),
    ("grep('timeout')", "Found 37 timeout warnings in the gateway."),
    ("calc('rps')", "Average request rate is about 4200 rps."),
    ("note('todo')", "Need to re-check the cache hit ratio later."),
    ("search('deploys')", "Three deploys today: web, api, worker."),
    ("open('grafana/cpu')", "CPU sits around 60 percent across the fleet."),
    ("note('hyp')", "Hypothesis: a downstream dependency is flapping."),
    ("ping('healthz')", "Health endpoint returns 200 for all pods."),
]
# Filler keeps each step a realistic length so the per-query re-prefill cost (and thus the KV-reuse
# saving) is measurable, instead of a 15-token toy step.
FILLER = (" Context: this step was part of the routine incident-response workflow; the engineer "
          "recorded the observation, cross-checked it against the dashboards, confirmed metrics were "
          "within the expected band, and proceeded without escalating further at this point.")


def build_trajectory(n_steps=48):
    steps = []
    di = 0
    for i in range(n_steps):
        if i in FACTS:
            act, obs, _, _ = FACTS[i]
        else:
            act, obs = DISTRACTORS[di % len(DISTRACTORS)]
            di += 1
        steps.append(f"<step {i}> {act} -> {obs}{FILLER}")
    return steps


QUERIES = [(meta[2], meta[3], gold_step) for gold_step, meta in FACTS.items()]


def split_wrap_nothink(llm):
    sent = "<<<SPLITPOINT>>>"
    try:
        full = llm.tok.apply_chat_template(
            [{"role": "user", "content": sent}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = llm.tok.apply_chat_template(
            [{"role": "user", "content": sent}], tokenize=False, add_generation_prompt=True)
    head, tail = full.split(sent)
    return head, tail


def build_full_ids(llm, header_text, seg_texts, kept_ids, question_text):
    """Reconstruct the token ids the SAME way prefill_full does (header + each segment tokenized
    SEPARATELY, then concatenated, then the question). Tokenizing the joined string instead would
    merge BPE tokens across segment boundaries and make the 'fresh' oracle a different token
    sequence -> spuriously large Δlogit even when the KV machinery is correct."""
    ids = list(llm.tok(header_text, add_special_tokens=False).input_ids)
    for i in kept_ids:
        ids += llm.tok(seg_texts[i], add_special_tokens=False).input_ids
    ids += list(llm.tok(question_text, add_special_tokens=False).input_ids)
    return torch.tensor([ids], dtype=torch.long, device=llm.device)


@torch.no_grad()
def first_logits_on_ids(llm, ids):
    cache = DynamicCache()
    pos = torch.arange(ids.shape[1], device=llm.device)
    out = llm.model(input_ids=ids, past_key_values=cache, use_cache=True,
                    cache_position=pos, attention_mask=torch.ones_like(ids))
    return out.logits[0, -1].float()


@torch.no_grad()
def first_logits_masked_full(llm, header_text, segment_texts, kept_ids, question_text):
    """Faithful drop oracle: fresh FULL prefill, then decode the question masking the dropped
    tokens, at original positions (question at total). subselect must match this."""
    cache, spans, total = llm.prefill_full(segment_texts, header_text)
    header_len = spans[0][0] if spans else 0
    keep_tok = list(range(header_len))
    for sid in sorted(kept_ids):
        keep_tok.extend(range(spans[sid][0], spans[sid][1]))
    qids = llm._ids(question_text)
    L = qids.shape[1]
    mask = torch.zeros(1, total + L, dtype=torch.long, device=llm.device)
    mask[0, torch.tensor(keep_tok, device=llm.device)] = 1
    mask[0, total:] = 1
    pos = torch.arange(total, total + L, device=llm.device)
    cpos = torch.arange(total, total + L, device=llm.device)
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=pos.unsqueeze(0), cache_position=cpos, attention_mask=mask)
    return out.logits[0, -1].float()


def kept_set(n_steps, gold_step, hot=4, n_distract=3):
    """Deterministic kept set: the gold step + the hot tail + a few earlier distractors.
    Guarantees answerability so accuracy reflects KV fidelity, not routing."""
    hot_ids = list(range(n_steps - hot, n_steps))
    extra = [s for s in (2, 9, 15, 21, 27)
             if s < n_steps and s not in hot_ids and s != gold_step][:n_distract]
    kept = sorted(set([gold_step] + hot_ids + extra))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=48)
    ap.add_argument("--hot", type=int, default=4)
    ap.add_argument("--ans_tokens", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3, help="efficiency timing repeats")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)

    steps = build_trajectory(args.n_steps)
    header_text = (head + "You are reviewing a completed agent trajectory. Use it to answer.\n\n"
                   "Trajectory:\n")
    segment_texts = [s + "\n" for s in steps]
    # only ask about facts that actually fit in this trajectory length
    queries = [q for q in QUERIES if q[2] < len(segment_texts)]

    # one full prefill, reused across all queries (the kv_select ingestion)
    t0 = time.time()
    full_cache, spans, total = llm.prefill_full(segment_texts, header_text)
    t_prefill_full = time.time() - t0
    header_len = spans[0][0]
    raw_chars = len(header_text) + sum(len(s) for s in segment_texts)
    print(f"\ntrajectory: {len(steps)} steps, {total} tok, prefill_full {t_prefill_full:.2f}s "
          f"(header {header_len} tok)\n")

    # ---- GATE-A: plumbing identity (keep ALL) ----
    print("="*72)
    print("GATE-A  plumbing identity: subselect(keep=ALL) vs fresh full prefill")
    print("="*72)
    all_ids = list(range(len(segment_texts)))
    a_match, a_diffs = 0, []
    for qname, gold, gstep in queries:
        q = f"\n\nQuestion: What is the {qname}?\nAnswer:" + tail
        fresh = first_logits_on_ids(llm, build_full_ids(llm, header_text, segment_texts, all_ids, q))
        sub = llm.first_logits_subselect(full_cache, spans, header_len, all_ids, q)
        same = int(sub.argmax()) == int(fresh.argmax())
        a_match += same
        a_diffs.append(float((sub - fresh).abs().max()))
    print(f"  argmax match: {a_match}/{len(queries)} | "
          f"max|Δlogit| mean={sum(a_diffs)/len(a_diffs):.4f} worst={max(a_diffs):.4f}")

    # ---- GATE-B (faithful drop) + INFO (compact) ----
    print("\n" + "="*72)
    print("GATE-B  faithful drop: subselect(subset) vs full-prefill+masked  |  INFO vs compact")
    print("="*72)
    b_match, b_diffs = 0, []
    c_match, c_diffs = 0, []
    for qname, gold, gstep in queries:
        kept = kept_set(len(segment_texts), gstep, args.hot)
        q = f"\n\nQuestion: What is the {qname}?\nAnswer:" + tail
        sub = llm.first_logits_subselect(full_cache, spans, header_len, kept, q)
        masked = first_logits_masked_full(llm, header_text, segment_texts, kept, q)
        compact = first_logits_on_ids(llm, build_full_ids(llm, header_text, segment_texts, kept, q))
        b_same = int(sub.argmax()) == int(masked.argmax())
        c_same = int(sub.argmax()) == int(compact.argmax())
        b_match += b_same
        c_match += c_same
        b_diffs.append(float((sub - masked).abs().max()))
        c_diffs.append(float((sub - compact).abs().max()))
    print(f"  GATE-B (vs masked-full): argmax {b_match}/{len(queries)} | "
          f"max|Δlogit| mean={sum(b_diffs)/len(b_diffs):.4f} worst={max(b_diffs):.4f}")
    print(f"  INFO   (vs compact):     argmax {c_match}/{len(queries)} | "
          f"max|Δlogit| mean={sum(c_diffs)/len(c_diffs):.4f} worst={max(c_diffs):.4f}")

    # ---- accuracy + efficiency ----
    print("\n" + "="*72)
    print("ACCURACY + EFFICIENCY  (kv_select vs re-prefill, same kept set)")
    print("="*72)
    acc_sel, acc_rep = 0, 0
    sel_ttft, rep_ttft = [], []
    sel_tok_per_q, rep_tok_per_q = 0, 0
    for qname, gold, gstep in queries:
        kept = kept_set(len(segment_texts), gstep, args.hot)
        q = f"\n\nQuestion: What is the {qname}?\nAnswer:" + tail
        # kv_select: reuse the one full prefill, subselect + decode the question only
        sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
        ans_sel, ti_sel, _ = llm.answer_from_cache(sub_cache, kpos, q, args.ans_tokens)
        # re-prefill baseline: header + kept text + question from scratch
        rep_text = header_text + "".join(segment_texts[i] for i in kept) + q
        ans_rep, ti_rep, _ = llm._greedy(DynamicCache(), 0, llm._ids(rep_text), args.ans_tokens)
        sel_ttft.append(ti_sel)
        rep_ttft.append(ti_rep)
        sel_tok_per_q += llm.count_tok(q)                       # only the question is fresh
        rep_tok_per_q += llm.count_tok(rep_text)                # whole context re-prefilled
        ok_sel = gold.lower() in ans_sel.lower()
        ok_rep = gold.lower() in ans_rep.lower()
        acc_sel += ok_sel
        acc_rep += ok_rep
        print(f"  [{qname:20s}] gold={gold:14s} | kv_select={'OK ' if ok_sel else 'MISS'} "
              f"({ans_sel.strip()[:32]!r}) | reprefill={'OK ' if ok_rep else 'MISS'} | "
              f"TTFT {ti_rep*1000:.0f}->{ti_sel*1000:.0f}ms")

    nq = len(queries)
    mean_sel = sum(sel_ttft) / nq
    mean_rep = sum(rep_ttft) / nq
    print("\n" + "-"*72)
    print(f"CORRECTNESS GATE-A (identity):  argmax {a_match}/{nq}  "
          f"-> {'PASS' if a_match >= nq-1 else 'FAIL'}")
    print(f"CORRECTNESS GATE-B (drop):      argmax {b_match}/{nq}  "
          f"-> {'PASS' if b_match >= nq-1 else 'FAIL'}")
    print(f"ACCURACY:   kv_select {acc_sel}/{nq}  vs  re-prefill {acc_rep}/{nq}  "
          f"(KV selection must not cost accuracy)")
    print(f"EFFICIENCY: per-query TTFT  re-prefill {mean_rep*1000:.0f}ms -> "
          f"kv_select {mean_sel*1000:.0f}ms  = {mean_rep/max(1e-9,mean_sel):.1f}x faster")
    print(f"            one-time prefill_full {t_prefill_full:.2f}s amortized over the episode")
    print(f"            prefill tokens/query: re-prefill {rep_tok_per_q//nq} -> "
          f"kv_select {sel_tok_per_q//nq}  (selected spans served from cache, not re-prefilled)")
    be = t_prefill_full / max(1e-9, (mean_rep - mean_sel))
    print(f"            break-even: ~{be:.1f} queries/episode (full-prefill cost vs per-query saving)")
    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
