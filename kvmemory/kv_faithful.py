"""Stronger KV-faithfulness battery (review #1): go beyond first-token argmax.

Per query on a controlled synthetic trajectory, at two selection scales, compare KV sub-selection to:
  (a) its OWN oracle -- a full-prefill-then-mask-dropped decode (what sub-selection actually computes),
      over the WHOLE generated answer, plus first-position logit-KL and answer margins; and
  (b) a compact re-prefill of the selected text (a DIFFERENT computation), to quantify how far apart
      position-preserving reuse and compact re-prefill really are.

Reports, per scale: kv-vs-oracle first-token argmax, full-sequence exact-match, mean/median logit-KL,
and the oracle's answer margin distribution; and the same kv-vs-compact numbers. This turns "100%
first-token argmax" into a sequence-level, distributional faithfulness claim.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
        python -m kvmemory.kv_faithful --gens 48
"""
from __future__ import annotations

import argparse
import statistics as st

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import (QUERIES, build_trajectory, build_full_ids, kept_set,
                                      split_wrap_nothink)


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp, lq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return float((lp.exp() * (lp - lq)).sum())


def margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits, 2).values
    return float(top2[0] - top2[1])


@torch.no_grad()
def masked_oracle_decode(llm, header, segs, kept_ids, question, max_tokens):
    """Greedy-decode the full-prefill-then-mask-dropped oracle for `max_tokens` (the sequence-level
    version of first_logits_masked_full): every decode step attends over kept-prefix + question +
    generated, dropped positions masked, at original positions."""
    cache, spans, total = llm.prefill_full(segs, header)
    header_len = spans[0][0] if spans else 0
    keep_tok = list(range(header_len))
    for sid in sorted(kept_ids):
        keep_tok.extend(range(spans[sid][0], spans[sid][1]))
    base = torch.zeros(1, total, dtype=torch.long, device=llm.device)
    base[0, torch.tensor(keep_tok, device=llm.device)] = 1
    cur = llm._ids(question)
    L = cur.shape[1]
    mask = torch.cat([base, torch.ones(1, L, dtype=torch.long, device=llm.device)], 1)
    pos = torch.arange(total, total + L, device=llm.device).unsqueeze(0)
    cpos = torch.arange(total, total + L, device=llm.device)
    first, out_ids = None, []
    for step in range(max_tokens):
        o = llm.model(input_ids=cur, past_key_values=cache, use_cache=True,
                      position_ids=pos, cache_position=cpos, attention_mask=mask)
        lg = o.logits[0, -1].float()
        if first is None:
            first = lg
        nxt = int(lg.argmax())
        out_ids.append(nxt)
        if nxt in llm.eos_ids:
            break
        cur = torch.tensor([[nxt]], device=llm.device)
        p = total + L + step
        pos = torch.tensor([[p]], device=llm.device)
        cpos = torch.tensor([p], device=llm.device)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=llm.device)], 1)
    return out_ids, first


@torch.no_grad()
def compact_decode(llm, ids, max_tokens):
    """Greedy decode a compact re-prefill of the selected text (a different computation)."""
    text, _, _ = llm._greedy(DynamicCache(), 0, ids, max_tokens)
    cache = DynamicCache()
    pos = torch.arange(ids.shape[1], device=llm.device)
    o = llm.model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos,
                  attention_mask=torch.ones_like(ids))
    return text, o.logits[0, -1].float()


def run_scale(llm, head, tail, n_steps, n_distract, gens, tag):
    seg_texts = build_trajectory(n_steps)  # each element is already a formatted step string
    header = (head + "You are reviewing a completed agent trajectory. Answer precisely.\n\nTrajectory:\n")
    full_cache, spans, total = llm.prefill_full(seg_texts, header)
    header_len = spans[0][0]
    queries = [q for q in QUERIES if q[2] < n_steps]

    o_arg = o_seq = c_arg = c_seq = 0
    o_kl, c_kl, margins = [], [], []
    n = 0
    for qname, gold, gstep in queries:
        kept = kept_set(n_steps, gstep, hot=4, n_distract=n_distract)
        q = f"\n\nQuestion: {qname}\nAnswer concisely:" + tail
        n += 1
        # kv sub-selection
        sub_cache, kpos = llm.subselect_cache(full_cache, spans, header_len, kept)
        kv_txt, _, _ = llm._greedy_pos(sub_cache, kpos, llm._ids(q), gens)
        kv_first = llm.first_logits_subselect(full_cache, spans, header_len, kept, q)
        # (a) oracle: full-prefill-then-mask, full sequence
        or_ids, or_first = masked_oracle_decode(llm, header, seg_texts, kept, q, gens)
        or_txt = llm.tok.decode(or_ids, skip_special_tokens=True)
        # (b) compact re-prefill of the selected text
        cids = build_full_ids(llm, header, seg_texts, kept, q)
        cp_txt, cp_first = compact_decode(llm, cids, gens)

        o_arg += int(kv_first.argmax()) == int(or_first.argmax())
        c_arg += int(kv_first.argmax()) == int(cp_first.argmax())
        o_seq += kv_txt.strip() == or_txt.strip()
        c_seq += kv_txt.strip() == cp_txt.strip()
        o_kl.append(kl(or_first, kv_first))
        c_kl.append(kl(cp_first, kv_first))
        margins.append(margin(or_first))

    print(f"\n[{tag}] n_steps={n_steps} ({total} tok) n_distract={n_distract} "
          f"(~{len(kept)} spans kept) | {n} queries, {gens}-token generations", flush=True)
    print(f"  kv vs ORACLE (full-prefill+mask -- what kv computes):")
    print(f"     first-token argmax : {o_arg}/{n}")
    print(f"     FULL-SEQUENCE match: {o_seq}/{n}   <-- beyond first token")
    print(f"     first-pos logit-KL : mean {st.mean(o_kl):.4f}  median {st.median(o_kl):.4f}  "
          f"max {max(o_kl):.4f}   [~0 = faithful]")
    print(f"  kv vs COMPACT re-prefill (a DIFFERENT computation):")
    print(f"     first-token argmax : {c_arg}/{n}")
    print(f"     FULL-SEQUENCE match: {c_seq}/{n}")
    print(f"     first-pos logit-KL : mean {st.mean(c_kl):.4f}  median {st.median(c_kl):.4f}   "
          f"[large = position-preserving != compact]")
    print(f"  oracle answer MARGIN (top1-top2 logit): mean {st.mean(margins):.2f}  "
          f"min {min(margins):.2f}   [low-margin cases are the hard ones]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=48)
    ap.add_argument("--tag", default="qwen3-8b")
    args = ap.parse_args()
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    # normal selection, then LARGE selection (many distractor spans = long selected evidence)
    run_scale(llm, head, tail, n_steps=48, n_distract=3, gens=args.gens, tag=args.tag)
    run_scale(llm, head, tail, n_steps=400, n_distract=80, gens=args.gens, tag=args.tag)
    print("FAITHFUL_DONE", flush=True)


if __name__ == "__main__":
    main()
