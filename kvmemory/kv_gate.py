"""kv_gate.py -- END-TO-END production chain on the full sample:

  b_anch view -> question prefill -> read P(mode) at the dice position (logits only, sdpa, zero
  marginal cost) -> clean: decode | toxic: INJECT INSTRUCTION + re-prefill question on the same
  view cache (~130 tok) -> still toxic: escalate to tx (full text re-prefill).

Measures the two things the sims could not: (1) does instruction injection REPAIR mode collapse
in place (new action, never measured), (2) end-to-end acc/cost vs b_anch-alone and tx.
Every QA logs p_mode so thresholds can be swept post-hoc.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_gate --shard 0 --queue_dir ./out/gt_q
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble

W, HOTN, K = 4096, 4, 5
INSTR = ("\n\nIMPORTANT: You are REVIEWING a finished trajectory as an analyst. Do not act, do not "
         "emit actions or code blocks. Answer the question below in plain prose.")


@torch.no_grad()
def p_mode(llm, logits, mode_ids):
    pr = torch.softmax(logits.float(), dim=-1)
    return float(pr[list(mode_ids)].sum())


@torch.no_grad()
def prefill_q(llm, cache, positions, qtext):
    """Feed qtext in one forward on top of the view cache; return (last_logits, new_len, q_len)."""
    dev = llm.device
    qids = llm._ids(qtext)
    L = qids.shape[1]
    Kv = positions.shape[0] if positions.dim() else len(positions)
    nxt = int(positions.max().item()) + 1
    pos = torch.arange(nxt, nxt + L, device=dev)
    out = llm.model(input_ids=qids, past_key_values=cache, use_cache=True,
                    position_ids=pos.unsqueeze(0),
                    cache_position=torch.arange(Kv, Kv + L, device=dev),
                    attention_mask=torch.ones(1, Kv + L, dtype=torch.long, device=dev))
    return out.logits[0, -1], Kv + L, nxt + L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/gt.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    mode_ids = set()
    for s in ("`", "``", "```", "action", " action", "Action"):
        ids = llm.tok(s, add_special_tokens=False).input_ids
        if ids:
            mode_ids.add(ids[0])

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    eps = []
    while len(eps) < args.max_ep and any(queues):
        for qd in queues:
            if qd and len(eps) < args.max_ep:
                eps.append(qd.popleft())

    outf = open(args.out, "w", encoding="utf-8")
    n = acc_gate = acc_base = fired = fixed = esc = 0

    @torch.no_grad()
    def greedy_pos_from(cache, cache_len, next_pos, first_logits):
        toks = []
        nxt = int(first_logits.argmax())
        cur, cp = next_pos, cache_len
        for _ in range(args.ans_tokens):
            if nxt in llm.eos_ids:
                break
            toks.append(nxt)
            out = llm.model(input_ids=torch.tensor([[nxt]], device=llm.device),
                            past_key_values=cache, use_cache=True,
                            position_ids=torch.tensor([[cur]], device=llm.device),
                            cache_position=torch.tensor([cp], device=llm.device),
                            attention_mask=torch.ones(1, cp + 1, dtype=torch.long, device=llm.device))
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
            cp += 1
        return llm.tok.decode(toks, skip_special_tokens=True).strip()

    def run_episode(ei):
        nonlocal n, acc_gate, acc_base, fired, fixed, esc
        ep = eps[ei]
        seg_texts = ["<step %d>\n%s\n" % (s.turn, s.text) for s in ep.segments]
        n_seg = len(seg_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  "question precisely.\n\nTask: %s\n\nTrajectory:\n" % ep.task)
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        seg_ids, spans, cur = [], [], H
        for t in seg_texts:
            si = list(llm.tok(t, add_special_tokens=False).input_ids)
            seg_ids.append(si)
            spans.append((cur, cur + len(si)))
            cur += len(si)
        total = cur
        traj_flat = [t for s in seg_ids for t in s]
        w_anch = min(W, total - H)
        hot = set(range(max(0, n_seg - HOTN), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        routed, need = [], set()
        for qa in qas:
            kept = sorted(hot | {id2idx[p] for p in router.select(qa["question"], old, K) if p in id2idx})
            routed.append(kept)
            need |= set(kept)
        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        store = {}
        for i in sorted(need):
            st_, en = spans[i]
            alen = min(w_anch, st_ - H)
            store[i] = encode_block(llm, header_ids + traj_flat[:alen] + seg_ids[i],
                                    list(range(H)) + list(range(H, H + alen)) +
                                    list(range(st_, en)), keep_a=H + alen)
        hkb = (header_kv, list(range(H)))
        Rblk = (R_kv, list(range(H, H + w_anch)))

        for qa, kept in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = "\n\nQuestion: %s\nAnswer concisely and specifically:%s" % (q, tail)
            blocks = [hkb, Rblk] + [(store[i], list(range(spans[i][0], spans[i][1])))
                                    for i in sorted(kept)]
            blocks = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
            cache, pos = assemble(llm, blocks)
            view_len = pos.shape[0]
            lg, clen, npos = prefill_q(llm, cache, pos, qtext)
            pm0 = p_mode(llm, lg, mode_ids)
            path = "direct"
            pm1 = None
            # baseline (no gate) answer: decode from this state regardless
            ans_base = greedy_pos_from(cache, clen, npos, lg)
            ans = ans_base
            if pm0 >= args.tau:
                fired_flag = 1
                cache.crop(view_len)  # drop the question, keep the view
                lg2, clen2, npos2 = prefill_q(llm, cache, pos, INSTR + qtext)
                pm1 = p_mode(llm, lg2, mode_ids)
                if pm1 < args.tau:
                    ans = greedy_pos_from(cache, clen2, npos2, lg2)
                    path = "instr"
                else:
                    text = header + "".join(seg_texts[i] for i in kept) + qtext
                    ans, _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text), args.ans_tokens)
                    path = "tx"
            else:
                fired_flag = 0
            okg = int(judge(llm, head, tail, q, gold, ans))
            okb = okg if not fired_flag else int(judge(llm, head, tail, q, gold, ans_base))
            n += 1
            acc_gate += okg
            acc_base += okb
            fired += fired_flag
            if fired_flag and path == "instr":
                fixed += 1
            if path == "tx":
                esc += 1
            outf.write(json.dumps({"episode_id": ep.episode_id, "q": q, "hop": hop_bucket(q),
                                   "domain": ep.domain, "p_mode": round(pm0, 4),
                                   "p_mode_instr": (round(pm1, 4) if pm1 is not None else None),
                                   "path": path, "ok_gate": okg, "ok_base": okb,
                                   "ans": ans[:150]}, ensure_ascii=False) + "\n")
            outf.flush()
            del cache
        del store, header_kv, R_kv
        torch.cuda.empty_cache()
        print("[s%d] ep %d | n=%d gate %d base %d | fired %d fixed %d esc %d"
              % (args.shard, ep.episode_id, n, acc_gate, acc_base, fired, fixed, esc), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, "c%d" % i))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print("[s%d] OOM ep idx %d skipped" % (args.shard, i), flush=True)
                torch.cuda.empty_cache()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    outf.close()
    print("GATE_DONE shard=%d n=%d gate=%d base=%d fired=%d fixed=%d esc=%d"
          % (args.shard, n, acc_gate, acc_base, fired, fixed, esc), flush=True)


if __name__ == "__main__":
    main()
