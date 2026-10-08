"""kv_klprobe.py -- distributional autopsy of the residual: does cached serving (b_hot) move
the answer DISTRIBUTION far from full recomputation, or only flip near-tied argmaxes?

Text-level autopsy (official K=12, tx-right/b_hot-wrong cell, n=257) already showed 98% fluent
wrong answers, 0 garbles -- the loss is wrong DETAILS, not broken language. This probe asks the
distribution-level question on the 8B mechanism subset, per QA, three servings of the SAME
reader: b_hot (cache), tx (routed text re-prefill, contiguous positions), full (entire episode
re-prefill = the reference distribution).

Measurements per QA and serving:
  - greedy answer + in-run judge (gives the outcome cells)
  - teacher-forced GOLD answer: per-token logprob (mean & sum)
  - first-answer-token: full-vocab entropy, KL(full || arm), argmax-vs-full agreement
  - b_hot's own greedy answer scored under full ("endorsement": is bh's answer plausible to
    the reference distribution?) and under b_hot itself (its own confidence)

Pre-registered readings:
  R1 if the residual were distribution damage: KL(full||bh) in the tx-right/bh-wrong cell >>
     KL in the both-right cell, entropy up.
  R2 if the residual is marginal integration deficit: mean-gold logprob mildly depressed,
     argmax flips with SMALL KL, and full assigns b_hot's wrong answers non-trivial mass
     (fluent competitors, not noise).
  R3 KL(full||tx) calibrates how much distribution shift routed TEXT serving itself causes;
     the interesting quantity is KL(full||bh) relative to that, not its absolute size.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_klprobe --shard 0 --queue_dir ./out/kp_q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh

ARMS = ["b_hot", "tx", "full"]


@torch.no_grad()
def feed_logits(llm, cache, plen, ids, positions):
    """Forward `ids` on top of cache (appends rows); returns logits (1, L, V)."""
    dev = llm.device
    out = llm.model(input_ids=torch.tensor([ids], dtype=torch.long, device=dev),
                    past_key_values=cache, use_cache=True,
                    position_ids=torch.tensor(positions, dtype=torch.long,
                                              device=dev).unsqueeze(0),
                    cache_position=torch.arange(plen, plen + len(ids), device=dev),
                    attention_mask=torch.ones(1, plen + len(ids), dtype=torch.long,
                                              device=dev))
    return out.logits


@torch.no_grad()
def score_target(llm, cache, plen, maxpos, qids, tgt_ids):
    """Teacher-force qids then tgt_ids. Returns (first_token_logits fp32 cpu (V,),
    per-token logprobs of tgt list). Cache must be cropped back by caller."""
    lq = len(qids)
    qpos = list(range(maxpos + 1, maxpos + 1 + lq))
    lg = feed_logits(llm, cache, plen, qids, qpos)
    first = lg[0, -1].float()
    lps = []
    logZ = torch.logsumexp(first, -1)
    lps.append((first[tgt_ids[0]] - logZ).item())
    if len(tgt_ids) > 1:
        tpos = list(range(qpos[-1] + 1, qpos[-1] + len(tgt_ids)))
        lg2 = feed_logits(llm, cache, plen + lq, tgt_ids[:-1], tpos)
        f2 = lg2[0].float()
        lz = torch.logsumexp(f2, -1)
        for t in range(len(tgt_ids) - 1):
            lps.append((f2[t, tgt_ids[t + 1]] - lz[t]).item())
    return first.cpu(), lps


def kl_ent(p_logits, q_logits):
    """KL(p||q) and H(p) from raw logit vectors (fp32 cpu)."""
    lp = torch.log_softmax(p_logits, -1)
    lq = torch.log_softmax(q_logits, -1)
    p = lp.exp()
    return float((p * (lp - lq)).sum()), float(-(p * lp).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--gold_cap", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/kp.json")
    ap.add_argument("--ans_out", default="./out/kp_ans.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    from collections import defaultdict, deque
    bydom = defaultdict(list)
    for e in load_episodes(args.data, max_tokens=args.max_tokens):
        bydom[e.domain].append(e)
    queues = [deque(bydom[d]) for d in sorted(bydom)]
    target = args.ep_offset + args.max_ep
    eps = []
    while len(eps) < target and any(queues):
        for qd in queues:
            if qd and len(eps) < target:
                eps.append(qd.popleft())
    eps = eps[args.ep_offset:]

    wid = args.shard
    acc = {a: 0 for a in ARMS}
    n = 0
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ei):
        nonlocal n
        ep = eps[ei]
        segment_texts = [f"<step {s.turn}>\n{s.text}\n" for s in ep.segments]
        n_seg = len(segment_texts)
        header = (head + "You are reviewing a completed agent trajectory. Use it to answer the "
                  f"question precisely.\n\nTask: {ep.task}\n\nTrajectory:\n")
        header_ids = list(llm.tok(header, add_special_tokens=False).input_ids)
        H = len(header_ids)
        all_seg_ids, spans, cur = [], [], H
        for txt in segment_texts:
            sids = list(llm.tok(txt, add_special_tokens=False).input_ids)
            all_seg_ids.append(sids)
            spans.append((cur, cur + len(sids)))
            cur += len(sids)
        total = cur
        traj_flat = [t for sids in all_seg_ids for t in sids]
        hot_idx = set(range(max(0, n_seg - args.hot), n_seg))
        old = [s for i, s in enumerate(ep.segments) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k)
                      if p in id2idx}
            k5 = sorted(hot_idx | picked)
            routed.append(k5)
            need |= set(k5)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            pos = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            st[i] = encode_block(llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                                 pos, keep_a=H + alen)
        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                s0, e0 = spans[i]
                for pth in range(max(H, s0), min(H + w_anch, e0)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted([b for b in blocks[1:] if b], key=lambda b: b[1][0])
            return assemble(llm, bs)

        # full cache: whole episode, contiguous, built ONCE per episode (chunked prefill)
        fc, fp = build([hkb])
        fl = H
        for a in range(0, len(traj_flat), 4096):
            chunk = traj_flat[a:a + 4096]
            prefill_fresh(llm, fc, fl, chunk, list(range(H + a, H + a + len(chunk))))
            fl += len(chunk)
        full_maxpos = H + len(traj_flat) - 1

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qt = llm._ids(qtext)
            qids = qt[0].tolist() if torch.is_tensor(qt) else list(qt)
            gids = list(llm.tok(gold, add_special_tokens=False).input_ids)[: args.gold_cap]
            if not gids:
                continue
            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold}
            try:
                rbl = rblk_for(k5)
                cold = [i for i in k5 if i not in hot_idx]
                # ---- b_hot serving ----
                bc, bp = build([hkb] + ([rbl] if rbl else [])
                               + [(st[i], list(range(spans[i][0], spans[i][1])))
                                  for i in cold])
                prefill_fresh(llm, bc, bp.shape[0], hot_ids, hot_pos)
                bl = bp.shape[0] + len(hot_ids)
                bpos = torch.cat([bp, torch.tensor(hot_pos, dtype=torch.long,
                                                   device=llm.device)])
                bmax = int(bpos.max().item())
                if not torch.is_tensor(qt):
                    qt = torch.tensor([qids], dtype=torch.long, device=llm.device)
                ans_bh, _, _ = llm._greedy_pos(bc, bpos, qt, args.ans_tokens)
                bc.crop(bl)
                f_bh, lp_gold_bh = score_target(llm, bc, bl, bmax, qids, gids)
                bc.crop(bl)
                bh_ids = list(llm.tok(ans_bh, add_special_tokens=False).input_ids)[:args.gold_cap]
                lp_own_bh = None
                if bh_ids:
                    _, lps = score_target(llm, bc, bl, bmax, qids, bh_ids)
                    lp_own_bh = sum(lps) / len(lps)
                del bc
                # ---- tx serving (routed text, contiguous re-prefill) ----
                tids = header_ids + [t for i in k5 for t in all_seg_ids[i]]
                tc, tp = build([hkb])
                tl = H
                for a0 in range(H, len(tids), 4096):
                    chunk = tids[a0:a0 + 4096]
                    prefill_fresh(llm, tc, tl, chunk, list(range(a0, a0 + len(chunk))))
                    tl += len(chunk)
                tmax = len(tids) - 1
                tpos = torch.arange(len(tids), device=llm.device)
                ans_tx, _, _ = llm._greedy_pos(tc, tpos, qt, args.ans_tokens)
                tc.crop(tl)
                f_tx, lp_gold_tx = score_target(llm, tc, tl, tmax, qids, gids)
                tc.crop(tl)
                del tc
                # ---- full serving (episode-shared cache, crop after use) ----
                fpos = torch.arange(fl, device=llm.device)
                ans_full, _, _ = llm._greedy_pos(fc, fpos, qt, args.ans_tokens)
                fc.crop(fl)
                f_full, lp_gold_full = score_target(llm, fc, fl, full_maxpos, qids, gids)
                fc.crop(fl)
                lp_bhans_full = None
                if bh_ids:
                    _, lps = score_target(llm, fc, fl, full_maxpos, qids, bh_ids)
                    lp_bhans_full = sum(lps) / len(lps)
                    fc.crop(fl)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            kl_fb, ent_f = kl_ent(f_full, f_bh)
            kl_ft, _ = kl_ent(f_full, f_tx)
            _, ent_b = kl_ent(f_bh, f_full)
            _, ent_t = kl_ent(f_tx, f_full)
            ans = {"b_hot": ans_bh, "tx": ans_tx, "full": ans_full}
            for a in ARMS:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
            row.update({
                "lp_gold_bh": sum(lp_gold_bh) / len(lp_gold_bh),
                "lp_gold_tx": sum(lp_gold_tx) / len(lp_gold_tx),
                "lp_gold_full": sum(lp_gold_full) / len(lp_gold_full),
                "lp_gold_bh_sum": sum(lp_gold_bh), "lp_gold_tx_sum": sum(lp_gold_tx),
                "lp_gold_full_sum": sum(lp_gold_full),
                "lp_bhans_under_bh": lp_own_bh, "lp_bhans_under_full": lp_bhans_full,
                "kl_full_bh": kl_fb, "kl_full_tx": kl_ft,
                "ent_full": ent_f, "ent_bh": ent_b, "ent_tx": ent_t,
                "agree_bh": int(int(f_full.argmax()) == int(f_bh.argmax())),
                "agree_tx": int(int(f_full.argmax()) == int(f_tx.argmax())),
                "gold_len": len(gids),
            })
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st, header_kv, R_kv, fc
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] KLPROBE QUEUE over {len(eps)} episodes", flush=True)
        for i in range(len(eps)):
            cdir = os.path.join(args.queue_dir, f"c{i}")
            try:
                os.mkdir(cdir)
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(i)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM on episode idx {i} -- skipped", flush=True)
                torch.cuda.empty_cache()
            open(os.path.join(cdir, "done"), "w").close()
    else:
        for i in range(args.shard, len(eps), args.nshards):
            run_episode(i)
    ansf.close()
    print(f"KLPROBE_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
