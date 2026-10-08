"""kv_ttft.py -- query-time cost microbenchmark for the paper's efficiency figure.

Per QA and per serving mode, stage-resolved wall times (CUDA-synchronized):
  tx      t_prefill (chunked, no logits) + t_ingest(question)          [fresh tok = view]
  glob    t_gather (subselect from a per-episode full prefill) + t_ingest   [amortized full prefill
          reported separately as t_fullprefill/episode]
  b_anch  t_assemble (CPU->GPU gather of cached blocks) + t_ingest     [fresh tok = question]
  b_hot   t_assemble + t_fresh(hot tail) + t_ingest                    [fresh tok = tail+question]
Every arm then decodes the same fixed number of tokens (t_decode reported for completeness).
Episodes are sampled across the size range; hot=2, K=5, within-window.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_ttft --out ./out/ttft_8b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--n_ep", type=int, default=12)
    ap.add_argument("--max_qa", type=int, default=4)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--dec_tokens", type=int, default=32)
    ap.add_argument("--out", default="./out/ttft.jsonl")
    args = ap.parse_args()

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    eps_all = sorted(load_episodes(args.data, max_tokens=args.max_tokens),
                     key=lambda e: e.total_tokens)
    idxs = [int(i * (len(eps_all) - 1) / max(1, args.n_ep - 1)) for i in range(args.n_ep)]
    eps = [eps_all[i] for i in sorted(set(idxs))]
    outf = open(args.out, "w", encoding="utf-8")

    def sync():
        torch.cuda.synchronize()

    for ep in eps:
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
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx}
            kept = sorted(hot_idx | picked)
            routed.append(kept)
            need |= set(kept)

        # write-side (offline, amortizable): stores + anchor
        sync(); t0 = time.time()
        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st_anch = {}
        for i in sorted(need):
            st, en = spans[i]
            alen = min(w_anch, st - H)
            st_anch[i] = encode_block(
                llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                list(range(H)) + list(range(H, H + alen)) + list(range(st, en)),
                keep_a=H + alen)
        sync(); t_write = time.time() - t0

        # glob's prerequisite: one full prefill per episode
        full_cache, _tl, t_fullprefill = prefill_chunked(llm, header, segment_texts)

        hkb = (header_kv, list(range(H)))
        hot_sorted = sorted(hot_idx)
        hot_ids = [t for i in hot_sorted for t in all_seg_ids[i]]
        hot_pos = [p for i in hot_sorted for p in range(spans[i][0], spans[i][1])]

        def rblk_for(kept):
            keep_mask = [True] * w_anch
            for i in kept:
                st, en = spans[i]
                for pth in range(max(H, st), min(H + w_anch, en)):
                    keep_mask[pth - H] = False
            ridx = [j for j in range(w_anch) if keep_mask[j]]
            if not ridx:
                return None
            rt = torch.tensor(ridx, dtype=torch.long)
            return ([(K.index_select(2, rt), V.index_select(2, rt)) for K, V in R_kv],
                    [H + j for j in ridx])

        def build(blocks):
            bs = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
            return assemble(llm, bs)

        for rep in range(2):          # rep 0 = extra warmup pass, rep 1 = measured
            for qa, kept in zip(qas, routed):
                q = qa["question"]
                qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
                qids = llm._ids(qtext)
                qlen = qids.shape[1]
                sel_tok = sum(spans[i][1] - spans[i][0] for i in kept)
                row = {"episode_id": ep.episode_id, "total_tok": total, "sel_tok": sel_tok,
                       "hot_tok": len(hot_ids), "qlen": qlen, "rep": rep,
                       "t_write": round(t_write, 3), "t_fullprefill": round(t_fullprefill, 3)}

                text = header + "".join(segment_texts[i] for i in kept) + qtext
                sync(); t0 = time.time()
                c_tx, tl2, _ = prefill_chunked(llm, header, [segment_texts[i] for i in kept])
                sync(); row["tx_prefill"] = round(time.time() - t0, 4)
                _, ti, td = llm._greedy(c_tx, tl2, qids, args.dec_tokens)
                row["tx_ingest"], row["tx_decode"] = round(ti, 4), round(td, 4)
                del c_tx

                sync(); t0 = time.time()
                sub_c, sub_pos = llm.subselect_cache(full_cache, spans, H, kept)
                sync(); row["glob_gather"] = round(time.time() - t0, 4)
                _, ti, td = llm._greedy_pos(sub_c, sub_pos, qids, args.dec_tokens)
                row["glob_ingest"], row["glob_decode"] = round(ti, 4), round(td, 4)
                del sub_c

                r5 = rblk_for(kept)
                ab = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept]
                sync(); t0 = time.time()
                c, p = build([hkb] + ([r5] if r5 else []) + ab)
                sync(); row["banch_assemble"] = round(time.time() - t0, 4)
                _, ti, td = llm._greedy_pos(c, p, qids, args.dec_tokens)
                row["banch_ingest"], row["banch_decode"] = round(ti, 4), round(td, 4)
                del c

                kept_old = [i for i in kept if i not in hot_idx]
                abo = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept_old]
                sync(); t0 = time.time()
                c, p = build([hkb] + ([r5] if r5 else []) + abo)
                sync(); t1 = time.time()
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                sync(); t2 = time.time()
                row["bhot_assemble"] = round(t1 - t0, 4)
                row["bhot_fresh"] = round(t2 - t1, 4)
                p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
                _, ti, td = llm._greedy_pos(c, p2, qids, args.dec_tokens)
                row["bhot_ingest"], row["bhot_decode"] = round(ti, 4), round(td, 4)
                del c

                if rep == 1:
                    outf.write(json.dumps(row) + "\n")
                    outf.flush()
        del st_anch, header_kv, R_kv, full_cache
        torch.cuda.empty_cache()
        print(f"ep {ep.episode_id} total={total} done", flush=True)
    outf.close()
    print("TTFT_DONE", flush=True)


if __name__ == "__main__":
    main()
