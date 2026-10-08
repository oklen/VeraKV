"""kv_locomo.py -- second benchmark family for the KV-level claims: LOCOMO multi-session dialogues.

Maps LOCOMO onto the b_hot protocol: event = SESSION (the natural per-event unit; ~20-35 per
conversation, each a dated sitting), hot tail = last `hot` sessions, router selects K sessions.
Original arms are the 2x2 table's: tx (text re-prefill of routed sessions), b_anch (session KV
encoded against [header; first-w anchor], anchor served), b_hot (b_anch stores + hot sessions
recomputed fresh at read). Harvest arms replicate the zero-write cell here: gr gathers the routed
rows out of ONE full-conversation rollout cache (no per-event stores at all), gr_it moves the
answer-format instruction before the question in text, gr_sk carries that same instruction as a
RoPE-rotated skill-KV block instead (kv_skill parity protocol, second family). Cats 1-4 scored
(cat 5 adversarial excluded, standard Mem0-J practice); judge = same paired LLM judge as the AMA
runs. YaRN 4x uniformly (a few convs exceed the native window).

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa SPRAG_ROPE_FACTOR=4.0 SPRAG_MAX_CTX=131072 \
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_locomo \
        --arms tx,b_hot,gr,gr_it,gr_sk \
        --queue_dir ./out/loc_q --out ./out/loco/loc_s0.json --ans_out ./out/loco/loc_ans_s0.jsonl
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
from kvmemory.locomo import load_locomo
from kvmemory.core import Segment
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_matrix import encode_block, assemble
from kvmemory.kv_floor import prefill_fresh
from kvmemory.kv_ow import prefill_chunked
from kvmemory.kv_globhot import gather_rows
from kvmemory.kv_distanchor import make_rotator, rotation_gate

INSTR = "Answer with as few words as possible (a name, a date, a short phrase)."
VPOS = 50000


def sessions_of(ep):
    """Group the per-turn segments into session-level texts (chronological)."""
    bysess = {}
    for s in ep.segments:
        bysess.setdefault(s.meta["session"], []).append(s)
    out = []
    for k in sorted(bysess):
        turns = bysess[k]
        ts = turns[0].meta.get("timestamp", "")
        body = "\n".join(t.text for t in turns)
        out.append((k, f"<session {k}  {ts}>\n{body}\n\n"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/locomo10.json")
    ap.add_argument("--arms", default="tx,b_anch,b_hot")
    ap.add_argument("--max_qa", type=int, default=120)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--ans_tokens", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/loco/loc.json")
    ap.add_argument("--ans_out", default="./out/loco/loc_ans.jsonl")
    args = ap.parse_args()
    ARMS = [a for a in args.arms.split(",") if a]
    GR_ARMS = [a for a in ARMS if a.startswith("gr")]
    ST_ARMS = [a for a in ARMS if a in ("b_anch", "b_hot")]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    eps = load_locomo(args.data)

    head_ids0 = list(llm.tok(head, add_special_tokens=False).input_ids)
    skill_kv, skill_len, rot_keys = None, 0, None
    if "gr_sk" in ARMS:
        instr_ids = list(llm.tok("\n\n" + INSTR, add_special_tokens=False).input_ids)
        skill_len = len(instr_ids)
        skill_kv = encode_block(llm, head_ids0 + instr_ids,
                                list(range(len(head_ids0))) + list(range(VPOS, VPOS + skill_len)),
                                keep_a=len(head_ids0))
        rot_keys = make_rotator(llm)
        rotation_gate(llm, rot_keys, delta=40000, p0=9000)
        print(f"[w{args.shard}] skill block: {skill_len} rows @ {VPOS}, rotation gate passed",
              flush=True)

    wid = args.shard
    acc = {a: 0 for a in ARMS}
    n = 0
    bycat = {}
    fresh_tok = {a: 0 for a in ARMS}
    ansf = open(args.ans_out, "w", encoding="utf-8")

    def run_episode(ep):
        nonlocal n
        sess = sessions_of(ep)
        n_seg = len(sess)
        segment_texts = [t for _, t in sess]
        seg_objs = [Segment(seg_id=str(k), turn=i, text=t, kind="session", meta={})
                    for i, (k, t) in enumerate(sess)]
        A, B = ep.speakers
        header = (head + f"You are reviewing a long multi-session conversation between {A} and "
                  f"{B}. Each line is timestamped. Use it to answer the question precisely.\n\n"
                  "Conversation:\n")
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
        old = [s for i, s in enumerate(seg_objs) if i not in hot_idx]
        id2idx = {s.seg_id: i for i, s in enumerate(seg_objs)}
        qas = [q for q in ep.qa if q.get("answer") is not None][: args.max_qa]
        w_anch = min(args.w, total - H)

        routed, need = [], set()
        for qa in qas:
            picked = {id2idx[p] for p in router.select(qa["question"], old, args.k) if p in id2idx}
            kept = sorted(hot_idx | picked)
            routed.append(kept)
            need |= set(kept)

        def qtexts(q):
            return {
                "std": ("\n\nQuestion: %s\nAnswer with as few words as possible (a name, a "
                        "date, a short phrase):" % q) + tail,
                "it": "\n\n" + INSTR + f"\n\nQuestion: {q}\nAnswer:" + tail,
                "bare": f"\n\nQuestion: {q}\nAnswer:" + tail,
            }

        ans = [dict() for _ in qas]

        # ---- pass 1: harvest arms over the single full-conversation rollout cache ----
        if GR_ARMS:
            t0 = time.time()
            full_cache, ftot, _ = prefill_chunked(llm, header, segment_texts, chunk=4096)
            assert ftot == total
            print(f"[w{wid}] conv {ep.conv_id} rollout cache {total} rows "
                  f"{time.time()-t0:.0f}s", flush=True)

            def gather_idx(kept):
                idx = set(range(H))
                covered = [spans[i] for i in kept]
                for p in range(H, H + w_anch):
                    if not any(st <= p < en for st, en in covered):
                        idx.add(p)
                for i in kept:
                    idx |= set(range(spans[i][0], spans[i][1]))
                return sorted(idx)

            for qi, (qa, kept) in enumerate(zip(qas, routed)):
                qt = qtexts(qa["question"])
                idx = gather_idx(kept)
                qlen = llm._ids(qt["std"]).shape[1]
                for a in GR_ARMS:
                    c = gather_rows(llm, full_cache, idx)
                    pos_list = list(idx)
                    if a == "gr_sk":
                        start = pos_list[-1] + 1
                        if start + skill_len + 600 < VPOS:
                            kv = rot_keys(skill_kv, start - VPOS)
                            for li, (K, V) in enumerate(kv):
                                c.update(K.to(llm.device, non_blocking=True),
                                         V.to(llm.device, non_blocking=True), li)
                            pos_list += list(range(start, start + skill_len))
                        qv = qt["bare"]
                    elif a == "gr_it":
                        qv = qt["it"]
                    else:
                        qv = qt["std"]
                    p = torch.tensor(pos_list, dtype=torch.long, device=llm.device)
                    ans[qi][a], _, _ = llm._greedy_pos(c, p, llm._ids(qv), args.ans_tokens)
                    fresh_tok[a] += qlen
                    del c
            del full_cache
            torch.cuda.empty_cache()

        # ---- pass 2: text + store-based arms ----
        st_anch, header_kv, R_kv = {}, None, None
        if ST_ARMS:
            t0 = time.time()
            header_kv = encode_block(llm, header_ids, list(range(H)))
            R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                                keep_a=H, keep_b=H + w_anch)
            for i in sorted(need):
                st, en = spans[i]
                alen = min(w_anch, st - H)
                st_anch[i] = encode_block(
                    llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                    list(range(H)) + list(range(H, H + alen)) + list(range(st, en)),
                    keep_a=H + alen)
            print(f"[w{wid}] conv {ep.conv_id} total={total} sess={n_seg} need={len(need)} "
                  f"qa={len(qas)} stores {time.time()-t0:.0f}s", flush=True)

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

        for qi, (qa, kept) in enumerate(zip(qas, routed)):
            qt = qtexts(qa["question"])
            qids = llm._ids(qt["std"])
            qlen = qids.shape[1]

            if "tx" in ARMS:
                text = header + "".join(segment_texts[i] for i in kept) + qt["std"]
                ans[qi]["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text),
                                                  args.ans_tokens)
                fresh_tok["tx"] += H + sum(spans[i][1] - spans[i][0] for i in kept) + qlen

            if "b_anch" in ARMS:
                r5 = rblk_for(kept)
                ab = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept]
                c, p = build([hkb] + ([r5] if r5 else []) + ab)
                ans[qi]["b_anch"], _, _ = llm._greedy_pos(c, p, qids, args.ans_tokens)
                fresh_tok["b_anch"] += qlen
                del c

            if "b_hot" in ARMS:
                r5 = rblk_for(kept)
                kept_old = [i for i in kept if i not in hot_idx]
                abo = [(st_anch[i], list(range(spans[i][0], spans[i][1]))) for i in kept_old]
                c, p = build([hkb] + ([r5] if r5 else []) + abo)
                c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
                p2 = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
                ans[qi]["b_hot"], _, _ = llm._greedy_pos(c, p2, qids, args.ans_tokens)
                fresh_tok["b_hot"] += len(hot_ids) + qlen
                del c

        if st_anch or header_kv:
            del st_anch, header_kv, R_kv
        torch.cuda.empty_cache()

        # ---- judge ----
        for qi, qa in enumerate(qas):
            q = qa["question"]
            gold = str(qa.get("answer", ""))
            cat = qa.get("category", 0)
            row = {"conv_id": ep.conv_id, "category": cat, "q": q, "gold": gold}
            for a in ARMS:
                ok = int(judge(llm, head, tail, q, gold, ans[qi].get(a, "")))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[qi].get(a, "")
            n += 1
            d = bycat.setdefault(cat, {a: 0 for a in ARMS})
            d.setdefault("_n", 0)
            d["_n"] += 1
            for a in ARMS:
                d[a] += row[a]
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        json.dump({"shard": wid, "arms": ARMS, "n": n, "acc": acc, "bycat": bycat,
                   "fresh_tok": fresh_tok}, open(args.out, "w"), indent=2)
        print(f"[w{wid}] conv {ep.conv_id} done | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in ARMS), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] LOCOMO QUEUE over {len(eps)} conversations", flush=True)
        for i in range(len(eps)):
            try:
                os.mkdir(os.path.join(args.queue_dir, f"c{i}"))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(eps[i])
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM conv idx {i} skipped", flush=True)
                torch.cuda.empty_cache()
    else:
        for ep in eps[args.shard::args.nshards]:
            run_episode(ep)
    ansf.close()

    nn = max(1, n)
    print("\n" + "=" * 60)
    print(f"LOCOMO w{wid}: {n} QA")
    for a in ARMS:
        print(f"  {a:7s} {acc[a]:4d}/{n} = {100*acc[a]/nn:5.1f}%  fresh/QA {fresh_tok[a]//nn}")
    print(f"LOC_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
