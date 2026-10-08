"""kv_frontier.py -- causal replay frontier vs matched fresh-selectors (C10).

Question: given a fixed served-evidence set and a fixed fresh-token budget, WHICH events
should be freshly recomputed over the assembled cache? Prior selectors: recency (our b_hot),
query-attention over tokens (ProphetKV/InfoFlow), attention propagation over groups (KEEP).
Ours: an explicit trajectory-structure dependency graph (signature co-occurrence over tool
args/paths/identifiers/entities) -> pick the events best CONNECTING routed evidence, the
episode tail, and the query; recompute them in chronological (topological) order.

Evidence-matching discipline (E2 lesson): ALL arms share the same served set
    kept_ext = picked(K) ∪ last2 ∪ top-B graph bridges   (arm-independent)
and differ ONLY in which 2 events of kept_ext are recomputed fresh:
    tx        text re-prefill of kept_ext (reference)
    f_none    no fresh (floor; all kept_ext served as anchored stores)
    f_recent  last 2 (= b_hot incumbent)
    f_random  seeded random 2
    f_qrel    top-2 by query-signature overlap
    f_attn    top-2 by query->cache attention mass (eager pass over the all-stores view;
              InfoFlow/ProphetKV-style signal at event granularity)
    f_front   top-2 by graph bridge score (connectivity to picked evidence + tail + query)

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_frontier --shard 0 --queue_dir ./out/fr_q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

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
from kvmemory.kv_floor import prefill_fresh

ARMS = ["tx", "f_none", "f_recent", "f_random", "f_qrel", "f_attn", "f_front"]
WORD_RE = re.compile(r"[A-Za-z]{4,}")
SIG_RE = re.compile(
    r"https?://\S+|/[\w./\-]{4,}|\b\w+_\w+\b|\b[a-z]+[A-Z]\w*\b|\"[^\"]{3,60}\"|'[^']{3,60}'|\b\d{3,}\b")


def build_sigs(texts):
    dfc = Counter()
    words_per = []
    for t in texts:
        ws = set(w.lower() for w in WORD_RE.findall(t))
        words_per.append(ws)
        dfc.update(ws)
    n = len(texts)
    common = {w for w, c in dfc.items() if c > max(2, 0.25 * n)}
    sigs = []
    for t, ws in zip(texts, words_per):
        s = set(m.group(0).strip("\"'").lower() for m in SIG_RE.finditer(t))
        s |= (ws - common)
        sigs.append(s)
    return sigs


def wov(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def qsig_of(question):
    s = set(m.group(0).strip("\"'").lower() for m in SIG_RE.finditer(question))
    s |= set(w.lower() for w in WORD_RE.findall(question))
    return s


@torch.no_grad()
def attn_event_scores(llm, cache, pos, qids, spans, events):
    """Query->cache attention mass per event (mean per token), via a throwaway eager pass."""
    core = getattr(llm.model, "model", llm.model)
    dev = llm.device
    kvlen = pos.shape[0]
    qlen = qids.shape[1]
    maxp = int(pos.max().item())
    old_impl = core.config._attn_implementation
    core.config._attn_implementation = "eager"
    try:
        out = core(input_ids=qids.to(dev), past_key_values=cache, use_cache=True,
                   output_attentions=True,
                   position_ids=torch.arange(maxp + 1, maxp + 1 + qlen, device=dev).unsqueeze(0),
                   cache_position=torch.arange(kvlen, kvlen + qlen, device=dev),
                   attention_mask=torch.ones(1, kvlen + qlen, dtype=torch.long, device=dev))
    finally:
        core.config._attn_implementation = old_impl
    if not out.attentions or out.attentions[0] is None:
        raise RuntimeError("no attentions returned (eager switch failed)")
    mass = torch.zeros(kvlen)
    for A in out.attentions:
        mass += A[0].float().mean(0).sum(0)[:kvlen].cpu()
    pos_cpu = pos.cpu()
    scores = {}
    for i in events:
        s0, e0 = spans[i]
        sel = (pos_cpu >= s0) & (pos_cpu < e0)
        ntok = int(sel.sum().item())
        scores[i] = float(mass[sel].sum().item()) / max(1, ntok)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--max_ep", type=int, default=103)
    ap.add_argument("--ep_offset", type=int, default=0)
    ap.add_argument("--max_qa", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--w", type=int, default=4096)
    ap.add_argument("--nbridge", type=int, default=3)
    ap.add_argument("--nfresh", type=int, default=2)
    ap.add_argument("--ans_tokens", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/fr.json")
    ap.add_argument("--ans_out", default="./out/fr_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

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
    acc = {a: 0 for a in RUN}
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
        last2 = set(range(max(0, n_seg - 2), n_seg))
        lastidx = n_seg - 1
        old = [s for i, s in enumerate(ep.segments) if i not in last2]
        id2idx = {s.seg_id: i for i, s in enumerate(ep.segments)}
        qas = ep.qa[: args.max_qa]
        w_anch = min(args.w, total - H)
        sigs = build_sigs(segment_texts)

        # per-QA routing + bridge extension (arm-independent)
        plans = []
        need = set()
        for qa in qas:
            q = qa["question"]
            picked = {id2idx[p] for p in router.select(q, old, args.k) if p in id2idx}
            qs = qsig_of(q)
            base = set(picked) | last2
            cand = [v for v in range(n_seg) if v not in base]
            conn = {v: (sum(wov(sigs[v], sigs[e]) for e in picked)
                        + wov(sigs[v], sigs[lastidx]) + wov(sigs[v], qs)) for v in cand}
            bridges = [v for v, s in sorted(conn.items(), key=lambda x: -x[1])[: args.nbridge]
                       if s > 0]
            kept_ext = sorted(base | set(bridges))
            plans.append((q, qa, picked, qs, bridges, kept_ext))
            need |= set(kept_ext)

        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        st = {}
        for i in sorted(need):
            s0, e0 = spans[i]
            alen = min(w_anch, s0 - H)
            st[i] = encode_block(
                llm, header_ids + traj_flat[:alen] + all_seg_ids[i],
                list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0)),
                keep_a=H + alen)
        hkb = (header_kv, list(range(H)))

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

        def serve(kept, fresh, rbl, qids_):
            keep = [i for i in kept if i not in fresh]
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(st[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            if fresh:
                ii = sorted(fresh)
                fids = [t for i in ii for t in all_seg_ids[i]]
                fpos = [pp for i in ii for pp in range(spans[i][0], spans[i][1])]
                c = prefill_fresh(llm, c, p.shape[0], fids, fpos)
                p = torch.cat([p, torch.tensor(fpos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        for qi, (q, qa, picked, qs, bridges, kept_ext) in enumerate(plans):
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            rbl = rblk_for(kept_ext)
            nf = min(args.nfresh, len(kept_ext))
            ans, fsets = {}, {}

            def top2(scores, excl=()):
                order = sorted((v for v in kept_ext if v not in excl),
                               key=lambda v: (-scores.get(v, 0.0), v))
                return set(order[:nf])

            try:
                if "tx" in RUN:
                    text = header + "".join(segment_texts[i] for i in kept_ext) + qtext
                    ans["tx"], _, _ = llm._greedy(DynamicCache(), 0, llm._ids(text),
                                                  args.ans_tokens)
                    fsets["tx"] = []
                if "f_none" in RUN:
                    ans["f_none"] = serve(kept_ext, set(), rbl, qids)
                    fsets["f_none"] = []
                if "f_recent" in RUN:
                    fr = set(sorted(kept_ext)[-nf:])
                    ans["f_recent"] = serve(kept_ext, fr, rbl, qids)
                    fsets["f_recent"] = sorted(fr)
                if "f_random" in RUN:
                    rng = random.Random(9973 * int(ep.episode_id) + qi)
                    fr = set(rng.sample(list(kept_ext), nf))
                    ans["f_random"] = serve(kept_ext, fr, rbl, qids)
                    fsets["f_random"] = sorted(fr)
                if "f_qrel" in RUN:
                    fr = top2({v: wov(sigs[v], qs) for v in kept_ext})
                    ans["f_qrel"] = serve(kept_ext, fr, rbl, qids)
                    fsets["f_qrel"] = sorted(fr)
                if "f_attn" in RUN:
                    c, p = build([hkb] + ([rbl] if rbl else [])
                                 + [(st[i], list(range(spans[i][0], spans[i][1])))
                                    for i in kept_ext])
                    sc = attn_event_scores(llm, c, p, qids, spans, kept_ext)
                    del c
                    fr = top2(sc)
                    ans["f_attn"] = serve(kept_ext, fr, rbl, qids)
                    fsets["f_attn"] = sorted(fr)
                if "f_front" in RUN:
                    fs = {v: (sum(wov(sigs[v], sigs[e]) for e in picked if e != v)
                              + wov(sigs[v], sigs[lastidx]) + wov(sigs[v], qs))
                          for v in kept_ext}
                    fr = top2(fs)
                    ans["f_front"] = serve(kept_ext, fr, rbl, qids)
                    fsets["f_front"] = sorted(fr)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold,
                   "kept_ext": kept_ext, "bridges": bridges,
                   "picked": sorted(picked), "n_seg": n_seg}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
                row["fresh_" + a] = fsets[a]
                row["ftok_" + a] = sum(spans[i][1] - spans[i][0] for i in fsets[a])
            n += 1
            ansf.write(json.dumps(row, ensure_ascii=False) + "\n")
            ansf.flush()
        del st, header_kv, R_kv
        torch.cuda.empty_cache()
        json.dump({"shard": wid, "arms": RUN, "n": n, "acc": acc}, open(args.out, "w"))
        print(f"[w{wid}] ep {ep.episode_id} {ep.domain} | n={n} | " +
              " ".join(f"{a} {acc[a]}" for a in RUN), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        print(f"[w{wid}] FRONTIER QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"FRONTIER_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
