"""kv_distanchor.py -- C3's third axis: WRITE-TIME DISTANCE between the conditioner (the
"virtual context" recorded inside each event store) and the event itself.

Hypothesis (user, 2026-07-13): the conditioner's influence on the event's KV is attention-
mediated and therefore RoPE-distance-gated. If the virtual context sits FAR from the event at
write time, the event barely attends to it -- both its benefit (boundary repair / semantic
conditioning) and its harm (attention dilution / poison) should switch off, converging to the
no-conditioner floor. If instead the effect is sink-existence (causal-order, distance-robust),
distance should not matter.

Code fact this isolates: the deployed recipe encodes stores with TRUTHFUL geometry --
anchor at native H..H+alen, event at native s0..e0 -- so a POSITION HOLE of
gap = max(0, s0-H-w) already exists and grows with event depth (0 for shallow events,
10-19k for deep ones). The dial below makes distance a controlled variable for the first time.

Design: 2 (content: true opening / dummy " the"*alen) x 3 (write distance) + floor, all arms
served IDENTICALLY (rblk always served, fresh last-2 tail, same routed evidence -- E2
discipline; only the stores' write-time geometry/content differs):

  t_nat  anchor@native H..H+alen, event@native            = deployed b_hot (bh_true; gate:
                                                             must reproduce .2209, 5th repl.)
  t_adj  anchor slid up to s0-alen..s0 (gap -> 0)           max write-time attention to anchor
  t_far  event written at s0+D..e0+D (D<=8192, ctx-capped), kept keys rotated R(-D) back to
         native s0 -> serving byte-identical, only write-time geometry differs
  d_nat / d_adj / d_far   same three geometries, conditioner = repeated-token dummy (E1's
         poison: at natural distance it scored BELOW none)
  none   [header; event] stores, rblk still served (content-access-equalized floor)

Pre-registered contrasts:
  P1  t_adj - t_nat  (primary on deep slice gapmax>=4096): does closing the natural hole help?
      If >0: free recipe upgrade (write cost unchanged). If <0: truthful geometry matters.
  P2  t_far - t_nat: further decay beyond the natural distance.
  P3  content effect (t - d) at adj vs far: distance-gated influence predicts it shrinks.
  P4  d_far - d_nat: the "conversely" half -- distance should mute the dummy poison
      (d_far -> none).
  P5  gate: t_nat == bh_true .2209 byte-level (same eps/router/params as kv_fixanchor).

RoPE re-rotation: cached keys are post-rotation, K_cached = s * R(p) k_hat (s = YaRN scalar if
any). Givens rotations at fixed frequencies compose additively, so R(-D) K_cached =
s * R(p-D) k_hat exactly; we use the model's own inv_freq buffer. A numeric gate at startup
encodes a probe span at p and p+D and checks the rotated keys match (bf16 noise only) and
values are position-invariant.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_distanchor --shard 0 --queue_dir ./out/dx_q
"""
from __future__ import annotations

import argparse
import json
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

ARMS = ["t_nat", "t_adj", "t_far", "d_nat", "d_adj", "d_far", "none"]


def make_rotator(llm):
    core = getattr(llm.model, "model", llm.model)
    inv = core.rotary_emb.inv_freq.detach().float().cpu()  # (head_dim/2,)

    def rot_keys(kv, delta):
        """R(delta) applied to cached (post-RoPE) keys; exact for RoPE/YaRN (scalar passes)."""
        if not delta:
            return kv
        ang = float(delta) * inv
        cos = torch.cos(ang).repeat(2).view(1, 1, 1, -1)  # HF rotate_half layout: cat(f, f)
        sin = torch.sin(ang).repeat(2).view(1, 1, 1, -1)
        out = []
        for K, V in kv:
            kf = K.float()
            h = kf.shape[-1] // 2
            rh = torch.cat([-kf[..., h:], kf[..., :h]], dim=-1)
            out.append(((kf * cos + rh * sin).to(K.dtype), V))
        return out

    return rot_keys


def rotation_gate(llm, rot_keys, delta=8192, p0=9000):
    """Three checks. (1) round-trip: R(-d)R(d)K == K up to bf16 quantization -- validates the
    rotation code. (2) layer-0 convention check: pre-RoPE layer-0 K is position-independent
    (pure embedding path), so rotating the far encode's layer-0 K back must match the near
    encode's layer-0 K almost exactly; a wrong convention/direction shows up as O(1) error
    here. Layer-0 V never sees RoPE and must be bit-equal. (3) deeper layers accumulate
    genuine bf16 forward cascade across positions (that cascade is part of the far-arm
    treatment, not a rotation error) -- printed as info only."""
    ids = list(llm.tok("The quick brown fox jumps over the lazy dog. " * 40,
                       add_special_tokens=False).input_ids)
    a = encode_block(llm, ids, list(range(p0, p0 + len(ids))))
    b = encode_block(llm, ids, list(range(p0 + delta, p0 + delta + len(ids))))
    ks = max(X[0].abs().max().item() for X, _ in a)
    rt = rot_keys(rot_keys(a, delta), -delta)
    rterr = max((X[0] - Y[0]).abs().max().item() for X, Y in zip(rt, a)) / max(ks, 1e-6)
    k0n, v0n = a[0]
    k0r, _ = rot_keys([b[0]], -delta)[0]
    s0 = k0n.abs().max().item()
    k0err = (k0r - k0n).abs().max().item() / max(s0, 1e-6)
    v0err = (b[0][1] - v0n).abs().max().item()
    deep = max((X[0] - Y[0]).abs().max().item() for X, Y in zip(rot_keys(b, -delta), a))
    print(f"[gate] roundtrip rel {rterr:.5f}; L0 K rel {k0err:.5f} (scale {s0:.1f}); "
          f"L0 V maxdiff {v0err:.6f}; deep-cascade K maxdiff {deep:.2f} (info only)",
          flush=True)
    if rterr > 0.02 or k0err > 0.02 or v0err > 1e-3:
        sys.exit(f"ROTATION GATE FAIL rt={rterr:.5f} k0={k0err:.5f} v0={v0err:.6f}")


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
    ap.add_argument("--far_delta", type=int, default=8192)
    ap.add_argument("--ctx_cap", type=int, default=31500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/dx.json")
    ap.add_argument("--ans_out", default="./out/dx_ans.jsonl")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    RUN = [a for a in ARMS if a in set(args.arms.split(","))]

    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()
    rot_keys = make_rotator(llm)
    if any(a.endswith("_far") for a in RUN):
        rotation_gate(llm, rot_keys, delta=args.far_delta)

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

    the_ids = list(llm.tok(" the", add_special_tokens=False).input_ids)

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
        st = {a: {} for a in RUN}
        gaps, deltas = {}, {}
        for i in sorted(need):
            s0, e0 = spans[i]
            span_ids = all_seg_ids[i]
            alen = min(w_anch, s0 - H)
            anc = traj_flat[:alen]
            dum = (the_ids * alen)[:alen]
            gaps[i] = max(0, s0 - H - alen)
            di = max(0, min(args.far_delta, args.ctx_cap - e0))
            deltas[i] = di
            pos_nat = list(range(H)) + list(range(H, H + alen)) + list(range(s0, e0))
            pos_adj = list(range(H)) + list(range(s0 - alen, s0)) + list(range(s0, e0))
            pos_far = (list(range(H)) + list(range(H, H + alen))
                       + list(range(s0 + di, e0 + di)))
            if "t_nat" in RUN:
                st["t_nat"][i] = encode_block(llm, header_ids + anc + span_ids, pos_nat,
                                              keep_a=H + alen)
            if "t_adj" in RUN:
                st["t_adj"][i] = encode_block(llm, header_ids + anc + span_ids, pos_adj,
                                              keep_a=H + alen)
            if "t_far" in RUN:
                st["t_far"][i] = rot_keys(encode_block(llm, header_ids + anc + span_ids,
                                                       pos_far, keep_a=H + alen), -di)
            if "d_nat" in RUN:
                st["d_nat"][i] = encode_block(llm, header_ids + dum + span_ids, pos_nat,
                                              keep_a=H + alen)
            if "d_adj" in RUN:
                st["d_adj"][i] = encode_block(llm, header_ids + dum + span_ids, pos_adj,
                                              keep_a=H + alen)
            if "d_far" in RUN:
                st["d_far"][i] = rot_keys(encode_block(llm, header_ids + dum + span_ids,
                                                       pos_far, keep_a=H + alen), -di)
            if "none" in RUN:
                st["none"][i] = encode_block(llm, header_ids + span_ids,
                                             list(range(H)) + list(range(s0, e0)), keep_a=H)
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

        def serve(stores, kept, rbl, qids_):
            keep = [i for i in kept if i not in hot_idx]
            c, p = build([hkb] + ([rbl] if rbl else [])
                         + [(stores[i], list(range(spans[i][0], spans[i][1]))) for i in keep])
            c = prefill_fresh(llm, c, p.shape[0], hot_ids, hot_pos)
            p = torch.cat([p, torch.tensor(hot_pos, dtype=torch.long, device=llm.device)])
            out, _, _ = llm._greedy_pos(c, p, qids_, args.ans_tokens)
            del c
            return out

        for qa, k5 in zip(qas, routed):
            q = qa["question"]
            gold = qa.get("answer", "") or ""
            qtext = f"\n\nQuestion: {q}\nAnswer concisely and specifically:" + tail
            qids = llm._ids(qtext)
            rbl = rblk_for(k5)
            ans = {}
            try:
                for a in RUN:
                    ans[a] = serve(st[a], k5, rbl, qids)
            except torch.OutOfMemoryError:
                print(f"[w{wid}] OOM QA skipped (ep {ep.episode_id})", flush=True)
                torch.cuda.empty_cache()
                continue

            cold = [i for i in k5 if i not in hot_idx]
            row = {"episode_id": ep.episode_id, "domain": ep.domain,
                   "qtype": qa.get("type", "?"), "hop": hop_bucket(q), "q": q, "gold": gold,
                   "gaps": [gaps[i] for i in cold], "deltas": [deltas[i] for i in cold]}
            for a in RUN:
                ok = int(judge(llm, head, tail, q, gold, ans[a]))
                acc[a] += ok
                row[a] = ok
                row["ans_" + a] = ans[a]
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
        print(f"[w{wid}] DISTANCHOR QUEUE over {len(eps)} episodes, arms={RUN}", flush=True)
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
    print(f"DISTANCHOR_DONE shard={wid} n={n}", flush=True)


if __name__ == "__main__":
    main()
