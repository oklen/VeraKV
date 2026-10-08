"""kv_attn.py -- attention-row probes for the conditioning mechanism (E0-E4).

Per (QA, arm) we step the question and the first ~24 generated tokens ONE TOKEN AT A TIME in eager
mode, and reduce each step's attention rows [1,H,1,K] to REGION MASSES:
  sink (first 4 header tok) / header / anchor / spans_cited / spans_other / qbody / instr
in three variants: raw all-head mean, retrieval-heads-only mean (E0 needle test locates them),
value-norm-weighted retrieval-heads ( a * ||v|| , view positions only), plus row entropy.

Arms: iso, roll, anch(A), b_anch(B), c_anch(C), b_mask(E2 intervention: B view but anchor positions
hard-masked during question+decode; generated answer is judged fresh).
The anchor KV is IDENTICAL in B and C (the episode opening has no predecessors), so any B-vs-C
attention difference on the anchor region is attributable purely to how the SPANS were encoded.

    SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=eager PYTHONPATH=. \
        CUDA_VISIBLE_DEVICES=0 python -m kvmemory.kv_attn --shard 0 --queue_dir ./out/at_q
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys

import torch
from transformers import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kvmemory.ama_bench import load_episodes
from kvmemory.components import LexicalRouter
from kvmemory.llm_hf import HFBackend, _iter_cache_kv
from kvmemory.kv_equiv import judge
from kvmemory.kv_select_smoke import split_wrap_nothink
from kvmemory.kv_scope import hop_bucket
from kvmemory.kv_matrix import encode_block, assemble, toks
from kvmemory.kv_write import banded_prefill

ARMS = ["iso", "roll", "anch", "b_anch", "c_anch", "b_mask"]
W, HOTN, K = 4096, 4, 5
_CIT = re.compile(r"\b(?:turns?|steps?)\s*#?\s*(\d+)", re.I)
_DEG = re.compile(r"^\s*`{0,3}\s*(action|click|scroll|press|type|hover|goto|go to|stop)\b|"
                  r"action \[arg\]|^\s*```", re.I)
REG = ["sink", "header", "anchor", "cited", "spans", "qbody", "instr"]


# ---------- E0: locate retrieval heads with a needle test ----------
@torch.no_grad()
def needle_heads(llm, head, tail, topk=40):
    rng = random.Random(7)
    fill = ["The maintenance crew logged a routine inspection of unit %d with no findings." % i
            for i in range(60)]
    scores = None
    for trial in range(6):
        code = "".join(rng.choice("0123456789") for _ in range(5))
        needle = "The vault access code is %s." % code
        sents = fill[:]
        pos = rng.randrange(10, 50)
        sents.insert(pos, needle)
        text = head + "Notes:\n" + " ".join(sents) + \
            "\n\nQuestion: What is the vault access code?\nAnswer with the code only:" + tail
        ids = llm._ids(text)
        # locate needle token span
        pre = head + "Notes:\n" + " ".join(sents[:pos])
        a = len(llm.tok(pre, add_special_tokens=False).input_ids)
        b = a + len(llm.tok(" " + needle, add_special_tokens=False).input_ids) + 2
        cache = DynamicCache()
        L = ids.shape[1]
        llm.model(input_ids=ids, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(L, device=llm.device),
                  attention_mask=torch.ones(1, L, dtype=torch.long, device=llm.device))
        cur, nxt = L, None
        out = llm.model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True,
                        cache_position=torch.tensor([cur], device=llm.device),
                        attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=llm.device),
                        output_attentions=True)
        # decode a few tokens, accumulate needle-region mass per (layer, head)
        for step in range(6):
            att = out.attentions  # tuple L x [1,H,1,Kl]
            m = torch.stack([a0[0, :, 0, a:b].sum(dim=-1) for a0 in att])  # [L,H]
            scores = m if scores is None else scores + m
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
            out = llm.model(input_ids=torch.tensor([[nxt]], device=llm.device),
                            past_key_values=cache, use_cache=True,
                            cache_position=torch.tensor([cur], device=llm.device),
                            attention_mask=torch.ones(1, cur + 1, dtype=torch.long, device=llm.device),
                            output_attentions=True)
    flat = scores.flatten()
    idx = torch.argsort(flat, descending=True)[:topk]
    H = scores.shape[1]
    return [(int(i // H), int(i % H)) for i in idx]


# ---------- probing ----------
def reduce_step(att, regid, rh_mask, vnorm, nreg, detail=False):
    """att: tuple(L)[1,H,1,Kl]; regid: LongTensor[Kl]; rh_mask: BoolTensor[L,H];
    vnorm: [L, H, Kview] or None beyond view. Returns dict of region vectors + entropies.
    detail=True additionally stores PER-LAYER region masses + entropy (decision positions only)."""
    Ll = len(att)
    A = torch.stack([a0[0, :, 0, :] for a0 in att]).float()  # [L,H,Kl] fp32 (eager emits bf16)
    Kl = A.shape[-1]
    rid = regid[:Kl]
    raw = torch.zeros(Ll, A.shape[1], nreg, device=A.device)
    raw.scatter_add_(2, rid.view(1, 1, -1).expand_as(A), A)
    ent = -(A.clamp_min(1e-9) * A.clamp_min(1e-9).log()).sum(-1)   # [L,H]
    out = {}
    out["raw_all"] = raw.mean(dim=(0, 1))
    out["raw_rh"] = raw[rh_mask].mean(dim=0) if rh_mask.any() else raw.mean(dim=(0, 1))
    if vnorm is not None:
        Kv = vnorm.shape[-1]
        Aw = A[:, :, :Kv] * vnorm
        vw = torch.zeros(Ll, A.shape[1], nreg, device=A.device)
        vw.scatter_add_(2, rid[:Kv].view(1, 1, -1).expand_as(Aw), Aw)
        s = vw.sum(-1, keepdim=True).clamp_min(1e-9)
        vw = vw / s
        out["vw_rh"] = vw[rh_mask].mean(dim=0) if rh_mask.any() else vw.mean(dim=(0, 1))
    out["ent_all"] = ent.mean()
    out["ent_rh"] = ent[rh_mask].mean() if rh_mask.any() else ent.mean()
    if detail:
        out["raw_L"] = raw.mean(dim=1)   # [L, nreg] per-layer all-head region masses
        out["ent_L"] = ent.mean(dim=1)   # [L]
    return {k: (v.tolist() if v.dim() else float(v)) for k, v in out.items()}


@torch.no_grad()
def probe_view(llm, cache, positions, regid_view, qtext, rh_mask, ans_tokens, mask_region=None):
    """Steps qtext then greedy tokens one-by-one with attention probing.
    mask_region: region id to hard-mask (additive -inf) during question+decode (E2)."""
    dev = llm.device
    nreg = len(REG)
    # value norms over the view (per layer, per KV-head expanded to query heads)
    vn = []
    nq_heads = llm.model.config.num_attention_heads
    nkv = llm.model.config.num_key_value_heads
    rep = nq_heads // nkv
    for _, K0, V0 in _iter_cache_kv(cache):
        v = V0[0].norm(dim=-1)                      # [kv, Kview]
        vn.append(v.repeat_interleave(rep, dim=0))  # [H, Kview]
    vnorm = torch.stack(vn).float()                  # [L,H,Kview]
    Kview = vnorm.shape[-1]
    regid = regid_view.tolist()
    qids = llm._ids(qtext)[0].tolist()
    qsplit = qtext.find("\nAnswer")
    nqbody = len(llm.tok(qtext[:qsplit], add_special_tokens=False).input_ids) if qsplit > 0 else len(qids)
    cur = Kview
    nxt_pos = int(positions.max().item()) + 1
    steps = []
    gen_ids = []
    tok_in = qids + [None] * ans_tokens
    nxt = None
    banned = torch.tensor([i for i, r in enumerate(regid) if r == mask_region], device=dev) \
        if mask_region is not None else None
    for si in range(len(qids) + ans_tokens):
        if si < len(qids):
            t = qids[si]
            regid.append(5 if si < nqbody else 6)
        else:
            if nxt is None or nxt in llm.eos_ids:
                break
            t = nxt
            gen_ids.append(t)
            regid.append(6)
        rt = torch.tensor(regid, dtype=torch.long, device=dev)
        if banned is not None and banned.numel():
            am = torch.zeros(1, 1, 1, cur + 1, dtype=llm.model.dtype, device=dev)
            am[0, 0, 0, banned] = torch.finfo(llm.model.dtype).min
        else:
            am = torch.ones(1, cur + 1, dtype=torch.long, device=dev)
        out = llm.model(input_ids=torch.tensor([[t]], device=dev), past_key_values=cache,
                        use_cache=True, position_ids=torch.tensor([[nxt_pos]], device=dev),
                        cache_position=torch.tensor([cur], device=dev),
                        attention_mask=am, output_attentions=True)
        is_dec = (si == len(qids) - 1) or (si == len(qids))   # last question tok / first gen tok
        red = reduce_step(out.attentions, rt, rh_mask, vnorm, nreg, detail=is_dec)
        red["tok"] = t
        red["phase"] = "q" if si < len(qids) else "g"
        if si == len(qids) - 1:  # the dice position: first-token distribution lives HERE
            pr = torch.softmax(out.logits[0, -1].float(), dim=-1)
            tv, ti = torch.topk(pr, 8)
            red["first_top8"] = [[int(i), round(float(v), 5)] for i, v in zip(ti, tv)]
        steps.append(red)
        nxt = int(out.logits[0, -1].argmax())
        cur += 1
        nxt_pos += 1
    ans = llm.tok.decode(gen_ids, skip_special_tokens=True).strip()
    return steps, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/ama_test.jsonl")
    ap.add_argument("--max_tokens", type=int, default=24000)
    ap.add_argument("--n_degen", type=int, default=40)
    ap.add_argument("--n_rand", type=int, default=40)
    ap.add_argument("--n_cited", type=int, default=30)
    ap.add_argument("--ans_tokens", type=int, default=24)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_dir", default="")
    ap.add_argument("--out", default="./out/at.jsonl")
    args = ap.parse_args()

    assert os.environ.get("SPRAG_ATTN_IMPL") == "eager", "attention probing requires eager"
    llm = HFBackend()
    llm.warmup()
    head, tail = split_wrap_nothink(llm)
    router = LexicalRouter()

    # ---- E0 (deterministic per shard) ----
    rheads = needle_heads(llm, head, tail)
    nL = llm.model.config.num_hidden_layers
    nH = llm.model.config.num_attention_heads
    rh_mask = torch.zeros(nL, nH, dtype=torch.bool, device=llm.device)
    for l, h in rheads:
        rh_mask[l, h] = True
    if args.shard == 0:
        json.dump(rheads, open("./out/retrieval_heads.json", "w"))

    # ---- sample selection from the matrix rows (deterministic) ----
    mrows = []
    for pfx in ("mx", "mxh"):
        for f in sorted(glob.glob("./out/%s_ans_s*.jsonl" % pfx)):
            for l in open(f):
                try:
                    mrows.append(json.loads(l))
                except Exception:
                    pass
    CACHE = ["iso", "dum_s", "dum_l", "fixnat", "ext", "randf", "sem", "roll", "anch"]

    def ndeg(r):
        return sum(1 for a in CACHE if _DEG.search((r.get("ans_" + a) or "").strip()))
    degset = [r for r in mrows if ndeg(r) >= 2][: args.n_degen]
    used = {(r["episode_id"], r["q"]) for r in degset}
    rng = random.Random(0)
    rest = [r for r in mrows if (r["episode_id"], r["q"]) not in used]
    rng.shuffle(rest)
    randset, per = [], {"0": 0, "1": 0, "2+": 0}
    for r in rest:
        if per[r["hop"]] < args.n_rand // 3 + 2 and len(randset) < args.n_rand:
            randset.append(r)
            per[r["hop"]] += 1
    used |= {(r["episode_id"], r["q"]) for r in randset}
    citedset = [r for r in rest if (r["episode_id"], r["q"]) not in used
                and _CIT.search(r["q"])][: args.n_cited]
    sample = ([(r, "degen") for r in degset] + [(r, "rand") for r in randset] +
              [(r, "cited") for r in citedset])
    byep = {}
    for r, st in sample:
        byep.setdefault(r["episode_id"], []).append((r, st))
    ep_ids = sorted(byep)
    print("[s%d] sample: %d QAs over %d episodes (degen %d rand %d cited %d) rheads=%d"
          % (args.shard, len(sample), len(ep_ids), len(degset), len(randset), len(citedset),
             len(rheads)), flush=True)

    eps_all = load_episodes(args.data, max_tokens=args.max_tokens)
    byid = {e.episode_id: e for e in eps_all}
    outf = open(args.out, "w", encoding="utf-8")

    def run_episode(eid):
        ep = byid[eid]
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
        turn2idx = {s.turn: i for i, s in enumerate(ep.segments)}
        need = set()
        qas = byep[eid]
        routed = {}
        for r, st in qas:
            kept = sorted(hot | {id2idx[p] for p in router.select(r["q"], old, K) if p in id2idx})
            routed[r["q"]] = kept
            need |= set(kept)
        header_kv = encode_block(llm, header_ids, list(range(H)))
        R_kv = encode_block(llm, header_ids + traj_flat[:w_anch], list(range(H + w_anch)),
                            keep_a=H, keep_b=H + w_anch)
        band, spans_b, _, _ = banded_prefill(llm, header, seg_texts, W)
        stores = {"iso": {}, "anch": {}, "roll": {}}
        for i in sorted(need):
            st_, en = spans[i]
            stores["iso"][i] = encode_block(llm, header_ids + seg_ids[i],
                                            list(range(H)) + list(range(st_, en)), keep_a=H)
            alen = min(w_anch, st_ - H)
            stores["anch"][i] = encode_block(
                llm, header_ids + traj_flat[:alen] + seg_ids[i],
                list(range(H)) + list(range(H, H + alen)) + list(range(st_, en)),
                keep_a=H + alen)
            kt = torch.arange(st_, en, device=llm.device)
            stores["roll"][i] = [(K0.index_select(2, kt).contiguous().cpu(),
                                  V0.index_select(2, kt).contiguous().cpu())
                                 for _, K0, V0 in _iter_cache_kv(band)]
        del band
        torch.cuda.empty_cache()
        hkb = (header_kv, list(range(H)))
        Rblk = (R_kv, list(range(H, H + w_anch)))

        for r, stratum in qas:
            q = r["q"]
            kept = routed[q]
            cited = {turn2idx[int(t)] for t in _CIT.findall(q) if int(t) in turn2idx}
            qtext = "\n\nQuestion: %s\nAnswer concisely and specifically:%s" % (q, tail)
            for arm in ARMS:
                mode = {"iso": "iso", "c_anch": "iso", "roll": "roll",
                        "anch": "anch", "b_anch": "anch", "b_mask": "anch"}[arm]
                blocks = [hkb]
                if arm in ("b_anch", "c_anch", "b_mask"):
                    blocks.append(Rblk)
                blocks += [(stores[mode][i], list(range(spans[i][0], spans[i][1])))
                           for i in sorted(kept)]
                blocks = [blocks[0]] + sorted(blocks[1:], key=lambda b: b[1][0])
                cache, pos = assemble(llm, blocks)
                regid = []
                for bi, b in enumerate(blocks):
                    p0 = b[1][0]
                    for pp in b[1]:
                        if bi == 0:
                            regid.append(0 if pp < 4 else 1)
                        elif b is Rblk:
                            regid.append(2)
                        else:
                            si2 = next(i for i in kept if spans[i][0] <= pp < spans[i][1])
                            regid.append(3 if si2 in cited else 4)
                rt = torch.tensor(regid, dtype=torch.long, device=llm.device)
                steps, ans = probe_view(llm, cache, pos, rt, qtext, rh_mask,
                                        args.ans_tokens,
                                        mask_region=(2 if arm == "b_mask" else None))
                ok = int(judge(llm, head, tail, q, r["gold"], ans)) if arm == "b_mask" \
                    else r.get({"b_anch": "b_anch", "c_anch": "c_anch", "anch": "anch",
                                "roll": "roll", "iso": "iso"}[arm], -1)
                outf.write(json.dumps({
                    "episode_id": eid, "q": q, "hop": r["hop"], "qtype": r["qtype"],
                    "domain": r["domain"], "stratum": stratum, "arm": arm, "ok": ok,
                    "ans": ans[:200], "degen": int(bool(_DEG.search(ans.strip()))),
                    "n_cited_in_view": len(cited & set(kept)),
                    "steps": steps}, ensure_ascii=False) + "\n")
                outf.flush()
                del cache
            torch.cuda.empty_cache()
        del stores, header_kv, R_kv
        torch.cuda.empty_cache()
        print("[s%d] ep %d done (%d QAs)" % (args.shard, eid, len(qas)), flush=True)

    if args.queue_dir:
        os.makedirs(args.queue_dir, exist_ok=True)
        for idx, eid in enumerate(ep_ids):
            try:
                os.mkdir(os.path.join(args.queue_dir, "c%d" % idx))
            except (FileExistsError, OSError):
                continue
            try:
                run_episode(eid)
            except torch.OutOfMemoryError:
                print("[s%d] OOM ep %d skipped" % (args.shard, eid), flush=True)
                torch.cuda.empty_cache()
    else:
        for eid in ep_ids[args.shard::args.nshards]:
            run_episode(eid)
    outf.close()
    print("ATTN_DONE shard=%d" % args.shard, flush=True)


if __name__ == "__main__":
    main()
