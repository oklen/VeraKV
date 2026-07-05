"""kvmemory.components — default pluggable implementations (the fork / 魔改 surface).

These embody the SPRAG findings as swappable defaults:
  * AutoGistSummarizer  — cheap extractive ColdIndex (action + observation head). The index must be
    cheap (computed once per old segment); the verbatim payload stays authoritative.
  * ModelPickRouter     — ask the model which past steps a query needs (EXP62: strongest deployable
    selector; query-conditioned at answer time, which a stateful agent is allowed to do).
  * LexicalRouter       — token-overlap fallback (fast, no model call).
  * TruncateCompressor  — text-level lossy floor (head+tail keep) for un-selected old segments.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .core import Segment


class LLM(Protocol):
    def generate(self, prompts: list[str], max_tokens: int = 256) -> list[str]: ...


class AutoGistSummarizer:
    """Cheap extractive gist: 'action=<...> obs=<...>'. One per old segment, computed once."""

    def __init__(self, obs_head: int = 160, act_head: int = 120):
        self.obs_head, self.act_head = obs_head, act_head

    def gist(self, seg: Segment) -> str:
        act = (seg.meta.get("action") or "").strip().replace("\n", " ")
        obs = (seg.meta.get("observation") or seg.text or "").strip().replace("\n", " ")
        g = f"action={act[:self.act_head]} " if act else ""
        g += f"obs={obs[:self.obs_head]}"
        return g.strip()


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class LexicalRouter:
    """Fast overlap router (no model call) — ranks old segments by token overlap with the query."""

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]:
        q = set(_tok(query))
        scored = [(len(q & set(_tok(s.gist or s.text))), s.turn, s.seg_id) for s in candidates]
        scored.sort(reverse=True)
        return [sid for _, _, sid in scored[:k]]


class ModelPickRouter:
    """Ask the model which past steps the query needs (EXP62 winner). Index = gists."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def build_prompt(self, query: str, candidates: list[Segment], k: int, max_index_chars: int = 40000) -> str:
        # The LLM index must fit the model context. Ultra-long trajectories have more old steps than
        # fit (the index alone can exceed the window) — keep the most RECENT that fit the char budget;
        # the long tail is covered by cheaper signals (lexical/embedding) when fused in a HybridRouter.
        lines, total = [], 0
        for s in sorted(candidates, key=lambda s: s.turn, reverse=True):
            line = f"[{s.turn}] {s.gist or s.text[:120]}"
            if lines and total + len(line) + 1 > max_index_chars:
                break
            lines.append(line)
            total += len(line) + 1
        index = "\n".join(reversed(lines))
        return (
            "You are selecting which past trajectory steps are needed to answer a question.\n"
            f"Past steps (index):\n{index}\n\n"
            f"Question: {query}\n\n"
            f"List up to {k} step numbers (from the index) whose FULL content is needed to answer, "
            "most relevant first, as a comma-separated list of integers. Only the numbers.\nAnswer:"
        )

    def parse(self, out: str, candidates: list[Segment], k: int) -> list[str]:
        by_turn = {s.turn: s.seg_id for s in candidates}
        picked, seen = [], set()
        for n in (int(x) for x in re.findall(r"-?\d+", out)):
            if n in by_turn and n not in seen:
                picked.append(by_turn[n]); seen.add(n)
            if len(picked) >= k:
                break
        return picked

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]:
        if not candidates:
            return []
        out = self.llm.generate([self.build_prompt(query, candidates, k)], max_tokens=32)[0]
        return self.parse(out, candidates, k)


class EmbeddingRouter:
    """Dense-retrieval router — the 'standard vector RAG over the trajectory' baseline.

    Embeds the query and each old segment's text, ranks by cosine, takes top-k. This is exactly the
    retrieval stage of Mem0 / A-Mem / AMA-Agent (embed the stored content, similarity top-K). We pit it
    against our LexicalRouter / ModelPickRouter on the same splice layout to answer the reviewer's
    question: does our routing beat plain vector retrieval? The embedder is any callable
    (list[str], is_query: bool) -> np.ndarray [n, d] L2-normalized; default = kvmemory.embed.QwenEmbedder.
    """

    def __init__(self, embedder, use_text: bool = True, max_cands: int = 0):
        self.embedder = embedder
        self.use_text = use_text        # embed full step text (true RAG) vs the gist index
        self.max_cands = max_cands      # 0=embed all; >0 = lexical-prefilter to this many before embedding
        self._cache: dict[str, object] = {}   # text -> embedding (embed each doc once per run)

    def _key(self, s: Segment) -> str:
        return s.text if self.use_text else (s.gist or s.text)

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]:
        if not candidates:
            return []
        import numpy as np
        if self.max_cands and len(candidates) > self.max_cands:
            # On a 1M-token trajectory there are thousands of old steps; embedding ALL of them blows up
            # GPU memory under the harness's concurrent-episode pool. Cheap lexical prefilter keeps the
            # query-relevant ones (recency-tiebroken), then dense reranks within that bounded set.
            q = set(_tok(query))
            candidates = sorted(
                candidates, key=lambda s: (len(q & set(_tok(s.gist or s.text))), s.turn), reverse=True
            )[:self.max_cands]
        keys = [self._key(s) for s in candidates]
        missing = [t for t in dict.fromkeys(keys) if t not in self._cache]
        if missing:
            for t, e in zip(missing, self.embedder(missing, is_query=False)):
                self._cache[t] = e
        D = np.stack([self._cache[t] for t in keys])      # [n, d]
        q = self.embedder([query], is_query=True)[0]       # [d]
        order = np.argsort(-(D @ q))[:k]
        return [candidates[i].seg_id for i in order]


class HybridRouter:
    """Rank-fusion router — combine several routers via Reciprocal Rank Fusion (RRF).

    §8 found lexical / embedding / model-pick are DOMAIN-complementary (model-pick tops spatial
    Game/WEB, embedding tops semantic SQL/EMBODIED, lexical is the free floor). RRF fuses their RANKED
    candidate lists with no score calibration — a segment any router ranks high floats up:

        fused(seg) = Σ_router  1 / (rrf_c + rank_router(seg))

    Take the top-k by fused score (ties broken toward recency). Sub-routers are pluggable: pass
    [LexicalRouter(), EmbeddingRouter(...), ModelPickRouter(...)] for the full panel, or any subset
    (e.g. lexical+model when no embedder is loaded). With one sub-router it degrades to that router.

    Why this is the case-study lever: kvmemory's AMA-Bench headline used the FREE LexicalRouter, §8
    showed the router is the headroom, and the head-to-head loss vs AMA-Agent concentrated on B-causal
    (multi-hop), where a single lexical pass under-recalls the antecedent steps. Fusing a wider,
    complementary candidate net is the cheap first attack before a learned / causal-graph router.
    """

    def __init__(self, routers: list, rrf_c: int = 60, per_router_k: int | None = None):
        if not routers:
            raise ValueError("HybridRouter needs >=1 sub-router")
        self.routers = routers
        self.rrf_c = rrf_c
        self.per_router_k = per_router_k    # candidates pulled from EACH sub-router before fusing

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]:
        if not candidates:
            return []
        m = self.per_router_k or max(k, 4 * k)      # wide net per router so fusion has signal
        from collections import defaultdict
        fused: dict = defaultdict(float)
        for r in self.routers:
            for rank, sid in enumerate(r.select(query, candidates, m)):
                fused[sid] += 1.0 / (self.rrf_c + rank)
        turn = {s.seg_id: s.turn for s in candidates}
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], -turn.get(kv[0], 0)))
        return [sid for sid, _ in ranked[:k]]


class MultiHopRouter:
    """Causal-/multi-hop-aware router: wrap a base router and EXPAND its picks along cheap 'antecedent'
    edges. AMA-Agent earns its B-Causal lead with a per-step LLM causal graph (§10.1); this approximates
    one with NO extra model calls — a multi-hop question needs not only the answer-bearing step but the
    steps that caused it, and three signals surface those:

      1. query-named steps — if the question cites "Step 7"/"Step 8", force those turns in verbatim
         (agent-trajectory questions routinely name the steps they ask about);
      2. temporal neighbors — a step's cause is usually adjacent (turn ± `temporal`);
      3. entity bridges — turns sharing salient tokens with a SEED (not just the query), i.e. the chain
         the query does not name explicitly (carries dialogue, where nothing is step-numbered).

    Seeds come from `base`; the expansion fills the REST of the same k budget (it trades low-value base
    picks for antecedents, not more context). Degrades per task: (1) fires on trajectories, (3) on dialogue.

    TESTED, OFF BY DEFAULT (FINDINGS §11.2). A budget-bound tradeoff, NOT a flat loss. On AMA-Bench's >200k
    tail: net −0.018 vs a SAME-SERVER hybrid+cap control (0.560 vs 0.577 — the old-server 0.619 was cross-
    build variance, re-baseline on the same server!). It LIFTS the B-causal multi-hop category it targets
    (+0.068) but ROBS C-state (−0.094) and A-recall, because under the forced 22k budget every antecedent it
    forces in evicts a hybrid pick → net slightly negative. On LOCOMO it's a literal no-op (selection-fail
    byte-identical to the hybrid base): `step N` never fires on dialogue and the base already fills k, and the
    entity/temporal links it would add are already captured by dense embedding. ⇒ keep the hybrid lexical+embed
    router as default; the open lever is a BUDGET-ADAPTIVE/learned router that adds antecedents WITHOUT
    evicting state/recall picks (spend extra context only when the question is multi-hop).
    """

    def __init__(self, base, temporal: int = 1, entity_bridge: bool = True, name_steps: bool = True):
        self.base = base
        self.temporal = temporal
        self.entity_bridge = entity_bridge
        self.name_steps = name_steps

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]:
        if not candidates:
            return []
        by_id = {s.seg_id: s for s in candidates}
        id_by_turn = {s.turn: s.seg_id for s in candidates}
        turn_of = {s.seg_id: s.turn for s in candidates}
        seeds = self.base.select(query, candidates, k)
        anchors = seeds[:max(2, k // 3)]                 # expand around the strongest seeds only

        named = []                                       # (1) steps/turns the QUERY names ("Turn 38", "steps 28-30")
        if self.name_steps:
            for m in re.finditer(r"(?:turn|step)s?\s*#?\s*(\d+)(?:\s*(?:[-–—]|to)\s*(\d+))?", query.lower()):
                a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
                if not (a <= b <= a + 12):               # ignore reversed / spuriously huge ranges
                    b = a
                for t in range(a, b + 1):
                    shuf = os.environ.get("SPRAG_PIN_SHUFFLE")   # ablation: corrupt the address (never resolve correctly)
                    if shuf == "2":                              # far-random wrong turn (deterministic)
                        ts = sorted(id_by_turn)
                        t2 = ts[(t * 7 + 13) % len(ts)]
                        t = t2 if t2 != t else ts[(t * 7 + 14) % len(ts)]
                    elif shuf:                                   # near-miss: off by 3
                        t = t + 3 if (t + 3) in id_by_turn else (t - 3)
                        if t not in id_by_turn:
                            continue
                    sid = id_by_turn.get(t)
                    if sid is not None and sid not in named:
                        named.append(sid)

        expand = []
        for sid in named + anchors:                      # (2) temporal neighbors of named + top seeds
            t = turn_of.get(sid)
            if t is None:
                continue
            for dt in range(1, self.temporal + 1):
                for tt in (t - dt, t + dt):
                    if tt in id_by_turn:
                        expand.append(id_by_turn[tt])

        if self.entity_bridge and anchors:               # (3) turns sharing salient tokens with anchors
            atoks: set = set()
            for sid in anchors:
                s = by_id.get(sid)
                if s is not None:
                    atoks |= set(_tok(s.gist or s.text))
            if atoks:
                scored = sorted(candidates, key=lambda s: len(atoks & set(_tok(s.gist or s.text))),
                                reverse=True)
                expand += [s.seg_id for s in scored[:k]]

        out, seen = [], set()                            # named → base seeds → expansions, dedup, to k
        for sid in named + seeds + expand:
            if sid in seen or sid not in by_id:
                continue
            out.append(sid)
            seen.add(sid)
            if len(out) >= k:
                break
        return out


class DenseGistSummarizer:
    """Model-generated DENSE gist: copy exact identifiers/numbers/SQL/paths VERBATIM, terse, no filler
    (EXP67 DENSE_CUE). One LLM call per segment (use gist_batch). Higher-signal than the extractive
    truncation — used to isolate 'is the FLOOR bad, or is the cheap gist bad?'."""

    PROMPT = (
        "Summarize this agent step into ONE dense line for a lookup index. Copy exact identifiers, "
        "numbers, table/column names, SQL, file paths, element IDs, and counts VERBATIM — do NOT "
        "paraphrase or truncate them. Terse, no filler, no full sentences.\n\nSTEP:\n{body}\n\nDense line:"
    )

    def __init__(self, llm: LLM, max_tokens: int = 96, max_chars: int = 3000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.max_chars = max_chars

    def gist_batch(self, segs: list[Segment]) -> list[str]:
        prompts = [self.PROMPT.format(body=s.text[:self.max_chars]) for s in segs]
        outs = self.llm.generate(prompts, max_tokens=self.max_tokens, batch_size=8)
        return [o.replace("\n", " ").strip() for o in outs]

    def gist(self, seg: Segment) -> str:
        return self.gist_batch([seg])[0]


class TruncateCompressor:
    """Text-level lossy floor: keep head+tail of an un-selected old segment up to budget."""

    def floor(self, seg: Segment, budget_tok: int) -> str:
        budget = max(40, budget_tok * 4)
        t = seg.text
        if len(t) <= budget:
            return t
        h = budget * 2 // 3
        return t[:h] + " …[trimmed]… " + t[-(budget - h):]
