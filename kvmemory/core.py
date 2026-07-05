"""kvmemory.core — recency-tiered verbatim agent memory with a generative router.

Survivor design from the SPRAG experiments (EXP56-71):
  * pure compression (h2o eviction / generative summary / their hybrid) loses sub-salient facts
    on multi-hop recall (EXP66 KILL; EXP67 long-context crossover never favors generative);
  * the real bottleneck is SELECTION, and an agent — being stateful — can keep old context around
    and RE-SELECT per query (EXP62 model-pick is the strongest deployable selector; EXP63 shows
    compressing OLD context is free for recency-biased loads).

So: keep recent turns verbatim (Hot), keep a cheap generative gist of old turns as the router
INDEX (Cold), and on each query let the router rehydrate the exact verbatim old spans it needs
(Warm), leaving the rest as gist/floor. The generative summary is the ROUTER, not the payload.

Backend-agnostic by design: `assemble()` returns a context STRING (text-level path — used for the
first-signal ablation and the AMA-Bench `--method` plugin). A KV-level backend (origpos splice from
EXP64-67) swaps in behind the same Segment/tier/router API to get compress-once / no-reprefill
efficiency without changing the policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

Tier = str  # "hot" | "warm" | "cold"


@dataclass
class Segment:
    """One event in the agent trajectory (a tool call, observation, or reasoning step)."""

    seg_id: str
    turn: int
    text: str                       # verbatim content — the authoritative payload
    kind: str = "step"              # step | observation | action | reasoning | system | question
    n_tok: int = 0
    tier: Tier = "hot"
    gist: Optional[str] = None      # ColdIndex entry: framework auto-gist OR the agent's own note
    meta: dict = field(default_factory=dict)


# ---- pluggable components (the fork / 魔改 surface) ----
class Summarizer(Protocol):
    """Produce the ColdIndex entry for an old segment. Default = framework auto-gist; swap for the
    agent's own scratchpad/Thought when the trajectory provides one."""

    def gist(self, seg: Segment) -> str: ...


class Router(Protocol):
    """Given a query and the old-segment index, pick which segments to rehydrate verbatim.
    Default = ModelPick (ask the model which gists the query needs; EXP62 winner)."""

    def select(self, query: str, candidates: list[Segment], k: int) -> list[str]: ...


class Compressor(Protocol):
    """Lossy floor for old segments the router did not select (text-level proxy of h2o eviction)."""

    def floor(self, seg: Segment, budget_tok: int) -> str: ...


@dataclass
class KVMemoryConfig:
    arm: str = "kvmemory"   # full | recent | gist | retrieval | kvmemory | kv_select
    hot_turns: int = 4      # recent turns kept verbatim (trailing-absorption window)
    k_rehydrate: int = 4    # how many old segments the router brings back verbatim
    floor_tok: int = 0      # per-seg lossy floor for un-selected old segs (0 = fall back to gist)
    per_step_cap_tok: int = 0  # query-focused cap on each rehydrated step (0=off). Under a HARD context
                               # budget (e.g. a 32k window minus the thinking reserve), capping each huge
                               # observation lets the same budget cover MORE distinct steps — the multi-hop
                               # chain — instead of a few giant ones. Only the routed-split path uses it.


class KVMemory:
    """Holds a trajectory as Segments across tiers and assembles a per-query context.

    Per-query loop (the heart):
      1. retier by recency (last `hot_turns` -> Hot/verbatim; older -> Warm/Cold);
      2. ensure each old segment has a gist (ColdIndex);
      3. router selects the old segments this query needs -> rehydrate verbatim (Warm);
      4. everything else -> gist (or lossy floor); Hot stays verbatim.
    """

    def __init__(
        self,
        cfg: KVMemoryConfig,
        *,
        summarizer: Optional[Summarizer] = None,
        router: Optional[Router] = None,
        compressor: Optional[Compressor] = None,
        count_tok: Optional[Callable[[str], int]] = None,
    ):
        self.cfg = cfg
        self.summarizer = summarizer
        self.router = router
        self.compressor = compressor
        self.count_tok = count_tok or (lambda s: max(1, len(s) // 4))
        self.segments: list[Segment] = []
        self.stats = {"queries": 0, "rehydrated": 0, "verbatim_tok": 0}

    def append(self, seg: Segment) -> None:
        if seg.n_tok == 0:
            seg.n_tok = self.count_tok(seg.text)
        self.segments.append(seg)

    # ---- internals ----
    def _retier(self) -> None:
        if not self.segments:
            return
        max_turn = max(s.turn for s in self.segments)
        for s in self.segments:
            s.tier = "hot" if (max_turn - s.turn) < self.cfg.hot_turns else "warm"

    def _ensure_gist(self, seg: Segment) -> str:
        if seg.gist is None and self.summarizer is not None:
            seg.gist = self.summarizer.gist(seg)
        return seg.gist or seg.text[:200]

    def _route(self, query: str, old: list[Segment]) -> set[str]:
        if not (self.router and old):
            return set()
        picked = set(self.router.select(query, old, self.cfg.k_rehydrate))
        self.stats["rehydrated"] += len(picked)
        return picked

    # ---- the per-query context ----
    def assemble(self, query: str) -> str:
        """Return the text context for `query` under the configured arm (text-level path)."""
        self.stats["queries"] += 1
        self._retier()
        old = [s for s in self.segments if s.tier != "hot"]
        arm = self.cfg.arm

        if arm == "full":
            mode = {s.seg_id: "verbatim" for s in self.segments}
        elif arm == "recent":
            mode = {s.seg_id: ("verbatim" if s.tier == "hot" else "drop") for s in self.segments}
        elif arm == "gist":
            for s in old:
                self._ensure_gist(s)
            mode = {s.seg_id: ("verbatim" if s.tier == "hot" else "gist") for s in self.segments}
        elif arm == "retrieval":
            # router over ALL old, no gist floor for the rest (pure retrieve-verbatim baseline)
            picked = self._route(query, old)
            mode = {s.seg_id: ("verbatim" if (s.tier == "hot" or s.seg_id in picked) else "drop")
                    for s in self.segments}
        else:  # "kvmemory": recency tiers + router-rehydrate + gist floor for the rest
            for s in old:
                self._ensure_gist(s)
            picked = self._route(query, old)
            mode = {s.seg_id: ("verbatim" if (s.tier == "hot" or s.seg_id in picked) else "gist")
                    for s in self.segments}

        return self._render(mode)

    def assemble_routed_split(self, query: str) -> tuple[str, str]:
        """Split the kvmemory-arm context for KV reuse on the query-DEPENDENT path.

        Returns (static_body, dynamic_body):
          static_body  = a recency overview identical for every query in the episode — recent turns
                         verbatim + every older turn as its gist. Prefill its KV ONCE and reuse it.
          dynamic_body = the few old turns the router picks for THIS query, verbatim. Re-encode per
                         query (cheap: k spans) and append after the cached static KV.

        The model sees the whole trajectory as a cached recency overview plus the full text of the
        steps this query needs. This is the reuse-friendly form of the `kvmemory` arm: the bulk (the
        overview) is compressed once and answered many times; only the per-query detail is fresh.
        Mirrors the text-level `kvmemory` arm (picked old turns are upgraded to verbatim) but lays the
        verbatim spans in an appendix so the overview prefix stays constant across queries.
        """
        self.stats["queries"] += 1
        self._retier()
        old = [s for s in self.segments if s.tier != "hot"]
        for s in old:
            self._ensure_gist(s)
        static_lines = []
        for s in self.segments:
            if s.tier == "hot":
                static_lines.append(f"<step {s.turn}>\n{s.text}")
            else:
                static_lines.append(f"<step {s.turn}> [summary] {self._ensure_gist(s)}")
        picked = self._route(query, old)
        pick = sorted((s for s in old if s.seg_id in picked), key=lambda s: s.turn)
        cap = self.cfg.per_step_cap_tok
        dyn_lines = [f"<step {s.turn}>\n{self._focus_trim(s.text, query, cap) if cap else s.text}" for s in pick]
        return "\n\n".join(static_lines), "\n\n".join(dyn_lines)

    def _focus_trim(self, text: str, query: str, cap_tok: int) -> str:
        """Query-focused trim of one rehydrated step to ~cap_tok tokens: keep the lines that lexically
        overlap the query (in original order), so a huge observation (a web page, a tool dump) collapses
        to its answer-bearing lines and the budget can hold MORE distinct steps. Falls back to head+tail
        when nothing overlaps (the cue may be non-lexical)."""
        if cap_tok <= 0 or self.count_tok(text) <= cap_tok:
            return text
        import re
        budget = cap_tok * 4  # char proxy (count_tok is ~chars/4)
        qset = set(re.findall(r"[a-z0-9]+", query.lower()))
        lines = text.split("\n")
        hits = [len(qset & set(re.findall(r"[a-z0-9]+", ln.lower()))) for ln in lines]
        if not any(hits):
            h = budget * 2 // 3
            return text[:h] + " …[trimmed]… " + text[-(budget - h):]
        keep, used = set(), 0
        for i in sorted(range(len(lines)), key=lambda i: -hits[i]):
            L = len(lines[i]) + 1
            if used + L > budget and keep:
                break
            keep.add(i); used += L
        return "\n".join(lines[i] for i in sorted(keep))

    def assemble_kv_select(self, query: str) -> tuple[list[int], list[str]]:
        """Return (kept_segment_indices, segment_texts) for the kv_select arm.

        Uses the same routing logic as the kvmemory arm but outputs:
          kept_segment_indices: list of segment INDICES (0-based into self.segments) that should
                                be kept in the sub-selected cache for this query. Includes both
                                hot-tier and router-picked old segments.
          segment_texts:        list of formatted segment texts for ALL segments, for prefill_full.

        The caller (HFBackend) does:
          1. prefill_full(segment_texts, header) ONCE per episode;
          2. subselect_cache(cache, spans, header_len, kept_ids) per query;
          3. answer_from_cache(sub_cache, kept_positions, question) per query.
        """
        self.stats["queries"] += 1
        self._retier()
        old = [s for s in self.segments if s.tier != "hot"]
        for s in old:
            self._ensure_gist(s)
        picked = self._route(query, old)
        kept_ids: list[int] = []
        for i, s in enumerate(self.segments):
            if s.tier == "hot" or s.seg_id in picked:
                kept_ids.append(i)
        segment_texts = [f"<step {s.turn}>\n{s.text}" for s in self.segments]
        return kept_ids, segment_texts

    def _render(self, mode: dict) -> str:
        lines = []
        for s in self.segments:
            m = mode.get(s.seg_id, "verbatim")
            if m == "drop":
                continue
            if m == "verbatim":
                body = s.text
                self.stats["verbatim_tok"] += s.n_tok
            elif m == "gist":
                body = f"[summary of step {s.turn}] {self._ensure_gist(s)}"
            elif m == "floor" and self.compressor:
                body = self.compressor.floor(s, self.cfg.floor_tok)
            else:
                body = s.text
            lines.append(f"<step {s.turn}>\n{body}")
        return "\n\n".join(lines)
