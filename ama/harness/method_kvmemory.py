"""kvmemory as an AMA-Bench method plugin.

Drop this in as `src/method/kvmemory.py` inside a clone of github.com/AMA-Bench/AMA-Bench, register it
in `src/method_register.py` (`"kvmemory": ("src.method.kvmemory", "KVMemoryMethod")`), and run it under
their exact backbone (Qwen3-32B / vLLM) + judge (Qwen3-32B, binary yes/no) so the result is directly
comparable to their published Table 5 (AMA-Agent 0.5722 avg, MemoRAG 0.4606, ...).

Their `BaseMethod` contract (verified from src/method/base_method.py + src/memory_interface.py):
  * memory_construction(traj_text: str, task: str = "") -> memory
      `traj_text` is the trajectory pre-flattened to "Step {i}:\nAction: {a}\nObservation: {o}\n\n".
  * memory_retrieve(memory, question: str) -> str
      return a per-query CONTEXT STRING; the interface wraps it into the prompt and calls the backbone.
  Do NOT name the method "longcontext" (that triggers a different list-mode path). The plain context
  string path (`_answer_question_with_index`) is what we want — identical prompt/judge to every baseline.

This is the SAME policy as our text-level `kvmemory` arm (core.KVMemory.assemble): recent turns verbatim,
old turns indexed by a cheap gist, router rehydrates the few old turns each query needs. The compact
selected context is precisely the advantage under their 32k backbone budget (trajectories reach ~1M tok).

Requires the `kvmemory` package importable (PYTHONPATH includes its parent, e.g. /home/tiger).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from src.method.base_method import BaseMethod  # noqa: provided by the AMA-Bench repo

import threading
from transformers import AutoTokenizer  # import at MODULE level (main thread) — importing this lazily

from kvmemory.core import KVMemory, KVMemoryConfig, Segment
from kvmemory.components import (
    AutoGistSummarizer, LexicalRouter, ModelPickRouter, EmbeddingRouter, HybridRouter, MultiHopRouter,
)

# inside a worker thread races transformers' lazy loader → "cannot import name 'AutoTokenizer'".
_TOK_LOCK = threading.Lock()
_TOK_CACHE: dict = {}


def _load_tokenizer(path: str):
    """Thread-safe, load-once tokenizer (the harness answers questions on a 48-way thread pool)."""
    if path not in _TOK_CACHE:
        with _TOK_LOCK:
            if path not in _TOK_CACHE:
                _TOK_CACHE[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOK_CACHE[path]


# Same hazard for the dense embedder: 8 concurrent episodes constructing QwenEmbedder() at once race
# transformers' meta-tensor materialization ("Cannot copy out of meta tensor"). Load it ONCE, locked.
_EMB_LOCK = threading.Lock()
_EMB_CACHE: dict = {}
# Bound CONCURRENT GPU embedding. The harness runs up to N episodes at once; on million-token
# trajectories each embed pass is large, so 8 of them in one process OOMs the embedder GPU. The
# semaphore decouples embed-concurrency from episode-concurrency (answer/judge stay fully parallel).
_EMB_SEM = threading.BoundedSemaphore(int(os.environ.get("SPRAG_EMBED_CONCURRENCY", "2")))


class _SemEmbedder:
    """Serialize-ish the embedder forward under a bounded semaphore (the embedder is a shared singleton)."""

    def __init__(self, emb):
        self._emb = emb

    def __call__(self, *a, **k):
        with _EMB_SEM:
            return self._emb(*a, **k)


def _load_embedder(path):
    key = path or "DEFAULT"
    if key not in _EMB_CACHE:
        with _EMB_LOCK:
            if key not in _EMB_CACHE:
                from kvmemory.embed import QwenEmbedder
                _EMB_CACHE[key] = QwenEmbedder(path)
    return _EMB_CACHE[key]

_STEP = re.compile(r"Step\s+(\d+)\s*:\s*\n?(.*?)(?=\nStep\s+\d+\s*:|\Z)", re.S)
_ACT = re.compile(r"Action:\s*(.*?)(?=\nObservation:|\Z)", re.S)
_OBS = re.compile(r"Observation:\s*(.*)", re.S)


class _ClientLLM:
    """Adapt the AMA-Bench ModelClient (.query(prompt, temperature, max_tokens)) to the .generate()
    interface ModelPickRouter expects. Lets the router reuse the harness backbone for query-aware picks."""

    def __init__(self, client):
        self.client = client

    def generate(self, prompts: list[str], max_tokens: int = 32, **_) -> list[str]:
        return [self.client.query(p, temperature=0.0, max_tokens=max_tokens) for p in prompts]


class KVMemoryMethod(BaseMethod):
    DEFAULTS = {"arm": "kvmemory", "hot_turns": 4, "k_rehydrate": 5, "router": "lexical",
                "hybrid_routers": ["lexical", "model"], "rrf_c": 60, "per_step_cap_tok": 0,
                "embed_max_cands": 1500,  # lexical-prefilter cap for the embed router (bounds GPU mem)
                "max_ctx_tokens": 22000, "tokenizer": "/tmp/Qwen3-32B"}  # TOKEN-cap under the model window

    def __init__(self, config_path: str | None = None, client=None, embedding_engine=None):
        self.client = client
        self._emb = None
        cfg = dict(self.DEFAULTS)
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg.update(json.load(f) if config_path.endswith(".json") else _yaml_load(f))
        self.cfg = cfg

    # ---- BaseMethod contract ----
    def memory_construction(self, traj_text: str, task: str = "") -> Any:
        mem = KVMemory(
            KVMemoryConfig(arm=self.cfg["arm"], hot_turns=int(self.cfg["hot_turns"]),
                           k_rehydrate=int(self.cfg["k_rehydrate"]),
                           per_step_cap_tok=int(self.cfg.get("per_step_cap_tok", 0))),
            summarizer=AutoGistSummarizer(),
            router=self._make_router(),
        )
        for seg in self._parse(traj_text):
            mem.append(seg)
        return mem

    def memory_retrieve(self, memory, question: str) -> str:
        if self.cfg["arm"] == "full":
            # full-context control: the raw trajectory, recency-truncated to the same token budget
            # (drop OLDEST first) -- no routing, no gists, no appendix; same reader/prompt/judge.
            body = "\n\n".join(f"<step {s.turn}>\n{s.text}" for s in memory.segments)
            tok = _load_tokenizer(self.cfg.get("tokenizer", "/tmp/Qwen3-32B"))
            max_tok = int(self.cfg.get("max_ctx_tokens", 22000))
            cc = max_tok * 6
            if len(body) > cc:
                body = body[-cc:]
            ids = tok(body, add_special_tokens=False).input_ids
            if len(ids) > max_tok:
                body = "...[older steps truncated]...\n" + tok.decode(ids[-max_tok:])
            return body
        # routed-split layout: a recency overview (hot verbatim + old gists) + an appendix of the
        # router-picked old turns VERBATIM (the evidence). TOKEN-cap the WHOLE thing under the model
        # window: keep the evidence appendix (trim only its tail if it alone overflows), then fill the
        # remaining budget with the overview's recent end.
        static, dyn = memory.assemble_routed_split(question)
        tok = _load_tokenizer(self.cfg.get("tokenizer", "/tmp/Qwen3-32B"))
        max_tok = int(self.cfg.get("max_ctx_tokens", 22000))
        cc = max_tok * 6  # cheap char pre-truncate so giant (300k-tok) contexts aren't tokenized in full
        if len(dyn) > cc:
            dyn = dyn[:cc]
        if len(static) > cc:
            static = static[-cc:]
        dyn_ids = tok(dyn, add_special_tokens=False).input_ids
        if len(dyn_ids) > max_tok:
            dyn = tok.decode(dyn_ids[:max_tok])
        dyn_block = ("\n\nFull text of the most relevant earlier steps:\n" + dyn) if dyn.strip() else ""
        budget = max_tok - len(tok(dyn_block, add_special_tokens=False).input_ids)
        if budget <= 0:
            return dyn_block.lstrip("\n")
        s_ids = tok(static, add_special_tokens=False).input_ids
        if len(s_ids) > budget:
            static = "…[older steps truncated]…\n" + tok.decode(s_ids[-budget:])
        return static + dyn_block

    # ---- helpers ----
    def _make_router(self):
        return self._router_by_name(self.cfg.get("router", "lexical"))

    def _router_by_name(self, name: str):
        if name == "lexical":
            return LexicalRouter()
        if name == "model":
            if self.client is None:
                raise ValueError("router='model' needs the harness client (pass --llm-server)")
            return ModelPickRouter(_ClientLLM(self.client))
        if name == "embed":
            return EmbeddingRouter(self._embedder(), max_cands=int(self.cfg.get("embed_max_cands", 1500)))
        if name == "hybrid":
            names = self.cfg.get("hybrid_routers", ["lexical", "model"])
            return HybridRouter([self._router_by_name(n) for n in names],
                                rrf_c=int(self.cfg.get("rrf_c", 60)))
        if name == "causal":
            base = self._router_by_name(self.cfg.get("causal_base", "hybrid"))
            return MultiHopRouter(base, temporal=int(self.cfg.get("causal_temporal", 1)),
                                  entity_bridge=bool(self.cfg.get("causal_entity_bridge", True)),
                                  name_steps=bool(self.cfg.get("causal_name_steps", True)))
        raise ValueError(f"unknown router {name!r}")

    def _embedder(self):
        return _SemEmbedder(_load_embedder(os.environ.get("SPRAG_EMBED_PATH")))

    @staticmethod
    def _parse(traj_text: str) -> list[Segment]:
        segs: list[Segment] = []
        for m in _STEP.finditer(traj_text or ""):
            turn, body = int(m.group(1)), m.group(2)
            a = (_ACT.search(body).group(1).strip() if _ACT.search(body) else "")
            o = (_OBS.search(body).group(1).strip() if _OBS.search(body) else "")
            text = "\n".join(x for x in (f"Action: {a}" if a else "", f"Observation: {o}" if o else "") if x)
            segs.append(Segment(seg_id=f"s{turn}", turn=turn, text=text or body.strip(),
                                meta={"action": a, "observation": o}))
        return segs


def _yaml_load(f):
    import yaml
    return yaml.safe_load(f) or {}
