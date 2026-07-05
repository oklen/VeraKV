"""kvmemory — recency-tiered verbatim agent memory with a generative router.

OSS reference implementation embodying the SPRAG survivor design (see core.py): keep recent
turns verbatim, index old turns by a cheap generative gist, and let a router rehydrate the exact
verbatim old spans each query needs. The generative summary routes; the verbatim payload answers.
"""
from .core import Segment, KVMemory, KVMemoryConfig, Router, Summarizer, Compressor
from .components import (
    AutoGistSummarizer, DenseGistSummarizer, ModelPickRouter, LexicalRouter, EmbeddingRouter,
    HybridRouter, MultiHopRouter, TruncateCompressor,
)

__all__ = [
    "Segment", "KVMemory", "KVMemoryConfig", "Router", "Summarizer", "Compressor",
    "AutoGistSummarizer", "DenseGistSummarizer", "ModelPickRouter", "LexicalRouter",
    "EmbeddingRouter", "HybridRouter", "MultiHopRouter", "TruncateCompressor",
]
