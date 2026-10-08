"""kvmemory.embed — a small dense embedder for the EmbeddingRouter baseline.

Qwen3-Embedding (last-token pooling + L2 norm), same model family as the agent — and the same family
AMA-Bench's AMA-Agent uses for its retrieval stage (Qwen3-4B-embedding). We use the 0.6B here as the
"standard vector retrieval" router baseline to compare against our lexical / model-pick routers.

Kept dependency-light (transformers only, no sentence-transformers) so it runs in the same env as the
HF backend. Loads one small copy on the given device (≈1.2GB bf16; fits beside the 57GB agent on one
80GB A100).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class QwenEmbedder:
    DEFAULT = None  # set SPRAG_EMBED_PATH to a local Qwen3-Embedding-0.6B snapshot
    # the retrieval task instruction is prepended to QUERIES only (Qwen3-Embedding convention)
    QUERY_INSTRUCT = "Given a question about an agent trajectory, retrieve the steps needed to answer it"

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 max_len: int = 1024, dtype=torch.bfloat16):
        model_path = model_path or os.environ.get("SPRAG_EMBED_PATH", self.DEFAULT)
        if not model_path:
            raise RuntimeError("Set SPRAG_EMBED_PATH to a local Qwen3-Embedding-0.6B directory.")
        # Pin the embedder to an explicit GPU. When the agent server already fills GPUs 0-N (and
        # CUDA_VISIBLE_DEVICES may not propagate into the harness thread that lazily builds this),
        # default "cuda:0" lands on the FULL server GPU → OOM. SPRAG_EMBED_DEVICE picks a free one.
        device = device or os.environ.get("SPRAG_EMBED_DEVICE", "cuda:0")
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"  # last-token pooling → last real token sits at index -1
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=False
        ).to(device).eval()  # low_cpu_mem_usage=False → real weights (not a meta tensor) so .to() works
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, texts: list[str], is_query: bool = False, batch_size: int = 16) -> np.ndarray:
        if is_query:
            texts = [f"Instruct: {self.QUERY_INSTRUCT}\nQuery: {t}" for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            enc = self.tok(b, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to(self.device)
            h = self.model(**enc).last_hidden_state          # [n, L, d]
            emb = h[:, -1]                                    # last-token pool (left-padded)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, 0)
