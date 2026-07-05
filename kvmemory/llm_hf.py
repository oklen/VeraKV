"""kvmemory.llm_hf — HuggingFace chat backend (Qwen3-30B-A3B etc.) with KV reuse.

Two roles:
  * generate()       — batched chat completion (used by the router + the text-level replay harness).
  * answer_many()    — compress-once-answer-many: prefill a trajectory PREFIX once, then answer every
                       question of the episode reusing that KV (transformers 5.x DynamicCache +
                       crop-back), instead of re-prefilling the whole context per question. This is
                       the framework's load-bearing efficiency claim.

One full model copy per visible GPU (30B bf16 ~57GB fits one 80GB A100); shard episodes across GPUs
with CUDA_VISIBLE_DEVICES + SHARD_ID. truncation_side='left' drops OLD context first, never the
trailing question. The SPRAG truncation lesson: never let a small budget silently empty the output.
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def _iter_cache_kv(cache):
    """Yield (layer_idx, K, V) across transformers versions: DynamicCache exposes .layers
    (newer) or parallel .key_cache/.value_cache lists (older)."""
    if hasattr(cache, "layers"):
        for i, layer in enumerate(cache.layers):
            yield i, layer.keys, layer.values
    else:
        for i, (K, V) in enumerate(zip(cache.key_cache, cache.value_cache)):
            yield i, K, V


class HFBackend:
    def __init__(self, model_path: str | None = None, dtype=torch.bfloat16, max_ctx: int | None = None):
        model_path = model_path or os.environ.get(
            "SPRAG_MODEL_PATH", "/tmp/Qwen3-30B-A3B-Instruct-2507"
        )
        # context cap (truncation_side='left' drops OLD turns first); raise via SPRAG_MAX_CTX for the
        # big SOFTWARE/OPENWORLD episodes (Qwen3-A3B supports long context natively).
        max_ctx = max_ctx or int(os.environ.get("SPRAG_MAX_CTX", "30000"))
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # optional stress-test knobs (review #1): attention kernel + RoPE (YaRN) scaling, env-gated so
        # default behaviour is unchanged.
        _kw = {}
        _attn = os.environ.get("SPRAG_ATTN_IMPL")  # sdpa | eager | flash_attention_2
        if _attn:
            _kw["attn_implementation"] = _attn
        _rope = os.environ.get("SPRAG_ROPE_FACTOR")  # e.g. "4.0" -> YaRN 4x (extends context)
        if _rope:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            _orig = getattr(_cfg, "max_position_embeddings", 32768)
            # preserve existing rope fields (esp. rope_theta) and just add YaRN scaling
            _base = getattr(_cfg, "rope_parameters", None) or getattr(_cfg, "rope_scaling", None) or {}
            _rp = dict(_base) if isinstance(_base, dict) else {}
            _rp.setdefault("rope_theta", getattr(_cfg, "rope_theta", 1000000.0))
            _rp["rope_type"] = "yarn"
            _rp["factor"] = float(_rope)
            _rp.setdefault("original_max_position_embeddings", _orig)
            _cfg.rope_scaling = _rp            # older transformers
            try:
                _cfg.rope_parameters = _rp      # transformers >=5.x
            except Exception:
                pass
            _kw["config"] = _cfg
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map={"": 0}, trust_remote_code=True, **_kw
            ).eval()
        except (ValueError, ImportError):   # no `accelerate` in this env: load on CPU then move
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True, **_kw
            ).to("cuda:0").eval()
        self.max_ctx = max_ctx
        self.device = self.model.device
        gc = getattr(self.model, "generation_config", None)
        eos = gc.eos_token_id if (gc and gc.eos_token_id is not None) else self.tok.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def count_tok(self, s: str) -> int:
        return len(self.tok(s, add_special_tokens=False).input_ids)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_tokens: int = 200, batch_size: int = 4) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
                )
                for p in batch
            ]
            enc = self.tok(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_ctx, add_special_tokens=False,
            ).to(self.model.device)
            gen = self.model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            for j in range(len(batch)):
                new = gen[j][enc.input_ids.shape[1]:]
                outs.append(self.tok.decode(new, skip_special_tokens=True).strip())
        return outs

    # ---- chat-template prefix/suffix split (so a static prefix can be cached) ----
    def chat_wrap(self, content: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    def split_wrap(self) -> tuple[str, str]:
        """(head, tail) of the chat template around the user content. The reusable prefix is
        head + <static trajectory context>; the per-query suffix is <question> + tail."""
        sent = "<<<SPLITPOINT>>>"
        head, tail = self.chat_wrap(sent).split(sent)
        return head, tail

    # ---- KV reuse: compress once, answer many ----
    def _ids(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    @torch.no_grad()
    def _greedy(self, cache: DynamicCache, prefix_len: int, new_ids: torch.Tensor,
                max_tokens: int) -> tuple[str, float, float]:
        """Greedy-decode from `cache` (already holding `prefix_len` tokens), feeding `new_ids` first.
        Mutates `cache` (appends the suffix + generated tokens); caller crops it back to reuse.
        Returns (text, t_ingest, t_decode): t_ingest = time to process new_ids + emit the first token
        (time-to-first-token — this is what KV reuse shrinks); t_decode = the rest of generation."""
        cur = prefix_len
        L = new_ids.shape[1]
        attn = torch.ones(1, cur + L, dtype=torch.long, device=self.device)
        pos = torch.arange(cur, cur + L, device=self.device)
        torch.cuda.synchronize(); t0 = time.time()
        out = self.model(input_ids=new_ids, past_key_values=cache, use_cache=True,
                         cache_position=pos, attention_mask=attn)
        nxt = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize(); t_ingest = time.time() - t0
        cur += L
        toks: list[int] = []
        t0 = time.time()
        for _ in range(max_tokens):
            if nxt in self.eos_ids:
                break
            toks.append(nxt)
            inp = torch.tensor([[nxt]], device=self.device)
            attn = torch.ones(1, cur + 1, dtype=torch.long, device=self.device)
            pos = torch.tensor([cur], device=self.device)
            out = self.model(input_ids=inp, past_key_values=cache, use_cache=True,
                             cache_position=pos, attention_mask=attn)
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
        torch.cuda.synchronize(); t_decode = time.time() - t0
        return self.tok.decode(toks, skip_special_tokens=True).strip(), t_ingest, t_decode

    @torch.no_grad()
    def warmup(self) -> None:
        """One throwaway prefill+decode so CUDA kernels/cudnn autotune before any timed run."""
        ids = self._ids("warmup")
        self._greedy(DynamicCache(), 0, ids, 4)

    @torch.no_grad()
    def prefill_prefix(self, prefix_text: str) -> tuple[DynamicCache, int]:
        """Prefill a static prefix ONCE; return (cache, prefix_len) to reuse across many queries."""
        ids = self._ids(prefix_text)
        cache = DynamicCache()
        pos = torch.arange(ids.shape[1], device=self.device)
        self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                   cache_position=pos, attention_mask=torch.ones_like(ids))
        return cache, ids.shape[1]

    @torch.no_grad()
    def answer_many(self, prefix_text: str, suffixes: list[str], max_tokens: int = 128,
                    reuse: bool = True) -> tuple[list[str], dict]:
        """Answer many suffixes sharing one `prefix_text` (the episode's static context).

        reuse=True  → prefill the prefix once, crop-back the KV between queries (compress-once-
                      answer-many: only each short suffix is re-encoded → tiny TTFT per query).
        reuse=False → re-prefill prefix+suffix from scratch per query (the cost this removes).
        Returns (answers, timing) with token-identical answers between the two modes. t_ingest is
        the per-query time-to-first-token (the prefill we shrink); t_decode is shared by both modes.
        """
        ans: list[str] = []
        t_ingest: list[float] = []
        t_decode: list[float] = []
        if reuse:
            torch.cuda.synchronize(); t0 = time.time()
            cache, Lp = self.prefill_prefix(prefix_text)
            torch.cuda.synchronize(); t_setup = time.time() - t0
            for sfx in suffixes:
                a, ti, td = self._greedy(cache, Lp, self._ids(sfx), max_tokens)
                cache.crop(Lp)  # reset to the cached prefix for the next query
                ans.append(a); t_ingest.append(ti); t_decode.append(td)
        else:
            pids = self._ids(prefix_text)
            Lp = pids.shape[1]
            t_setup = 0.0
            for sfx in suffixes:
                full = torch.cat([pids, self._ids(sfx)], dim=1)
                a, ti, td = self._greedy(DynamicCache(), 0, full, max_tokens)
                ans.append(a); t_ingest.append(ti); t_decode.append(td)
        n = len(suffixes)
        return ans, {
            "prefix_tok": Lp, "n": n, "reuse": reuse,
            "t_setup": t_setup,                                   # one-time prefix prefill (reuse only)
            "t_ingest_mean": sum(t_ingest) / max(1, n),           # per-query TTFT
            "t_decode_mean": sum(t_decode) / max(1, n),           # per-query decode (mode-independent)
            "t_total": t_setup + sum(t_ingest) + sum(t_decode),   # end-to-end for all n queries
            # tokens pushed through the prefill path (the work we save) — decode-independent:
            "prefill_tok_processed": (Lp + sum(self.count_tok(s) for s in suffixes)) if reuse
            else (n * Lp + sum(self.count_tok(s) for s in suffixes)),
        }

    # ---- KV sub-selection: real cache reuse for selected spans (IMPL_PLAN_B) ----

    @torch.no_grad()
    def _greedy_pos(self, cache: DynamicCache, kept_positions: torch.Tensor,
                    new_ids: torch.Tensor, max_tokens: int) -> tuple[str, float, float]:
        """Like _greedy but uses explicit position_ids for non-contiguous cached positions.

        kept_positions: 1D LongTensor of the original absolute positions of tokens already in cache.
        new_ids: (1, L) token ids to feed now (question tokens).
        Position ids for new_ids start at max(kept_positions)+1; cache_position indexes the appended
        slots starting at len(kept_positions). Decode tokens continue from there.
        """
        cache_len = kept_positions.shape[0]
        L = new_ids.shape[1]
        next_pos = int(kept_positions.max().item()) + 1
        pos = torch.arange(next_pos, next_pos + L, device=self.device)
        cache_pos = torch.arange(cache_len, cache_len + L, device=self.device)
        attn = torch.ones(1, cache_len + L, dtype=torch.long, device=self.device)
        torch.cuda.synchronize(); t0 = time.time()
        out = self.model(input_ids=new_ids, past_key_values=cache, use_cache=True,
                         position_ids=pos.unsqueeze(0), cache_position=cache_pos,
                         attention_mask=attn)
        nxt = int(out.logits[0, -1].argmax())
        torch.cuda.synchronize(); t_ingest = time.time() - t0
        cur = cache_len + L
        cur_pos = next_pos + L
        toks: list[int] = []
        t0 = time.time()
        for _ in range(max_tokens):
            if nxt in self.eos_ids:
                break
            toks.append(nxt)
            inp = torch.tensor([[nxt]], device=self.device)
            attn = torch.ones(1, cur + 1, dtype=torch.long, device=self.device)
            p = torch.tensor([[cur_pos]], device=self.device)
            cp = torch.tensor([cur], device=self.device)
            out = self.model(input_ids=inp, past_key_values=cache, use_cache=True,
                             position_ids=p, cache_position=cp, attention_mask=attn)
            nxt = int(out.logits[0, -1].argmax())
            cur += 1
            cur_pos += 1
        torch.cuda.synchronize(); t_decode = time.time() - t0
        return self.tok.decode(toks, skip_special_tokens=True).strip(), t_ingest, t_decode

    @torch.no_grad()
    def prefill_full(self, segment_texts: list[str], header_text: str = ""
                     ) -> tuple[DynamicCache, list[tuple[int, int]], int]:
        """Prefill header + all segments ONCE. Return (cache, segment_spans, total_len).

        segment_spans[i] = (start_tok, end_tok) of segment i in the cached sequence.
        Header occupies [0, header_len). Tokenize segments separately (add_special_tokens=False)
        to know boundaries, concat ids, single forward at positions [0..total).
        """
        all_ids: list[int] = []
        if header_text:
            hids = self.tok(header_text, add_special_tokens=False).input_ids
            all_ids.extend(hids)
        header_len = len(all_ids)
        spans: list[tuple[int, int]] = []
        for txt in segment_texts:
            sids = self.tok(txt, add_special_tokens=False).input_ids
            start = len(all_ids)
            all_ids.extend(sids)
            spans.append((start, len(all_ids)))
        total = len(all_ids)
        ids_t = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        cache = DynamicCache()
        pos = torch.arange(total, device=self.device)
        self.model(input_ids=ids_t, past_key_values=cache, use_cache=True,
                   cache_position=pos, attention_mask=torch.ones_like(ids_t))
        return cache, spans, total

    @torch.no_grad()
    def subselect_cache(self, cache: DynamicCache, segment_spans: list[tuple[int, int]],
                        header_len: int, kept_ids: list[int]
                        ) -> tuple[DynamicCache, torch.Tensor]:
        """Build a new DynamicCache holding header + kept segments' KV slices.

        Gathers along seq (dim=2) per layer. Keeps tokens in temporal order.
        Returns (sub_cache, kept_positions) where kept_positions are the ORIGINAL
        absolute positions of the retained tokens (for building question position_ids).
        """
        # Build the list of kept token indices (original absolute positions)
        keep_indices: list[int] = list(range(header_len))  # always keep the header
        for sid in sorted(kept_ids):  # temporal order
            start, end = segment_spans[sid]
            keep_indices.extend(range(start, end))
        keep_t = torch.tensor(keep_indices, dtype=torch.long, device=self.device)
        # Gather K/V along dim=2 for each layer
        new_cache = DynamicCache()
        for i, K0, V0 in _iter_cache_kv(cache):
            K = K0.index_select(2, keep_t).contiguous()
            V = V0.index_select(2, keep_t).contiguous()
            new_cache.update(K, V, i)
        # kept_positions = the original absolute positions (same as keep_indices for position-preserving)
        kept_positions = keep_t  # these ARE the original positions
        return new_cache, kept_positions

    @torch.no_grad()
    def answer_from_cache(self, sub_cache: DynamicCache, kept_positions: torch.Tensor,
                          question_text: str, max_tokens: int = 64
                          ) -> tuple[str, float, float]:
        """Decode the question on top of sub_cache using position-preserving positions.

        Question tokens get position_ids starting at max(kept_positions)+1.
        Returns (answer_text, t_ingest, t_decode).
        """
        qids = self._ids(question_text)
        return self._greedy_pos(sub_cache, kept_positions, qids, max_tokens)

    @torch.no_grad()
    def first_logits_subselect(self, cache: DynamicCache, segment_spans: list[tuple[int, int]],
                               header_len: int, kept_ids: list[int],
                               question_text: str) -> torch.Tensor:
        """Get first-token logits via KV sub-selection (for correctness comparison).

        Sub-selects from a full cache, feeds the question, returns the last-position logits.
        Does NOT mutate the original cache (builds a new sub_cache).
        """
        sub_cache, kept_positions = self.subselect_cache(cache, segment_spans, header_len, kept_ids)
        qids = self._ids(question_text)
        cache_len = kept_positions.shape[0]
        L = qids.shape[1]
        next_pos = int(kept_positions.max().item()) + 1
        pos = torch.arange(next_pos, next_pos + L, device=self.device)
        cache_pos = torch.arange(cache_len, cache_len + L, device=self.device)
        attn = torch.ones(1, cache_len + L, dtype=torch.long, device=self.device)
        # use_cache=True (not False): with use_cache=False some HF versions skip reading the provided
        # past_key_values entirely → the question would not attend to the cached spans. sub_cache is a
        # per-query throwaway (index_select copy of the full cache), so appending to it is harmless.
        out = self.model(input_ids=qids, past_key_values=sub_cache, use_cache=True,
                         position_ids=pos.unsqueeze(0), cache_position=cache_pos,
                         attention_mask=attn)
        return out.logits[0, -1].float()

    def _expand_cache(self, cache: DynamicCache, n: int) -> DynamicCache:
        """Replicate a batch-1 prefix cache across n sequences. HF has no shared-prefix paged
        attention, so the shared KV is materialized per sequence here — the real byte saving lives in
        vLLM/SGLang; we only need this to batch the decode."""
        new = DynamicCache()
        for i, K0, V0 in _iter_cache_kv(cache):
            new.update(K0.repeat(n, 1, 1, 1), V0.repeat(n, 1, 1, 1), i)
        return new

    @torch.no_grad()
    def answer_batched(self, prefix_text: str, suffixes: list[str],
                       max_tokens: int = 128) -> tuple[list[str], dict]:
        """Compress-once-answer-many with BATCHED decode: prefill the shared prefix once, replicate
        its KV across the n queries, and decode all n in parallel (one forward per token for the whole
        batch) instead of one query at a time. Decode dominates end-to-end, so batching it is the
        lever that turns the per-query TTFT win into a wall-clock win.

        Left-pads the suffixes and passes the FULL prefix+suffix attention mask so HF derives correct
        (padding-aware) RoPE positions on top of the cached prefix.
        """
        torch.cuda.synchronize(); t0 = time.time()
        pcache, Lp = self.prefill_prefix(prefix_text)
        n = len(suffixes)
        cache = self._expand_cache(pcache, n)
        enc = self.tok(suffixes, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_ctx, add_special_tokens=False).to(self.device)
        Ls = enc.input_ids.shape[1]
        full_mask = torch.cat(
            [torch.ones(n, Lp, dtype=enc.attention_mask.dtype, device=self.device),
             enc.attention_mask], dim=1)
        gen = self.model.generate(
            input_ids=enc.input_ids, attention_mask=full_mask, past_key_values=cache,
            max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        torch.cuda.synchronize(); t_total = time.time() - t0
        ans = [self.tok.decode(gen[j][Ls:], skip_special_tokens=True).strip() for j in range(n)]
        return ans, {"n": n, "prefix_tok": Lp, "t_total": t_total, "t_per_q": t_total / max(1, n)}
