"""Make retrieve_tokcap.py: AMA-Agent's released retrieve.py with ONE change -- the final context cap
(max_context_length = max_model_len - max_tokens = 23,808) is counted in reader tokens instead of
characters. Only `_synthesize` (the final assembly) changes; every replaced line is asserted to occur
exactly once, so the copy cannot drift silently from the released file.

    python ama/paired/make_tokcap.py <released retrieve.py> <out retrieve_tokcap.py>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()

HELPERS = '''
# == Paired re-run 2026-10-08, arm A2: the one change ==========================================
# The final context cap (max_context_length = max_model_len - max_tokens) is counted in reader
# TOKENS (Qwen3-32B tokenizer) instead of characters. Same 70% head / 30% tail rule, same budget
# arithmetic; every other line of this file is the released retrieve.py.
import os as _os
import threading as _threading

_TOK = None
_TOK_LOCK = _threading.Lock()


def _tokenizer():
    global _TOK
    if _TOK is None:
        with _TOK_LOCK:
            if _TOK is None:
                from transformers import AutoTokenizer
                _TOK = AutoTokenizer.from_pretrained(_os.environ.get("AMA_TOKCAP_TOKENIZER", "Qwen/Qwen3-32B"))
    return _TOK


def _ntok(text: str) -> int:
    if not text:
        return 0
    tok = _tokenizer()
    with _TOK_LOCK:
        return len(tok(text, add_special_tokens=False).input_ids)


def _truncate_tokens(text: str, max_tokens: int) -> str:
    """truncate_trajectory_text (utils.py) with its max_length counted in tokens."""
    tok = _tokenizer()
    with _TOK_LOCK:
        ids = tok(text, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens:
            return text
        head_length = int(max_tokens * 0.7)
        tail_length = max_tokens - head_length
        return tok.decode(ids[:head_length]) + "\\n...\\n" + tok.decode(ids[-tail_length:])
# ==============================================================================================

'''

REPL = [
    ("    _overhead = len(task) + 1024\n",
     "    _overhead = _ntok(task) + 1024\n"),
    ("        - len(evidence_body)\n        - len(code_search_section)\n        - len(pinned_section)\n",
     "        - _ntok(evidence_body)\n        - _ntok(code_search_section)\n        - _ntok(pinned_section)\n"),
    ("    if len(state_mem_str) > _state_mem_budget:\n        state_mem_str = truncate_trajectory_text(state_mem_str, _state_mem_budget)\n",
     "    if _ntok(state_mem_str) > _state_mem_budget:\n        state_mem_str = _truncate_tokens(state_mem_str, _state_mem_budget)\n"),
    ("    return truncate_trajectory_text(context, max_context_length)\n",
     "    return _truncate_tokens(context, max_context_length)\n"),
]

a = t.index("\ndef _synthesize(")
b = t.index("\ndef ", a + 1)
body = t[a:b]
for old, new in REPL:
    n = body.count(old)
    assert n == 1, (old, n)
    body = body.replace(old, new)
out = t[:a] + "\n" + HELPERS + body.lstrip("\n") + t[b:]
out = out.replace('"""\nMemory Retrieval Module for AMA-Agent',
                  '"""\nMemory Retrieval Module for AMA-Agent -- PAIRED RE-RUN COPY (arm A2): final context cap in tokens.', 1)
open(dst, "w", encoding="utf-8").write(out)
print("ok", len(t), "->", len(out))
