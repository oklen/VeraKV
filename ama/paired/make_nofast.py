"""Make retrieve_nofast.py: AMA-Agent's released retrieve.py with ONE change -- the sufficiency check may
never end with a direct answer. Where the released code returns the check's inline answer (the fast
path), the copy proceeds as it does when the check asks for more: the keyword-search round (NEED_CODE
branch, with its own 60 s budget), then the final context (summary + pinned + evidence, char cap as
released) and the reader. A hook records what the released code would have returned at that point, so
one run gives both the released answer (A1 replicate) and A3 on the same calls.

    python ama/paired/make_nofast.py <released retrieve.py> <out retrieve_nofast.py>
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
t = open(src, encoding="utf-8").read()

OLD = """            inline_answer = _extract_inline_answer(sufficiency_response or "")
            if inline_answer:
                return (
                    f"{DIRECT_ANSWER_PREFIX}{inline_answer}{DIRECT_ANSWER_SUFFIX}\\n"
                    f"{context}"
                )
            return context
"""
NEW = """            inline_answer = _extract_inline_answer(sufficiency_response or "")
            if inline_answer:
                # == Paired re-run 2026-10-08, arm A3: the one change ==================================
                # The released code returns the inline answer here (fast path). A3 never lets the check
                # end with a direct answer: it proceeds as when the check asks for more evidence -- the
                # keyword-search round (NEED_CODE branch, its own 60 s budget) -- then assembles the final
                # context and lets the reader answer. FAST_PATH_HOOK only records what the released code
                # would have returned here (for the same-run A1 replicate); it changes nothing.
                if FAST_PATH_HOOK is not None:
                    FAST_PATH_HOOK(f"{DIRECT_ANSWER_PREFIX}{inline_answer}{DIRECT_ANSWER_SUFFIX}\\n{context}")
                _need_code = True
                break
                # ======================================================================================
            return context
"""
assert t.count(OLD) == 1, "fast-path block not found exactly once"
t = t.replace(OLD, NEW)
HOOK = '''DIRECT_ANSWER_SUFFIX = "<<<END_AMA_DIRECT_ANSWER>>>"

# Paired re-run copy (arm A3): set by the driver to record the released fast-path return value.
FAST_PATH_HOOK = None
'''
assert t.count('DIRECT_ANSWER_SUFFIX = "<<<END_AMA_DIRECT_ANSWER>>>"\n') == 1
t = t.replace('DIRECT_ANSWER_SUFFIX = "<<<END_AMA_DIRECT_ANSWER>>>"\n', HOOK, 1)
t = t.replace('"""\nMemory Retrieval Module for AMA-Agent',
              '"""\nMemory Retrieval Module for AMA-Agent -- PAIRED RE-RUN COPY (arm A3): no direct answer from the sufficiency check.', 1)
open(dst, "w", encoding="utf-8").write(t)
print("ok")
