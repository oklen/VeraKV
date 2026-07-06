"""Agentic readers for KVMemory (env-gated via AMA_AGENTIC_READER).

Two modes (AMA_AGENTIC_MODE):
  * "code"       : reason + optional Python compute over the SELECTED evidence, then answer.
                   Diagnoses compute-bound failures (evidence present, reader can't compute).
  * "reretrieve" : self-evaluate evidence sufficiency step-by-step; if something is missing,
                   emit a search query and re-retrieve (model-driven); the sufficiency
                   ANALYSIS is preserved and shown to the final reader. This moves the
                   structured reasoning into a principled self-eval step (subsumes the
                   answer-prompt) and fixes retrieval misses.
"""
import os, re, subprocess, tempfile

MODE = os.environ.get("AMA_AGENTIC_MODE", "code")
MAX_STEPS = int(os.environ.get("AMA_AGENTIC_STEPS", "3"))
CODE_TIMEOUT = int(os.environ.get("AMA_AGENTIC_TIMEOUT", "8"))
# 32k window: reserve room for the completion so multi-step prompts never overflow.
CTX_CHARS = int(os.environ.get("AMA_AGENTIC_CTXCHARS", "96000"))   # ~24k tokens
COMP_TOK = int(os.environ.get("AMA_AGENTIC_COMP", "2048"))
# worker /tmp is per-login-container (not shared) -> write debug to shared CephFS so it's readable.
LOGBASE = os.environ.get("AMA_AGENTIC_LOG", "/tmp/sel")

def _cap(c):
    return c if len(c) <= CTX_CHARS else (c[:CTX_CHARS] + "\n...[evidence truncated]")

# ----- entry point -----
def agentic_answer(client, question, context, max_tokens, extract_final_answer, method=None, memory=None):
    if MODE == "plan2":
        return _plan_answer(client, question, context, max_tokens, extract_final_answer, method, memory, v2=True)
    if MODE == "handoff":
        return _handoff_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "adapt":
        return _adapt_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "packet2":
        return _packet_answer(client, question, context, max_tokens, extract_final_answer, method, memory, v2=True)
    if MODE == "packet":
        return _packet_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "facts":
        return _facts_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "extlines":
        return _extlines_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "loop":
        return _loop_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "lossy":
        return _lossy_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "plan":
        return _plan_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "tools":
        return _tools_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "state":
        return _state_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "decouple":
        return _decouple_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "gateonly":
        return _gated_answer(client, question, context, max_tokens, extract_final_answer, method, memory, noretrieve=True)
    if MODE == "gated":
        return _gated_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "select":
        return _select_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    if MODE == "reretrieve" and method is not None and memory is not None:
        return _reretrieve_answer(client, question, context, max_tokens, extract_final_answer, method, memory)
    return _code_answer(client, question, context, max_tokens, extract_final_answer)


# ===================== mode: gated (structured-first; re-retrieve ONLY when model flags insufficiency) =====================
# Default path is byte-for-byte the harness structured prompt + FULL max_tokens (no compression, no cap),
# so it cannot regress below structured. A minimal escape clause lets the model request more evidence.
GATED_ESCAPE = (
    "\n\nIf — and ONLY if — the context is genuinely missing a specific fact required to answer, "
    "do not guess: instead output a single line `INSUFFICIENT: <a precise search query for the missing "
    "fact>` and nothing else. Otherwise, answer normally in the format below."
)

def _structured_instr():
    f = os.environ.get("AMA_ANSWER_INSTR_FILE")
    if f and os.path.exists(f):
        try:
            return open(f).read().strip()
        except Exception:
            pass
    return "Provide a direct and concise answer."

def _harness_prompt(context, question, instr):
    return ("%s\n\n## Questions\nQuestion 1: %s\n\n## Instructions\n%s\n\nAnswer[1]: [your answer here]"
            % (context, question, instr))

GCAP = int(os.environ.get("AMA_GATED_CTXCHARS", "84000"))  # ~21k tokens; leaves completion room in 32k

def _gcap(c):
    return c if len(c) <= GCAP else (c[:GCAP] + "\n...[context truncated]")

def _gated_answer(client, question, context, max_tokens, xfa, method, memory, noretrieve=False):
    base = _structured_instr()
    resp = ""; ans = None; fired = "ANS"
    if os.environ.get("AMA_AGENTIC_DBG"):  # step-recall diagnostic: is the asked step in our context?
        try:
            asked = re.findall(r"(?:step|turn)s?\s*#?\s*(\d+)", question.lower())
            present = [n for n in asked if ("<step %s>" % n) in context]
            open(LOGBASE + "_recall.log", "a").write(
                "asked=%s present=%s ctxlen=%d q=%s\n" % (asked, present, len(context), question[:60]))
        except Exception:
            pass
    try:
        # pass 1: byte-for-byte the harness structured prompt + escape, FULL budget (cannot regress)
        resp = client.query(_harness_prompt(context, question, base + GATED_ESCAPE),
                            temperature=0.0, max_tokens=max_tokens)
        ins = re.search(r"INSUFFICIENT:\s*(.+)", resp)
        ans = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        if ins and not (ans and resp.rfind("Answer[1]:") > resp.rfind("INSUFFICIENT:")):
            if noretrieve:
                # gate-only ablation: the gate fired, but we DON'T re-retrieve -> answer over the
                # ORIGINAL context. Isolates the escape-clause perturbation from the re-retrieve action.
                resp = client.query(_harness_prompt(context, question, base),
                                    temperature=0.0, max_tokens=max_tokens)
                ans = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
                fired = "GATEONLY"
            else:
                q2 = ins.group(1).strip().strip('"').strip()[:200]
                try:
                    more = method.memory_retrieve(memory, q2) if (method is not None and memory is not None) else ""
                except Exception:
                    more = ""
                aug = context + ("\n\nFull text of additional retrieved steps:\n" + more
                                 if (more and more[:200] not in context) else "")
                # pass 2: NO escape, capped context so augmentation can't overflow the window
                resp = client.query(_harness_prompt(_gcap(aug), question, base),
                                    temperature=0.0, max_tokens=min(max_tokens, 4096))
                ans = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
                fired = "RETRIEVE:" + q2[:50]
    except Exception:
        # robust: never crash the shard -> one capped structured call
        fired = "FALLBACK"
        try:
            resp = client.query(_harness_prompt(_gcap(context), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
            ans = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        except Exception:
            resp = ""; ans = None
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("gated=%s q=%s\n" % (fired, question[:70]))
        except Exception:
            pass
    txt = ("###Answer: %s" % ans.group(1).strip()) if ans else resp
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": resp}


# ===================== mode: select (model-as-selector + reason + optional re-query) =====================
SELECT_INSTR = (
    "You are given numbered evidence chunks (each tagged [E#]) and one question.\n"
    "Think step by step about how to answer using ONLY these chunks, then output EXACTLY:\n"
    "THINKING: <concise step-by-step reasoning that reaches the answer if the evidence allows>\n"
    "KEEP: E<id>, E<id>, ...   (ONLY the chunk ids that are essential evidence for the answer)\n"
    "STATUS: ANSWERABLE            (if the kept chunks are sufficient)\n"
    "   -- OR, if a required fact is absent from the evidence, replace the STATUS line with --\n"
    "STATUS: NEED <one short precise search query for the single most important missing fact>\n"
)

def _chunk(context, size=2600):
    # Prefer STEP boundaries ("<step N>") so the model can select the exact step a question asks about
    # (SOFTWARE/trajectory point-lookups). Hard-split any oversized step; fall back to fixed-size when
    # the evidence isn't step-structured.
    parts = [p for p in re.split(r"(?=<step \d+>)", context) if p.strip()]
    if len(parts) >= 3:
        out = []
        for p in parts:
            if len(p) <= 2 * size:
                out.append(p)
            else:
                out += [p[i:i + size] for i in range(0, len(p), size)]
        return out
    cs = [context[i:i + size] for i in range(0, len(context), size)]
    return cs if cs else [""]

def _select_answer(client, question, context, max_tokens, xfa, method, memory):
    comp = min(max_tokens, COMP_TOK)
    chunks = _chunk(context)
    thinking_all = ""
    kept = []
    for rnd in range(2):
        numbered = "\n\n".join("[E%d] %s" % (i, chunks[i]) for i in range(len(chunks)))
        p = ("## Evidence chunks\n%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s"
             % (_cap(numbered), question, SELECT_INSTR))
        resp = client.query(p, temperature=0.0, max_tokens=comp)
        tm = re.search(r"THINKING:\s*(.*?)(?:\n\s*KEEP:|\n\s*STATUS:|\Z)", resp, re.DOTALL)
        thinking_all += ((tm.group(1).strip() if tm else resp.strip())[:2500]) + "\n"
        km = re.search(r"KEEP:\s*([^\n]*)", resp)
        ids = sorted(set(int(x) for x in re.findall(r"E(\d+)", km.group(1)))) if km else []
        for i in ids:
            if 0 <= i < len(chunks):
                kept.append(chunks[i])
        nm = re.search(r"STATUS:\s*NEED\s*(.+)", resp)
        if os.environ.get("AMA_AGENTIC_DBG"):
            try:
                open(LOGBASE + "_dbg.log", "a").write(
                    "rnd=%d nchunks=%d kept=%s status=%s q=%s\n"
                    % (rnd, len(chunks), ids, ("NEED" if nm else "ANS"), question[:70]))
            except Exception:
                pass
        if nm and rnd == 0 and method is not None and memory is not None:
            q2 = nm.group(1).strip().strip('"').strip()[:200]
            try:
                more = method.memory_retrieve(memory, q2)
            except Exception:  # noqa
                more = ""
            if more:
                chunks = _chunk(more)   # round 2 selects over the re-retrieved evidence
                continue
        break
    kept_text = "\n\n".join(kept) if kept else _cap(context)
    fp = ("## Selected evidence\n%s\n\n## Reasoning\n%s\n\n## Question\nQuestion 1: %s\n\n"
          "## Instructions\nUsing the selected evidence and reasoning above, answer precisely.\n"
          "Answer[1]: [your answer here]" % (kept_text[:70000], thinking_all[:8000], question))
    resp = client.query(fp, temperature=0.0, max_tokens=comp)
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG") == "2":
        try:
            import json as _j
            open(LOGBASE + "_full.jsonl", "a").write(_j.dumps({
                "q": question[:220], "ctx_len": len(context), "n_chunks": len(chunks),
                "kept_ids": ids, "status": ("NEED" if nm else "ANS"), "kept_len": len(kept_text),
                "thinking": thinking_all[:1800], "answer_raw": resp[:900], "final_ans": txt[:400],
            }) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": thinking_all}


# ===================== mode: lossy (same-pipeline payload swap: summarize ONLY the verbatim evidence appendix) =====================
# Reviewer ask: the oracle payload ablation (Fig. payload) uses a stripped-down single-step setting whose
# scale (~37%) is far below the main table. This mode isolates the payload IN the deployed pipeline: same
# QA, same overview, same router-selected steps, same reader/prompt/judge — only the evidence appendix is
# re-encoded as an LLM summary instead of verbatim. Delta vs the structured anchor = system-level payload cost.
LOSSY_MARK = "\n\nFull text of the most relevant earlier steps:\n"
LOSSY_SUMM = ("Rewrite each step below as a 2-3 sentence summary that preserves the key facts (what was "
              "done, on what, and the outcome). Keep the '<step N>' headers, one per step, in order. Do "
              "NOT answer any question. Output only the summaries.")

def _lossy_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    newctx = context; swapped = 0
    if LOSSY_MARK in context:
        static, dyn = context.split(LOSSY_MARK, 1)
        try:
            respS = client.query("## Steps\n%s\n\n## Instructions\n%s" % (dyn, LOSSY_SUMM),
                                 temperature=0.0, max_tokens=min(max_tokens, 3072))
            clean = (respS.split("</think>")[-1] if "</think>" in respS else respS).strip()
            if clean:
                newctx = static + "\n\nSummary of the most relevant earlier steps:\n" + clean[:14000]
                swapped = 1
        except Exception:
            newctx = context
    try:
        resp = client.query(_harness_prompt(newctx, question, base), temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(newctx), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("lossy=%d q=%s\n" % (swapped, question[:60]))
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


# ===================== mode: packet (RAMP MVP: reorganize the appendix into a reasoning-affordant view) =====================
# Tests whether the structured-prompt gain can be moved into MEMORY PRESENTATION: the compiler reorganizes
# the SAME selected verbatim evidence into need-groups / occurrence lists / timelines (NO conclusions, NO
# instructions to the reader), and the answer call uses the DEFAULT harness prompt. Guard: if the compile
# fails or drops the appendix, fall back to the raw context.
PACKET_INSTR = (
    "You are a memory-view compiler. Reorganize the evidence steps below into a reasoning-ready view for "
    "the question. Hard rules:\n"
    "1. Do NOT answer the question. Do NOT state conclusions, inferences, or recommendations.\n"
    "2. Preserve decisive content VERBATIM: copy values, commands, file paths, matrices, error messages, "
    "names, counts, and step indices exactly as written. Never paraphrase them.\n"
    "3. Organize, don't create: group the evidence by the sub-need of the question it serves; within a "
    "group keep temporal order; every content line cites its step number and quotes verbatim.\n"
    "4. If the question asks how many / which steps / list occurrences: add an [Occurrences] section that "
    "enumerates EVERY matching event, one line per step, with the verbatim trigger line.\n"
    "5. If the question involves state changing over time: add a [Timeline] section, one line per "
    "(entity, step) with the verbatim value or change.\n"
    "6. If spans conflict, list both under [Conflicting evidence] with step numbers. If something the "
    "question needs is not in the evidence, say so under [Missing evidence].\n"
    "Output ONLY the view, in this format:\n"
    "[Evidence needs]\n- ...\n[Evidence groups]\nGroup A (need): <step N> \"verbatim\" ...\n"
    "[Occurrences] (if applicable)\n[Timeline] (if applicable)\n[Conflicting evidence] (if any)\n"
    "[Missing evidence] (if any)"
)

# ===================== mode: handoff (answer-accountable memory-side reasoning; goal experiment) =====================
# Plan-poisoning case study diagnosis: memory-side reasoning fails NOT because reasoning cannot run
# upstream, but because it is generated without answer-accountability (absence commitments freeze early;
# modality misreads are never self-corrected against the question; sub-question framing displaces the
# question's own rubric) and consumed with authority. Test: the memory service runs the FULL structured
# answer-directed derivation itself at full budget; the artifact is handed to the DEFAULT reader as a
# prior analysis to verify. AMA_HANDOFF_STRIP=1 deterministically removes the final-answer line from the
# SAME upstream generation, separating "the answer transfers" from "the derivation transfers".
HANDOFF_DEFAULT = "Provide a direct and concise answer."
HANDOFF_TRUST = ("Provide a direct and concise answer. The context ends with a prior analysis of this "
                 "question by the memory service; briefly verify it against the evidence and, unless it "
                 "plainly contradicts the evidence, give its final answer.")

# Blocker-2 attack (goal follow-up): under the plain structured instruction the derivation happens inside
# <think> and 99.2% of visible artifacts begin "Answer[1]:" -- only the product transfers. AMA_HANDOFF_EXPORT=1
# appends an explicit visible-format demand plus a schematic few-shot to the SAME structured instruction,
# forcing the derivation (verbatim evidence quotes + numbered reasoning) into the artifact itself.
# AMA_HANDOFF_VERIFY=1 replaces blanket trust with a verification procedure whose default outcome is adoption:
# the reader can CHECK an exported derivation instead of re-deriving, which a bare answer never allowed.
HANDOFF_EXPORT_SUFFIX = """

Your visible reply (the part after your private thinking) must lay out the full worked derivation, not just the final answer, in exactly this format:

Evidence:
- step <N>: "<verbatim quote from that step>" -- <what it establishes>
(2-6 bullets; only the steps that decide the answer)

Derivation:
1. <sub-question> -> <conclusion, citing step numbers>
(numbered lines; end with the check that no other step overturns the conclusion)

Answer[1]: <final answer>

Example of the required reply shape (illustrative only, unrelated to the actual context):

Evidence:
- step 12: "You open the cabinet 2. In it, you see a mug 1." -- mug 1 first seen in cabinet 2
- step 19: "You put the mug 1 in/on the desk 1." -- mug 1 moved to desk 1

Derivation:
1. Where is mug 1 last placed? -> step 19 puts it on desk 1, after step 12.
2. Does any step after 19 move mug 1? -> no later step touches mug 1.

Answer[1]: desk 1"""

HANDOFF_VERIFY = ("Provide a direct and concise answer. The context ends with a worked derivation of this "
                  "question by the memory service: evidence quotes, its reasoning steps, and its final answer. "
                  "Do not re-derive the answer from scratch. Instead, check the derivation: confirm its quotes "
                  "appear in the evidence above and that no step it overlooked contradicts its conclusion. If it "
                  "survives these checks, answer with its final answer. Depart from it only if you can name the "
                  "specific quote or step that is wrong, and then answer with the correction.")

# S2 variants: PRSXP measures the export tax directly (upstream export pass answered in place, no relay);
# PRAV relays the untouched structured answer under a verify-adopt reader (no export demand, no tax);
# PRXC is a minimal-footprint export (answer + deciding step numbers only, no derivation prose).
HANDOFF_EXPORT_COMPACT = ("\n\nEnd your visible reply with exactly one line in this form:\n"
                          "Answer[1]: <final answer> [evidence: steps <N>, <M>, ...]\n"
                          "listing only the step numbers that decide the answer.")

# Tax-decomposition controls (audit found SXPH answers 3.5x shorter than anchor: the terse few-shot
# compressed final answers, confounding reasoning damage with completeness-credit loss at the judge).
# EXPORT=3: brevity demand alone, no derivation export -- prices the compression penalty by itself.
# EXPORT=4: same Evidence/Derivation export, but the Answer line and few-shot demand a COMPLETE final
# answer -- the residual is the process tax with verbosity restored.
HANDOFF_BREVITY_SUFFIX = ("\n\nKeep your final answer brief: state only the conclusion, in at most three "
                          "sentences. Do not enumerate your reasoning or list evidence in the answer.")

HANDOFF_EXPORT_SUFFIX_FULL = """

Your visible reply (the part after your private thinking) must lay out the full worked derivation AND a complete final answer, in exactly this format:

Evidence:
- step <N>: "<verbatim quote from that step>" -- <what it establishes>
(2-6 bullets; only the steps that decide the answer)

Derivation:
1. <sub-question> -> <conclusion, citing step numbers>
(numbered lines; end with the check that no other step overturns the conclusion)

Answer[1]: <your COMPLETE final answer -- write it exactly as fully as you would if the sections above did not exist: full sentences, every required part, step citations included. Do not compress it.>

Example of the required reply shape (illustrative only, unrelated to the actual context):

Evidence:
- step 12: "You open the cabinet 2. In it, you see a mug 1." -- mug 1 first seen in cabinet 2
- step 19: "You put the mug 1 in/on the desk 1." -- mug 1 moved to desk 1

Derivation:
1. Where is mug 1 last placed? -> step 19 puts it on desk 1, after step 12.
2. Does any step after 19 move mug 1? -> no later step touches mug 1.

Answer[1]: The mug 1 ends up on the desk 1. It was first seen inside cabinet 2 when the agent opened it at step 12, and at step 19 the agent explicitly put the mug 1 in/on the desk 1; no step after 19 moves or removes it, so its final location is the desk 1."""

# Reverse test (does more output fully recover?): a shared maximal-completeness answer demand,
# applied WITH the export format (EXPORT=5) and WITHOUT it (EXPORT=6). If the two arms score the
# same at comparable answer lengths, the derivation format itself is harmless; any gap that
# survives length-matching is the isolated process tax. EXPORT=6 also probes the ceiling: does
# demanding more-complete answers beat the anchor's natural output?
ANSWER_COMPLETENESS = (
    "Your final answer must be COMPLETE and self-contained: address every part of the question, "
    "enumerate every required element with step citations, and write it at the length the question "
    "deserves -- typically 5-12 full sentences (or a structured multi-part answer). A short summary "
    "is NOT acceptable.")

HANDOFF_EXPORT_SUFFIX_V = """

Your visible reply (the part after your private thinking) must lay out the full worked derivation AND a complete final answer, in exactly this format:

Evidence:
- step <N>: "<verbatim quote from that step>" -- <what it establishes>
(2-6 bullets; only the steps that decide the answer)

Derivation:
1. <sub-question> -> <conclusion, citing step numbers>
(numbered lines; end with the check that no other step overturns the conclusion)

Answer[1]: <""" + ANSWER_COMPLETENESS + """>

Example of the required reply shape (illustrative only, unrelated to the actual context):

Evidence:
- step 12: "You open the cabinet 2. In it, you see a mug 1." -- mug 1 first seen in cabinet 2
- step 19: "You put the mug 1 in/on the desk 1." -- mug 1 moved to desk 1

Derivation:
1. Where is mug 1 last placed? -> step 19 puts it on desk 1, after step 12.
2. Does any step after 19 move mug 1? -> no later step touches mug 1.

Answer[1]: The mug 1 ends up on the desk 1. It was first observed inside cabinet 2 when the agent opened that cabinet at step 12, which establishes where it started. At step 19 the agent explicitly performed "put the mug 1 in/on the desk 1", which is the last action affecting the mug anywhere in the trajectory. No step after 19 picks the mug up again, moves it, or removes it, and no other container is ever associated with it afterwards. Its final location is therefore the desk 1, per steps 12 and 19."""

HANDOFF_VERBOSE_ONLY = "\n\n" + ANSWER_COMPLETENESS

# Relay-stage completeness protection (AMA_HANDOFF_FULLANS=1): the relay readers' "direct and
# concise" prefix compresses their final answers (522-666 med chars) onto the same penalty curve
# that explained the export tax -- the 0.593 relay ceiling is answer-form, not friction. This rider
# swaps the prefix and demands the adopted answer be re-emitted in full.
HANDOFF_READER_FULLANS = (" When you give the final answer, give it IN FULL: address every part of the "
                          "question, enumerate every required element with step citations, and write it at "
                          "the length the question deserves -- typically 5-12 full sentences (or a structured "
                          "multi-part answer). Do not compress it into a one-line summary.")

HANDOFF_VERIFY_LITE = ("Provide a direct and concise answer. The context ends with the memory service's prior "
                       "analysis of this question, ending in its final answer and the steps it relied on. Do not "
                       "re-derive the answer from scratch: check those steps against the evidence above, and if "
                       "nothing you can name contradicts the answer, give its final answer as yours. Depart from "
                       "it only if you can cite the specific step that contradicts it, and then answer with the "
                       "correction.")

WORKSHEET_INSTR = (
    "You are the memory service preparing an ANSWER WORKSHEET for a colleague who will write the final "
    "answer. Using ONLY the context above, produce:\n"
    "Requirements: a numbered list of every distinct element a complete answer must provide.\n"
    "Then, for EACH requirement:\n"
    "- Evidence: the exact raw lines that decide it, COPIED VERBATIM from the context in double quotes, "
    "each with its <step N> number. Copy characters exactly; never paraphrase inside quotes.\n"
    "- Working: a very detailed, numbered sequence of small steps from the quoted evidence to what it "
    "establishes for this requirement -- spell out every comparison, count, and check explicitly.\n"
    "Do NOT write the final overall answer. Keep each quote under 25 words and exact.")

def _handoff_answer(client, question, context, max_tokens, xfa, method, memory):
    if os.environ.get("AMA_HANDOFF_MANIFEST") == "2":
        # DSCAF: per-requirement worksheet with verbatim evidence pre-pasted + very detailed working.
        # Containment guard (packet lesson): any non-verbatim quote voids the whole worksheet.
        wp = (context + "\n\n## Question\n" + question + "\n\n## Task\n" + WORKSHEET_INSTR
              + "\n\nWorksheet:")
        ws = ""
        try:
            wresp = client.query(wp, temperature=0.0, max_tokens=max_tokens)
            ws = (wresp.split("</think>")[-1] if "</think>" in wresp else wresp).strip()
        except Exception:
            ws = ""
        qs = re.findall(r'"([^"\n]{12,200})"', ws)
        qok = (sum(1 for s in qs if s in context) / len(qs)) if qs else 1.0
        ws_ok = bool(ws) and qok == 1.0 and len(ws) < 9000
        newctx = (context + "\n\n## Answer worksheet from the memory service (requirements, the raw "
                  "evidence for each quoted verbatim from the trajectory, and detailed working)\n"
                  + ws[:6000]) if ws_ok else context
        try:
            resp = client.query(_harness_prompt(newctx, question, HANDOFF_DEFAULT),
                                temperature=0.0, max_tokens=min(max_tokens, 6144))
        except Exception:
            resp = ""
        m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
        if os.environ.get("AMA_AGENTIC_DBG") == "2":
            try:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "art": ws[:2600], "ws_ok": ws_ok, "quotes_ok": qok,
                     "nq": len(qs), "ans": txt[:300]}) + "\n")
            except Exception:
                pass
        return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}
    if os.environ.get("AMA_HANDOFF_MANIFEST"):
        # memory-side presentation probe: append a checklist of what the question asks for
        # (question-requirements only -- generated WITHOUT the context, so evidence fabrication is
        # structurally impossible; no imperatives -- presentation, not instruction smuggling).
        # Reader keeps the untouched default instruction.
        mp = ("Read the question below. List the distinct elements a complete answer must provide, "
              "as a short numbered list (one line each). Do NOT answer the question. Do NOT mention "
              "the trajectory, steps, or any evidence.\n\nQuestion: %s\n\nRequirements:" % question)
        man = ""
        try:
            mresp = client.query(mp, temperature=0.0, max_tokens=1536)
            man = (mresp.split("</think>")[-1] if "</think>" in mresp else mresp).strip()
        except Exception:
            man = ""
        if man and "step " not in man.lower() and len(man) < 1500:
            newctx = (context + "\n\n## What the question asks for (a checklist of elements a "
                      "complete answer must cover)\n" + man)
        else:
            man = ""
            newctx = context
        try:
            resp = client.query(_harness_prompt(newctx, question, HANDOFF_DEFAULT),
                                temperature=0.0, max_tokens=min(max_tokens, 6144))
        except Exception:
            resp = ""
        m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
        if os.environ.get("AMA_AGENTIC_DBG") == "2":
            try:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "art": man[:1500], "ans": txt[:300]}) + "\n")
            except Exception:
                pass
        return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}
    strip = bool(os.environ.get("AMA_HANDOFF_STRIP"))
    export = os.environ.get("AMA_HANDOFF_EXPORT", "")
    up_instr = _structured_instr()   # run with PROMPT=structured: the exact headline instruction
    if export == "2":
        up_instr = up_instr + HANDOFF_EXPORT_COMPACT
    elif export == "3":
        up_instr = up_instr + HANDOFF_BREVITY_SUFFIX
    elif export == "4":
        up_instr = up_instr + HANDOFF_EXPORT_SUFFIX_FULL
    elif export == "5":
        up_instr = up_instr + HANDOFF_EXPORT_SUFFIX_V
    elif export == "6":
        up_instr = up_instr + HANDOFF_VERBOSE_ONLY
    elif export == "7":
        # completeness with the conflicting "concise" prefix REPLACED, not just appended-over
        up_instr = (up_instr.replace("Provide a direct and concise answer.",
                                     "Provide a complete, well-supported answer.")
                    + HANDOFF_VERBOSE_ONLY)
    elif export:
        up_instr = up_instr + HANDOFF_EXPORT_SUFFIX
    art = ""
    try:
        respU = client.query(_harness_prompt(context, question, up_instr), temperature=0.0, max_tokens=max_tokens)
        art = (respU.split("</think>")[-1] if "</think>" in respU else respU).strip()
        if strip and art:
            mm = re.search(r"Answer\[1\]\s*:", art)
            if mm:
                art = art[:mm.start()].rstrip()
            art = re.sub(r"(?is)\n[^\n]*final answer[^\n]*$", "", art).rstrip()
    except Exception:
        art = ""
    if os.environ.get("AMA_HANDOFF_NORELAY"):
        # export-tax probe: answer in place from the upstream pass itself, no reader relay
        m0 = re.search(r"Answer\[1\]:\s*(.+?)$", art, re.DOTALL)
        txt0 = ("###Answer: %s" % m0.group(1).strip()) if m0 else art
        if os.environ.get("AMA_AGENTIC_DBG") == "2" and art:
            try:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "art": art[:2600], "up_ans": (m0.group(1).strip()[:300] if m0 else None),
                     "ans": txt0[:300]}) + "\n")
            except Exception:
                pass
        return {"final_answer": xfa(txt0, mcq_mode=False), "reasoning_trace": ""}
    if art:
        newctx = (context + "\n\n## Prior analysis of this question (produced by the memory service; "
                  "verify it against the evidence above before relying on it)\n" + art[:8000])
    else:
        newctx = context
    try:
        rd_instr = ((HANDOFF_VERIFY if export in ("1", "4", "5") else HANDOFF_VERIFY_LITE)
                    if os.environ.get("AMA_HANDOFF_VERIFY")
                    else HANDOFF_TRUST if os.environ.get("AMA_HANDOFF_TRUST") else HANDOFF_DEFAULT)
        if os.environ.get("AMA_HANDOFF_FULLANS"):
            rd_instr = rd_instr.replace("Provide a direct and concise answer.",
                                        "Provide a complete, well-supported answer.") + HANDOFF_READER_FULLANS
        resp = client.query(_harness_prompt(newctx, question, rd_instr),
                            temperature=0.0, max_tokens=min(max_tokens, 6144))
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(newctx), question, HANDOFF_DEFAULT),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("handoff=%d strip=%d q=%s\n" % (int(bool(art)), int(strip), question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2" and art:
                import json as _j
                mu = re.search(r"Answer\[1\]:\s*(.+?)$", art, re.DOTALL)
                qs = re.findall(r'"([^"\n]{12,200})"', art)
                qok = (sum(1 for s in qs if s in context) / len(qs)) if qs else None
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "art": art[:2600], "up_ans": (mu.group(1).strip()[:300] if mu else None),
                     "quotes_ok": qok, "nq": len(qs), "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


# ===================== mode: adapt (deterministic instruction routing; rules from the 26-case study) =====================
# The fixed structured instruction fixes 318 questions and breaks 184 (churn 3.7x its net) -- it is a blunt
# instrument. ADAPT routes a per-question instruction by auditable regex rules derived from the annotated
# case study (analysis/adapt_case_study.md): protective direct-answer for counterfactual/recommendation/
# negative-why; enumerate-then-count for bounded single counts; summary-only tally for long-range
# histograms; chronological ledger for state questions; verbatim-copy for exact-value recall; the proven
# structured instruction (verbatim, via AMA_ANSWER_INSTR_FILE) for everything else. R1 deliberately omits
# the cite-your-step demand: citation-forcing is the measured failure mechanism on negative-why questions.
_ONLY = "Answer using ONLY the context above. "
ADAPT_RULES = [
    ("R1", r"\bif\b.{0,80}\bhad\b|\bwould (it|the|have|this|that)\b|what should|should the agent|most reasonable|why did no|why (was|were) [^?]{0,40} not\b|why didn'?t|counterfactual",
     _ONLY + "Give the single most direct answer, directly and concisely; do not enumerate sub-questions; "
             "prefer the simplest explanation consistent with the trajectory; do not invent specifics."),
    ("R3", r"(what|which) (types|kinds)\b|how frequent|\bfrequency\b|how often",
     _ONLY + "Tally by scanning the whole range and report the final counts only; do not list every step "
             "individually; do not invent counts."),
    ("R2", r"how many times|how many [^?]{0,60}\b(before|between|until|after|at step|first|last|prior)\b",
     _ONLY + "First list every matching occurrence with its step number (cite only steps you can quote), "
             "then give the count of that list as the answer."),
    ("R4", r"how did the state|location histor|state change[sd]?\b|throughout the trajectory|state of [^?]{0,40} change",
     _ONLY + "Reconstruct the state chronologically: one line per step where it changes, citing each step "
             "verbatim; add nothing beyond the cited steps."),
    ("R5", r"\bwhat exact|\bexactly what|\bthe exact\b",
     _ONLY + "Locate the exact item asked for and copy it VERBATIM from the context, citing its step; do "
             "not generalize or abstract."),
]
_ADAPT_RX = None

def _adapt_route(question):
    global _ADAPT_RX
    if _ADAPT_RX is None:
        _ADAPT_RX = [(n, re.compile(rx, re.I), ins) for n, rx, ins in ADAPT_RULES]
    for name, rx, ins in _ADAPT_RX:
        if rx.search(question):
            return name, ins
    return "R6", None   # None -> use the structured instruction file (bit-identical to the anchor)

def _adapt_answer(client, question, context, max_tokens, xfa, method, memory):
    rule, instr = _adapt_route(question)
    base = instr if instr is not None else _structured_instr()
    try:
        resp = client.query(_harness_prompt(context, question, base), temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(context), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("adapt=%s q=%s\n" % (rule, question[:60]))
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


PACKET2_INSTR = (
    "You are a memory-view compiler. Reorganize the evidence steps below into a reasoning-ready view for "
    "the question. The steps below are a SELECTED SUBSET of a longer trajectory. Hard rules:\n"
    "1. Do NOT answer the question and do NOT state conclusions, inferences, or recommendations.\n"
    "2. Quote VERBATIM only: every content line cites its step number and copies text exactly as it "
    "appears IN THAT STEP. Never paraphrase, never merge, never re-word values, commands, paths, errors, "
    "names, counts, or indices.\n"
    "3. Executed events are ONLY what a step's 'Action:' field states. Text listing available/possible "
    "actions inside an observation is an OPTION, not an event; never cite an option as something that "
    "happened.\n"
    "4. Organize, don't create: group evidence by the sub-need of the question it serves; within a group "
    "keep temporal order. If unsure where something belongs, put the WHOLE step under [Ungrouped steps, "
    "kept verbatim]. Every step given to you must appear at least once (grouped or ungrouped).\n"
    "5. If the question asks how many / which steps / list occurrences: add an [Occurrences] section "
    "enumerating every matching event among the steps above, one line per step, verbatim trigger line.\n"
    "6. If the question involves state changing over time: add a [Timeline] section, one line per "
    "(entity, step) with the VERBATIM value only. Never compute changes, deltas, or differences.\n"
    "7. NEVER state that something is absent, missing, or does not exist. These steps are a subset; "
    "absence here proves nothing. If you cannot support a need, simply omit it.\n"
    "Output ONLY the view:\n[Evidence needs]\n- ...\n[Evidence groups]\nGroup A (need): <step N> "
    "\"verbatim\" ...\n[Occurrences] (if applicable)\n[Timeline] (if applicable)\n[Ungrouped steps, "
    "kept verbatim] (if any)"
)

def _norm_ws(t):
    return " ".join(t.replace("\\n", " ").split())

def _packet_faithful(clean, dyn):
    """Deterministic containment guard: every quoted string (>=12 chars) must appear verbatim in the
    source appendix; if it cites <step N>, it must appear within THAT step's span. Any violation -> False."""
    import re as _re
    steps = {}
    for b in _re.split(r"\n(?=<step \d+>)", dyn.strip()):
        m = _re.match(r"<step (\d+)>", b)
        if m:
            steps[m.group(1)] = _norm_ws(b)
    alln = _norm_ws(dyn)
    bad = 0
    for m in _re.finditer(r'(?:<step (\d+)>[^"\n]*)?"([^"]{12,}?)"', clean):
        sid, q = m.group(1), _norm_ws(m.group(2))
        if len(q) < 12:
            continue
        hay = steps.get(sid, alln) if sid else alln
        if q not in hay:
            if sid and q in alln:
                bad += 1   # verbatim somewhere, but not in the cited step (cross-step borrowing)
            else:
                bad += 1   # pure fabrication or paraphrase
    return bad == 0

def _packet_answer(client, question, context, max_tokens, xfa, method, memory, v2=False):
    base = _structured_instr()   # PROMPT=plain -> harness default instruction
    newctx = context; swapped = 0
    if LOSSY_MARK in context:
        static, dyn = context.split(LOSSY_MARK, 1)
        try:
            instr = PACKET2_INSTR if v2 else PACKET_INSTR
            respS = client.query("## Question\n%s\n\n## Evidence steps\n%s\n\n## Instructions\n%s"
                                 % (question, dyn, instr),
                                 temperature=0.0, max_tokens=min(max_tokens, 3072))
            clean = (respS.split("</think>")[-1] if "</think>" in respS else respS).strip()
            if v2 and clean and not _packet_faithful(clean, dyn):
                clean = ""   # containment guard tripped -> fall back to raw context
            if clean and "[Evidence" in clean:
                newctx = (static + "\n\nReasoning view of the most relevant earlier steps "
                          "(verbatim quotes, organized by evidence need):\n" + clean[:14000])
                swapped = 1
        except Exception:
            newctx = context
    try:
        resp = client.query(_harness_prompt(newctx, question, base), temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(newctx), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("packet%s=%d q=%s\n" % ("2" if v2 else "", swapped, question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2" and swapped:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "packet": newctx.split("organized by evidence need):\n")[-1][:1600],
                     "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


# ===================== mode: facts (same-pipeline payload swap: appendix -> extracted atomic facts) =====================
# Completes the same-pipeline payload family (verbatim / summary / facts / extractive-lines / gist-only):
# Mem0-style write-time fact extraction applied to the SAME router-selected steps, same 14k-char cap as lossy.
FACTS_SUMM = ("Rewrite each step below as a list of ATOMIC FACTS -- short standalone statements, one fact "
              "each (an action taken, a parameter, a produced value, an outcome). Keep the '<step N>' "
              "headers, one block per step, in order, 2-5 facts per step. Do NOT answer any question. "
              "Output only the fact lists.")

def _facts_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    newctx = context; swapped = 0
    if LOSSY_MARK in context:
        static, dyn = context.split(LOSSY_MARK, 1)
        try:
            respS = client.query("## Steps\n%s\n\n## Instructions\n%s" % (dyn, FACTS_SUMM),
                                 temperature=0.0, max_tokens=min(max_tokens, 3072))
            clean = (respS.split("</think>")[-1] if "</think>" in respS else respS).strip()
            if clean:
                newctx = static + "\n\nAtomic facts extracted from the most relevant earlier steps:\n" + clean[:14000]
                swapped = 1
        except Exception:
            newctx = context
    try:
        resp = client.query(_harness_prompt(newctx, question, base), temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(newctx), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("facts=%d q=%s\n" % (swapped, question[:60]))
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


# ===================== mode: extlines (same-pipeline payload swap: appendix -> query-relevant ORIGINAL lines) =====================
# Extractive-compression control: keep only the top query-relevant LINES of each selected step, verbatim
# (deterministic, no LLM, no paraphrase; same 14k cap). If this holds up while summary/facts drop, the
# poison is paraphrase, not compression; if it drops too, whole-span context matters.
EXTLINES_K = int(os.environ.get("AMA_EXTLINES_K", "6"))

def _ext_lines(question, dyn, per_step=EXTLINES_K):
    qtok = set(t for t in re.findall(r"[a-z0-9_./-]+", question.lower()) if len(t) >= 3)
    out = []
    for b in re.split(r"\n(?=<step \d+>)", dyn.strip()):
        lines = b.splitlines()
        if not lines:
            continue
        head, body = lines[0], [l for l in lines[1:]]
        scored = sorted(
            ((len(qtok & set(re.findall(r"[a-z0-9_./-]+", l.lower()))), -i, i) for i, l in enumerate(body)),
            reverse=True)
        keep = set(i for _, _, i in scored[:per_step])
        if body:
            keep.add(0)   # always keep the Action line
        kept = [body[i] for i in sorted(keep) if i < len(body)]
        omitted = len(body) - len(kept)
        blk = head + (("\n" + "\n".join(kept)) if kept else "")
        if omitted > 0:
            blk += "\n[... %d original lines omitted ...]" % omitted
        out.append(blk)
    return "\n\n".join(out)[:14000]

def _extlines_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    newctx = context; swapped = 0
    if LOSSY_MARK in context:
        static, dyn = context.split(LOSSY_MARK, 1)
        ext = _ext_lines(question, dyn)
        if ext:
            newctx = (static + "\n\nQuery-relevant original lines of the most relevant earlier steps "
                      "(verbatim; other lines omitted):\n" + ext)
            swapped = 1
    try:
        resp = client.query(_harness_prompt(newctx, question, base), temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(newctx), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("extlines=%d q=%s\n" % (swapped, question[:60]))
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": ""}


# ===================== mode: loop (iterative answer production over the verbatim store) =====================
# The SOFTWARE dissection localizes the AMA-Agent gap to its ITERATIVE answer loop (retrieve -> inspect ->
# verify -> compose): a process, not a format. This mode gives our single-pass reader that process: up to
# LOOP_ROUNDS ReAct rounds of deterministic lookups (get_steps / find_steps) over the lossless store, with a
# running investigation transcript, then a verified answer. No code execution (disclosed as a boundary).
LOOP_ROUNDS = int(os.environ.get("AMA_LOOP_ROUNDS", "4"))
LOOP_INSTR = (
    "Answer by ITERATIVE INVESTIGATION. You have two deterministic tools over the FULL stored trajectory "
    "(every step, verbatim -- beyond the excerpt above):\n"
    "  get_steps(i, j, ...)   -> exact Action/Observation of steps i, j, ... verbatim\n"
    "  find_steps(\"pattern\", mode=\"keyword\"|\"regex\"|\"action\") -> scan ALL steps, return matching step numbers\n"
    "Each round, output:\n"
    "  THOUGHT: what you know so far, what is still missing or unverified\n"
    "then EITHER up to 4 tool lines like:\n"
    "  TOOL: get_steps(38, 40)\n"
    "  TOOL: find_steps(\"runtests.py\", mode=\"keyword\")\n"
    "OR, once every value/step-index in your answer has been verified against exact step contents:\n"
    "  FINAL: <the answer, concise and exact>\n"
    "Verify before answering: locate with find_steps, read with get_steps, cross-check counts and values."
)

def _loop_answer(client, question, context, max_tokens, xfa, method, memory):
    transcript = ""; ncalls = 0; rounds = 0; final = None
    hdr = "%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s" % (context, question, LOOP_INSTR)
    if memory is not None and getattr(memory, "segments", None):
        for rnd in range(LOOP_ROUNDS):
            rounds = rnd + 1
            try:
                resp = client.query(hdr + transcript + "\n\n## Round %d\n" % rounds,
                                    temperature=0.0, max_tokens=1536)
            except Exception:
                break
            clean = resp.split("</think>")[-1] if "</think>" in resp else resp
            mF = re.search(r"FINAL:\s*(.+?)\s*$", clean, re.DOTALL)
            calls = _parse_tool_lines(clean)
            transcript += "\n\n## Round %d (yours)\n%s" % (rounds, clean.strip()[:2200])
            if mF and not calls:
                final = mF.group(1).strip()
                break
            if not calls:
                break
            outs = []
            for name, args in calls[:4]:
                ncalls += 1
                outs.append(_tool_get(memory, args) if name == "get_steps" else _tool_find(memory, args))
            transcript += "\n\n## Tool observations (round %d)\n%s" % (rounds, "\n\n".join(outs)[:6000])
            transcript = transcript[-20000:]
    base = _structured_instr()
    if final and not os.environ.get("AMA_LOOP_COMPOSE"):
        txt = "###Answer: %s" % final
    else:
        evid = context + (("\n\n## Investigation transcript (deterministic lookups over the full stored trajectory)"
                           + transcript) if transcript else "")
        try:
            resp = client.query(_harness_prompt(evid, question, base), temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            try:
                resp = client.query(_harness_prompt(_gcap(evid), question, base),
                                    temperature=0.0, max_tokens=min(max_tokens, 4096))
            except Exception:
                resp = ""
        m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("loop r=%d calls=%d fin=%d q=%s\n"
                                                  % (rounds, ncalls, int(final is not None), question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2":
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "rounds": rounds, "calls": ncalls,
                     "tr": transcript[-1500:], "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": transcript[:4000]}


# ===================== mode: plan (EvidencePlan — move the structured-reasoning gain into the memory service) =====================
# Test: is the structured-prompt gain (+5.1~5.8pp) a reader-owned effect, or cacheable reasoning state the
# MEMORY service can produce? Call P (planner) decomposes + cites + locally concludes over the selected
# evidence WITHOUT answering; the plan is prepended as a memory-service artifact; the final answer call uses
# the harness's DEFAULT instruction (run with PROMPT=plain). If plan+default ~= structured, reasoning is
# movable into memory; if not, the unified law extends: reasoning must happen in the answering pass itself.
PLAN_INSTR = (
    "You are the memory service's evidence planner. Do NOT produce the final answer to the question. "
    "Break the question into the sub-questions it requires. For each sub-question, quote the exact "
    "evidence from the context (values, step indices, outcomes, verbatim strings) and state a one-line "
    "local conclusion. Then state how the pieces compose. Use exactly this format:\n"
    "Sub-question 1: ...\n  Evidence: <step N> \"...\"\n  Local conclusion: ...\n"
    "Sub-question 2: ...\n  Evidence: ...\n  Local conclusion: ...\n"
    "Composition: <how the local conclusions combine; what the final answer must mention>"
)

PLAN2_INSTR = (
    "You are the memory service's evidence planner. Do NOT produce the final answer. "
    "Break the question into the sub-questions it requires; for each, quote exact evidence "
    "(values, step indices, outcomes, verbatim strings) and state a one-line local conclusion; "
    "then state how the pieces compose. Hard rules learned from audited failures:\n"
    "(a) MIRROR THE QUESTION'S OWN STRUCTURE AND MODALITY. If it asks for preconditions of success, "
    "plan preconditions -- not what failed. If it asks 'which turn first X', the plan's job is to FIND "
    "that turn, not to survey X.\n"
    "(b) NEVER conclude something is absent. If you have not found it, write 'not located in the "
    "selected evidence (may exist elsewhere)' and name what to look for -- absence claims poison the "
    "answer downstream.\n"
    "(c) Collect ONLY facts the question needs; do not add side-observations, they mis-weight the "
    "composition.\n"
    "(d) For counts, list every occurrence with its step number BEFORE stating the number.\n"
    "Worked contrast:\n"
    "Question: 'Before step 9 could compile successfully, what preconditions had to be true?'\n"
    "WRONG framing: 'Sub-question 1: What failed at step 9?' (plans the failure, not the asked "
    "preconditions)\n"
    "RIGHT framing: 'Sub-question 1: What does the step-9 query require to be true about the "
    "referenced columns? Evidence: <step 9> \"...\" Local conclusion: precondition P1 = the column "
    "exists in every joined table.'\n"
    "Question: 'In which turn did the agent first extract detail D?'\n"
    "WRONG: 'Local conclusion: no step mentions D.' RIGHT: 'Not located in the selected evidence "
    "(may exist elsewhere); candidate turns discussing the source of D: <step 4> \"...\", <step 9> "
    "\"...\" -- the earliest that contains D verbatim is the answer.'\n"
    "Use exactly this output format:\n"
    "Sub-question 1: ...\n  Evidence: <step N> \"...\"\n  Local conclusion: ...\n"
    "Sub-question 2: ...\n  Evidence: ...\n  Local conclusion: ...\n"
    "Composition: <how the local conclusions answer THE QUESTION AS ASKED>"
)

def _plan_answer(client, question, context, max_tokens, xfa, method, memory, v2=False):
    base = _structured_instr()   # with PROMPT=plain this is the harness default instruction
    plan = ""
    try:
        respP = client.query("%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s"
                             % (context, question, (PLAN2_INSTR if v2 else PLAN_INSTR)),
                             temperature=0.0, max_tokens=(max_tokens if v2 else min(max_tokens, 2048)))
        clean = respP.split("</think>")[-1] if "</think>" in respP else respP
        plan = clean.strip()[:8000]
    except Exception:
        plan = ""
    if plan:
        evid = ("%s\n\n## Evidence plan (prepared by the memory service from the context above)\n%s"
                % (context, plan))
        comp = min(max_tokens, 4096)
    else:
        evid = context; comp = max_tokens
    try:
        resp = client.query(_harness_prompt(evid, question, base), temperature=0.0, max_tokens=comp)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(evid), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("plan=%dch q=%s\n" % (len(plan), question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2" and plan:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "plan": plan[:1800], "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": plan}


# ===================== mode: tools (deterministic access API over the lossless substrate) =====================
# Unifying law from the ablations: a reader-side mechanism helps iff it injects information/computation the
# context+model cannot already produce. Self-generated scaffolds (select/state/self-judgment) inject nothing
# -> <=0. These tools DO inject: program-computed, verbatim, exhaustive access to the FULL stored trajectory
# (our memory keeps every step verbatim — the router is just a selector on top of a lossless store).
# Wiring lessons applied: the ANSWER call stays byte-for-byte pure structured (fused-escape cost -2.6pp);
# tool DECISION is a separate call; tool results enter as an extra EVIDENCE section, never as instruction.
TOOLS_DECIDE = (
    "You also have two DETERMINISTIC tools that query the FULL stored trajectory (every step, verbatim — "
    "beyond the excerpt above):\n"
    "  get_steps(i, j, ...)   -> exact Action/Observation of steps i, j, ... verbatim\n"
    "  find_steps(\"pattern\", mode=\"keyword\"|\"regex\"|\"action\") -> scan ALL steps, return matching step "
    "numbers and their count\n"
    "Most questions are already answerable from the evidence above. Call tools ONLY if the answer needs the "
    "exact verbatim content of specific steps, an exhaustive search, or a count over the full trajectory.\n"
    "If needed, output ONLY tool-call lines, one per line, exact numbers comma-separated, e.g.:\n"
    "TOOL: get_steps(38, 40)\n"
    "TOOL: find_steps(\"runtests.py\", mode=\"keyword\")\n"
    "Otherwise output exactly: NONE"
)

def _parse_tool_lines(resp):
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"TOOL:\s*(get_steps|find_steps)\s*\((.*?)\)\s*$", resp, re.M)]

def _tool_get(memory, args):
    turns = []
    for a, b in re.findall(r"(\d+)\s*(?:-|to)\s*(\d+)", args):   # expand small ranges "38-41"
        a, b = int(a), int(b)
        if a <= b <= a + 12:
            turns += list(range(a, b + 1))
    rest = re.sub(r"\d+\s*(?:-|to)\s*\d+", "", args)
    turns += [int(x) for x in re.findall(r"\d+", rest)]
    segs = {s.turn: s for s in getattr(memory, "segments", [])}
    out = []
    for t in list(dict.fromkeys(turns))[:8]:
        s = segs.get(t)
        out.append("<step %d>\n%s" % (t, s.text) if s is not None else "<step %d> NOT FOUND" % t)
    return "\n\n".join(out)[:9000] if out else "(no valid step numbers)"

def _tool_find(memory, args):
    m = re.search(r'["\'](.+?)["\']', args)
    if not m:
        return "(no pattern)"
    pat = m.group(1)
    mm = re.search(r'mode\s*=\s*["\']?(\w+)', args)
    mode = (mm.group(1).lower() if mm else "keyword")
    rx = None
    if mode == "regex":
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            return "invalid regex: %s" % pat
    hits = []
    for s in getattr(memory, "segments", []):
        meta = getattr(s, "meta", None) or {}
        a = meta.get("action", "") or ""
        o = meta.get("observation", "") or ""
        hay = a if mode == "action" else (a + "\n" + o)
        ok = bool(rx.search(hay)) if rx else (pat.lower() in hay.lower())
        if ok:
            hits.append(s.turn)
    return "find_steps(%r, mode=%s): %d matching steps -> %s" % (pat, mode, len(hits), hits[:60])

def _tools_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    tool_all = ""; ncalls = 0
    if memory is not None and getattr(memory, "segments", None):
        try:
            hdr = "%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s" % (context, question, TOOLS_DECIDE)
            resp = client.query(hdr, temperature=0.0, max_tokens=1024)
            for rnd in range(2):
                calls = _parse_tool_lines(resp)
                if not calls:
                    break
                outs = []
                for name, args in calls[:4]:
                    ncalls += 1
                    outs.append(_tool_get(memory, args) if name == "get_steps" else _tool_find(memory, args))
                tool_all += ("\n\n" if tool_all else "") + "\n\n".join(outs)
                tool_all = tool_all[:12000]
                if rnd == 0:
                    resp = client.query(hdr + "\n\n## Tool output\n" + tool_all +
                                        "\n\nIf you need more lookups, output more TOOL: lines. Otherwise output NONE.",
                                        temperature=0.0, max_tokens=1024)
        except Exception:
            pass
    if tool_all:
        evid = ("%s\n\n## Exact trajectory lookups (computed deterministically from the full stored trajectory)\n%s"
                % (context, tool_all))
        comp = min(max_tokens, 4096)   # evidence grew; keep the total under the 32k window
    else:
        evid = context; comp = max_tokens
    try:
        resp = client.query(_harness_prompt(evid, question, base), temperature=0.0, max_tokens=comp)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(evid), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("tools=%d q=%s\n" % (ncalls, question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2" and tool_all:
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "tool_out": tool_all[:1800], "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": tool_all}


# ===================== mode: state (state-aware memory, L1: query-conditioned state-timeline scaffold) =====================
# The recall probe showed 100% step recall but reader-side failures on state-tracking / causal questions
# (e.g. "win-block x: -3 -> -2 => which way did the agent move?"). This offloads STATE TRACKING from the
# backbone: a separate call extracts a compact per-entity timeline, prepended (additively — full substrate
# kept) so the reader reads the computed state delta instead of reconstructing it from raw steps.
STATE_EXTRACT = (
    "You are a state-tracking assistant. Do NOT answer the question. Identify the entities / variables / "
    "objects the question depends on (e.g. a block's position, a file's contents, a test's status, an "
    "object's location, a variable's value). From the evidence, output a COMPACT timeline covering ONLY "
    "those entities — one line per (entity, step) where its value is established or CHANGES, in this form:\n"
    "  <entity> | step <N> | <value or change>\n"
    "Cover the steps the question references and their immediate neighbours. Copy exact "
    "values/coordinates/identifiers/paths VERBATIM. Output ONLY the timeline lines."
)

def _state_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    timeline = ""
    try:
        respS = client.query("%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s"
                            % (context, question, STATE_EXTRACT),
                            temperature=0.0, max_tokens=min(max_tokens, 2048))
        clean = respS.split("</think>")[-1] if "</think>" in respS else respS
        lines = [l.strip() for l in clean.splitlines() if l.count("|") >= 2]
        timeline = "\n".join(lines[:40])
    except Exception:
        timeline = ""
    if timeline:
        prompt = ("%s\n\n## State timeline (tracked for this question)\n%s\n\n## Questions\nQuestion 1: %s\n\n"
                  "## Instructions\n%s\n\nAnswer[1]: [your answer here]" % (context, timeline, question, base))
    else:
        prompt = _harness_prompt(context, question, base)
    try:
        resp = client.query(prompt, temperature=0.0, max_tokens=max_tokens)
    except Exception:
        try:
            resp = client.query(_harness_prompt(_gcap(context), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
        except Exception:
            resp = ""
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("state=%dlines q=%s\n" % (len(timeline.splitlines()), question[:60]))
            if os.environ.get("AMA_AGENTIC_DBG") == "2":
                import json as _j
                open(LOGBASE + "_full.jsonl", "a").write(_j.dumps(
                    {"q": question[:200], "timeline": timeline[:1600], "ans": txt[:300]}) + "\n")
        except Exception:
            pass
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": timeline}


# ===================== mode: decouple (separate sufficiency gate from the answer prompt) =====================
# Attribution showed the FUSED escape clause perturbs the answer (-2.6pp on flagship) while the re-retrieve
# ACTION is net-positive (+0.4pp). So: answer with the UNPERTURBED structured prompt (Call A), run the
# sufficiency check as a SEPARATE call (Call B) that never touches A, and only re-answer when B flags a gap.
DECOUPLE_GATE = (
    "You are a fact-sufficiency checker. Do NOT answer the question. Decide ONLY whether the evidence "
    "above already contains every specific fact the question requires.\n"
    "Decompose the question into the specific facts/values/steps it needs and check each against the "
    "evidence. If a required fact is genuinely ABSENT, output exactly one line:\n"
    "NEED: <a precise search query for the single most important missing fact>\n"
    "If the evidence contains everything needed, output exactly: OK\n"
    "Output only `NEED: ...` or `OK`."
)

def _gate_log(fired, question):
    if os.environ.get("AMA_AGENTIC_DBG"):
        try:
            open(LOGBASE + "_dbg.log", "a").write("decouple=%s q=%s\n" % (fired, question[:70]))
        except Exception:
            pass

def _decouple_answer(client, question, context, max_tokens, xfa, method, memory):
    base = _structured_instr()
    fired = "OK"; respA = ""; ansA = None
    try:
        # Call A: UNPERTURBED structured answer (identical to the structured baseline)
        respA = client.query(_harness_prompt(context, question, base), temperature=0.0, max_tokens=max_tokens)
        ansA = re.search(r"Answer\[1\]:\s*(.+?)$", respA, re.DOTALL)
        # Call B: SEPARATE sufficiency gate (does NOT influence A's answer)
        respB = client.query("%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s"
                             % (context, question, DECOUPLE_GATE),
                             temperature=0.0, max_tokens=min(max_tokens, 2048))
        ins = re.search(r"NEED:\s*(.+)", respB)
        if ins and not (re.search(r"\bOK\b", respB) and respB.rfind("OK") > respB.rfind("NEED:")):
            q2 = ins.group(1).strip().strip('"').strip()[:200]
            try:
                more = method.memory_retrieve(memory, q2) if (method is not None and memory is not None) else ""
            except Exception:
                more = ""
            if more and more[:200] not in context:
                aug = _gcap(context + "\n\nFull text of additional retrieved steps:\n" + more)
                respC = client.query(_harness_prompt(aug, question, base),
                                     temperature=0.0, max_tokens=min(max_tokens, 4096))
                ansC = re.search(r"Answer\[1\]:\s*(.+?)$", respC, re.DOTALL)
                _gate_log("RETRIEVE:" + q2[:40], question)
                txt = ("###Answer: %s" % ansC.group(1).strip()) if ansC else respC
                return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": respC}
    except Exception:
        fired = "FALLBACK"
        try:
            respA = client.query(_harness_prompt(_gcap(context), question, base),
                                temperature=0.0, max_tokens=min(max_tokens, 4096))
            ansA = re.search(r"Answer\[1\]:\s*(.+?)$", respA, re.DOTALL)
        except Exception:
            respA = ""; ansA = None
    _gate_log(fired, question)
    txt = ("###Answer: %s" % ansA.group(1).strip()) if ansA else respA
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": respA}


# ===================== mode: reretrieve =====================
SUFF_INSTR = (
    "First, assess whether the evidence is sufficient, step by step:\n"
    "1. Decompose the question into the specific facts/values/steps it requires.\n"
    "2. For each requirement, check the evidence above and QUOTE the exact supporting span, "
    "or mark it MISSING.\n"
    "3. If (and only if) a required piece is not present in the evidence, output one line:\n"
    "   NEED: <a short precise search query for the missing piece>\n"
    "   Otherwise output the single line: SUFFICIENT\n"
    "Write this decomposition-and-check analysis explicitly and concisely."
)

def _reretrieve_answer(client, question, context, max_tokens, xfa, method, memory):
    analysis_all = ""
    comp = min(max_tokens, COMP_TOK)
    for _ in range(MAX_STEPS - 1):
        p = ("## Evidence\n%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s"
             % (_cap(context), question, SUFF_INSTR))
        analysis = client.query(p, temperature=0.0, max_tokens=comp)
        analysis_all += analysis.strip() + "\n"
        m = re.search(r"NEED:\s*(.+)", analysis)
        pre = analysis.split("NEED:")[0][-60:] if m else ""
        if m and "SUFFICIENT" not in pre:
            q2 = m.group(1).strip().strip('"').strip()[:200]
            try:
                more = method.memory_retrieve(memory, q2)
            except Exception as e:  # noqa
                more = ""
            if more:
                context += "\n\n## Additional evidence (self-query: %s)\n%s" % (q2, more)
            continue
        break
    # final answer, with the preserved sufficiency analysis in context
    fp = ("## Evidence\n%s\n\n## Analysis\n%s\n\n## Question\nQuestion 1: %s\n\n"
          "## Instructions\nUsing the evidence and your analysis above, answer precisely.\n"
          "Answer[1]: [your answer here]" % (_cap(context), analysis_all[:8000], question))
    resp = client.query(fp, temperature=0.0, max_tokens=comp)
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": analysis_all}


# ===================== mode: code =====================
TOOL_INSTR = (
    "Answer the question using the evidence above. If the question requires computation over "
    "long or code-heavy evidence (counting, aggregation, exact matching, pattern search, sorting, "
    "deduplication), you MAY write ONE Python code block:\n"
    "```python\n# `context` (str) holds all selected evidence; re, collections, math are imported.\n"
    "print(...)\n```\n"
    "The stdout is returned to you as Tool output; then continue. Up to %d code blocks. "
    "When done, output exactly:\nAnswer[1]: <your final answer>\n" % MAX_STEPS
)

def _run_code(code, context):
    ctxpath = spath = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as cf:
            cf.write(context); ctxpath = cf.name
        script = ("context=open(%r,encoding='utf-8').read()\nimport re,collections,math,json,itertools\n%s\n"
                  % (ctxpath, code))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as sf:
            sf.write(script); spath = sf.name
        r = subprocess.run(["python3", spath], capture_output=True, text=True, timeout=CODE_TIMEOUT)
        out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if (r.returncode and r.stderr) else "")
        return out.strip()[:4000] if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: code timed out"
    except Exception as e:  # noqa
        return "ERROR: %s" % e
    finally:
        for p in (ctxpath, spath):
            try:
                if p: os.unlink(p)
            except Exception:
                pass

def _code_answer(client, question, context, max_tokens, xfa):
    comp = min(max_tokens, COMP_TOK)
    convo = "%s\n\n## Question\nQuestion 1: %s\n\n## Instructions\n%s" % (_cap(context), question, TOOL_INSTR)
    for _ in range(MAX_STEPS):
        resp = client.query(convo, temperature=0.0, max_tokens=comp)
        code = re.findall(r"```python\s*(.*?)```", resp, re.DOTALL)
        ans = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
        if code and not (ans and resp.rfind("Answer[1]:") > resp.rfind("```")):
            obs = _run_code(code[-1], context)
            convo += ("\n%s\n\n## Tool output\n%s\n\n## Continue (write another ```python``` block "
                      "or give Answer[1]:)\n" % (resp, obs))
            continue
        if ans:
            return {"final_answer": xfa("###Answer: %s" % ans.group(1).strip(), mcq_mode=False),
                    "reasoning_trace": convo}
        return {"final_answer": xfa(resp, mcq_mode=False), "reasoning_trace": convo}
    resp = client.query(convo[:120000] + "\nStop using tools. Now output only: Answer[1]: <final answer>",
                        temperature=0.0, max_tokens=comp)
    m = re.search(r"Answer\[1\]:\s*(.+?)$", resp, re.DOTALL)
    txt = ("###Answer: %s" % m.group(1).strip()) if m else resp
    return {"final_answer": xfa(txt, mcq_mode=False), "reasoning_trace": convo}
