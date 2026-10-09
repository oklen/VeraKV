"""Paired AMA-Bench re-run (2026-10-08): AMA-Agent vs VeraKV on the same episodes, server and judge
(write-up: docs/AMA_AGENT_PAIRED.md).

Arms, all on one Qwen3-32B vLLM instance per shard and the official harness code paths:
  A1  AMA-Agent exactly as released: configs/ama_agent.yaml + src/method/ama_agent*, answered through
      MemoryQAInterface.answer_question (including its direct-answer fast path).
  A2  A1 with one change: the final context cap counted in tokens (retrieve_tokcap.py, a code copy of
      retrieve.py). A2 shares A1's memory, similarity retrieval and sufficiency/code-search calls; only
      the final context differs, so only the reader call differs. Where A1 answers from the sufficiency
      judgment (direct path) the final context is never read, so A2's answer is A1's by construction.
  V1  VeraKV deployed config (cfg_flagship.json: causal router over hybrid lexical + model-pick, step-pin).
  V2  VeraKV lexical router + step-pin (cfg_kvmem_causal.json).
  V1f V1 with the model-pick call's thinking switched off (cfg_flagship_nothink.json); added for the
      model-pick fix re-run (docs/MODEL_PICK_FIX.md).
Reader prompt: the harness default ("Provide a direct and concise answer."), or the file named by
AMA_ANSWER_INSTR_FILE (the fix re-run's structured reader); each record says which. Judge: the official
compute_llm_as_judge against the same Qwen3-32B server, as in the 0.5954 runs.

Writes, per shard:
  calls.jsonl  every chat completion: tag (arm/ep/qi), stage, exact prompt, raw response, usage, latency
  q.jsonl      per question and arm: answer, verdict, the exact prompt that produced the answer, the
               retrieved context and its token length, AMA-Agent path stats
  status.txt   progress, rewritten every minute

    python ama/paired/run.py --ama-root <AMA-Bench checkout> --cfg-dir <configs> --data <open-ended .jsonl> \
        --port 8060 --shard 0 --nshards 4 --out runs/full/s0 [--arms V1,V2,A | --arms A3 --mem-dir runs/full/s0/mem --reuse-mem | --arms V1,V1f,V2]

  --ama-root  an AMA-Bench checkout with this repo's ama/harness/ files and the generated
              retrieve_tokcap.py / retrieve_nofast.py in src/method/ama_agent_core/ (make_tokcap.py, make_nofast.py)
  --cfg-dir   qwen_<port>.yaml (AMA-Bench's configs/qwen3-32B.yaml with vllm_port set), ama_agent.yaml
              (AMA-Bench's configs), cfg_flagship.json, cfg_flagship_nothink.json and cfg_kvmem_causal.json (this repo's ama/configs)
  AMA_TOKCAP_TOKENIZER  the Qwen3-32B tokenizer (a local path or a Hugging Face id; default Qwen/Qwen3-32B)
"""
import argparse
import contextvars
import copy
import json
import os
import pickle
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ap = argparse.ArgumentParser()
ap.add_argument("--ama-root", required=True, help="AMA-Bench checkout (patched, see the docstring)")
ap.add_argument("--pkgs", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                help="folder that contains the kvmemory package (default: this repository)")
ap.add_argument("--cfg-dir", required=True)
ap.add_argument("--data", required=True, help="AMA-Bench open-ended test file (.jsonl)")
ap.add_argument("--domains", default="WEB,EMBODIED_AI,Game,TEXT2SQL,SOFTWARE")
ap.add_argument("--episodes", default="", help="comma list of episode ids (smoke test); sharded like the rest")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--emb-port", type=int, default=8003)
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="V1,V2,A")
ap.add_argument("--build-conc", type=int, default=4)
ap.add_argument("--q-conc", type=int, default=24)
ap.add_argument("--mem-dir", default="", help="AMA-Agent memory pickles (A_<ep>.pkl); default <out>/mem")
ap.add_argument("--reuse-mem", action="store_true", help="never build AMA-Agent memory; a missing pickle is an infra failure")
args = ap.parse_args()

sys.path.insert(0, args.pkgs)
sys.path.insert(0, args.ama_root)
os.makedirs(args.out, exist_ok=True)
WORK = os.path.join(args.out, "work")
MEMDIR = args.mem_dir or os.path.join(args.out, "mem")
os.makedirs(WORK, exist_ok=True)
os.makedirs(MEMDIR, exist_ok=True)
os.chdir(WORK)  # AMA-Agent's code search writes ./tmp/mem_exec_*

import yaml  # noqa: E402
from src.model_client import ModelClient  # noqa: E402
from src.memory_interface import MemoryQAInterface  # noqa: E402
from utils.embedding import EmbeddingEngine  # noqa: E402
from utils.evaluation_metrics import compute_llm_as_judge  # noqa: E402
import src.method.ama_agent_core.construct as C  # noqa: E402
import src.method.ama_agent_core.retrieve as R  # noqa: E402
import src.method.ama_agent_core.retrieve_tokcap as RT  # noqa: E402
import src.method.ama_agent_core.retrieve_nofast as RN  # noqa: E402

import tempfile  # noqa: E402

_orig_mkdtemp = tempfile.mkdtemp


def _mkdtemp_abs(*a, **kw):
    """Python 3.12 semantics on our 3.11 env: mkdtemp returns an absolute path even for a relative dir.
    AMA-Agent's code search (utils._run_keyword_search) depends on it: it makes tmpdir with dir="tmp" and
    runs `python <tmpdir>/script.py` with cwd=<tmpdir>; with a relative tmpdir the script is never found
    and every code search returns "can't open file"."""
    return os.path.abspath(_orig_mkdtemp(*a, **kw))


tempfile.mkdtemp = _mkdtemp_abs

TAG = contextvars.ContextVar("TAG", default=None)      # {"arm","ep","qi"}
CALLS = contextvars.ContextVar("CALLS", default=None)  # list collecting this question-arm's calls
SYN = contextvars.ContextVar("SYN", default=None)      # A1/A2 contexts captured in _synthesize


class _Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, rec):
        line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            self.f.write(line + "\n")
            self.f.flush()


CALL_LOG = _Log(os.path.join(args.out, "calls.jsonl"))
Q_LOG = _Log(os.path.join(args.out, "q.jsonl"))


def stage_of(prompt):
    p = prompt or ""
    if p.startswith("You are an expert evaluator."):
        return "judge"
    if p.startswith("You are routing a question about an agent trajectory"):
        return "suff"
    if p.startswith("You are helping to extract relevant information from a trajectory to answer a question by writing Python code."):
        return "codegen"
    if p.startswith("You are presented with a section of agent trajectory"):
        return "build"
    if p.startswith("You are selecting which past trajectory steps are needed"):
        return "modelpick"
    if "## Questions\nQuestion 1:" in p and p.rstrip().endswith("Answer[1]: [your answer here]"):
        return "reader"
    return "other"


def wrap_client(mc):
    """Log every chat completion made through this ModelClient: the exact messages sent (after the
    client's own fallback truncation, if any), the raw response, server-side token usage, latency."""
    comp = mc.client.chat.completions
    orig = comp.create

    def create(*a, **kw):
        t0 = time.time()
        resp, err = None, None
        try:
            resp = orig(*a, **kw)
            return resp
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:3000]
            raise
        finally:
            msgs = kw.get("messages") or []
            prompt = msgs[0]["content"] if msgs else ""
            rec = dict(TAG.get() or {})
            rec.update(stage=stage_of(prompt), t0=round(t0, 3), dt=round(time.time() - t0, 3),
                       max_tokens=kw.get("max_tokens"), extra_body=kw.get("extra_body"), prompt=prompt, err=err)
            if resp is not None:
                try:
                    rec["response"] = resp.choices[0].message.content
                    rec["finish"] = resp.choices[0].finish_reason
                    rec["usage"] = [resp.usage.prompt_tokens, resp.usage.completion_tokens]
                except Exception:  # noqa: BLE001
                    rec["response"] = None
            CALL_LOG.write(rec)
            lst = CALLS.get()
            if lst is not None:
                lst.append(rec)

    comp.create = create
    assert mc.client.chat.completions.create is create, "openai client does not cache its resources"
    return mc


class _CtxPool(ThreadPoolExecutor):
    """construct.py's ThreadPoolExecutor, carrying the submitting thread's tags into its workers."""

    def submit(self, fn, *a, **kw):
        return super().submit(contextvars.copy_context().run, fn, *a, **kw)


C.ThreadPoolExecutor = _CtxPool

_orig_synthesize = R._synthesize


def _synthesize_both(**kw):
    """A1 = the released _synthesize (returned, so the released flow continues unchanged);
    A2 = the token-cap copy's _synthesize on the same arguments (captured for the A2 reader call)."""
    a1 = _orig_synthesize(**kw)
    st = SYN.get()
    if st is not None:
        st["a1"] = a1
        st["n_synth"] += 1
        try:
            st["a2"] = RT._synthesize(**kw)
        except Exception:  # noqa: BLE001
            st["a2"] = None
            st["a2_err"] = traceback.format_exc()[-2000:]
    return a1


R._synthesize = _synthesize_both


FAST = contextvars.ContextVar("FAST", default=None)     # A3: what the released fast path would return


def _fast_hook(direct):
    """retrieve_nofast calls this where the released code would return the check's inline answer."""
    st = FAST.get()
    if st is not None:
        st["a1_direct"] = direct
        lst = CALLS.get()
        st["a1_ncalls"] = len(lst) if lst is not None else None


RN.FAST_PATH_HOOK = _fast_hook


class _A3Method:
    """AMAAgentMethod.memory_retrieve, same arguments, but through retrieve_nofast (arm A3)."""

    def __init__(self, m):
        self.m = m

    def memory_retrieve(self, memory, question):
        m = self.m
        return RN.memory_retrieve(memory=memory.to_dict(), question=question, call_llm_func=m._call_llm,
                                  top_k=m.top_k, embed_engine=m.embedding_engine,
                                  max_context_length=m.max_model_length - m.max_tokens)


class _FixedContext:
    """Stand-in method that returns a given context, so A2's reader call goes through the official
    answer_question prompt and answer parsing."""

    def __init__(self, ctx):
        self.ctx = ctx

    def memory_retrieve(self, memory, question):
        return self.ctx


# ---------------------------------------------------------------- setup
cfg = args.cfg_dir
qyaml = os.path.join(cfg, "qwen_%d.yaml" % args.port)
ama_yaml = os.path.join(cfg, "ama_agent.yaml")
arms = [a.strip() for a in args.arms.split(",") if a.strip()]


def mk():
    return wrap_client(ModelClient(config_path=qyaml, server_type="vllm"))


clients = {a: mk() for a in arms}
judge_client = mk()
ifaces = {}
V_CFG = {"V1": "cfg_flagship.json", "V1f": "cfg_flagship_nothink.json", "V2": "cfg_kvmem_causal.json"}
V_ARMS = [a for a in ("V1", "V1f", "V2") if a in arms]
for _a in V_ARMS:
    ifaces[_a] = MemoryQAInterface(client=clients[_a], method_name="kvmemory",
                                   method_config=os.path.join(cfg, V_CFG[_a]), subset="openend")
READER_INSTR = os.path.basename(os.environ.get("AMA_ANSWER_INSTR_FILE", "")) or "default"
print("VeraKV arms %s, reader instruction: %s" % (V_ARMS, READER_INSTR), flush=True)
if "A" in arms or "A3" in arms:
    ec = yaml.safe_load(open(ama_yaml))["embedding_engine"]
    emb = EmbeddingEngine(model_name=ec.get("model_name"), base_url="http://127.0.0.1:%d/v1" % args.emb_port,
                          api_key=ec.get("api_key", "EMPTY"), batch_size=ec.get("batch_size", 8),
                          max_length=ec.get("max_length", 512), auto_launch=False, host="127.0.0.1",
                          port=args.emb_port)
    ifaces["A"] = MemoryQAInterface(client=clients["A" if "A" in clients else "A3"], method_name="ama_agent", method_config=ama_yaml,
                                    subset="openend", embedding_engine=emb)
    _meth = ifaces["A"].method
    _orig_mr = _meth.memory_retrieve

    def _spy_mr(memory, question):
        out = _orig_mr(memory, question)
        st = SYN.get()
        if st is not None:
            st["raw"] = out
        return out

    _meth.memory_retrieve = _spy_mr
    print("AMA-Agent: top_k=%s session_size=%s causal=%s max_tokens=%s max_model_length=%s -> cap %s"
          % (_meth.top_k, _meth.session_size, _meth.causal, _meth.max_tokens, _meth.max_model_length,
             _meth.max_model_length - _meth.max_tokens), flush=True)

from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained(os.environ.get("AMA_TOKCAP_TOKENIZER", "Qwen/Qwen3-32B"))
TOK_LOCK = threading.Lock()


def ntok(s):
    if not s:
        return 0
    with TOK_LOCK:
        return len(TOK(s, add_special_tokens=False).input_ids)


# ---------------------------------------------------------------- episodes
doms = set(args.domains.split(","))
eps = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
eps = sorted((e for e in eps if e.get("domain") in doms), key=lambda e: e["episode_id"])
if args.episodes:
    want = {int(x) for x in args.episodes.split(",") if x.strip()}
    eps = [e for e in eps if e["episode_id"] in want]
eps = [e for i, e in enumerate(eps) if i % args.nshards == args.shard]
print("shard %d/%d: %d episodes, %d questions, arms %s" % (
    args.shard, args.nshards, len(eps), sum(len(e["qa_pairs"]) for e in eps), arms), flush=True)

done = set()
qpath = os.path.join(args.out, "q.jsonl")
if os.path.exists(qpath):
    for l in open(qpath, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if not r.get("infra"):
            done.add((r["arm"], r["ep"], r["qi"]))
print("resume: %d arm-questions already done" % len(done), flush=True)

STATS = {"t0": time.time(), "built": 0, "build_fail": 0, "q_done": 0, "q_total": 0, "by_arm": {}}
SLOCK = threading.Lock()


def note(arm, score, infra):
    with SLOCK:
        s = STATS["by_arm"].setdefault(arm, [0, 0, 0])
        s[0] += 1
        s[1] += 1 if score == 1.0 else 0
        s[2] += 1 if infra else 0


def status_loop():
    while True:
        with SLOCK:
            el = time.time() - STATS["t0"]
            lines = ["shard %d elapsed %.0fs built %d (fail %d) questions %d/%d" % (
                args.shard, el, STATS["built"], STATS["build_fail"], STATS["q_done"], STATS["q_total"])]
            for a, (n, c, inf) in sorted(STATS["by_arm"].items()):
                lines.append("  %s n=%d acc=%.4f infra=%d" % (a, n, c / max(1, n), inf))
        with open(os.path.join(args.out, "status.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        time.sleep(60)


threading.Thread(target=status_loop, daemon=True).start()

# ---------------------------------------------------------------- phase 1: memories
MEM = {}


def build(ep):
    eid = ep["episode_id"]
    out = {}
    TAG.set({"arm": "build", "ep": eid, "qi": None})
    CALLS.set(None)
    for a in V_ARMS:
        out[a] = ifaces[a].memory_construction(ep.get("trajectory", []), ep.get("task", ""))
    if "A" in ifaces:
        p = os.path.join(MEMDIR, "A_%d.pkl" % eid)
        if os.path.exists(p):
            out["A"] = pickle.load(open(p, "rb"))
        elif args.reuse_mem:
            out["A"] = None
            out["A_err"] = "memory pickle missing (--reuse-mem, no rebuild): %s" % p
            print("NO MEMORY ep %d" % eid, flush=True)
        else:
            TAG.set({"arm": "A", "ep": eid, "qi": None})
            t0 = time.time()
            try:
                m = ifaces["A"].memory_construction(ep.get("trajectory", []), ep.get("task", ""))
                with open(p + ".tmp", "wb") as f:
                    pickle.dump(m, f)
                os.replace(p + ".tmp", p)
                out["A"] = m
                print("built A ep %d in %.0fs state_mem %d chars" % (eid, time.time() - t0, len(str(m.state_mem or ""))), flush=True)
            except Exception:  # noqa: BLE001
                out["A"] = None
                out["A_err"] = traceback.format_exc()[-3000:]
                print("BUILD FAIL ep %d\n%s" % (eid, out["A_err"]), flush=True)
    return eid, out


with ThreadPoolExecutor(max_workers=args.build_conc) as ex:
    for fut in as_completed([ex.submit(build, e) for e in eps]):
        eid, out = fut.result()
        MEM[eid] = out
        with SLOCK:
            STATS["built"] += 1
            STATS["build_fail"] += 1 if ("A" in ifaces and out.get("A") is None) else 0
print("phase 1 done in %.0fs" % (time.time() - STATS["t0"]), flush=True)


# ---------------------------------------------------------------- phase 2: questions
def last_answer_call(calls):
    for c in reversed(calls):
        if c.get("stage") in ("reader", "suff"):
            return c
    return None


def base_rec(arm, ep, qi):
    q = ep["qa_pairs"][qi]
    return {"arm": arm, "ep": ep["episode_id"], "qi": qi, "uuid": q.get("question_uuid"),
            "domain": ep.get("domain"), "task_type": ep.get("task_type", "unknown"),
            "qtype": q.get("type") or "unknown", "question": q.get("question", ""), "gold": q.get("answer", "")}


def judge(arm, ep, qi, pred):
    q = ep["qa_pairs"][qi]
    calls = []
    TAG.set({"arm": arm, "ep": ep["episode_id"], "qi": qi, "judge": True})
    CALLS.set(calls)
    score = compute_llm_as_judge(question=q.get("question", ""), golden_answer=q.get("answer", ""),
                                 predicted_answer=pred, judge_client=judge_client,
                                 task_description=ep.get("task", ""), task_type=ep.get("task_type", "unknown"),
                                 episode_id=str(ep["episode_id"]))
    jr = calls[-1].get("response") if calls else None
    return score, jr


def answer(arm, iface, mem, ep, qi):
    calls = []
    TAG.set({"arm": arm, "ep": ep["episode_id"], "qi": qi})
    CALLS.set(calls)
    t0 = time.time()
    try:
        res, err = iface.answer_question(ep["qa_pairs"][qi]["question"], mem), None
    except Exception:  # noqa: BLE001
        res, err = None, traceback.format_exc()[-3000:]
    return res, calls, err, time.time() - t0


def finish(rec, pred, calls, err, dt, ctx, judge_arm=None, copy_from=None):
    """Fill the common fields, judge (or copy a verdict), write the record."""
    ac = last_answer_call(calls)
    rec.update(pred=pred, err=err, dt=round(dt, 2), context=ctx, ctx_chars=len(ctx or ""),
               ctx_tokens=ntok(ctx), n_calls=len(calls),
               calls=[[c.get("stage"), (c.get("usage") or [None, None])[0], (c.get("usage") or [None, None])[1],
                       c.get("dt"), c.get("finish")] for c in calls],
               answer_stage=ac.get("stage") if ac else None,
               answer_prompt=ac.get("prompt") if ac else None,
               answer_response=ac.get("response") if ac else None,
               answer_prompt_tokens=(ac.get("usage") or [None])[0] if ac else None,
               reader_instr=READER_INSTR,
               pick=[[(c.get("response") or "")[:300], c.get("finish"), c.get("extra_body")]
                     for c in calls if c.get("stage") == "modelpick"])
    if pred is None:
        rec.update(score=None, infra=True, judge_response=None)
    elif copy_from is not None:
        rec.update(score=copy_from["score"], infra=copy_from["infra"], judge_response=copy_from.get("judge_response"),
                   judge_copied=True)
    else:
        try:
            s, jr = judge(judge_arm or rec["arm"], ep_by_id[rec["ep"]], rec["qi"], pred)
            rec.update(score=s, infra=s is None, judge_response=jr)
        except Exception:  # noqa: BLE001
            rec.update(score=None, infra=True, judge_response=None, judge_err=traceback.format_exc()[-2000:])
    Q_LOG.write(rec)
    note(rec["arm"], rec["score"], rec["infra"])
    return rec


ep_by_id = {e["episode_id"]: e for e in eps}


CAP_CUT_CHARS = int(23808 * 0.7) + 5 + (23808 - int(23808 * 0.7))  # a context cut by the released char cap


def code_search_kind(ctx):
    i = (ctx or "").find("# Code Search Result\n")
    if i < 0:
        return "none"
    res = ctx[i + len("# Code Search Result\n"):]
    if res.startswith("timeout"):
        return "timeout"
    if res.startswith("error:"):
        return "error"
    if res.startswith("Keyword search: empty") or res.startswith("Keyword search: no code"):
        return "empty"
    return "result"


def run_a3(ep, qi, mem):
    """A3 and, on the same calls, the released behaviour (A1r): run retrieve_nofast once. Where the released
    code would have answered from the sufficiency check, the hook records that return value; A1r's answer is
    parsed from it by the official answer_question (no model call), while A3 carries on to the keyword-search
    round, the final context and the reader. Where there was no fast path, both ran the same calls."""
    rec3, rec1 = base_rec("A3", ep, qi), base_rec("A1r", ep, qi)
    if mem.get("A") is None:
        for r in (rec3, rec1):
            finish(r, None, [], mem.get("A_err", "no memory"), 0.0, None)
        return
    st = {"a1_direct": None, "a1_ncalls": None}
    FAST.set(st)
    ic3 = copy.copy(ifaces["A"])
    ic3.method = _A3Method(ifaces["A"].method)
    res3, calls3, err3, dt3 = answer("A3", ic3, mem["A"], ep, qi)
    FAST.set(None)
    fast = st["a1_direct"] is not None
    ctx3 = res3["reasoning_trace"] if res3 else None
    stages = [c.get("stage") for c in calls3]
    info = {"a1_fast": fast, "stages": stages, "n_suff": stages.count("suff"), "n_codegen": stages.count("codegen"),
            "code_search": code_search_kind(ctx3), "a3_cut": len(ctx3 or "") == CAP_CUT_CHARS,
            "a3_reader_called": "reader" in stages}
    rec3["ama"] = info
    rec1["ama"] = dict(info)
    r3 = finish(rec3, res3["final_answer"] if res3 else None, calls3, err3, dt3, ctx3)
    if fast:
        ic1 = copy.copy(ifaces["A"])
        ic1.method = _FixedContext(st["a1_direct"])
        res1, _, err1, _ = answer("A1r", ic1, mem["A"], ep, qi)
        n = st["a1_ncalls"]
        calls1 = calls3[:n] if n is not None else []
        m = re.match(r"<<<AMA_DIRECT_ANSWER>>>(.*?)<<<END_AMA_DIRECT_ANSWER>>>\n", st["a1_direct"], re.DOTALL)
        ctx1 = st["a1_direct"][m.end():] if m else st["a1_direct"]
        finish(rec1, res1["final_answer"] if res1 else None, calls1, err1, dt3, ctx1)
    elif res3 is None:
        finish(rec1, None, [], err3, 0.0, None)
    else:
        # no fast path: the released code and A3 made the same calls and gave the same answer
        finish(rec1, res3["final_answer"], calls3, None, dt3, ctx3, copy_from=r3)


def run_question(ep, qi):
    eid = ep["episode_id"]
    mem = MEM.get(eid, {})
    for arm in V_ARMS:
        if (arm, eid, qi) in done:
            continue
        res, calls, err, dt = answer(arm, ifaces[arm], mem.get(arm), ep, qi)
        rec = base_rec(arm, ep, qi)
        finish(rec, res["final_answer"] if res else None, calls, err, dt, res["reasoning_trace"] if res else None)
    if "A3" in arms and not ((("A3", eid, qi) in done) and (("A1r", eid, qi) in done)):
        run_a3(ep, qi, mem)
    if "A" in arms and not ((("A1", eid, qi) in done) and (("A2", eid, qi) in done)):
        rec1, rec2 = base_rec("A1", ep, qi), base_rec("A2", ep, qi)
        if mem.get("A") is None:
            for r in (rec1, rec2):
                finish(r, None, [], mem.get("A_err", "no memory"), 0.0, None)
            return
        st = {"a1": None, "a2": None, "raw": None, "n_synth": 0}
        SYN.set(st)
        res1, calls1, err1, dt1 = answer("A", ifaces["A"], mem["A"], ep, qi)
        SYN.set(None)
        raw = st.get("raw") or ""
        direct = raw.startswith(R.DIRECT_ANSWER_PREFIX)
        path = {"direct": direct, "n_synth": st["n_synth"],
                "stages": [c.get("stage") for c in calls1],
                "a2_err": st.get("a2_err")}
        rec1["ama"] = path
        rec2["ama"] = path
        r1 = finish(rec1, res1["final_answer"] if res1 else None, calls1, err1, dt1, st.get("a1"))
        if res1 is None:
            finish(rec2, None, [], err1, 0.0, None)
        elif direct:
            # The sufficiency call answered; the final context is never read. Same answer as A1.
            rec2["a2_context_unread"] = True
            finish(rec2, res1["final_answer"], calls1, None, 0.0, st.get("a2"), copy_from=r1)
        elif st.get("a2") is None:
            finish(rec2, None, [], st.get("a2_err") or "no A2 context", 0.0, None)
        else:
            ic2 = copy.copy(ifaces["A"])
            ic2.method = _FixedContext(st["a2"])
            res2, calls2, err2, dt2 = answer("A2", ic2, mem["A"], ep, qi)
            finish(rec2, res2["final_answer"] if res2 else None, calls2, err2, dt2, st["a2"])


work = []
nq = max(len(e["qa_pairs"]) for e in eps) if eps else 0
for qi in range(nq):
    for e in eps:
        if qi < len(e["qa_pairs"]):
            work.append((e, qi))
STATS["q_total"] = len(work)


def _run(e, qi):
    try:
        run_question(e, qi)
    except Exception:  # noqa: BLE001
        print("QUESTION FAIL ep %d qi %d\n%s" % (e["episode_id"], qi, traceback.format_exc()[-3000:]), flush=True)
    with SLOCK:
        STATS["q_done"] += 1


with ThreadPoolExecutor(max_workers=args.q_conc) as ex:
    futs = [ex.submit(_run, e, qi) for e, qi in work]
    for f in as_completed(futs):
        f.result()

with SLOCK:
    el = time.time() - STATS["t0"]
    summ = {a: {"n": n, "acc": c / max(1, n), "infra": inf} for a, (n, c, inf) in STATS["by_arm"].items()}
open(os.path.join(args.out, "DONE"), "w").write(json.dumps({"elapsed": el, "arms": summ}) + "\n")
print("SHARD DONE %.0fs %s" % (el, json.dumps(summ)), flush=True)
