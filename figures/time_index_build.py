"""Time VeraKV's memory build for one AMA-Bench episode with this repository's own code. No model call.

    python figures/time_index_build.py <ama_test.jsonl> <episode_id> [repeats]

Builds the trajectory text the AMA-Bench harness hands a memory (Step N / Action / Observation, as in
ama/harness/memory_interface.py), runs KVMemoryMethod.memory_construction from ama/harness/method_kvmemory.py
with ama/configs/cfg_kvmem_causal.json (split into steps), then computes every step's index line, the first
characters of its action and observation that the overview shows. Prints every run and the median.

ama/harness/method_kvmemory.py imports AMA-Bench's BaseMethod and transformers' AutoTokenizer; neither is used
to build the memory, so if they are not installed, empty stand-ins are registered before the import."""
import json
import os
import statistics
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    import src.method.base_method  # noqa: F401  (AMA-Bench, if installed)
except ImportError:
    for name in ("src", "src.method", "src.method.base_method"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["src.method.base_method"].BaseMethod = object
try:
    import transformers  # noqa: F401
except ImportError:
    sys.modules["transformers"] = types.ModuleType("transformers")
    sys.modules["transformers"].AutoTokenizer = None
from ama.harness.method_kvmemory import KVMemoryMethod  # noqa: E402

path, episode_id = sys.argv[1], int(sys.argv[2])
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 9
ep = next(e for e in map(json.loads, open(path, encoding="utf-8")) if e["episode_id"] == episode_id)
parts = []
for s in ep["trajectory"]:
    parts += ["Step %s:" % s.get("turn_idx", 0), "Action: %s" % s.get("action", ""),
              "Observation: %s" % s.get("observation", ""), ""]
traj_text = "\n".join(parts)
method = KVMemoryMethod(os.path.join(ROOT, "ama", "configs", "cfg_kvmem_causal.json"))

times = []
for _ in range(repeats):
    t0 = time.perf_counter()
    mem = method.memory_construction(traj_text, ep.get("task", ""))
    lines = [mem._ensure_gist(s) for s in mem.segments]
    times.append(time.perf_counter() - t0)
assert len(mem.segments) == len(ep["trajectory"]) == len(lines)
print("episode %d: %d steps, %d characters of trajectory text" % (episode_id, len(lines), len(traj_text)))
print("runs (ms):", ", ".join("%.2f" % (1000 * t) for t in times))
print("median %.2f ms" % (1000 * statistics.median(times)))
print(json.dumps({"episode": episode_id, "steps": len(lines), "repeats": repeats,
                  "median_ms": round(1000 * statistics.median(times), 2),
                  "runs_ms": [round(1000 * t, 2) for t in times]}))
