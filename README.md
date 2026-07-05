# KVMemory — verbatim-selection memory for long-horizon agents

Companion code, configs, and raw prediction artifacts for the paper
[`paper/kvmemory.pdf`](paper/kvmemory.pdf) (*KVMemory: gist routes, verbatim answers*).

## What is new / what is reproduced / what is NOT claimed

**New (the paper's claims):**
- **Payload fidelity dominates.** On dense agent trajectories, selecting raw spans and re-injecting them *verbatim* beats every published lossy-construction memory system; matched-evidence ablations make the payload's role causal (re-encoding the *same* selected evidence costs 8–14pp isolated, −5.1pp in-pipeline).
- **A memory × reader confound audit.** A same-batch factorial separates memory-side gains (routing +1.4/+1.7pp) from reader-side gains (one answer instruction, +5.1–5.8pp). Memory comparisons that do not control the reader largely measure reader differences.
- **Reader-side saturation law.** Eight reader-side mechanisms (compression, gating, scaffolds, deterministic tools, pre-computed reasoning, re-retrieval) tested over a fixed memory: a mechanism helps iff it supplies *specific, query-addressable* information the context is missing; reasoning must happen in the answering pass.
- **Structural address resolution.** Step references are dereferenced deterministically (a second access path beside content routing); robust to near-miss corruption, degrades gracefully under adversarial corruption.
- **Position-preserving KV sub-selection.** Selected spans are served by gathering their cached KV at original positions — faithful to a full-prefill-then-mask oracle, TTFT 1.6–10.9× (prototype microbenchmark).

**Reproduced here:** every AMA-Bench number in the paper's tables maps to a raw
prediction file in [`results/`](results/) (official harness, Qwen3-32B reader +
judge, full open-ended split). See [`results/MANIFEST.md`](results/MANIFEST.md)
for the tag → paper-table mapping, including all negative results.

**NOT claimed:**
- No official leaderboard placement yet (submission pending; numbers are our runs of the official harness).
- No LOCOMO SOTA (positioned as an out-of-domain sanity check; file-first systems report higher under their own setups).
- No production serving result (efficiency section = single-request prototype microbenchmark; no vLLM/SGLang integration yet).
- No privacy guarantee for cached KV (contextualized KV carries influence from unselected history; see the paper's Limitations).

## Layout

```
paper/       kvmemory.tex + .bib + style files + compiled PDF (self-contained; compiles with tectonic or pdflatex)
kvmemory/    the memory package: core.py (store, tiers, assembly, budget), components.py (routers incl. structural address resolution)
ama/         AMA-Bench integration: harness/ (2 patched files), configs/ (all method configs), agentic_reader.py (reader-mode ablations), answer instructions
analysis/    cluster_ci.py (QA-level + episode-cluster bootstrap), pin_split.py (step-pin mechanism signature)
results/     mu_merged_<TAG>.json — raw per-question predictions + judge scores for every run; mu_<TAG>_done — run manifests (n, acc, CI)
scripts/     run_ama.sh — end-to-end reproduction on one 8×A100 node
```

## Reproducing the AMA-Bench numbers

Requirements: one 8×A100-80GB node, [AMA-Bench](https://github.com/AMA-Bench/AMA-Bench),
vLLM ≥ 0.8, Qwen3-32B weights.

```bash
git clone https://github.com/AMA-Bench/AMA-Bench
cp ama/harness/method_kvmemory.py  AMA-Bench/src/method/kvmemory.py
cp ama/harness/memory_interface.py AMA-Bench/src/memory_interface.py   # adds the env-gated answer-instruction + reader hooks
cp -r kvmemory <dir-on-PYTHONPATH>/
bash scripts/run_ama.sh cfg_flagship.json MYTAG structured        # -> results/mu_merged_MYTAG.json
bash scripts/run_ama.sh cfg_flagship.json MYTAG_DEFAULT plain     # the default-reader (memory-only) cell
python3 analysis/cluster_ci.py                                     # QA-level + episode-cluster CIs
```

Key environment variables (all optional, documented in `ama/maxutil_run.sh`):
`AMA_ANSWER_INSTR_FILE` (structured answer instruction; the paper's reader-side
variable), `AMA_AGENTIC_READER` + `AMA_AGENTIC_MODE` (reader-mode ablations:
`select / gated / decouple / state / tools / plan / lossy`), `SPRAG_PIN_SHUFFLE`
(structural-address corruption ablations).

## Honest-measurement notes

- AMA-Bench ships **only a test split**; all tuning necessarily happened on it (true for every system on its leaderboard). Our defenses: a single global config across domains, consistency across three benchmark families, and full release of configs + predictions + judge outputs here.
- Same-day replicate noise is ~0.5pp; cross-day ~2pp; SOFTWARE-domain batch noise ±4pp. All paper deltas come from same-batch pairs.
- Episode-cluster bootstrap CIs are ~30% wider than QA-level; the paper reports both.

## Contact

Zefeng Cai — see the paper title page.
