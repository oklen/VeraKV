# VeraKV

**Agent memory that hands the model the original lines, and #1 on AMA-Bench when submitted.**

| AMA-Bench, official harness, all 2,496 questions | Accuracy |
|---|---|
| **VeraKV** — verified on the public leaderboard, #1 as of 2026-07-15 | **0.6478** |
| HybridFocus (leaderboard entry) | 0.6246 |
| VeraKV, memory only (the harness's default reader) | 0.5954 |
| AMA-Agent (purpose-built for the benchmark) | 0.5722 |
| Long-context Qwen3-32B, no memory | 0.52 |
| MemoRAG | 0.4606 |
| HippoRAG2 | 0.4480 |
| MemGPT | 0.3304 |
| Mem0 | 0.2104 |

- **Wins 5 of 6 AMA-Bench domains** against the purpose-built AMA-Agent.
- **The memory alone beats AMA-Agent:** 0.5954 vs 0.5722 with the harness's own default reader, before any
  reader-side change.
- **Beats dedicated dialogue memory on LOCOMO:** J = 0.704 against Mem0's 0.671 and Zep's 0.660, under
  LOCOMO's public gpt-4o-mini protocol.
- **Original lines beat rewritten memory by 8–14 points.** With matched evidence, reader and judge,
  serving an LLM summary or extracted facts instead of the verbatim step costs 8–14 pp.
- **First token up to 10.9× faster.** Gathering the selected spans' cached KV instead of re-reading them
  as text keeps time-to-first-token flat at about 40 ms. The speedup is 1.6× to 10.9× as the evidence
  grows to 4k tokens, and 32× at 32k. The first token is identical to a full-prefill oracle.

A later leaderboard entry (0.6975, dated 2026-09-09) has since overtaken VeraKV, which now ranks #2.
Under the gpt-4o-mini judge, prompting LOCOMO's full history scores 0.756. These and the other
protocol caveats are in the paper and in [Honest-measurement notes](#honest-measurement-notes).

**What it is:** a memory layer for long-running agents. It keeps the raw trajectory, routes each
question to the steps it needs (lexical overlap plus step-number pins), and serves those steps
**verbatim** in a filled budget. It can serve them as text, or as their own cached KV at their original
positions.

This repository has the code, configurations, prompts, raw predictions and judge outputs behind every
number in the paper, [`paper/kvmemory.pdf`](paper/kvmemory.pdf) (*When Does KV Reuse Work for Agent
Memory? A Controlled Study of Payload Fidelity, Store Conditioning, and Cache Harvesting — with a
Reproducible State-of-the-Art System*; source in [`paper/kvmemory.tex`](paper/kvmemory.tex)).

## Findings in detail

- **Payload fidelity, measured on agent trajectories.** Most memory systems keep the raw history; what
  differs is what the reader is served (Mem0's search returns its extracted facts; AMA-Agent's released
  code serves its top-5 raw turns by embedding similarity together with an LLM-written state summary).
  With matched cited evidence, reader and judge, serving an LLM summary or extracted facts instead of the
  verbatim step costs 8–14pp; in the deployed pipeline a deterministic swap to the query-relevant
  *original lines* costs −2.6pp versus −5.1 / −5.9pp for generative re-encodings. AMA-Bench's own needle
  control prices the same choice: serving MemoryBank's constructed memory of the answer-bearing turns
  instead of the turns drops BabyAI accuracy from 0.46 to 0.27 (−41% relative).
- **The reader is a separable axis.** A same-batch memory × reader factorial: routing adds +3.1–3.5pp at
  either reader, a one-sentence structured answer instruction adds +5.1–5.5pp at either router. The
  instruction can travel inside the memory's return value, and as a rotated skill-KV block it ties the
  text instruction.
- **Within the context window, KV gather is exact and costs less, not more accurate.** Gathering the
  selected spans' cached KV at their original positions reproduces a full-prefill-then-mask oracle and is
  accuracy-equivalent to re-prefilling the same spans as text (17.3% vs 17.4%, 896 QA); first-token
  latency decouples from evidence size.
- **Beyond the window, a characterization.** Independently encoded event stores collapse on the official
  harness (−13.2pp paired, n=2431); anchored stores with a fresh two-event tail recover 72%; two scoped
  regularities (evidence converts to accuracy only through joint encoding; write-time conditioning must
  match what is served, in content and position); six fresh-budget selectors, including a KEEP-style
  one, leave recency tied or ahead; harvesting the agent's own rollout cache replaces the ~5× store-write
  cost at parity (8B/32B, Qwen/Llama, AMA-Bench/LOCOMO); a mixed deployment policy ties the routed-text
  carrier baseline on the full official harness (.397 vs .397).
- **The system.** The memory stack alone (harness default reader) scores 0.5954 against the purpose-built
  AMA-Agent's 0.5722; with the disclosed structured answer instruction it reaches the verified 0.6478.
  The one domain it does not lead, SOFTWARE, is a coverage ceiling on temporal-scan questions (verifying
  "which step first/last" needs the whole 60–100k-token trajectory), not a retrieval or reasoning gap.

## Layout

```
paper/      kvmemory.tex + .bib + style files + compiled PDF (30 pp; tectonic or pdflatex)
kvmemory/   the memory package (core, components, routers, HF backend with KV sub-selection) and one
            module per experiment, each runnable as `python -m kvmemory.<name>` (usage in its docstring)
ama/        AMA-Bench integration: harness/ (2 patched files), configs/, agentic_reader.py (reader-mode
            ablations), answer instructions, maxutil_run.sh (vLLM launch + sharding + merge),
            br_judge.py / bg_judge.py (official judge for the KV-bridge answers)
analysis/   scripts that turn run outputs into the paper's numbers (see the table below)
results/    raw per-question predictions + judge scores (mu_merged_<TAG>.json, run manifests mu_<TAG>_done),
            results/oow/ (KV-serving and out-of-window runs), results/probes/ (synthetic probes),
            smaller outputs; index in results/MANIFEST.md
scripts/    run_ama.sh — end-to-end official-harness run on one 8×A100 node
submissions/ the leaderboard submission file
data/       where the benchmark files go (not shipped; see data/README.md)
```

## Setup

Two environments were used (exact versions in [ENVIRONMENT.md](ENVIRONMENT.md)):

- **Official AMA-Bench harness** (text pipeline, every leaderboard-style number): vLLM serving
  Qwen3-32B as reader and judge.
- **KV experiments** (`kvmemory/kv_*.py`): HuggingFace transformers, one model copy per GPU; Qwen3-8B,
  Qwen3-32B (shard with `SPRAG_DEVICE_MAP=auto`), Llama-3.1-8B-Instruct.

Environment variables used by the modules: `SPRAG_MODEL_PATH` (local model directory; required),
`SPRAG_EMBED_PATH` (Qwen3-Embedding-0.6B, for the embedding / hybrid routers), `SPRAG_ATTN_IMPL=sdpa`
(required for the block-diagonal masks of the store experiments; `eager` for attention probes),
`SPRAG_ROPE_FACTOR=4.0` + `SPRAG_MAX_CTX=131072` (YaRN for over-window runs). Outputs default to `./out/`.

## Reproducing

**Official-harness runs (AMA-Bench, Qwen3-32B reader + judge).** Requirements: one 8×A100-80GB node,
[AMA-Bench](https://github.com/AMA-Bench/AMA-Bench), vLLM, Qwen3-32B weights.

```bash
git clone https://github.com/AMA-Bench/AMA-Bench
cp ama/harness/method_kvmemory.py  AMA-Bench/src/method/kvmemory.py      # + register "kvmemory" in src/method_register.py
cp ama/harness/memory_interface.py AMA-Bench/src/memory_interface.py     # env-gated answer-instruction + reader hooks
export AMA_BENCH=$PWD/AMA-Bench MODEL=/path/to/Qwen3-32B VENV=/path/to/vllm_env
bash scripts/run_ama.sh cfg_kvmem_causal.json MYTAG structured   # lexical+pin memory, structured reader -> results/mu_merged_MYTAG.json
bash scripts/run_ama.sh cfg_kvmem_causal.json MYTAG_DEF plain     # the default-reader (memory-only) cell
python analysis/cluster_ci.py                                     # QA-level + episode-cluster CIs
```

`ama/maxutil_run.sh` documents the reader-mode switches (`AMA_AGENTIC_READER`, `AMA_AGENTIC_MODE`) and
the pin-corruption ablation (`SPRAG_PIN_SHUFFLE`). The submitted 0.6478 entry used `cfg_flagship.json`
(lexical + model-pick fusion + pin); App. "The deployed router" shows the fusion is score-neutral, and
`cfg_kvmem_causal.json` (lexical + step-pin) is the released default.

**KV and out-of-window experiments (HF transformers).** The AMA-Bench test file goes to
`data/ama_test.jsonl` (see [data/README.md](data/README.md)). Each module's docstring carries its run
line, e.g.

```bash
SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    python -m kvmemory.kv_globhot --shard 0 --queue_dir ./out/gh_q
```

The mechanism-subset runs use 103 within-window episodes (≤24k tokens, drawn round-robin across domains;
n=824 QA) and an 8-way
work queue (`--queue_dir`, one process per GPU). The official-harness bridge generates answers with
`kvmemory.kv_ama_bridge` (Qwen3-32B, YaRN 4×) and scores them with AMA-Bench's own judge via
`ama/br_judge.py` (tx / iso / b_hot) or `ama/bg_judge.py` (deployment map).

**Paired statistics from the released outputs.**

```bash
python analysis/oow_paired.py results/oow/official_k12.jsonl.gz iso-tx b_hot-tx     # Table "oow", K=12
python analysis/oow_paired.py results/oow/gh.jsonl.gz glob_hotR-bh glob_R-glob     # harvest cell
python analysis/oow_paired.py results/oow/rv.jsonl.gz --arms                        # list arms
```

`oow_paired.py` reproduces accuracies, discordant counts and exact McNemar p-values exactly; its
episode-clustered bootstrap (10,000 draws, seed 0) can differ from a paper CI in the last digit.

## Experiments → scripts (results added in this release)

| Paper | What | Script(s) | Output / analysis |
|---|---|---|---|
| §3.4, Tab. "family" | appendix → original lines / facts; full trajectory recency-truncated | `ama/agentic_reader.py` (`extlines`, `facts`), `ama/configs/cfg_full22.json` | `results/mu_merged_{EXTLW,FACTSW,FULLTR}.json` |
| §3.4, Fig. "payload" | oracle-evidence payload ablation (verbatim / summary / facts, 910 QA) + Llama-3.1-8B re-judge | `kvmemory/kv_payload.py` (vLLM client), `kvmemory/kv_payload_hf.py`, `kvmemory/cross_judge.py` | — |
| App. D | memory-assembly 2×2 (selection × gist overview), per-domain split | `ama/configs/cfg_arm_{recent,gist,retrieval}.json`, `cfg_kvmem_lex.json` | `results/mu_merged_ASM_*.json`, `analysis/assembly_2x2.py` |
| §3.1, App. E | SOFTWARE temporal-scan coverage ceiling (8B and 32B readers) | `kvmemory/sw_mechanism.py` | `results/sw_mechanism/` |
| App. F | deployed-router audit: same-batch replication, lexical+pin 0.6466 vs model-pick hybrid+pin 0.6538, paired −0.8pp [−2.3, +0.8]; a later model-pick run on the upgraded serving stack, 0.6410 (not paired) | `ama/configs/cfg_kvmem_causal.json`, `cfg_flagship.json` | `results/mu_merged_RESTRLEXF.json` (lexical+pin), `mu_merged_RESTRF2.json` (model-pick+pin, same batch), `mu_merged_RESTRF4.json` (model-pick+pin, later run); flagship replicates `RESTRF`, `RESTRF2`, `RESTRF3` |
| App. G | step-number deletion / pin-only robustness (+ Llama-3.1-8B re-judge) | `kvmemory/kv_stepablate.py`, `kvmemory/cross_judge.py` | — |
| §3.5, App. H | mask-oracle gate (synthetic), 0.7 vs 13.5-logit gap, TTFT vs evidence size | `kvmemory/kv_select_smoke.py` | — |
| §3.5, App. H | sequence-level faithfulness, logit-KL, eager/SDPA, YaRN 4× | `kvmemory/kv_faithful.py` | — |
| §3.5, App. H | KV-vs-text end-to-end parity: 144 QA and 896 QA (17.3% vs 17.4%) | `kvmemory/kv_equiv.py` | `results/kv_equiv_144/`, `results/kv_equiv_896/` |
| §3.5 | unselected-upstream perturbation (gathered KV changes 111/168, text 0) | `kvmemory/kv_sidechannel.py` | `results/kv_sidechannel/` |
| §3.7 | storage cost and TTFT to 64k selected tokens; prefill saved over a run | `kvmemory/kv_efficiency.py`, `kvmemory/bench_kv.py` | — |
| §3.7 | resident full-trajectory prefix-cache baseline | `kvmemory/kv_pfx.py` | `results/kv_cost/pfx_8b.json`, `analysis/pfx_ttft.py` |
| §3.6 | answering protocol as a rotated skill-KV block | `kvmemory/kv_skill.py` | `results/oow/sk.jsonl.gz` |
| §3.6 | query-routed skill library (replacement vs additive; text vs KV) | `kvmemory/kv_skillib.py` | `results/oow/sl.jsonl.gz` |
| §4, Tab. "oow" | official harness: tx / iso / anchored+tail at K=12 and K=5 | `kvmemory/kv_ama_bridge.py`, `ama/br_judge.py` | `results/oow/official_k{12,5}.jsonl.gz` |
| §4 | over-window cell (30–100k, YaRN) at 8B and 32B | `kvmemory/kv_owfloor.py` | `results/oow/owf{8,32}.jsonl.gz` |
| §4 | stage-resolved query cost, write cost | `kvmemory/kv_ttft.py` | `results/kv_cost/ttft_{8b,32b}.jsonl`, `analysis/ttft_stages.py` |
| §4 | evidence matching (tx_plus), conditioner content, full-reading ceiling, tail dial (8B, 32B) | `kvmemory/kv_review.py` | `results/oow/rv{,32}.jsonl.gz` |
| §4 | generated-digest conditioner and other floor arms | `kvmemory/kv_floor.py` | `results/oow/fl.jsonl.gz` |
| §4 | harvest 2×2 (gathered rows, opening rows, fresh tail); Llama-3.1-8B replication | `kvmemory/kv_globhot.py` | `results/oow/gh{,_llama}.jsonl.gz` |
| §4 | agent-style cache header; 32B harvest vs store vs full reading | `kvmemory/kv_hxfer.py` | `results/oow/hx{,32}.jsonl.gz` |
| §4 | LOCOMO, sessions as events (8B, 32B; harvest and skill arms) | `kvmemory/kv_locomo.py` | `results/oow/loc{8,32}.jsonl.gz` |
| §4 | distributional autopsy (first-token KL) | `kvmemory/kv_klprobe.py` | `results/oow/kl.jsonl.gz`, `analysis/kl_probe.py` |
| §4 | fallback triggers (mode gate, self-verification, attention landing, voting) | `kvmemory/kv_gate.py`, `kv_verify.py`, `kv_attn.py`, `kv_matrix.py` | `results/oow/mx.jsonl.gz`, `analysis/{vote_analysis,cascade_ceiling}.py` |
| §4 Reg. I | K=5 vs K=12 per carrier; same extra events to both paths | `kvmemory/kv_ama_bridge.py`, `kvmemory/kv_evext.py` | `results/oow/official_k*.jsonl.gz`, `results/oow/ev.jsonl.gz` |
| §4 Reg. II | note-transplant test (delimiter vs random anchor rows) | `kvmemory/kv_notesel.py` | `results/oow/nt.jsonl.gz` |
| §4 Reg. II | frozen global conditioner / conditioner geometry | `kvmemory/kv_fixanchor.py`, `kvmemory/kv_distanchor.py` | `results/oow/{fx,dx}.jsonl.gz` |
| §4 Reg. II | EPIC-style boundary recompute; dependency-graph conditioner | `kvmemory/kv_epic.py`, `kvmemory/kv_depanchor.py` | `results/oow/{ep,da}.jsonl.gz` |
| §4 | six fresh-budget selectors (recency, graph, attention, query, random; KEEP-style) | `kvmemory/kv_frontier.py`, `kvmemory/kv_keepsel.py` | `results/oow/{fr,kp}.jsonl.gz` |
| §4 | deployment map on the official harness (gsk vs anchored stores vs text) | `kvmemory/kv_ama_bridge.py --arms gsk,g_txt`, `ama/bg_judge.py` | `results/oow/official_deploymap.jsonl.gz` |
| §4 probe 1–3 | phantom conclusions, erasure boundary, sink vs dependency | `kvmemory/kv_phantom.py` (+ `kv_vartrack.py`) | `results/probes/phvt/`, `analysis/probes/phvt_analyze.py` |
| §4 probe 4 | four-arm decomposition; incomplete routing | `kvmemory/kv_vardecomp.py`, `kv_predigest.py`, `kv_predigest_sweep.py` | `results/probes/{vd,pd,pds}/`, `analysis/probes/` |
| §4 probe 4 | independent decoys, menu readout, entropy | `kvmemory/kv_decoyctl.py`, `kv_decoyctl2.py`, `kv_sweepmenu.py` | `results/probes/{dc,d2,sm}/` |
| §4 probe 4 | computed verdicts, predicate binding, write-time notes | `kvmemory/kv_mater.py`, `kv_refcarrier.py`, `kv_noteknob2.py` | `results/probes/{mt,rc,nk2}/`, `analysis/probes/nk2_analyze.py` |
| §3.2 | LOCOMO, Qwen3-32B reader/judge; public gpt-4o-mini protocol | `kvmemory/locomo_eval.py`, `kvmemory/locomo_gpt4omini.py` | — |
| §3.3 | LongMemEval-S (128k YaRN full vs routed at 28k) | `kvmemory/longmemeval_eval.py` | — |
| App. I | Llama-3.1-70B re-judge of a full official run | `analysis/rejudge_llama70b.py`, `ama/configs/judge_llama70b.yaml` | `results/judge_llama70b/ama_rejudged.json` |

The shared modules `kv_scope`, `kv_replay`, `kv_write`, `kv_matrix`, `kv_ow`, `kv_floor`, `kv_globhot`,
`kv_distanchor` and `kv_orphan` hold the store-encoding, assembly and gather primitives the
out-of-window runs import; each is also runnable as the experiment it was written for. The four synthetic
probes are the starting point of the companion paper
[Compute Globally, Materialize Locally](https://github.com/oklen/Compute-Globally-Materialize-Locally),
which carries the extended versions. `gpt-4o-mini` runs need an OpenAI-compatible endpoint
(`LOCOMO_API_BASE_URL`, `LOCOMO_API_KEY`); LOCOMO/LongMemEval runs with a local reader need a vLLM server.

## Honest-measurement notes

- AMA-Bench ships only a public test split; the leaderboard configuration was developed on it, so the
  paper labels 0.6478 an exploratory engineering result. The out-of-window recipes were developed on a
  103-episode mechanism subset before one evaluation on the full official set.
- Same-day replicate noise is ~0.5pp, cross-day ~2pp, SOFTWARE batch noise ±4pp; every reported paired delta is
  same-batch; the one cross-batch reading (App. F: 0.6466 vs 0.6410) is labeled as such.
- The serving-cell protocol of §4 (fixed-K retrieval, no overview, official default answer prompt) is
  not the deployed pipeline; its text arm (.397) is a routed-text carrier baseline about 25pp below the
  headline system. Never compare the two across protocols (paper App. "Protocol ledger").
- KV gather is exact only within the positional window; every headline accuracy comes from the text
  pipeline. Claims for the cache path are about first-token latency, not answering faster.

## License

Code is released under the MIT License (see `LICENSE`). Benchmarks and datasets keep their own
licenses: dataset files are not included (see `data/README.md`), and the per-question result files quote
AMA-Bench questions and reference answers only so the scores can be audited.

## Contact

Zefeng Cai — see the paper title page.
