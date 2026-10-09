# Paired re-run against AMA-Agent

Code for the comparison in [docs/AMA_AGENT_PAIRED.md](../../docs/AMA_AGENT_PAIRED.md): AMA-Agent and
VeraKV on the same AMA-Bench questions, the same Qwen3-32B server and the same judge. The per-question
records are in [results/ama_paired/](../../results/ama_paired/).

| File | What it does |
|---|---|
| `run.py` | Runs the arms on one shard. It builds AMA-Agent's memory, answers every question with each arm, judges each answer with the official `compute_llm_as_judge`, and logs every model call. |
| `make_tokcap.py` | Writes `retrieve_tokcap.py` (arm A2) from AMA-Agent's released `retrieve.py`. |
| `make_nofast.py` | Writes `retrieve_nofast.py` (arm A3) from the same file. |
| `analyze.py` | Readout of the main run: A1, A2, V1, V2. |
| `a3_analyze.py` | Readout of the A3 run against the main run. |
| `path_splits.py` | The context-cap split: A2 − A1 where the reader read a context that was cut. |
| `arm_cost.py` | Model calls, tokens and wall time per question for each arm. |
| `make_label_items.py`, `label.py`, `label_summary.py` | Build the disagreement items, label them with a model, then summarise the labels and draw the hand-check sample. |
| `modelpick_gates.py`, `modelpick_analyze.py` | Gates and readout of the model-pick fix re-run ([docs/MODEL_PICK_FIX.md](../../docs/MODEL_PICK_FIX.md)): arms V1, V1f, V2 under two readers. |

## Check the numbers from the released records

No GPU or model is needed. Each command prints the numbers in the write-up.

```bash
python ama/paired/analyze.py results/ama_paired/full --july results --data data/ama_test.jsonl
python ama/paired/a3_analyze.py results/ama_paired/a3 results/ama_paired/full
python ama/paired/path_splits.py results/ama_paired/full
python ama/paired/arm_cost.py results/ama_paired/full results/ama_paired/a3
python ama/paired/label_summary.py results/ama_paired/labels.jsonl --sample-out handcheck.jsonl
```

- `--july results` compares V1 with the July runs of the same configuration (`results/mu_merged_FPA.json`
  and `mu_merged_FPB.json`).
- `--data` adds the splits by trajectory length and is optional (see [data/README.md](../../data/README.md)).
- `analyze.py --out` and `a3_analyze.py --out` write the JSON files in `results/ama_paired/`.
- `handcheck.jsonl` comes out identical to `results/ama_paired/handcheck_sample.jsonl`.
- The labelling items cannot be rebuilt from the released records, which leave out the full reader prompts.

## Run it again

We used one 8×A100-80GB node. The main run took about 2.5 hours and the A3 run about 1.5 hours.

**1. AMA-Bench with VeraKV's harness files and the two generated arms.** We used AMA-Bench commit
`ddfd319`.

```bash
git clone https://github.com/AMA-Bench/AMA-Bench && git -C AMA-Bench checkout ddfd319
cp ama/harness/method_kvmemory.py  AMA-Bench/src/method/kvmemory.py   # + register "kvmemory" in src/method_register.py
cp ama/harness/memory_interface.py AMA-Bench/src/memory_interface.py
C=AMA-Bench/src/method/ama_agent_core
python ama/paired/make_tokcap.py $C/retrieve.py $C/retrieve_tokcap.py
python ama/paired/make_nofast.py $C/retrieve.py $C/retrieve_nofast.py
```

Both generators assert that every line they replace occurs exactly once, so they refuse a `retrieve.py`
that has changed.

**2. A config folder.**

```bash
mkdir -p cfg
for p in 8060 8061 8062 8063; do
  sed -e "s/^vllm_port: .*/vllm_port: $p/" -e "s#^model: .*#model: \"/path/to/Qwen3-32B\"#" \
      AMA-Bench/configs/qwen3-32B.yaml > cfg/qwen_$p.yaml
done
cp AMA-Bench/configs/ama_agent.yaml ama/configs/cfg_flagship.json ama/configs/cfg_kvmem_causal.json \
   ama/configs/cfg_flagship_nothink.json cfg/
```

**3. Servers.** We used vLLM 0.28.0 with torch 2.13 and CUDA 12.9 on Python 3.11.

```bash
# AMA-Agent's similarity retrieval; configs/ama_agent.yaml expects it on port 8003
CUDA_VISIBLE_DEVICES=7 vllm serve /path/to/Qwen3-Embedding-4B --runner pooling \
    --served-model-name Qwen/Qwen3-Embedding-4B --port 8003 --gpu-memory-utilization 0.21 --max-model-len 32768 &
# reader and judge for every arm: four Qwen3-32B instances, tensor parallel 2
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2*g)),$((2*g+1)) vllm serve /path/to/Qwen3-32B --served-model-name /path/to/Qwen3-32B \
      --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.72 --max-num-seqs 128 \
      --port $((8060+g)) &
done
```

- The served name must equal `model:` in the `cfg/qwen_<port>.yaml` files.
- Thinking mode stays at Qwen3's default (on). AMA-Bench's client does not read the config's
  `enable_thinking` key.
- On Python 3.11, tensor parallelism needed `from __future__ import annotations` at the top of
  `flashinfer/comm/fd_exchange.py`.
- We set `VLLM_USE_FLASHINFER_SAMPLER=0`. Every call is greedy.

**4. The runs, one driver per Qwen3-32B instance.**

```bash
export AMA_TOKCAP_TOKENIZER=/path/to/Qwen3-32B
unset AMA_ANSWER_INSTR_FILE AMA_AGENTIC_READER SPRAG_PIN_SHUFFLE   # the harness's default reader
for g in 0 1 2 3; do   # arms A1 and A2 ("A"), V1, V2
  python ama/paired/run.py --ama-root AMA-Bench --cfg-dir cfg --data data/ama_test.jsonl \
      --port $((8060+g)) --shard $g --nshards 4 --out runs/full/s$g &
done; wait
for g in 0 1 2 3; do   # arm A3 and its same-run replicate A1r, on the main run's AMA-Agent memory
  python ama/paired/run.py --ama-root AMA-Bench --cfg-dir cfg --data data/ama_test.jsonl \
      --port $((8060+g)) --shard $g --nshards 4 --out runs/a3/s$g \
      --arms A3 --mem-dir runs/full/s$g/mem --reuse-mem &
done; wait
```

Each shard writes three files:
- `q.jsonl`: one record per question and arm;
- `calls.jsonl`: every model call, with its exact prompt and raw response;
- `status.txt`: progress.

The analysis commands above take `runs/full` and `runs/a3` in place of `results/ama_paired/full` and
`results/ama_paired/a3`.

**5. Labels.**

```bash
python ama/paired/make_label_items.py runs/full items.jsonl
export LABEL_BASE_URL=<endpoint serving /chat/completions> LABEL_API_KEY=<key> LABEL_MODEL=<model>   # we used GPT-5.6-Sol
python ama/paired/label.py items.jsonl labels.jsonl
python ama/paired/label_summary.py labels.jsonl --sample-out handcheck.jsonl
```

`label.py` sends `reasoning_effort: high`, retries once at `medium` if that fails and records which one
it used. It can be stopped and restarted: items already in `labels.jsonl` are skipped.

## The model-pick fix re-run

[docs/MODEL_PICK_FIX.md](../../docs/MODEL_PICK_FIX.md) re-runs the deployed configuration with its
model-pick call fixed. The arms are V1 (`cfg_flagship.json`), V1f (`cfg_flagship_nothink.json`) and V2
(`cfg_kvmem_causal.json`). Each runs under the harness's default reader and under the structured
instruction, on all six domains.

Check the numbers from the released records:

```bash
python ama/paired/modelpick_analyze.py results/modelpick_fix --paired1008 results/ama_paired/full
```

To run it again, use the same setup as above. The fix is the `pick_thinking` switch in
`ama/harness/method_kvmemory.py`, so step 1 already installs it. The embedder is not needed. We gave the
Qwen3-32B servers `--gpu-memory-utilization 0.85`, and the run took about 7 hours on one node. Each
instance gets two drivers:

```bash
export AMA_TOKCAP_TOKENIZER=/path/to/Qwen3-32B
unset AMA_ANSWER_INSTR_FILE AMA_AGENTIC_READER SPRAG_PIN_SHUFFLE
D=WEB,EMBODIED_AI,Game,TEXT2SQL,SOFTWARE,OPENWORLD_QA
for g in 0 1 2 3; do
  A="--ama-root AMA-Bench --cfg-dir cfg --data data/ama_test.jsonl --domains $D --arms V1,V1f,V2
     --port $((8060+g)) --shard $g --nshards 4 --q-conc 32"
  python ama/paired/run.py $A --out runs/mpfix/def/s$g &
  AMA_ANSWER_INSTR_FILE=ama/dec_instr.txt python ama/paired/run.py $A --out runs/mpfix/str/s$g &
done; wait
python ama/paired/modelpick_gates.py runs/mpfix --full
python ama/paired/modelpick_analyze.py runs/mpfix --paired1008 results/ama_paired/full
```

The gates need the reader prompts, which only a fresh run's records hold. The script checks four things:
- every V1f pick call carried `enable_thinking: false`;
- every V1 pick call still stopped inside `<think>`;
- each reader got its own instruction;
- infrastructure failures stayed under 2%.
