#!/bin/bash
# Max-util AMA run: NINST x TP=2 vLLM (all 8 GPUs), NINST-way episode shard, merge+score.
# args: CFG TAG NINST PROMPT(structured|plain|quote) [EMBED_DEV(cuda:N|cpu|none)] [SRC] [AGENTIC] [AMODE] [DBG] [PINSHUF]
#   CFG  method config: a path, or a basename resolved against $REPO/ama/configs/
#   SRC  AMA-Bench dataset sub-dir under $AMA_BENCH/dataset/ (default: test)
# Paths (override via environment; scripts/run_ama.sh sets them):
#   AMA_BENCH  AMA-Bench checkout with ama/harness/* copied in   (default: $HOME/AMA-Bench)
#   MODEL      local Qwen3-32B weights, also the vLLM served-model-name (default: $HOME/models/Qwen3-32B)
#   VENV       python env that has vllm + the AMA-Bench requirements (default: $HOME/vllm_env)
#   EMBED_PATH local Qwen3-Embedding-0.6B (only when EMBED_DEV != none)
#   OUT        where shard outputs, mu_merged_<TAG>.json and mu_<TAG>_done land (default: $REPO/results)
#   SCRATCH    logs / per-port yaml (default: /tmp)
set -u
CFG=$1; TAG=$2; NINST=$3; PROMPT=$4; EMBED_DEV=${5:-none}; SRC=${6:-test}; AGENTIC=${7:-0}; AMODE=${8:-code}; DBG=${9:-1}; PINSHUF=${10:-0}
if [ "$PINSHUF" != "0" ]; then export SPRAG_PIN_SHUFFLE=$PINSHUF; else unset SPRAG_PIN_SHUFFLE; fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
AB=${AMA_BENCH:-$HOME/AMA-Bench}
MODEL=${MODEL:-$HOME/models/Qwen3-32B}
VENV=${VENV:-$HOME/vllm_env}
OUT=${OUT:-$REPO/results}
SCRATCH=${SCRATCH:-/tmp}
case "$CFG" in */*) CFGP=$CFG ;; *) CFGP=$REPO/ama/configs/$CFG ;; esac
mkdir -p "$OUT"
cd "$AB"
LOG=$SCRATCH/mu_$TAG.log
echo "MU_START $TAG $(date) ninst=$NINST prompt=$PROMPT embed=$EMBED_DEV cfg=$CFGP src=$SRC agentic=$AGENTIC" > $LOG
rm -f $OUT/mu_${TAG}_done
# --- optional host workaround (CUDAFIX=1): some of our nodes shipped a truncated libcuda stub on the
#     default path; symlink the largest (intact) libcuda/libnvidia-ml into $SCRATCH/cudafix ---
if [ "${CUDAFIX:-0}" = "1" ]; then
  mkdir -p $SCRATCH/cudafix
  for lib in libcuda libnvidia-ml; do
    LIBSRC=$(find /usr/lib /usr/lib64 /lib -type f -name "$lib.so*" 2>/dev/null | while read f; do echo "$(stat -c%s "$f") $f"; done | sort -n | tail -1 | awk '{print $2}')
    [ -n "$LIBSRC" ] && { ln -sf "$LIBSRC" $SCRATCH/cudafix/$lib.so.1; ln -sf "$LIBSRC" $SCRATCH/cudafix/$lib.so; }
  done
  export LD_LIBRARY_PATH=$SCRATCH/cudafix:${LD_LIBRARY_PATH:-}
  # link-time fix: ld reads the system libcuda directly during triton JIT builds
  export LIBRARY_PATH=$SCRATCH/cudafix:${LIBRARY_PATH:-}
  for TD in $VENV/lib/python3.*/site-packages/triton/backends/nvidia/lib; do
    [ -d "$TD" ] && ln -sf "$(readlink -f $SCRATCH/cudafix/libcuda.so 2>/dev/null || echo $SCRATCH/cudafix/libcuda.so)" "$TD/libcuda.so" 2>/dev/null
  done
fi
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
# cleanup of this recipe's own servers: vLLM V1 spawns worker subprocesses that survive pkill "vllm serve"
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "$VENV/bin/python -c from multiprocessing" 2>/dev/null
pkill -9 -f multiproc_executor 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null
# KILL_GPU_PROCS=1 additionally kills EVERY process holding a GPU (dedicated node only)
if [ "${KILL_GPU_PROCS:-0}" = "1" ]; then
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | xargs -r kill -9 2>/dev/null
fi
sleep 6
for _w in $(seq 1 15); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1); [ -z "$U" ] && break; [ "$U" -lt 2000 ] && break; sleep 5; done
# --- launch NINST vLLM (TP=2 each, GPUs [2g,2g+1], port 8060+g) ---
for g in $(seq 0 $((NINST-1))); do
  G0=$((2*g)); G1=$((2*g+1)); PORT=$((8060+g))
  CUDA_VISIBLE_DEVICES=$G0,$G1 nohup $VENV/bin/vllm serve $MODEL \
    --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --port $PORT --served-model-name $MODEL --enforce-eager --disable-log-requests \
    > $SCRATCH/vs_${TAG}_$g.log 2>&1 &
done
# --- wait for all instances up (<=20min) ---
ALLUP=1
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g)); UP=0
  for i in $(seq 1 200); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen3-32B && { UP=1; break; }; sleep 6; done
  echo "PORT $PORT up=$UP" >> $LOG; [ $UP -eq 0 ] && ALLUP=0
done
if [ $ALLUP -eq 0 ]; then echo "SRVFAIL $TAG $(date)" > $OUT/mu_${TAG}_done; exit 1; fi
# --- split episodes NINST ways (split on \n only) + per-port yaml ---
echo "SPLIT_START $(date)" >> $LOG
$VENV/bin/python - "$NINST" "$TAG" "$SRC" "$AB" >> $LOG 2>&1 <<'PY'
import os,sys
NG=int(sys.argv[1]); TAG=sys.argv[2]; SRC=sys.argv[3]; AB=sys.argv[4]
src="%s/dataset/%s/open_end_qa_set.jsonl" % (AB, SRC)
lines=[l for l in open(src,encoding='utf-8') if l.strip()]
for s in range(NG):
    d="%s/dataset/test_%s_%d" % (AB,TAG,s); os.makedirs(d,exist_ok=True)
    with open("%s/open_end_qa_set.jsonl" % d,"w",encoding='utf-8') as g:
        for i in range(len(lines)):
            if i%NG==s: g.write(lines[i] if lines[i].endswith(chr(10)) else lines[i]+chr(10))
print("SPLIT_OK",NG,len(lines))
PY
if [ ! -f "dataset/test_${TAG}_0/open_end_qa_set.jsonl" ]; then echo "SPLITFAIL $TAG $(date)" > $OUT/mu_${TAG}_done; pkill -9 -f "vllm serve"; exit 1; fi
# AMA-Bench's configs/qwen3-32B.yaml must name the same served model ($MODEL); only the port is rewritten here
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g)); sed "s/vllm_port: 8056/vllm_port: $PORT/" configs/qwen3-32B.yaml > $SCRATCH/qwen_$PORT.yaml
done
# --- env: prompt + embed ---
export PYTHONPATH=$REPO:$AB VERAKV_AMA_DIR=$REPO/ama MODEL
if [ "$PROMPT" = "structured" ]; then export AMA_ANSWER_INSTR_FILE=$REPO/ama/dec_instr.txt
elif [ "$PROMPT" = "quote" ]; then export AMA_ANSWER_INSTR_FILE=$REPO/ama/quote_instr.txt
else unset AMA_ANSWER_INSTR_FILE; fi
if [ "$EMBED_DEV" != "none" ]; then export SPRAG_EMBED_PATH=${EMBED_PATH:?set EMBED_PATH to a local Qwen3-Embedding-0.6B} SPRAG_EMBED_DEVICE=$EMBED_DEV; fi
if [ "$AGENTIC" = "1" ]; then export AMA_AGENTIC_READER=1 AMA_AGENTIC_MODE=$AMODE AMA_AGENTIC_DBG=$DBG AMA_AGENTIC_LOG=$OUT/sel_$TAG; rm -f $OUT/sel_${TAG}_dbg.log $OUT/sel_${TAG}_full.jsonl; else unset AMA_AGENTIC_READER; fi
# --- run NINST shards in parallel (wait only on shard PIDs, not the vLLM servers) ---
PIDS=()
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g))
  $VENV/bin/python src/run.py --llm-server vllm --llm-config $SCRATCH/qwen_$PORT.yaml \
    --subset openend --method kvmemory --method-config $CFGP \
    --test-dir dataset/test_${TAG}_$g --output-dir $OUT/mu_out_${TAG}_$g \
    --judge-config $SCRATCH/qwen_$PORT.yaml --judge-server vllm --evaluate True \
    --max-concurrency-episodes 8 --max-concurrency-questions-per-episode 4 \
    > $SCRATCH/run_${TAG}_$g.log 2>&1 &
  PIDS+=($!)
done
echo "SHARD_PIDS ${PIDS[*]} $(date)" >> $LOG
wait "${PIDS[@]}"
pkill -9 -f "vllm serve" 2>/dev/null
# --- merge + accuracy + CI ---
$VENV/bin/python - "$NINST" "$TAG" "$OUT" <<'PY'
import json,glob,sys,random
NG=int(sys.argv[1]); TAG=sys.argv[2]; OUT=sys.argv[3]; random.seed(0)
allr=[]; miss=[]
for s in range(NG):
    fs=glob.glob(f"{OUT}/mu_out_{TAG}_{s}/results_*.json")
    if not fs: miss.append(s); continue
    allr+=json.load(open(fs[0]))["results"]
n=len(allr); acc=sum(1 for r in allr if r["score"]==1.0)/n if n else 0
if n:
    bs=sorted(sum(1 for r in (allr[random.randrange(n)] for _ in range(n)) if r["score"]==1.0)/n for _ in range(2000))
    ci=f"[{bs[50]:.4f},{bs[1950]:.4f}]"
else: ci="[na]"
line=f"MU_{TAG} n={n} acc={acc:.4f} CI{ci} miss={miss}"
print(line)
json.dump(allr, open(f"{OUT}/mu_merged_{TAG}.json","w"))
open(f"{OUT}/mu_{TAG}_done","w").write(line+"\n")
PY
echo "MU_DONE $TAG $(date)" >> $LOG
