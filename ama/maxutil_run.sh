#!/bin/bash
# Max-util AMA run: NINST x TP=2 vLLM (all 8 GPUs), NINST-way episode shard, merge+score.
# args: CFG_BASENAME TAG NINST PROMPT(structured|plain) [EMBED_DEV(cuda:N|cpu|none)]
set -u
CFG=$1; TAG=$2; NINST=$3; PROMPT=$4; EMBED_DEV=${5:-none}; SRC=${6:-test}; AGENTIC=${7:-0}; AMODE=${8:-code}; DBG=${9:-1}; PINSHUF=${10:-0}
if [ "$PINSHUF" != "0" ]; then export SPRAG_PIN_SHUFFLE=$PINSHUF; else unset SPRAG_PIN_SHUFFLE; fi
BASE=/home/tiger; AB=/home/tiger/AMA-Bench
cd $AB
LOG=/tmp/mu_$TAG.log
echo "MU_START $TAG $(date) ninst=$NINST prompt=$PROMPT embed=$EMBED_DEV cfg=$CFG src=$SRC agentic=$AGENTIC" > $LOG
rm -f $BASE/mu_${TAG}_done
# --- cudafix: symlink intact (largest) libcuda + libnvidia-ml ---
mkdir -p /tmp/cudafix
for lib in libcuda libnvidia-ml; do
  LIBSRC=$(find /usr/lib /usr/lib64 /lib -type f -name "$lib.so*" 2>/dev/null | while read f; do echo "$(stat -c%s "$f") $f"; done | sort -n | tail -1 | awk '{print $2}')
  [ -n "$LIBSRC" ] && { ln -sf "$LIBSRC" /tmp/cudafix/$lib.so.1; ln -sf "$LIBSRC" /tmp/cudafix/$lib.so; }
done
CU=$(find /usr/lib -type f -name "libcuda.so*" 2>/dev/null | while read f; do echo "$(stat -c%s "$f") $f"; done | sort -n | tail -1 | awk '{print $2}')
ln -sf "$CU" /tmp/vllm_env/lib/python3.10/site-packages/triton/backends/nvidia/lib/libcuda.so 2>/dev/null
export LD_LIBRARY_PATH=/tmp/cudafix:${LD_LIBRARY_PATH:-}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
# aggressive cleanup: vLLM V1 spawns VllmWorker subprocs that survive pkill "vllm serve"
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "/tmp/vllm_env" 2>/dev/null; pkill -9 -f multiproc_executor 2>/dev/null
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | xargs -r kill -9 2>/dev/null
sleep 6
# --- launch NINST vLLM (TP=2 each, GPUs [2g,2g+1], port 8060+g) ---
for g in $(seq 0 $((NINST-1))); do
  G0=$((2*g)); G1=$((2*g+1)); PORT=$((8060+g))
  CUDA_VISIBLE_DEVICES=$G0,$G1 nohup /tmp/vllm_env/bin/vllm serve /tmp/Qwen3-32B \
    --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --port $PORT --served-model-name /tmp/Qwen3-32B --enforce-eager --disable-log-requests \
    > /tmp/vs_${TAG}_$g.log 2>&1 &
done
# --- wait for all instances up (<=20min) ---
ALLUP=1
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g)); UP=0
  for i in $(seq 1 200); do curl -s http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen3-32B && { UP=1; break; }; sleep 6; done
  echo "PORT $PORT up=$UP" >> $LOG; [ $UP -eq 0 ] && ALLUP=0
done
if [ $ALLUP -eq 0 ]; then echo "SRVFAIL $TAG $(date)" > $BASE/mu_${TAG}_done; exit 1; fi
# --- split episodes NINST ways (split on \n only) + per-port yaml ---
echo "SPLIT_START $(date)" >> $LOG
/tmp/vllm_env/bin/python - "$NINST" "$TAG" "$SRC" >> $LOG 2>&1 <<'PY'
import os,sys
NG=int(sys.argv[1]); TAG=sys.argv[2]; SRC=sys.argv[3]
src="/home/tiger/AMA-Bench/dataset/%s/open_end_qa_set.jsonl" % SRC
lines=[l for l in open(src,encoding='utf-8') if l.strip()]
for s in range(NG):
    d="/home/tiger/AMA-Bench/dataset/test_%s_%d" % (TAG,s); os.makedirs(d,exist_ok=True)
    with open("%s/open_end_qa_set.jsonl" % d,"w",encoding='utf-8') as g:
        for i in range(len(lines)):
            if i%NG==s: g.write(lines[i] if lines[i].endswith(chr(10)) else lines[i]+chr(10))
print("SPLIT_OK",NG,len(lines))
PY
if [ ! -f "dataset/test_${TAG}_0/open_end_qa_set.jsonl" ]; then echo "SPLITFAIL $TAG $(date)" > $BASE/mu_${TAG}_done; pkill -9 -f "vllm serve"; exit 1; fi
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g)); sed "s/vllm_port: 8056/vllm_port: $PORT/" configs/qwen3-32B.yaml > /tmp/qwen_$PORT.yaml
done
# --- env: prompt + embed ---
export PYTHONPATH=/home/tiger:/home/tiger/AMA-Bench
if [ "$PROMPT" = "structured" ]; then export AMA_ANSWER_INSTR_FILE=/home/tiger/dec_instr.txt
elif [ "$PROMPT" = "quote" ]; then export AMA_ANSWER_INSTR_FILE=/home/tiger/quote_instr.txt
else unset AMA_ANSWER_INSTR_FILE; fi
if [ "$EMBED_DEV" != "none" ]; then export SPRAG_EMBED_PATH=/tmp/Qwen3-Embedding-0.6B SPRAG_EMBED_DEVICE=$EMBED_DEV; fi
if [ "$AGENTIC" = "1" ]; then export AMA_AGENTIC_READER=1 AMA_AGENTIC_MODE=$AMODE AMA_AGENTIC_DBG=$DBG AMA_AGENTIC_LOG=$BASE/sel_$TAG; rm -f $BASE/sel_${TAG}_dbg.log $BASE/sel_${TAG}_full.jsonl; else unset AMA_AGENTIC_READER; fi
# --- run NINST shards in parallel (wait only on shard PIDs, not the vLLM servers) ---
PIDS=()
for g in $(seq 0 $((NINST-1))); do
  PORT=$((8060+g))
  /tmp/vllm_env/bin/python src/run.py --llm-server vllm --llm-config /tmp/qwen_$PORT.yaml \
    --subset openend --method kvmemory --method-config $BASE/$CFG \
    --test-dir dataset/test_${TAG}_$g --output-dir $BASE/mu_out_${TAG}_$g \
    --judge-config /tmp/qwen_$PORT.yaml --judge-server vllm --evaluate True \
    --max-concurrency-episodes 8 --max-concurrency-questions-per-episode 4 \
    > /tmp/run_${TAG}_$g.log 2>&1 &
  PIDS+=($!)
done
echo "SHARD_PIDS ${PIDS[*]} $(date)" >> $LOG
wait "${PIDS[@]}"
pkill -9 -f "vllm serve" 2>/dev/null
# --- merge + accuracy + CI ---
/tmp/vllm_env/bin/python - "$NINST" "$TAG" <<'PY'
import json,glob,sys,random
NG=int(sys.argv[1]); TAG=sys.argv[2]; random.seed(0)
allr=[]; miss=[]
for s in range(NG):
    fs=glob.glob(f"/home/tiger/mu_out_{TAG}_{s}/results_*.json")
    if not fs: miss.append(s); continue
    allr+=json.load(open(fs[0]))["results"]
n=len(allr); acc=sum(1 for r in allr if r["score"]==1.0)/n if n else 0
if n:
    bs=sorted(sum(1 for r in (allr[random.randrange(n)] for _ in range(n)) if r["score"]==1.0)/n for _ in range(2000))
    ci=f"[{bs[50]:.4f},{bs[1950]:.4f}]"
else: ci="[na]"
line=f"MU_{TAG} n={n} acc={acc:.4f} CI{ci} miss={miss}"
print(line)
json.dump(allr, open(f"/home/tiger/mu_merged_{TAG}.json","w"))
open(f"/home/tiger/mu_{TAG}_done","w").write(line+"\n")
PY
echo "MU_DONE $TAG $(date)" >> $LOG
