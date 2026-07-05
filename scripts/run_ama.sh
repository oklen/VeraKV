#!/bin/bash
# End-to-end AMA-Bench reproduction on one 8xA100-80GB node.
#   usage: bash scripts/run_ama.sh <config.json> <TAG> <structured|plain|quote> [split] [agentic] [amode]
# Launches 4x vLLM (TP=2, all 8 GPUs), shards the episode set 4-way through the OFFICIAL
# src/run.py (generation + LLM-as-judge), merges, and writes results/mu_merged_<TAG>.json.
#
# Prerequisites (adjust the paths below to your layout):
#   - AMA-Bench checkout with the two files from ama/harness/ copied in
#   - kvmemory/ package on PYTHONPATH
#   - Qwen3-32B weights at $MODEL
#   - vllm binary in $VENV
set -eu
CFG=${1:?config json (see ama/configs/)}
TAG=${2:?run tag}
PROMPT=${3:-structured}
SRC=${4:-test}
AGENTIC=${5:-0}
AMODE=${6:-code}

REPO=$(cd "$(dirname "$0")/.." && pwd)
export AMA_BENCH=${AMA_BENCH:-$HOME/AMA-Bench}
export MODEL=${MODEL:-/tmp/Qwen3-32B}
export VENV=${VENV:-/tmp/vllm_env}
export PYTHONPATH=$REPO:$AMA_BENCH:${PYTHONPATH:-}

# The paper's reader-side variable, disclosed and ablated (factorial):
case "$PROMPT" in
  structured) export AMA_ANSWER_INSTR_FILE=$REPO/ama/dec_instr.txt ;;
  quote)      export AMA_ANSWER_INSTR_FILE=$REPO/ama/quote_instr.txt ;;
  *)          unset AMA_ANSWER_INSTR_FILE ;;
esac
if [ "$AGENTIC" = "1" ]; then
  export AMA_AGENTIC_READER=1 AMA_AGENTIC_MODE=$AMODE
fi

# Delegate to the battle-tested runner (documented inline; identical to the paper's runs).
exec bash "$REPO/ama/maxutil_run.sh" "$CFG" "$TAG" 4 "$PROMPT" none "$SRC" "$AGENTIC" "$AMODE" 1
