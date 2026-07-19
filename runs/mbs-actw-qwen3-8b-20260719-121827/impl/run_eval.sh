#!/usr/bin/env bash
# Entrypoint contract for paper-reprise's from-scratch runner:
#   bash impl/run_eval.sh <claim_id>   -> prints `acc_norm: <pct>` (and `acc:`)
#   bash impl/run_eval.sh --smoke      -> tiny self-test (few samples), proves it runs
# Reads: $PAPER_REPRISE_MODEL (base model path), ${PAPER_REPRISE_TASKS:-hellaswag},
#        ${PAPER_REPRISE_GPUS:-1} (GPU count; which GPUs via CUDA_VISIBLE_DEVICES).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--smoke" ]; then
  CLAIM="qwen3-8b-mxfp4-mbs-h-hellaswag"   # exercise the heaviest path (dynamic MBS)
  SMOKE="--smoke"
else
  CLAIM="${1:?usage: run_eval.sh <claim_id> | --smoke}"
  SMOKE=""
fi

export PAPER_REPRISE_TASKS="${PAPER_REPRISE_TASKS:-hellaswag}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# Default to a single GPU (8B fits one H200); operator can override CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$HERE"
python eval_one.py "$CLAIM" $SMOKE
