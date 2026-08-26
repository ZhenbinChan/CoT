#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${INPUT_FILE:?Set INPUT_FILE to the manual-cases JSON file}"
: "${RUN_NAME:?Set RUN_NAME for this manual intervention run}"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/share/nlp/chenzhenbin/Workspaces/LLMs/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./final_results/case_study}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

extra_args=()
if [[ -n "${DATA_NAME:-}" ]]; then
  extra_args+=(--data_name "${DATA_NAME}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  extra_args+=(--max_samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" 15_manual_cot_intervention.py \
  --input_file "${INPUT_FILE}" \
  --run_name "${RUN_NAME}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --output_root "${OUTPUT_ROOT}" \
  "${extra_args[@]}"
