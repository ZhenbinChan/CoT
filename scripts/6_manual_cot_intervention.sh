#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Manual-experiment configuration.  Edit these three values when switching
# from the smoke test to the full experiment.
INPUT_FILE="/Users/chenzhenbin/Downloads/CoT/letter_removed_manal_cases_smoke.json"
RUN_NAME="letter_removed_manual_smoke"
MAX_SAMPLES="1"

MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/share/nlp/chenzhenbin/Workspaces/LLMs/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./final_results/case_study}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -- 15_manual_cot_intervention.py \
  --input_file "${INPUT_FILE}" \
  --run_name "${RUN_NAME}" \
  --model_name "${MODEL_NAME}" \
  --model_path "${MODEL_PATH}" \
  --output_root "${OUTPUT_ROOT}"

if [ -n "${DATA_NAME:-}" ]; then
  set -- "$@" --data_name "${DATA_NAME}"
fi
if [ -n "${MAX_SAMPLES}" ]; then
  set -- "$@" --max_samples "${MAX_SAMPLES}"
fi

exec "${PYTHON_BIN}" "$@"
