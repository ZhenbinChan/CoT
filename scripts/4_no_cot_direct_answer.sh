#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_NAME="${DATA_NAME:-mmlu_redux}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/share/nlp/chenzhenbin/Workspaces/LLMs/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./final_results/case_study}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

subsets=()
if (( $# > 0 )); then
  subsets=("$@")
elif [[ -n "${SUBSETS:-}" ]]; then
  read -r -a subsets <<< "${SUBSETS}"
else
  shopt -s nullglob
  filter_files=("results/${DATA_NAME}/${MODEL_NAME}_"*_filter_right.json)
  shopt -u nullglob
  for filter_file in "${filter_files[@]}"; do
    filename="${filter_file##*/}"
    subset="${filename#${MODEL_NAME}_}"
    subset="${subset%_filter_right.json}"
    subsets+=("${subset}")
  done
fi

if (( ${#subsets[@]} == 0 )); then
  echo "No filter files found for ${DATA_NAME}/${MODEL_NAME}." >&2
  exit 1
fi

extra_args=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  extra_args+=(--max_samples "${MAX_SAMPLES}")
fi

for subset in "${subsets[@]}"; do
  filter_file="results/${DATA_NAME}/${MODEL_NAME}_${subset}_filter_right.json"
  output_file="${OUTPUT_ROOT}/no_cot/${DATA_NAME}/${MODEL_NAME}/${subset}/records.json"
  if [[ ! -f "${filter_file}" ]]; then
    echo "Missing filter file: ${filter_file}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_file}" ]]; then
    echo "Skipping existing No-CoT result: ${output_file}"
    continue
  fi
  "${PYTHON_BIN}" 13_no_cot_direct_answer.py \
    --data_name "${DATA_NAME}" \
    --sub_set "${subset}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    "${extra_args[@]}"
done

"${PYTHON_BIN}" 13_no_cot_direct_answer.py \
  --aggregate_only \
  --data_name "${DATA_NAME}" \
  --model_name "${MODEL_NAME}" \
  --output_root "${OUTPUT_ROOT}"
