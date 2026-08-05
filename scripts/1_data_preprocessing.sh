#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Batch data preprocessing for MMLU-Redux.
# It runs 1_prepare_dataset.py for each subset and creates:
#   results/${DATA_NAME}/${MODEL_NAME}_${subset}_filter_right.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_NAME="${DATA_NAME:-mmlu_redux}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

export HF_ENDPOINT

DEFAULT_SUBSETS=(
  global_facts
  high_school_mathematics
  college_mathematics
  high_school_computer_science
  college_computer_science
  high_school_biology
  college_biology
  professional_law
  college_physics
  machine_learning
  sociology
  us_foreign_policy
  computer_security
  conceptual_physics
  econometrics
  business_ethics
  clinical_knowledge
  electrical_engineering
  elementary_mathematics
  formal_logic
)

if [[ "$#" -gt 0 ]]; then
  SUBSETS=("$@")
else
  SUBSETS=("${DEFAULT_SUBSETS[@]}")
fi

mkdir -p log "results/${DATA_NAME}"

echo "============================================================"
echo "Data preprocessing"
echo "data_name : ${DATA_NAME}"
echo "model_name: ${MODEL_NAME}"
echo "subsets   : ${SUBSETS[*]}"
echo "root      : ${PROJECT_ROOT}"
echo "============================================================"

for subset in "${SUBSETS[@]}"; do
  output_file="results/${DATA_NAME}/${MODEL_NAME}_${subset}_filter_right.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${output_file}" ]]; then
    echo "[Skip] ${subset}: ${output_file} already exists"
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[Run] ${subset}"
  echo "------------------------------------------------------------"

  python3 1_prepare_dataset.py \
    --data_name "${DATA_NAME}" \
    --sub_set "${subset}" \
    --model_name "${MODEL_NAME}"

  echo "[Done] ${subset}"
done

echo "============================================================"
echo "All preprocessing jobs finished."
echo "============================================================"
