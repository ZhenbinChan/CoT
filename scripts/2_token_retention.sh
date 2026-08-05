#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Batch token-retention experiments for MMLU-Redux.
# For each subset, it extracts attention when needed and retains the
# top-attention 10%, 20%, and 30% of CoT tokens/whitespace-delimited words.
# Prerequisite:
#   results/${DATA_NAME}/${MODEL_NAME}_${subset}_filter_right.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_NAME="${DATA_NAME:-mmlu_redux}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/2024133105/Workspaces/llms/Qwen3-8B}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

TOP_K_PERCENTAGES=(0.1 0.2 0.3)

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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[Error] Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p log "results/${DATA_NAME}/att_grad_before"

attention_finished() {
  local subset="$1"
  [[ -f "results/${DATA_NAME}/att_grad_before/${MODEL_NAME}_${subset}_rate1_all_att_list.json" ]] \
    && [[ -f "results/${DATA_NAME}/att_grad_before/${MODEL_NAME}_${subset}_rate1_words_list.json" ]]
}

experiment_finished() {
  local subset="$1"
  local percentage

  for percentage in "${TOP_K_PERCENTAGES[@]}"; do
    if [[ ! -f "final_results/maintain_only_topk/filter/unmask/${MODEL_NAME}-${DATA_NAME}-${subset}-att-right-top_percentage${percentage}.json" ]]; then
      return 1
    fi
  done

  return 0
}

echo "============================================================"
echo "Token-retention experiments"
echo "data_name : ${DATA_NAME}"
echo "model_name: ${MODEL_NAME}"
echo "model_path: ${MODEL_PATH}"
echo "retention : ${TOP_K_PERCENTAGES[*]}"
echo "subsets   : ${SUBSETS[*]}"
echo "root      : ${PROJECT_ROOT}"
echo "============================================================"

for subset in "${SUBSETS[@]}"; do
  filter_file="results/${DATA_NAME}/${MODEL_NAME}_${subset}_filter_right.json"

  if [[ ! -f "${filter_file}" ]]; then
    echo "[Error] Missing prerequisite for ${subset}: ${filter_file}" >&2
    exit 1
  fi

  echo "------------------------------------------------------------"
  echo "[Subset] ${subset}"
  echo "------------------------------------------------------------"

  if [[ "${SKIP_EXISTING}" == "1" ]] && attention_finished "${subset}"; then
    echo "[Skip] Attention already exists: ${subset}"
  else
    echo "[Run] Extract attention: ${subset}"
    python3 get_att.py \
      --data_name "${DATA_NAME}" \
      --sub_set "${subset}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}"
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]] && experiment_finished "${subset}"; then
    echo "[Skip] Token-retention results already exist: ${subset}"
    continue
  fi

  echo "[Run] Retain top-attention tokens: ${subset}"
  python3 4_maintain_only_topk_words.py \
    --data_name "${DATA_NAME}" \
    --sub_set "${subset}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}"

  echo "[Done] ${subset}"
done

echo "============================================================"
echo "All token-retention experiments finished."
echo "============================================================"
