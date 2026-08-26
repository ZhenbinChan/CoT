#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 CONDITION [SUBSET ...]" >&2
  exit 2
fi

CONDITION="$1"
shift
case "${CONDITION}" in
  letter_kept|letter_removed|correct_option_words_removed|permuted_letter_kept|permuted_letter_removed) ;;
  *)
    echo "Unsupported Top-k condition: ${CONDITION}" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_NAME="${DATA_NAME:-mmlu_redux}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
MODEL_PATH="${MODEL_PATH:-/share/nlp/chenzhenbin/Workspaces/LLMs/${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./final_results/case_study}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RANDOM_SEED="${RANDOM_SEED:-51}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

read -r -a ratios <<< "${RATIOS:-0.1 0.2 0.3}"
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
  attention_file="results/${DATA_NAME}/att_grad_before/${MODEL_NAME}_${subset}_rate1_all_att_list.json"
  words_file="results/${DATA_NAME}/att_grad_before/${MODEL_NAME}_${subset}_rate1_words_list.json"
  if [[ ! -f "${filter_file}" ]]; then
    echo "Missing filter file: ${filter_file}" >&2
    exit 1
  fi
  if [[ ! -f "${attention_file}" || ! -f "${words_file}" ]]; then
    echo "Attention inputs missing for ${subset}; running get_att.py."
    "${PYTHON_BIN}" get_att.py \
      --data_name "${DATA_NAME}" \
      --sub_set "${subset}" \
      --model_name "${MODEL_NAME}" \
      --model_path "${MODEL_PATH}"
  fi

  ratios_to_run=()
  for ratio in "${ratios[@]}"; do
    result_file="${OUTPUT_ROOT}/topk_leakage/${DATA_NAME}/${MODEL_NAME}/${subset}/${CONDITION}/top_percentage${ratio}.json"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${result_file}" ]]; then
      echo "Skipping existing ${CONDITION} ${subset} ratio ${ratio}."
    else
      ratios_to_run+=("${ratio}")
    fi
  done
  if (( ${#ratios_to_run[@]} == 0 )); then
    continue
  fi

  "${PYTHON_BIN}" 14_topk_answer_leakage.py \
    --condition "${CONDITION}" \
    --ratios "${ratios_to_run[@]}" \
    --data_name "${DATA_NAME}" \
    --sub_set "${subset}" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --output_root "${OUTPUT_ROOT}" \
    --random_seed "${RANDOM_SEED}" \
    "${extra_args[@]}"
done

"${PYTHON_BIN}" 14_topk_answer_leakage.py \
  --aggregate_only \
  --condition "${CONDITION}" \
  --ratios "${ratios[@]}" \
  --data_name "${DATA_NAME}" \
  --model_name "${MODEL_NAME}" \
  --output_root "${OUTPUT_ROOT}"
