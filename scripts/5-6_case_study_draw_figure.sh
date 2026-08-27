#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_NAME="${DATA_NAME:-mmlu_redux}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
RESULT_ROOT="${RESULT_ROOT:-${OUTPUT_ROOT:-./final_results/case_study}}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-${RESULT_ROOT}/analysis/${DATA_NAME}/${MODEL_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DPI="${DPI:-300}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

read -r -a ratios <<< "${RATIOS:-0.1 0.2 0.3}"
read -r -a figure_formats <<< "${FIGURE_FORMATS:-png svg pdf}"

if (( ${#ratios[@]} == 0 )); then
  echo "RATIOS must contain at least one maintain ratio." >&2
  exit 2
fi
if (( ${#figure_formats[@]} == 0 )); then
  echo "FIGURE_FORMATS must contain at least one output format." >&2
  exit 2
fi

command=(
  "${PYTHON_BIN}"
  16_summarize_case_study_results.py
  --result_root "${RESULT_ROOT}"
  --data_name "${DATA_NAME}"
  --model_name "${MODEL_NAME}"
  --ratios "${ratios[@]}"
  --output_dir "${ANALYSIS_OUTPUT_DIR}"
  --figure_formats "${figure_formats[@]}"
  --dpi "${DPI}"
)
if [[ "${SKIP_FIGURES}" == "1" ]]; then
  command+=(--skip_figures)
fi
if (( $# > 0 )); then
  command+=("$@")
fi

"${command[@]}"
