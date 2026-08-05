# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python experiment workspace for CoT faithfulness analysis. Root-level numbered scripts are the main workflow stages:

- `1_prepare_dataset.py`: builds `datasets/` JSON files and `results/.../*_filter_right.json`.
- `get_att.py`: extracts word-level attention before intervention.
- `run_two_experiments.py` and `run_pipeline.py`: run the current sentence-level experiment pipeline.
- `2_get_attention_and_gradient.py` through `12_maintain_model_selected_sentences.py`: older or extended intervention, recovery, metric, and analysis experiments.
- `utils.py`: shared CLI parsing, model loading, attention utilities, prompt templates, and text helpers.

Generated artifacts live in `results/`, `final_results/`, `log/`, `analysis_pictures/`, `metric_pictures/`, `tsne_data/`, and `umap_data/`. Treat these as experiment outputs unless intentionally updating published results.

## Build, Test, and Development Commands

Use the `cot` conda environment when available:

```bash
conda run -n cot python 1_prepare_dataset.py --data_name mmlu_redux --sub_set global_facts --model_name Qwen3-0.6B
conda run -n cot python get_att.py --data_name mmlu_redux --sub_set global_facts --model_name Qwen3-0.6B
conda run -n cot python run_two_experiments.py --data_name mmlu_redux --sub_set global_facts --model_name Qwen3-0.6B
```

For the batch pipeline, ensure `*_filter_right.json` files already exist, then run:

```bash
conda run -n cot python run_pipeline.py
```

GPU execution is required because model loading in `utils.py` targets `cuda:0`.

## Coding Style & Naming Conventions

Use Python 3, four-space indentation, and snake_case for functions and variables. Keep new helpers in `utils.py` only when they are reused by multiple scripts; otherwise keep logic local to the experiment script. Preserve existing filename conventions for staged scripts and output files, especially `{model_name}_{subset}_rate1_...`.

## Testing Guidelines

There is no formal test suite. Before committing code changes, run a small subset such as `global_facts` or a shortened local sample, and check that expected JSON files are produced. For syntax-only validation, run:

```bash
python -m py_compile *.py
bash -n scripts/*.sh
```

## Commit & Pull Request Guidelines

Git history currently uses minimal messages like `first commit`; prefer clearer imperative summaries such as `Add Qwen3 preprocessing script` or `Fix sentence attention aggregation`. Pull requests should describe the experiment stage affected, list commands run, note model/subset choices, and mention any large generated outputs intentionally changed.

## Security & Configuration Tips

Do not add API keys or tokens to source files. Move credentials currently needed by API calls into environment variables before sharing changes. Verify local model paths under `/2024133105/Workspaces/llms` before launching long GPU jobs.


# Testing principles
1. ```conda activate cot``` to activate the environment
2. run the script
