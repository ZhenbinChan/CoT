"""Run the direct-answer baseline without any chain-of-thought context."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from case_study_utils import (
    atomic_write_csv,
    atomic_write_json,
    build_direct_generation_args,
    build_no_cot_prompt,
    build_prediction_summary,
    get_option_token_ids,
    get_sample_choices,
    load_json,
    score_and_generate_direct_answer,
    write_prediction_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-CoT direct-answer baseline for case analysis."
    )
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--sub_set")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path")
    parser.add_argument(
        "--output_root", default="./final_results/case_study"
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def make_record(
    sample: dict[str, Any],
    sample_index: int,
    model: Any,
    tokenizer: Any,
    generation_args: dict[str, Any],
    option_token_ids: dict[str, int],
) -> dict[str, Any]:
    truth = str(sample["truth"]).strip().upper()
    base_input_text = sample["input_text"]
    prompt = build_no_cot_prompt(base_input_text)
    prediction_fields = score_and_generate_direct_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        truth=truth,
        generation_args=generation_args,
        option_token_ids=option_token_ids,
    )
    return {
        "sample_index": sample_index,
        "experiment_condition": "no_cot",
        "question": sample.get("question"),
        "choices": get_sample_choices(sample),
        "source_choices_field": sample["choices"],
        "truth": truth,
        "base_input_text": base_input_text,
        "input_text_with_CoT": prompt,
        "original_CoT": sample.get("CoT", ""),
        "original_explanation": sample.get("output_text", ""),
        "original_prediction": sample.get("prediction"),
        "legacy_nothink_prediction": sample.get("nothink_prediction"),
        **prediction_fields,
    }


def run_experiment(args: argparse.Namespace) -> None:
    if not args.sub_set:
        raise ValueError("--sub_set is required unless --aggregate_only is used")
    if not args.model_path:
        raise ValueError("--model_path is required for inference")

    input_path = Path("results") / args.data_name / (
        f"{args.model_name}_{args.sub_set}_filter_right.json"
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Filter file not found: {input_path}")
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a JSON list in {input_path}")
    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max_samples must be non-negative")
        samples = samples[: args.max_samples]

    logging.info("Loading model from %s", args.model_path)
    from tqdm import tqdm
    from utils import load_tokenizer_and_model

    tokenizer, model = load_tokenizer_and_model(model_path=args.model_path)
    model.eval()
    generation_args = build_direct_generation_args(tokenizer)
    option_token_ids = get_option_token_ids(tokenizer)

    records = []
    for sample_index, sample in enumerate(
        tqdm(samples, desc=f"No-CoT {args.sub_set}")
    ):
        records.append(
            make_record(
                sample=sample,
                sample_index=sample_index,
                model=model,
                tokenizer=tokenizer,
                generation_args=generation_args,
                option_token_ids=option_token_ids,
            )
        )

    output_directory = (
        Path(args.output_root)
        / "no_cot"
        / args.data_name
        / args.model_name
        / args.sub_set
    )
    summary = write_prediction_bundle(
        output_directory=output_directory,
        records=records,
        metadata={
            "experiment": "no_cot_direct_answer",
            "data_name": args.data_name,
            "model_name": args.model_name,
            "subset": args.sub_set,
            "source_file": str(input_path),
            "max_samples": args.max_samples,
            "generation": {
                "prompt_suffix": r" \boxed{",
                "do_sample": False,
                "max_new_tokens": 8,
                "repetition_penalty": 1.2,
            },
        },
    )
    logging.info(
        "Saved %d records to %s (accuracy %.4f)",
        len(records),
        output_directory,
        summary["accuracy"],
    )


def aggregate_results(args: argparse.Namespace) -> None:
    model_directory = (
        Path(args.output_root)
        / "no_cot"
        / args.data_name
        / args.model_name
    )
    record_paths = sorted(model_directory.glob("*/records.json"))
    if not record_paths:
        raise FileNotFoundError(
            f"No subset records found under {model_directory}"
        )

    all_records: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    for record_path in record_paths:
        subset = record_path.parent.name
        records = load_json(record_path)
        all_records.extend(records)
        summary = build_prediction_summary(records)
        subset_rows.append(
            {
                "subset": subset,
                "total": summary["total"],
                "correct_count": summary["correct_count"],
                "accuracy": summary["accuracy"],
                "option_argmax_accuracy": summary["option_argmax_accuracy"],
                "average_truth_probability_raw": summary[
                    "average_truth_probability_raw"
                ],
                "average_truth_probability_normalized": summary[
                    "average_truth_probability_normalized"
                ],
            }
        )

    aggregate = build_prediction_summary(
        all_records,
        metadata={
            "experiment": "no_cot_direct_answer",
            "data_name": args.data_name,
            "model_name": args.model_name,
            "subset_count": len(record_paths),
            "aggregation": "micro",
            "subsets": subset_rows,
        },
    )
    total_row = {
        "subset": "__micro_total__",
        "total": aggregate["total"],
        "correct_count": aggregate["correct_count"],
        "accuracy": aggregate["accuracy"],
        "option_argmax_accuracy": aggregate["option_argmax_accuracy"],
        "average_truth_probability_raw": aggregate[
            "average_truth_probability_raw"
        ],
        "average_truth_probability_normalized": aggregate[
            "average_truth_probability_normalized"
        ],
    }
    atomic_write_json(model_directory / "aggregate_summary.json", aggregate)
    atomic_write_csv(
        model_directory / "aggregate_summary.csv", subset_rows + [total_row]
    )
    logging.info(
        "Saved aggregate summary for %d subsets to %s",
        len(record_paths),
        model_directory,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.aggregate_only:
        aggregate_results(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
