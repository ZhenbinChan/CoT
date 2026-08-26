"""Evaluate manually edited CoT variants against a freshly scored baseline."""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from case_study_utils import (
    OPTION_LABELS,
    atomic_write_json,
    build_direct_generation_args,
    build_reasoning_direct_prompt,
    get_option_token_ids,
    load_json,
    score_and_generate_direct_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare manually edited CoT variants with their baselines."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_name")
    parser.add_argument(
        "--output_root", default="./final_results/case_study"
    )
    parser.add_argument("--max_samples", type=int)
    return parser.parse_args()


def get_base_input_text(case: dict[str, Any]) -> str:
    explicit = case.get("base_input_text")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("base_input_text must be a non-empty string")
        return explicit

    legacy = case.get("input_text_with_CoT")
    if not isinstance(legacy, str):
        raise ValueError(
            "Case lacks base_input_text and has no legacy "
            "input_text_with_CoT fallback"
        )
    think_positions = [match.start() for match in re.finditer(r"<think>", legacy)]
    if len(think_positions) != 1:
        raise ValueError(
            "Cannot uniquely extract base_input_text: expected exactly one "
            f"<think> in input_text_with_CoT, found {len(think_positions)}"
        )
    return legacy[: think_positions[0]]


def validate_cases(
    cases: Any, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(cases, list):
        raise TypeError("Manual intervention input must be a top-level JSON list")
    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max_samples must be non-negative")
        cases = cases[: args.max_samples]
    if not cases:
        raise ValueError("Manual intervention input contains no selected cases")

    data_names = {case.get("data_name") for case in cases}
    if args.data_name:
        data_name = args.data_name
        unexpected = data_names - {None, data_name}
        if unexpected:
            raise ValueError(
                f"Input data_name values do not match --data_name: {unexpected}"
            )
    else:
        if None in data_names or len(data_names) != 1:
            raise ValueError(
                "Without --data_name, every case must contain the same "
                "non-empty data_name"
            )
        data_name = str(next(iter(data_names)))

    seen_case_ids: set[str] = set()
    for case_index, case in enumerate(cases):
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"Case {case_index} has no case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        if case.get("model_name") not in (None, args.model_name):
            raise ValueError(
                f"Case {case_id} model_name {case.get('model_name')!r} "
                f"does not match {args.model_name!r}"
            )
        truth = str(case.get("truth", "")).strip().upper()
        if truth not in OPTION_LABELS:
            raise ValueError(f"Case {case_id} has invalid truth {truth!r}")
        if not isinstance(case.get("baseline"), dict):
            raise ValueError(f"Case {case_id} has no baseline object")
        variants = case.get("variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Case {case_id} must have at least one variant")
        seen_variant_ids: set[str] = set()
        for variant in variants:
            variant_id = str(variant.get("variant_id", "")).strip()
            if not variant_id:
                raise ValueError(f"Case {case_id} has a variant without variant_id")
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", variant_id)
                or variant_id in {".", ".."}
            ):
                raise ValueError(
                    f"Case {case_id} variant_id {variant_id!r} is not a safe "
                    "output-directory name"
                )
            if variant_id in seen_variant_ids:
                raise ValueError(
                    f"Case {case_id} has duplicate variant_id {variant_id!r}"
                )
            seen_variant_ids.add(variant_id)
            if "cot" not in variant:
                raise ValueError(
                    f"Case {case_id} variant {variant_id} lacks cot"
                )
        get_base_input_text(case)
    return cases, data_name


def score_reasoning(
    base_input_text: str,
    truth: str,
    reasoning: dict[str, Any],
    model: Any,
    tokenizer: Any,
    generation_args: dict[str, Any],
    option_token_ids: dict[str, int],
) -> dict[str, Any]:
    prompt, rendered_cot, rendered_explanation = build_reasoning_direct_prompt(
        base_input_text=base_input_text,
        cot=reasoning.get("cot", ""),
        explanation=reasoning.get("explanation", ""),
    )
    prediction_fields = score_and_generate_direct_answer(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        truth=truth,
        generation_args=generation_args,
        option_token_ids=option_token_ids,
    )
    return {
        "cot": reasoning.get("cot", ""),
        "explanation": reasoning.get("explanation", ""),
        "rendered_CoT": rendered_cot,
        "rendered_explanation": rendered_explanation,
        "input_text_with_CoT": prompt,
        **prediction_fields,
    }


def build_variant_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    before_correct = sum(record["baseline_correct"] for record in records)
    after_correct = sum(record["correct"] for record in records)
    mean_normalized_delta = (
        sum(
            record["truth_probability_normalized_delta_vs_baseline"]
            for record in records
        )
        / total
        if total
        else None
    )
    mean_raw_delta = (
        sum(
            record["truth_probability_raw_delta_vs_baseline"]
            for record in records
        )
        / total
        if total
        else None
    )
    return {
        "total": total,
        "before_correct_count": before_correct,
        "after_correct_count": after_correct,
        "before_accuracy": before_correct / total if total else 0.0,
        "after_accuracy": after_correct / total if total else 0.0,
        "accuracy_delta": (after_correct - before_correct) / total if total else 0.0,
        "average_truth_probability_raw_delta": mean_raw_delta,
        "average_truth_probability_normalized_delta": mean_normalized_delta,
        "wrong_to_correct": sum(
            not record["baseline_correct"] and record["correct"]
            for record in records
        ),
        "correct_to_wrong": sum(
            record["baseline_correct"] and not record["correct"]
            for record in records
        ),
        "prediction_unchanged": sum(
            not record["prediction_changed_vs_baseline"] for record in records
        ),
        "prediction_changed": sum(
            record["prediction_changed_vs_baseline"] for record in records
        ),
    }


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_file)
    cases, data_name = validate_cases(load_json(input_path), args)
    output_directory = (
        Path(args.output_root)
        / "manual"
        / data_name
        / args.model_name
        / args.run_name
    )
    output_filenames = {
        "records.json",
        "baseline_records.json",
        "baseline_good_cases.json",
        "baseline_bad_cases.json",
        "variant_records.json",
        "variant_good_cases.json",
        "variant_bad_cases.json",
        "improved_cases.json",
        "degraded_cases.json",
        "summary.json",
    }
    input_resolved = input_path.resolve()
    direct_output_targets = {
        (output_directory / filename).resolve()
        for filename in output_filenames
    }
    variant_ids = {
        str(variant["variant_id"])
        for case in cases
        for variant in case["variants"]
    }
    variant_output_targets = {
        (output_directory / "by_variant" / variant_id / filename).resolve()
        for variant_id in variant_ids
        for filename in {
            "records.json",
            "good_cases.json",
            "bad_cases.json",
            "improved_cases.json",
            "degraded_cases.json",
            "summary.json",
        }
    }
    if input_resolved in direct_output_targets | variant_output_targets:
        raise ValueError("An output file would overwrite the manual input file")

    logging.info("Loading model from %s", args.model_path)
    from tqdm import tqdm
    from utils import load_tokenizer_and_model

    tokenizer, model = load_tokenizer_and_model(model_path=args.model_path)
    model.eval()
    generation_args = build_direct_generation_args(tokenizer)
    option_token_ids = get_option_token_ids(tokenizer)

    case_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    variant_records: list[dict[str, Any]] = []
    variants_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case_index, case in enumerate(tqdm(cases, desc="Manual CoT cases")):
        case_id = str(case["case_id"])
        truth = str(case["truth"]).strip().upper()
        base_input_text = get_base_input_text(case)
        common = {
            "case_id": case_id,
            "case_index": case_index,
            "sample_index": case.get("sample_index"),
            "data_name": data_name,
            "model_name": args.model_name,
            "subset": case.get("subset"),
            "truth": truth,
            "base_input_text": base_input_text,
        }
        baseline_scored = score_reasoning(
            base_input_text=base_input_text,
            truth=truth,
            reasoning=case["baseline"],
            model=model,
            tokenizer=tokenizer,
            generation_args=generation_args,
            option_token_ids=option_token_ids,
        )
        baseline_record = {
            **common,
            "record_type": "baseline",
            **baseline_scored,
        }
        baseline_records.append(baseline_record)

        case_variant_records = []
        for variant in case["variants"]:
            variant_id = str(variant["variant_id"])
            variant_scored = score_reasoning(
                base_input_text=base_input_text,
                truth=truth,
                reasoning=variant,
                model=model,
                tokenizer=tokenizer,
                generation_args=generation_args,
                option_token_ids=option_token_ids,
            )
            raw_deltas = {
                label: (
                    variant_scored["option_probabilities_raw"][label]
                    - baseline_scored["option_probabilities_raw"][label]
                )
                for label in OPTION_LABELS
            }
            normalized_deltas = {
                label: (
                    variant_scored["option_probabilities_normalized"][label]
                    - baseline_scored["option_probabilities_normalized"][label]
                )
                for label in OPTION_LABELS
            }
            variant_record = {
                **common,
                "record_type": "variant",
                "variant_id": variant_id,
                "edit_note": variant.get("edit_note", ""),
                "baseline_prediction": baseline_scored["prediction"],
                "baseline_correct": baseline_scored["correct"],
                "baseline_option_probabilities_raw": baseline_scored[
                    "option_probabilities_raw"
                ],
                "baseline_option_probabilities_normalized": baseline_scored[
                    "option_probabilities_normalized"
                ],
                "baseline_truth_probability_raw": baseline_scored[
                    "truth_probability_raw"
                ],
                "baseline_truth_probability_normalized": baseline_scored[
                    "truth_probability_normalized"
                ],
                "baseline_option_top1_top2_margin": baseline_scored[
                    "option_top1_top2_margin"
                ],
                **variant_scored,
                "prediction_changed_vs_baseline": (
                    variant_scored["prediction"] != baseline_scored["prediction"]
                ),
                "correct_delta_vs_baseline": (
                    variant_scored["correct"] - baseline_scored["correct"]
                ),
                "option_probabilities_raw_delta_vs_baseline": raw_deltas,
                "option_probabilities_normalized_delta_vs_baseline": (
                    normalized_deltas
                ),
                "truth_probability_raw_delta_vs_baseline": (
                    variant_scored["truth_probability_raw"]
                    - baseline_scored["truth_probability_raw"]
                ),
                "truth_probability_normalized_delta_vs_baseline": (
                    variant_scored["truth_probability_normalized"]
                    - baseline_scored["truth_probability_normalized"]
                ),
                "option_top1_top2_margin_delta_vs_baseline": (
                    variant_scored["option_top1_top2_margin"]
                    - baseline_scored["option_top1_top2_margin"]
                ),
            }
            variant_records.append(variant_record)
            variants_by_id[variant_id].append(variant_record)
            case_variant_records.append(variant_record)

        case_records.append(
            {
                **common,
                "baseline": baseline_record,
                "variants": case_variant_records,
            }
        )

    baseline_good = [record for record in baseline_records if record["correct"]]
    baseline_bad = [record for record in baseline_records if not record["correct"]]
    variant_good = [record for record in variant_records if record["correct"]]
    variant_bad = [record for record in variant_records if not record["correct"]]
    improved = [
        record
        for record in variant_records
        if not record["baseline_correct"] and record["correct"]
    ]
    degraded = [
        record
        for record in variant_records
        if record["baseline_correct"] and not record["correct"]
    ]
    summary_by_variant = {
        variant_id: build_variant_summary(records)
        for variant_id, records in sorted(variants_by_id.items())
    }
    summary = {
        "experiment": "manual_cot_intervention",
        "run_name": args.run_name,
        "data_name": data_name,
        "model_name": args.model_name,
        "input_file": str(input_path),
        "case_count": len(case_records),
        "variant_prediction_count": len(variant_records),
        "generation": {
            "prompt_suffix": r" \boxed{",
            "do_sample": False,
            "max_new_tokens": 8,
            "repetition_penalty": 1.2,
        },
        "by_variant_id": summary_by_variant,
    }

    atomic_write_json(output_directory / "records.json", case_records)
    atomic_write_json(output_directory / "baseline_records.json", baseline_records)
    atomic_write_json(output_directory / "baseline_good_cases.json", baseline_good)
    atomic_write_json(output_directory / "baseline_bad_cases.json", baseline_bad)
    atomic_write_json(output_directory / "variant_records.json", variant_records)
    atomic_write_json(output_directory / "variant_good_cases.json", variant_good)
    atomic_write_json(output_directory / "variant_bad_cases.json", variant_bad)
    atomic_write_json(output_directory / "improved_cases.json", improved)
    atomic_write_json(output_directory / "degraded_cases.json", degraded)
    atomic_write_json(output_directory / "summary.json", summary)

    for variant_id, records in variants_by_id.items():
        variant_directory = output_directory / "by_variant" / variant_id
        atomic_write_json(variant_directory / "records.json", records)
        atomic_write_json(
            variant_directory / "good_cases.json",
            [record for record in records if record["correct"]],
        )
        atomic_write_json(
            variant_directory / "bad_cases.json",
            [record for record in records if not record["correct"]],
        )
        atomic_write_json(
            variant_directory / "improved_cases.json",
            [
                record
                for record in records
                if not record["baseline_correct"] and record["correct"]
            ],
        )
        atomic_write_json(
            variant_directory / "degraded_cases.json",
            [
                record
                for record in records
                if record["baseline_correct"] and not record["correct"]
            ],
        )
        atomic_write_json(
            variant_directory / "summary.json",
            summary_by_variant[variant_id],
        )
    logging.info(
        "Saved %d cases and %d variant predictions to %s",
        len(case_records),
        len(variant_records),
        output_directory,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
