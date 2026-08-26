"""Run top-attention-token answer-leakage case-study conditions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from case_study_utils import (
    add_no_cot_comparison,
    atomic_write_csv,
    atomic_write_json,
    build_direct_generation_args,
    build_no_cot_prompt,
    build_prediction_summary,
    build_reasoning_direct_prompt,
    extract_reasoning_candidates,
    filter_candidates,
    get_correct_option_terms,
    get_option_token_ids,
    get_sample_choices,
    load_json,
    make_fixed_choice_permutation,
    replace_prompt_choices,
    score_and_generate_direct_answer,
    select_top_attention_candidates,
    selected_candidates_to_text,
    write_prediction_bundle,
)


CONDITIONS = (
    "letter_kept",
    "letter_removed",
    "correct_option_words_removed",
    "permuted_letter_kept",
    "permuted_letter_removed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Top-k answer-leakage experiments for CoT case analysis."
    )
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--sub_set")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path")
    parser.add_argument(
        "--output_root", default="./final_results/case_study"
    )
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--random_seed", type=int, default=51)
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def ratio_name(ratio: float) -> str:
    return f"{ratio:g}"


def load_no_cot_records(
    output_root: str,
    data_name: str,
    model_name: str,
    subset: str,
) -> dict[int, dict[str, Any]]:
    path = (
        Path(output_root)
        / "no_cot"
        / data_name
        / model_name
        / subset
        / "records.json"
    )
    if not path.is_file():
        logging.warning(
            "No-CoT records are absent at %s; non-permuted delta fields "
            "will be null. Run scripts/4_no_cot_direct_answer.sh first.",
            path,
        )
        return {}
    records = load_json(path)
    return {int(record["sample_index"]): record for record in records}


def load_inputs(args: argparse.Namespace) -> tuple[list[Any], list[Any], list[Any]]:
    result_directory = Path("results") / args.data_name
    filter_path = result_directory / (
        f"{args.model_name}_{args.sub_set}_filter_right.json"
    )
    attention_directory = result_directory / "att_grad_before"
    attention_path = attention_directory / (
        f"{args.model_name}_{args.sub_set}_rate1_all_att_list.json"
    )
    words_path = attention_directory / (
        f"{args.model_name}_{args.sub_set}_rate1_words_list.json"
    )
    for path in (filter_path, attention_path, words_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    samples = load_json(filter_path)
    all_attention = load_json(attention_path)
    all_words = load_json(words_path)
    if not isinstance(samples, list):
        raise TypeError(f"Expected a JSON list in {filter_path}")
    if not (len(samples) == len(all_attention) == len(all_words)):
        raise ValueError(
            "filter/attention/words sample counts differ: "
            f"{len(samples)} != {len(all_attention)} != {len(all_words)}"
        )
    for sample_index, (attention, words) in enumerate(
        zip(all_attention, all_words)
    ):
        if len(attention) != len(words):
            raise ValueError(
                f"Sample {sample_index} attention/words lengths differ: "
                f"{len(attention)} != {len(words)}"
            )

    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max_samples must be non-negative")
        samples = samples[: args.max_samples]
        all_attention = all_attention[: args.max_samples]
        all_words = all_words[: args.max_samples]
    return samples, all_attention, all_words


def removal_configuration(
    condition: str,
    choices: list[str],
    truth: str,
) -> tuple[bool, set[str], str]:
    remove_letters = condition in {
        "letter_removed",
        "correct_option_words_removed",
        "permuted_letter_removed",
    }
    if condition == "correct_option_words_removed":
        correct_choice = choices["ABCD".index(truth)]
        terms = get_correct_option_terms(correct_choice)
        return True, terms, correct_choice
    return remove_letters, set(), choices["ABCD".index(truth)]


def run_experiment(args: argparse.Namespace) -> None:
    if not args.sub_set:
        raise ValueError("--sub_set is required unless --aggregate_only is used")
    if not args.model_path:
        raise ValueError("--model_path is required for inference")
    if len(set(args.ratios)) != len(args.ratios):
        raise ValueError(f"Duplicate ratios are not allowed: {args.ratios}")
    if any(ratio < 0 or ratio > 1 for ratio in args.ratios):
        raise ValueError(f"All ratios must be within [0, 1]: {args.ratios}")

    samples, all_attention, all_words = load_inputs(args)
    no_cot_by_index = load_no_cot_records(
        output_root=args.output_root,
        data_name=args.data_name,
        model_name=args.model_name,
        subset=args.sub_set,
    )

    logging.info("Loading model from %s", args.model_path)
    from tqdm import tqdm
    from utils import load_tokenizer_and_model

    tokenizer, model = load_tokenizer_and_model(model_path=args.model_path)
    model.eval()
    generation_args = build_direct_generation_args(tokenizer)
    option_token_ids = get_option_token_ids(tokenizer)

    records_by_ratio: dict[float, list[dict[str, Any]]] = {
        ratio: [] for ratio in args.ratios
    }
    is_permuted = args.condition.startswith("permuted_")

    for sample_index, (sample, attention, words) in enumerate(
        tqdm(
            zip(samples, all_attention, all_words),
            total=len(samples),
            desc=f"{args.condition} {args.sub_set}",
        )
    ):
        original_truth = str(sample["truth"]).strip().upper()
        original_choices = get_sample_choices(sample)
        source_choices_field = sample["choices"]
        original_base_input_text = sample["input_text"]
        candidates = extract_reasoning_candidates(
            all_output_text=sample["all_output_text"],
            words=words,
            attention=attention,
        )
        remove_letters, correct_terms, correct_choice_text = (
            removal_configuration(
                args.condition, original_choices, original_truth
            )
        )
        eligible_candidates, removed_candidates = filter_candidates(
            candidates,
            remove_option_labels=remove_letters,
            remove_terms=correct_terms,
        )

        base_input_text = original_base_input_text
        truth = original_truth
        permutation_fields: dict[str, Any] = {}
        no_cot_reference = no_cot_by_index.get(sample_index)
        no_cot_reference_kind = "original_prompt_saved_baseline"
        if no_cot_reference is not None:
            if no_cot_reference.get("truth") != original_truth:
                raise ValueError(
                    f"No-CoT truth mismatch at sample {sample_index}: "
                    f"{no_cot_reference.get('truth')} != {original_truth}"
                )
            if no_cot_reference.get("base_input_text") != original_base_input_text:
                raise ValueError(
                    f"No-CoT prompt mismatch at sample {sample_index}; "
                    "sample_index alignment is not safe"
                )
        if is_permuted:
            permutation_fields = make_fixed_choice_permutation(
                choices=original_choices,
                truth=original_truth,
                seed=args.random_seed + sample_index,
            )
            base_input_text = replace_prompt_choices(
                input_text=original_base_input_text,
                original_choices=original_choices,
                permuted_choices=permutation_fields["permuted_choices"],
            )
            truth = permutation_fields["truth"]
            # Re-score the no-CoT baseline on the same permuted prompt.  The
            # saved original-prompt baseline is not label-comparable after the
            # correct option moves to a new letter.
            permuted_no_cot_prompt = build_no_cot_prompt(base_input_text)
            no_cot_reference = {
                **score_and_generate_direct_answer(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=permuted_no_cot_prompt,
                    truth=truth,
                    generation_args=generation_args,
                    option_token_ids=option_token_ids,
                ),
                "sample_index": sample_index,
                "truth": truth,
                "input_text_with_CoT": permuted_no_cot_prompt,
            }
            no_cot_reference_kind = "permuted_prompt_recomputed"

        for ratio in args.ratios:
            selected, target_top_k = select_top_attention_candidates(
                eligible_candidates, ratio
            )
            selected_cot, selected_explanation = selected_candidates_to_text(
                selected
            )
            prompt, rendered_cot, rendered_explanation = (
                build_reasoning_direct_prompt(
                    base_input_text=base_input_text,
                    cot=selected_cot,
                    explanation=selected_explanation,
                )
            )
            prediction_fields = score_and_generate_direct_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                truth=truth,
                generation_args=generation_args,
                option_token_ids=option_token_ids,
            )
            record = {
                "sample_index": sample_index,
                "experiment_condition": args.condition,
                "ratio": ratio,
                "question": sample.get("question"),
                "choices": (
                    permutation_fields.get("permuted_choices")
                    if is_permuted
                    else original_choices
                ),
                "truth": truth,
                "original_truth": original_truth,
                "original_choices": original_choices,
                "source_choices_field": source_choices_field,
                "base_input_text": base_input_text,
                "original_base_input_text": original_base_input_text,
                "input_text_with_CoT": prompt,
                "original_CoT": sample.get("CoT", ""),
                "original_explanation": sample.get("output_text", ""),
                "original_prediction": sample.get("prediction"),
                "modified_CoT": rendered_cot,
                "modified_explanation": rendered_explanation,
                "retained_cot_text": selected_cot,
                "retained_explanation_text": selected_explanation,
                "original_candidate_count": len(candidates),
                "candidate_count": len(eligible_candidates),
                "candidate_count_after_filter": len(eligible_candidates),
                "target_top_k": target_top_k,
                "actual_top_k": len(selected),
                "actual_retained_ratio": (
                    len(selected) / len(eligible_candidates)
                    if eligible_candidates
                    else 0.0
                ),
                "selected_global_indices": [
                    item["global_index"] for item in selected
                ],
                "selected_words": [item["word"] for item in selected],
                "selected_attention": [
                    item["attention"] for item in selected
                ],
                "selected_segments": [item["segment"] for item in selected],
                "selected_candidates": selected,
                "removed_global_indices": [
                    item["global_index"] for item in removed_candidates
                ],
                "removed_words": [
                    item["word"] for item in removed_candidates
                ],
                "removed_candidates": removed_candidates,
                "correct_option_text": correct_choice_text,
                "correct_option_terms": sorted(correct_terms),
                "last_two_sentences_removed": False,
                "candidate_scope": "full_cot_and_post_cot_explanation",
                "retained_selection_source_condition": (
                    args.condition.removeprefix("permuted_")
                    if is_permuted
                    else args.condition
                ),
                "no_cot_reference_kind": no_cot_reference_kind,
                **permutation_fields,
                **prediction_fields,
            }
            add_no_cot_comparison(record, no_cot_reference)
            records_by_ratio[ratio].append(record)

    output_directory = (
        Path(args.output_root)
        / "topk_leakage"
        / args.data_name
        / args.model_name
        / args.sub_set
        / args.condition
    )
    for ratio, records in records_by_ratio.items():
        filename = f"top_percentage{ratio_name(ratio)}.json"
        summary = write_prediction_bundle(
            output_directory=output_directory,
            records=records,
            record_filename=filename,
            metadata={
                "experiment": "topk_answer_leakage",
                "condition": args.condition,
                "ratio": ratio,
                "data_name": args.data_name,
                "model_name": args.model_name,
                "subset": args.sub_set,
                "random_seed": args.random_seed,
                "max_samples": args.max_samples,
                "last_two_sentences_removed": False,
                "candidate_scope": "full_cot_and_post_cot_explanation",
                "selection_rounding": "round(candidate_count * ratio)",
                "generation": {
                    "prompt_suffix": r" \boxed{",
                    "do_sample": False,
                    "max_new_tokens": 8,
                    "repetition_penalty": 1.2,
                },
            },
        )
        logging.info(
            "Saved %s ratio %s: %d records, accuracy %.4f",
            args.condition,
            ratio_name(ratio),
            len(records),
            summary["accuracy"],
        )


def aggregate_results(args: argparse.Namespace) -> None:
    model_directory = (
        Path(args.output_root)
        / "topk_leakage"
        / args.data_name
        / args.model_name
    )
    ratio_summaries: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for ratio in args.ratios:
        ratio_text = ratio_name(ratio)
        record_paths = sorted(
            model_directory.glob(
                f"*/{args.condition}/top_percentage{ratio_text}.json"
            )
        )
        if not record_paths:
            logging.warning(
                "No records found for %s ratio %s under %s",
                args.condition,
                ratio_text,
                model_directory,
            )
            continue
        all_records: list[dict[str, Any]] = []
        subset_rows = []
        for record_path in record_paths:
            subset = record_path.parents[1].name
            records = load_json(record_path)
            all_records.extend(records)
            summary = build_prediction_summary(records)
            row = {
                "condition": args.condition,
                "ratio": ratio,
                "subset": subset,
                "total": summary["total"],
                "correct_count": summary["correct_count"],
                "accuracy": summary["accuracy"],
                "option_argmax_accuracy": summary["option_argmax_accuracy"],
                "accuracy_gain_vs_no_cot": summary[
                    "accuracy_gain_vs_no_cot"
                ],
                "average_truth_probability_normalized": summary[
                    "average_truth_probability_normalized"
                ],
                "average_truth_probability_normalized_delta_vs_no_cot": summary[
                    "average_truth_probability_normalized_delta_vs_no_cot"
                ],
            }
            subset_rows.append(row)
            csv_rows.append(row)
        aggregate = build_prediction_summary(
            all_records,
            metadata={
                "condition": args.condition,
                "ratio": ratio,
                "aggregation": "micro",
                "subset_count": len(record_paths),
                "subsets": subset_rows,
            },
        )
        ratio_summaries[ratio_text] = aggregate
        csv_rows.append(
            {
                "condition": args.condition,
                "ratio": ratio,
                "subset": "__micro_total__",
                "total": aggregate["total"],
                "correct_count": aggregate["correct_count"],
                "accuracy": aggregate["accuracy"],
                "option_argmax_accuracy": aggregate[
                    "option_argmax_accuracy"
                ],
                "accuracy_gain_vs_no_cot": aggregate[
                    "accuracy_gain_vs_no_cot"
                ],
                "average_truth_probability_normalized": aggregate[
                    "average_truth_probability_normalized"
                ],
                "average_truth_probability_normalized_delta_vs_no_cot": aggregate[
                    "average_truth_probability_normalized_delta_vs_no_cot"
                ],
            }
        )

    if not ratio_summaries:
        raise FileNotFoundError(
            f"No results found for condition {args.condition!r}"
        )
    aggregate_document = {
        "experiment": "topk_answer_leakage",
        "condition": args.condition,
        "data_name": args.data_name,
        "model_name": args.model_name,
        "aggregation": "micro",
        "by_ratio": ratio_summaries,
    }
    atomic_write_json(
        model_directory / f"{args.condition}_aggregate_summary.json",
        aggregate_document,
    )
    atomic_write_csv(
        model_directory / f"{args.condition}_aggregate_summary.csv",
        csv_rows,
    )
    logging.info("Saved aggregate summaries under %s", model_directory)


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
