"""Shared helpers for the CoT case-analysis experiments.

The helpers in this module are intentionally independent from the legacy
intervention scripts.  New experiments can therefore use a consistent direct
answer scorer and output schema without changing existing v1-v4 results.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence


OPTION_LABELS = ("A", "B", "C", "D")

# A dependency-free English stop-word list.  It is deliberately limited to
# function words so that content-bearing option terms are not over-filtered.
ENGLISH_STOP_WORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am",
        "an", "and", "any", "are", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can",
        "could", "did", "do", "does", "doing", "down", "during", "each",
        "few", "for", "from", "further", "had", "has", "have", "having",
        "he", "her", "here", "hers", "herself", "him", "himself", "his",
        "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
        "now", "of", "off", "on", "once", "only", "or", "other", "our",
        "ours", "ourselves", "out", "over", "own", "same", "she", "should",
        "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this",
        "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "would", "you", "your",
        "yours", "yourself", "yourselves",
    }
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Write JSON without leaving a partially written target file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(
    path: str | Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if fieldnames is None:
        ordered_fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered_fields:
                    ordered_fields.append(key)
        fieldnames = ordered_fields
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def strip_outer_think_tags(text: str | None) -> str:
    if not text:
        return ""
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def build_no_cot_prompt(base_input_text: str) -> str:
    return base_input_text + r" \boxed{"


def build_reasoning_direct_prompt(
    base_input_text: str,
    cot: str | None,
    explanation: str | None,
) -> tuple[str, str, str]:
    cot_text = strip_outer_think_tags(cot)
    explanation_text = (explanation or "").strip()
    rendered_cot = f"<think>{cot_text}</think>"
    prompt = base_input_text + rendered_cot
    if explanation_text:
        prompt += " " + explanation_text
    prompt += r" \boxed{"
    return prompt, rendered_cot, explanation_text


def build_direct_generation_args(tokenizer: Any) -> dict[str, Any]:
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return {
        "do_sample": False,
        "max_new_tokens": 8,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "use_cache": True,
        "repetition_penalty": 1.2,
    }


def get_option_token_ids(tokenizer: Any) -> dict[str, int]:
    option_token_ids: dict[str, int] = {}
    for label in OPTION_LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Option label {label!r} is not a single token: {token_ids}. "
                "This experiment measures next-token option probabilities and "
                "therefore requires one token per option label."
            )
        option_token_ids[label] = int(token_ids[0])
    if len(set(option_token_ids.values())) != len(OPTION_LABELS):
        raise ValueError(f"Option token ids are not unique: {option_token_ids}")
    return option_token_ids


def _move_inputs_to_model(inputs: Any, model: Any) -> dict[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    return {key: value for key, value in inputs.items()}


def extract_raw_option_probabilities(
    vocabulary_probabilities: Any,
    option_token_ids: dict[str, int],
) -> dict[str, float]:
    probabilities = {}
    for label, token_id in option_token_ids.items():
        value = vocabulary_probabilities[token_id]
        probabilities[label] = float(
            value.item() if hasattr(value, "item") else value
        )
    return probabilities


def normalize_option_probabilities(
    raw_probabilities: dict[str, float],
) -> tuple[dict[str, float], float]:
    if set(raw_probabilities) != set(OPTION_LABELS):
        raise ValueError(
            "Option probabilities must contain exactly A/B/C/D: "
            f"{sorted(raw_probabilities)}"
        )
    option_probability_mass = sum(raw_probabilities.values())
    if option_probability_mass > 0:
        normalized = {
            label: raw_probabilities[label] / option_probability_mass
            for label in OPTION_LABELS
        }
    else:
        normalized = {label: 0.0 for label in OPTION_LABELS}
    return normalized, option_probability_mass


def _parse_generated_prediction(raw_generated_text: str) -> tuple[str | None, str]:
    boxed_output_text = r"\boxed{" + raw_generated_text
    matches = re.findall(
        r"\\boxed\{\s*([ABCD])\s*\}", boxed_output_text, re.IGNORECASE
    )
    if matches:
        return matches[-1].upper(), boxed_output_text

    # Preserve the strict boxed-prefix generation protocol while tolerating a
    # missing closing brace within the eight-token generation budget.
    fallback = re.match(r"\s*([ABCD])(?=\s|\}|\W|$)", raw_generated_text)
    prediction = fallback.group(1).upper() if fallback else None
    return prediction, boxed_output_text


def score_and_generate_direct_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    truth: str,
    generation_args: dict[str, Any] | None = None,
    option_token_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Score A/B/C/D at the boxed prefix, then run the direct generation."""
    import torch

    truth = str(truth).strip().upper()
    if truth not in OPTION_LABELS:
        raise ValueError(f"Unsupported truth label: {truth!r}")

    option_token_ids = option_token_ids or get_option_token_ids(tokenizer)
    generation_args = generation_args or build_direct_generation_args(tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = _move_inputs_to_model(encoded, model)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
        vocabulary_probabilities = torch.softmax(logits, dim=-1)
        raw_probabilities = extract_raw_option_probabilities(
            vocabulary_probabilities, option_token_ids
        )
        outputs = model.generate(**inputs, **generation_args)

    normalized_probabilities, option_probability_mass = (
        normalize_option_probabilities(raw_probabilities)
    )

    ranked_options = sorted(
        OPTION_LABELS,
        key=lambda label: normalized_probabilities[label],
        reverse=True,
    )
    option_argmax_prediction = ranked_options[0]
    option_top1_probability = normalized_probabilities[ranked_options[0]]
    option_top1_top2_margin = (
        option_top1_probability - normalized_probabilities[ranked_options[1]]
    )

    input_length = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0, input_length:]
    raw_generated_text = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()
    prediction, boxed_output_text = _parse_generated_prediction(
        raw_generated_text
    )
    output_text = (
        f"\\boxed{{{prediction}}}" if prediction else boxed_output_text
    )

    return {
        "option_token_ids": option_token_ids,
        "option_probabilities_raw": raw_probabilities,
        "option_probabilities_normalized": normalized_probabilities,
        "option_probability_mass": option_probability_mass,
        "option_argmax_prediction": option_argmax_prediction,
        "truth_probability_raw": raw_probabilities[truth],
        "truth_probability_normalized": normalized_probabilities[truth],
        "option_top1_probability": option_top1_probability,
        "option_top1_top2_margin": option_top1_top2_margin,
        "raw_generated_text": raw_generated_text,
        "boxed_output_text": boxed_output_text,
        "output_text": output_text,
        "prediction": prediction,
        "correct": int(prediction == truth),
    }


def split_choices(choices: str | Sequence[str]) -> list[str]:
    # Dataset preparation joins choices with a single ``|``.  Preserve ``||``
    # operators that legitimately occur inside programming-language options.
    values = (
        re.split(r"(?<!\|)\|(?!\|)", choices)
        if isinstance(choices, str)
        else list(choices)
    )
    if len(values) != 4:
        raise ValueError(f"Expected four choices, got {len(values)}: {values}")
    return [str(value) for value in values]


def extract_prompt_choices(input_text: str) -> list[str]:
    """Extract the four option texts that were actually shown to the model."""
    matches = re.findall(
        r"^[ \t]*([ABCD])\.[ \t]*(.*?)[ \t]*$",
        input_text,
        re.MULTILINE,
    )
    labels = [label for label, _ in matches]
    if labels != list(OPTION_LABELS):
        raise ValueError(
            "Expected exactly one ordered A./B./C./D. option block in "
            f"input_text, found labels {labels}"
        )
    return [choice for _, choice in matches]


def get_sample_choices(sample: dict[str, Any]) -> list[str]:
    """Return prompt choices, which are authoritative for model inference.

    A few legacy filter files contain option text with ``||``.  Their original
    preprocessing split that field incorrectly when rendering the prompt, so
    the serialized ``choices`` field can differ from what the model saw.  Case
    analysis must intervene on the actual prompt rather than silently changing
    the underlying question.
    """
    return extract_prompt_choices(sample["input_text"])


def extract_reasoning_candidates(
    all_output_text: str,
    words: Sequence[str],
    attention: Sequence[float],
) -> list[dict[str, Any]]:
    """Return whitespace-word candidates from CoT and explanation segments."""
    if len(words) != len(attention):
        raise ValueError(
            f"words/attention length mismatch: {len(words)} != {len(attention)}"
        )

    word_matches = list(re.finditer(r"\S+", all_output_text))
    text_words = [match.group(0) for match in word_matches]
    if text_words != list(words):
        raise ValueError(
            "words_list does not match the whitespace tokenization of "
            "all_output_text"
        )

    cot_match = re.search(r"<think>(.*?)</think>", all_output_text, re.DOTALL)
    explanation_match = re.search(
        r"</think>(.*?)\\boxed", all_output_text, re.DOTALL
    )
    segments = (
        ("cot", cot_match),
        ("explanation", explanation_match),
    )

    candidates: list[dict[str, Any]] = []
    for segment_name, match in segments:
        if match is None:
            continue
        segment_start, segment_end = match.span(1)
        for global_index, word_match in enumerate(word_matches):
            if (
                word_match.start() >= segment_start
                and word_match.end() <= segment_end
            ):
                candidates.append(
                    {
                        "global_index": global_index,
                        "word": words[global_index],
                        "attention": float(attention[global_index]),
                        "segment": segment_name,
                    }
                )

    return sorted(candidates, key=lambda item: item["global_index"])


def _normalized_label_token(word: str) -> str:
    normalized = unicodedata.normalize("NFKC", word).strip()
    normalized = normalized.replace("*", "").replace("_", "")
    normalized = re.sub(r"^[^A-Za-z0-9]+", "", normalized)
    normalized = re.sub(r"[^A-Za-z0-9]+$", "", normalized)
    return normalized.upper()


def is_option_label_token(word: str) -> bool:
    return _normalized_label_token(word) in OPTION_LABELS


def normalize_lexemes(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-z0-9]+", normalized)


def get_correct_option_terms(choice_text: str) -> set[str]:
    return {
        term
        for term in normalize_lexemes(choice_text)
        if term not in ENGLISH_STOP_WORDS
    }


def candidate_matches_terms(candidate_word: str, terms: set[str]) -> bool:
    return bool(set(normalize_lexemes(candidate_word)) & terms)


def filter_candidates(
    candidates: Sequence[dict[str, Any]],
    remove_option_labels: bool = False,
    remove_terms: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    remove_terms = remove_terms or set()
    for candidate in candidates:
        reason: str | None = None
        if remove_option_labels and is_option_label_token(candidate["word"]):
            reason = "option_label"
        elif remove_terms and candidate_matches_terms(
            candidate["word"], remove_terms
        ):
            reason = "correct_option_term"

        if reason is None:
            eligible.append(dict(candidate))
        else:
            removed_candidate = dict(candidate)
            removed_candidate["removal_reason"] = reason
            removed.append(removed_candidate)
    return eligible, removed


def select_top_attention_candidates(
    candidates: Sequence[dict[str, Any]], ratio: float
) -> tuple[list[dict[str, Any]], int]:
    if ratio < 0 or ratio > 1:
        raise ValueError(f"Top-k ratio must be within [0, 1], got {ratio}")
    target_top_k = round(len(candidates) * ratio)
    ranked = sorted(
        candidates,
        key=lambda item: (-item["attention"], -item["global_index"]),
    )
    selected = sorted(
        ranked[:target_top_k], key=lambda item: item["global_index"]
    )
    return [dict(item) for item in selected], target_top_k


def selected_candidates_to_text(
    selected: Sequence[dict[str, Any]],
) -> tuple[str, str]:
    cot_words = [
        item["word"] for item in selected if item["segment"] == "cot"
    ]
    explanation_words = [
        item["word"]
        for item in selected
        if item["segment"] == "explanation"
    ]
    return " ".join(cot_words), " ".join(explanation_words)


def make_fixed_choice_permutation(
    choices: Sequence[str], truth: str, seed: int
) -> dict[str, Any]:
    original_choices = split_choices(choices)
    truth = truth.strip().upper()
    if truth not in OPTION_LABELS:
        raise ValueError(f"Unsupported truth label: {truth!r}")
    correct_original_index = OPTION_LABELS.index(truth)
    generator = random.Random(seed)
    permutation = list(range(4))
    for _ in range(100):
        generator.shuffle(permutation)
        correct_new_index = permutation.index(correct_original_index)
        if permutation != list(range(4)) and correct_new_index != correct_original_index:
            break
    else:
        raise RuntimeError("Failed to construct a non-trivial option permutation")

    permuted_choices = [original_choices[index] for index in permutation]
    new_truth = OPTION_LABELS[permutation.index(correct_original_index)]
    original_to_new_indices = [
        permutation.index(original_index) for original_index in range(4)
    ]
    original_to_new_label = {
        OPTION_LABELS[original_index]: OPTION_LABELS[
            permutation.index(original_index)
        ]
        for original_index in range(4)
    }
    new_label_to_original_label = {
        OPTION_LABELS[new_index]: OPTION_LABELS[original_index]
        for new_index, original_index in enumerate(permutation)
    }
    return {
        "original_choices": original_choices,
        "permuted_choices": permuted_choices,
        "original_truth": truth,
        "truth": new_truth,
        "permutation_new_to_original_indices": permutation,
        "permutation_original_to_new_indices": original_to_new_indices,
        "original_to_new_label": original_to_new_label,
        "new_label_to_original_label": new_label_to_original_label,
    }


def replace_prompt_choices(
    input_text: str,
    original_choices: Sequence[str],
    permuted_choices: Sequence[str],
) -> str:
    original_choices = split_choices(original_choices)
    permuted_choices = split_choices(permuted_choices)
    actual_choices = extract_prompt_choices(input_text)
    if actual_choices != original_choices:
        raise ValueError(
            "Provided original choices do not exactly match the prompt option "
            f"block: {original_choices} != {actual_choices}"
        )

    option_line_pattern = re.compile(
        r"^[ \t]*([ABCD])\.[ \t]*(.*?)[ \t]*$", re.MULTILINE
    )
    matches = list(option_line_pattern.finditer(input_text))
    if [match.group(1) for match in matches] != list(OPTION_LABELS):
        raise ValueError("Prompt option block is missing or occurs more than once")

    updated = input_text
    for match, label, choice in reversed(
        list(zip(matches, OPTION_LABELS, permuted_choices))
    ):
        original_line = match.group(0)
        indentation = original_line[: len(original_line) - len(original_line.lstrip())]
        replacement = f"{indentation}{label}. {choice}"
        updated = updated[: match.start()] + replacement + updated[match.end() :]
    return updated


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def build_prediction_summary(
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(records)
    correct_count = sum(int(record.get("correct", 0) == 1) for record in records)
    option_argmax_correct_count = sum(
        int(record.get("option_argmax_prediction") == record.get("truth"))
        for record in records
    )
    missing_prediction_count = sum(
        int(record.get("prediction") is None) for record in records
    )
    summary = {
        **(metadata or {}),
        "total": total,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "option_argmax_correct_count": option_argmax_correct_count,
        "option_argmax_accuracy": (
            option_argmax_correct_count / total if total else 0.0
        ),
        "missing_prediction_count": missing_prediction_count,
        "average_truth_probability_raw": _mean(
            float(record["truth_probability_raw"])
            for record in records
            if record.get("truth_probability_raw") is not None
        ),
        "average_truth_probability_normalized": _mean(
            float(record["truth_probability_normalized"])
            for record in records
            if record.get("truth_probability_normalized") is not None
        ),
        "average_option_top1_probability": _mean(
            float(record["option_top1_probability"])
            for record in records
            if record.get("option_top1_probability") is not None
        ),
        "average_option_top1_top2_margin": _mean(
            float(record["option_top1_top2_margin"])
            for record in records
            if record.get("option_top1_top2_margin") is not None
        ),
    }

    delta_records = [
        record
        for record in records
        if record.get("truth_probability_normalized_delta_vs_no_cot") is not None
    ]
    if delta_records:
        summary.update(
            {
                "no_cot_comparable_count": len(delta_records),
                "accuracy_gain_vs_no_cot": _mean(
                    float(record["correct"] - record["no_cot_correct"])
                    for record in delta_records
                ),
                "average_truth_probability_normalized_delta_vs_no_cot": _mean(
                    float(
                        record[
                            "truth_probability_normalized_delta_vs_no_cot"
                        ]
                    )
                    for record in delta_records
                ),
                "average_truth_probability_raw_delta_vs_no_cot": _mean(
                    float(record["truth_probability_raw_delta_vs_no_cot"])
                    for record in delta_records
                ),
            }
        )
    else:
        summary.update(
            {
                "no_cot_comparable_count": 0,
                "accuracy_gain_vs_no_cot": None,
                "average_truth_probability_normalized_delta_vs_no_cot": None,
                "average_truth_probability_raw_delta_vs_no_cot": None,
            }
        )
    return summary


def write_prediction_bundle(
    output_directory: str | Path,
    records: Sequence[dict[str, Any]],
    record_filename: str = "records.json",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    stem = Path(record_filename).stem
    if stem == "records":
        good_name = "good_cases.json"
        bad_name = "bad_cases.json"
        summary_name = "summary.json"
    else:
        good_name = f"{stem}_good_cases.json"
        bad_name = f"{stem}_bad_cases.json"
        summary_name = f"{stem}_summary.json"

    good_cases = [record for record in records if record.get("correct") == 1]
    bad_cases = [record for record in records if record.get("correct") != 1]
    summary = build_prediction_summary(records, metadata=metadata)
    atomic_write_json(output_directory / record_filename, list(records))
    atomic_write_json(output_directory / good_name, good_cases)
    atomic_write_json(output_directory / bad_name, bad_cases)
    atomic_write_json(output_directory / summary_name, summary)
    return summary


def add_no_cot_comparison(
    record: dict[str, Any], no_cot_record: dict[str, Any] | None
) -> None:
    if no_cot_record is None:
        record.update(
            {
                "no_cot_prediction": None,
                "no_cot_correct": None,
                "correct_delta_vs_no_cot": None,
                "prediction_changed_vs_no_cot": None,
                "no_cot_option_probabilities_raw": None,
                "no_cot_option_probabilities_normalized": None,
                "option_probabilities_raw_delta_vs_no_cot": None,
                "option_probabilities_normalized_delta_vs_no_cot": None,
                "truth_probability_raw_delta_vs_no_cot": None,
                "truth_probability_normalized_delta_vs_no_cot": None,
                "option_top1_top2_margin_delta_vs_no_cot": None,
            }
        )
        return

    raw_deltas = {
        label: (
            record["option_probabilities_raw"][label]
            - no_cot_record["option_probabilities_raw"][label]
        )
        for label in OPTION_LABELS
    }
    normalized_deltas = {
        label: (
            record["option_probabilities_normalized"][label]
            - no_cot_record["option_probabilities_normalized"][label]
        )
        for label in OPTION_LABELS
    }
    record.update(
        {
            "no_cot_prediction": no_cot_record.get("prediction"),
            "no_cot_correct": no_cot_record.get("correct"),
            "correct_delta_vs_no_cot": (
                record["correct"] - no_cot_record["correct"]
            ),
            "prediction_changed_vs_no_cot": (
                record.get("prediction") != no_cot_record.get("prediction")
            ),
            "no_cot_option_probabilities_raw": no_cot_record[
                "option_probabilities_raw"
            ],
            "no_cot_option_probabilities_normalized": no_cot_record[
                "option_probabilities_normalized"
            ],
            "option_probabilities_raw_delta_vs_no_cot": raw_deltas,
            "option_probabilities_normalized_delta_vs_no_cot": (
                normalized_deltas
            ),
            "truth_probability_raw_delta_vs_no_cot": (
                record["truth_probability_raw"]
                - no_cot_record["truth_probability_raw"]
            ),
            "truth_probability_normalized_delta_vs_no_cot": (
                record["truth_probability_normalized"]
                - no_cot_record["truth_probability_normalized"]
            ),
            "option_top1_top2_margin_delta_vs_no_cot": (
                record["option_top1_top2_margin"]
                - no_cot_record["option_top1_top2_margin"]
            ),
        }
    )
