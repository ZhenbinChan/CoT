"""Dependency-free unit tests for the case-study experiment helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from case_study_utils import (
    add_no_cot_comparison,
    build_reasoning_direct_prompt,
    candidate_matches_terms,
    extract_reasoning_candidates,
    extract_raw_option_probabilities,
    filter_candidates,
    get_correct_option_terms,
    get_sample_choices,
    is_option_label_token,
    load_json,
    make_fixed_choice_permutation,
    normalize_option_probabilities,
    replace_prompt_choices,
    select_top_attention_candidates,
    selected_candidates_to_text,
    split_choices,
    write_prediction_bundle,
)


class CaseStudyUtilsTest(unittest.TestCase):
    def test_full_reasoning_scope_keeps_final_sentences(self) -> None:
        text = "prompt <think> first. final cot. </think> conclusion words \\boxed{A}"
        words = text.split()
        attention = [index / 10 for index in range(len(words))]
        candidates = extract_reasoning_candidates(text, words, attention)
        self.assertEqual(
            [item["word"] for item in candidates],
            ["first.", "final", "cot.", "conclusion", "words"],
        )
        # The final CoT/explanation words are still present and no sentence-tail
        # deletion is applied.

    def test_choice_parsing_preserves_double_pipe_operator(self) -> None:
        source = "equal|not equal|less || greater|FALSE"
        self.assertEqual(
            split_choices(source),
            ["equal", "not equal", "less || greater", "FALSE"],
        )
        sample = {
            "choices": source,
            "input_text": "A. equal\nB. not equal\nC. less\nD. \n",
        }
        self.assertEqual(
            get_sample_choices(sample), ["equal", "not equal", "less", ""]
        )

    def test_option_label_detection_is_standalone(self) -> None:
        for word in ("A.", "(B)", "**C**", "[D]", "A"):
            self.assertTrue(is_option_label_token(word), word)
        for word in ("According", "Data", "ABCD", "choiceA", "B-cell"):
            self.assertFalse(is_option_label_token(word), word)

    def test_correct_option_terms_and_exact_matching(self) -> None:
        terms = get_correct_option_terms(
            "The PRIMARY-production rate is 42.5 percent."
        )
        self.assertEqual(
            terms, {"primary", "production", "rate", "42", "5", "percent"}
        )
        self.assertTrue(candidate_matches_terms("PRIMARY-production,", terms))
        self.assertTrue(candidate_matches_terms("42.5", terms))
        self.assertFalse(candidate_matches_terms("productionism", terms))
        self.assertFalse(candidate_matches_terms("the", terms))

    def test_prefilter_recomputes_top_k_and_selects_full_target(self) -> None:
        candidates = [
            {
                "global_index": index,
                "word": word,
                "attention": float(index),
                "segment": "cot",
            }
            for index, word in enumerate(
                ["A.", "reason", "B", "alpha", "beta", "gamma"]
            )
        ]
        eligible, removed = filter_candidates(
            candidates, remove_option_labels=True
        )
        selected, target = select_top_attention_candidates(eligible, 0.5)
        self.assertEqual(len(eligible), 4)
        self.assertEqual(len(removed), 2)
        self.assertEqual(target, 2)
        self.assertEqual(len(selected), target)
        self.assertEqual(
            [item["global_index"] for item in selected], [4, 5]
        )

    def test_fixed_permutation_moves_truth_and_is_reproducible(self) -> None:
        choices = ["alpha", "beta", "gamma", "delta"]
        first = make_fixed_choice_permutation(choices, "C", 58)
        second = make_fixed_choice_permutation(choices, "C", 58)
        self.assertEqual(first, second)
        self.assertNotEqual(first["permuted_choices"], choices)
        self.assertNotEqual(first["truth"], "C")
        new_truth_index = "ABCD".index(first["truth"])
        self.assertEqual(first["permuted_choices"][new_truth_index], "gamma")

    def test_prompt_options_and_truth_are_updated_together(self) -> None:
        choices = ["alpha", "beta", "gamma", "delta"]
        prompt = "Question\nA. alpha\nB. beta\nC. gamma\nD. delta\nAnswer:"
        permutation = make_fixed_choice_permutation(choices, "B", 51)
        updated = replace_prompt_choices(
            prompt, choices, permutation["permuted_choices"]
        )
        for label, choice in zip(
            "ABCD", permutation["permuted_choices"]
        ):
            self.assertIn(f"{label}. {choice}", updated)
        truth_index = "ABCD".index(permutation["truth"])
        self.assertEqual(
            permutation["permuted_choices"][truth_index], "beta"
        )

    def test_retained_reasoning_is_identical_after_option_permutation(self) -> None:
        selected = [
            {
                "global_index": 4,
                "word": "therefore",
                "attention": 0.8,
                "segment": "cot",
            },
            {
                "global_index": 8,
                "word": "evidence",
                "attention": 0.7,
                "segment": "explanation",
            },
        ]
        cot, explanation = selected_candidates_to_text(selected)
        original = "A. one\nB. two\nC. three\nD. four\n"
        permuted = replace_prompt_choices(
            original,
            ["one", "two", "three", "four"],
            ["three", "four", "one", "two"],
        )
        _, rendered_before, explanation_before = build_reasoning_direct_prompt(
            original, cot, explanation
        )
        _, rendered_after, explanation_after = build_reasoning_direct_prompt(
            permuted, cot, explanation
        )
        self.assertEqual(rendered_before, rendered_after)
        self.assertEqual(explanation_before, explanation_after)

    def test_raw_and_normalized_probabilities(self) -> None:
        vocabulary_probabilities = [0.01, 0.1, 0.2, 0.3, 0.15, 0.24]
        token_ids = {"A": 1, "B": 2, "C": 3, "D": 4}
        raw = extract_raw_option_probabilities(
            vocabulary_probabilities, token_ids
        )
        self.assertEqual(raw, {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.15})
        normalized, mass = normalize_option_probabilities(raw)
        self.assertAlmostEqual(mass, 0.75)
        self.assertLess(abs(sum(normalized.values()) - 1.0), 1e-6)
        self.assertAlmostEqual(normalized["C"], 0.4)

    def test_no_cot_comparison_saves_all_probability_deltas(self) -> None:
        no_cot = {
            "prediction": "A",
            "correct": 0,
            "option_probabilities_raw": {
                "A": 0.1,
                "B": 0.2,
                "C": 0.3,
                "D": 0.1,
            },
            "option_probabilities_normalized": {
                "A": 1 / 7,
                "B": 2 / 7,
                "C": 3 / 7,
                "D": 1 / 7,
            },
            "truth_probability_raw": 0.2,
            "truth_probability_normalized": 2 / 7,
            "option_top1_top2_margin": 1 / 7,
        }
        record = {
            "prediction": "B",
            "correct": 1,
            "option_probabilities_raw": {
                "A": 0.05,
                "B": 0.4,
                "C": 0.1,
                "D": 0.05,
            },
            "option_probabilities_normalized": {
                "A": 1 / 12,
                "B": 8 / 12,
                "C": 2 / 12,
                "D": 1 / 12,
            },
            "truth_probability_raw": 0.4,
            "truth_probability_normalized": 8 / 12,
            "option_top1_top2_margin": 0.5,
        }
        add_no_cot_comparison(record, no_cot)
        self.assertEqual(record["correct_delta_vs_no_cot"], 1)
        self.assertAlmostEqual(
            record["option_probabilities_raw_delta_vs_no_cot"]["B"], 0.2
        )
        self.assertAlmostEqual(
            record["truth_probability_normalized_delta_vs_no_cot"],
            8 / 12 - 2 / 7,
        )

    def test_good_bad_bundle_is_complete_partition(self) -> None:
        records = [
            {
                "sample_index": 0,
                "truth": "A",
                "prediction": "A",
                "correct": 1,
                "option_argmax_prediction": "A",
                "truth_probability_raw": 0.2,
                "truth_probability_normalized": 0.4,
                "option_top1_probability": 0.4,
                "option_top1_top2_margin": 0.1,
            },
            {
                "sample_index": 1,
                "truth": "B",
                "prediction": "C",
                "correct": 0,
                "option_argmax_prediction": "C",
                "truth_probability_raw": 0.1,
                "truth_probability_normalized": 0.2,
                "option_top1_probability": 0.5,
                "option_top1_top2_margin": 0.2,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            summary = write_prediction_bundle(output, records)
            good = load_json(output / "good_cases.json")
            bad = load_json(output / "bad_cases.json")
            self.assertEqual(summary["total"], 2)
            self.assertEqual(len(good) + len(bad), len(records))
            self.assertEqual(
                {item["sample_index"] for item in good + bad}, {0, 1}
            )


if __name__ == "__main__":
    unittest.main()
