"""Plot pooled probability metrics for the six case-study conditions.

One 1 x 3 figure is produced for each maintain ratio.  Within every figure,
the x-axis contains the six experiment conditions and the three panels show
the pooled mean raw truth probability, normalized truth probability, and
top1-top2 option probability margin.  No-CoT is repeated as a fixed,
ratio-independent baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROBABILITY_METRICS = OrderedDict(
    (
        (
            "average_truth_probability_raw",
            "Truth probability\n(raw vocabulary)",
        ),
        (
            "average_truth_probability_normalized",
            "Truth probability\n(A/B/C/D normalized)",
        ),
        (
            "average_option_top1_top2_margin",
            "Top1–top2\nprobability margin",
        ),
    )
)

EXPERIMENTS = OrderedDict(
    (
        ("no_cot", "No CoT"),
        ("letter_kept", "Letter kept"),
        ("letter_removed", "Letter removed"),
        (
            "correct_option_words_removed",
            "Correct-option words removed",
        ),
        ("permuted_letter_kept", "Permuted + letter kept"),
        ("permuted_letter_removed", "Permuted + letter removed"),
    )
)

TOPK_CONDITIONS = tuple(key for key in EXPERIMENTS if key != "no_cot")

EXPERIMENT_COLORS = {
    "No CoT": "#5B6573",
    "Letter kept": "#4C78A8",
    "Letter removed": "#9ECAE1",
    "Correct-option words removed": "#9C89B8",
    "Permuted + letter kept": "#F2A65A",
    "Permuted + letter removed": "#F6C98D",
}

EXPERIMENT_HATCHES = {
    "No CoT": "",
    "Letter kept": "///",
    "Letter removed": "\\\\",
    "Correct-option words removed": "xx",
    "Permuted + letter kept": "..",
    "Permuted + letter removed": "++",
}

EXPERIMENT_TICK_LABELS = {
    "No CoT": "No CoT",
    "Letter kept": "Letter\nkept",
    "Letter removed": "Letter\nremoved",
    "Correct-option words removed": "Correct-option\nwords removed",
    "Permuted + letter kept": "Permuted +\nletter kept",
    "Permuted + letter removed": "Permuted +\nletter removed",
}

SOURCE_TABLE_COLUMNS = (
    "condition",
    "experiment_name",
    "maintain_ratio",
    "total",
    "subset_count",
    *PROBABILITY_METRICS.keys(),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three pooled probability metrics across the six case-study "
            "conditions, with one 1x3 figure per maintain ratio."
        )
    )
    parser.add_argument(
        "--result_root", default="./final_results/case_study"
    )
    parser.add_argument("--data_name", default="mmlu_redux")
    parser.add_argument("--model_name", default="Qwen3-8B")
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3]
    )
    parser.add_argument("--output_dir")
    parser.add_argument(
        "--figure_formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png", "svg", "pdf"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip_figures", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def format_ratio(ratio: float) -> str:
    return f"{ratio:g}"


def validated_probability_metrics(
    summary: dict[str, Any], path: Path
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in PROBABILITY_METRICS:
        if metric not in summary:
            raise KeyError(f"Missing {metric!r} in {path}")
        value = summary[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Expected numeric {metric!r} in {path}, got {value!r}"
            )
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Expected {metric!r} within [0, 1] in {path}, got {value}"
            )
        values[metric] = value
    return values


def validated_counts(
    summary: dict[str, Any], path: Path
) -> tuple[int, int]:
    total = summary.get("total")
    subset_count = summary.get("subset_count")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise ValueError(f"Invalid total in {path}: {total!r}")
    if (
        not isinstance(subset_count, int)
        or isinstance(subset_count, bool)
        or subset_count <= 0
    ):
        raise ValueError(
            f"Invalid subset_count in {path}: {subset_count!r}"
        )
    return total, subset_count


def collect_probability_rows(
    result_root: Path,
    data_name: str,
    model_name: str,
    ratios: list[float],
) -> list[dict[str, Any]]:
    no_cot_path = (
        result_root
        / "no_cot"
        / data_name
        / model_name
        / "aggregate_summary.json"
    )
    if not no_cot_path.is_file():
        raise FileNotFoundError(f"Missing No-CoT aggregate summary: {no_cot_path}")
    no_cot_summary = load_json(no_cot_path)
    no_cot_metadata = (
        no_cot_summary.get("data_name"),
        no_cot_summary.get("model_name"),
        no_cot_summary.get("aggregation"),
    )
    expected_no_cot_metadata = (data_name, model_name, "micro")
    if no_cot_metadata != expected_no_cot_metadata:
        raise ValueError(
            f"No-CoT metadata mismatch in {no_cot_path}: expected "
            f"{expected_no_cot_metadata}, got {no_cot_metadata}"
        )
    reference_total, reference_subset_count = validated_counts(
        no_cot_summary, no_cot_path
    )
    no_cot_metrics = validated_probability_metrics(
        no_cot_summary, no_cot_path
    )

    topk_root = result_root / "topk_leakage" / data_name / model_name
    topk_summaries: dict[str, tuple[Path, dict[str, Any]]] = {}
    for condition in TOPK_CONDITIONS:
        path = topk_root / f"{condition}_aggregate_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Top-k aggregate summary: {path}")
        summary = load_json(path)
        metadata = (
            summary.get("condition"),
            summary.get("data_name"),
            summary.get("model_name"),
            summary.get("aggregation"),
        )
        expected_metadata = (condition, data_name, model_name, "micro")
        if metadata != expected_metadata:
            raise ValueError(
                f"Top-k metadata mismatch in {path}: expected "
                f"{expected_metadata}, got {metadata}"
            )
        if not isinstance(summary.get("by_ratio"), dict):
            raise TypeError(f"Expected by_ratio object in {path}")
        topk_summaries[condition] = (path, summary)

    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        rows.append(
            {
                "condition": "no_cot",
                "experiment_name": EXPERIMENTS["no_cot"],
                "maintain_ratio": ratio,
                "total": reference_total,
                "subset_count": reference_subset_count,
                **no_cot_metrics,
            }
        )
        ratio_key = format_ratio(ratio)
        for condition in TOPK_CONDITIONS:
            path, aggregate = topk_summaries[condition]
            ratio_summary = aggregate["by_ratio"].get(ratio_key)
            if not isinstance(ratio_summary, dict):
                raise KeyError(f"Missing ratio {ratio_key!r} in {path}")
            if ratio_summary.get("condition") != condition:
                raise ValueError(
                    f"Condition mismatch for ratio {ratio_key} in {path}"
                )
            reported_ratio = ratio_summary.get("ratio")
            if reported_ratio != ratio:
                raise ValueError(
                    f"Ratio mismatch in {path}: expected {ratio}, "
                    f"got {reported_ratio!r}"
                )
            total, subset_count = validated_counts(ratio_summary, path)
            if (total, subset_count) != (
                reference_total,
                reference_subset_count,
            ):
                raise ValueError(
                    f"Comparison population mismatch for ratio {ratio_key} "
                    f"in {path}: expected total/subsets "
                    f"{reference_total}/{reference_subset_count}, got "
                    f"{total}/{subset_count}"
                )
            rows.append(
                {
                    "condition": condition,
                    "experiment_name": EXPERIMENTS[condition],
                    "maintain_ratio": ratio,
                    "total": total,
                    "subset_count": subset_count,
                    **validated_probability_metrics(ratio_summary, path),
                }
            )

    expected_rows = len(ratios) * len(EXPERIMENTS)
    if len(rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} probability rows, produced {len(rows)}"
        )
    return rows


def write_source_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SOURCE_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def plot_probability_figures(
    rows: list[dict[str, Any]],
    ratios: list[float],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required for figures. Install it with "
            "`python -m pip install matplotlib`."
        ) from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    experiment_names = list(EXPERIMENTS.values())
    by_key = {
        (row["experiment_name"], row["maintain_ratio"]): row for row in rows
    }
    output_paths: list[Path] = []
    for ratio in ratios:
        figure, axes = plt.subplots(
            1,
            len(PROBABILITY_METRICS),
            figsize=(15.2, 5.3),
            sharey=True,
        )
        for panel_index, (metric, panel_title) in enumerate(
            PROBABILITY_METRICS.items()
        ):
            axis = axes[panel_index]
            values = [
                by_key[(experiment_name, ratio)][metric]
                for experiment_name in experiment_names
            ]
            bars = axis.bar(
                range(len(experiment_names)),
                values,
                width=0.72,
                color=[
                    EXPERIMENT_COLORS[name] for name in experiment_names
                ],
                edgecolor="#2F3540",
                linewidth=0.6,
                zorder=3,
            )
            for bar, experiment_name in zip(bars, experiment_names):
                bar.set_hatch(EXPERIMENT_HATCHES[experiment_name])
            axis.bar_label(
                bars,
                labels=[f"{value:.1%}" for value in values],
                padding=3,
                fontsize=7.3,
                color="#30343B",
            )
            axis.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)
            axis.set_xticks(
                range(len(experiment_names)),
                [
                    EXPERIMENT_TICK_LABELS[name]
                    for name in experiment_names
                ],
            )
            axis.tick_params(axis="x", labelsize=7.1, length=0, pad=7)
            axis.tick_params(axis="y", labelsize=8)
            axis.set_ylim(0.0, 1.12)
            axis.set_yticks([value / 10 for value in range(0, 11, 2)])
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.set_axisbelow(True)
            axis.grid(
                axis="y", color="#D9DEE7", linewidth=0.7, alpha=0.85
            )
            axis.text(
                -0.08,
                1.04,
                chr(ord("a") + panel_index),
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                ha="right",
                va="bottom",
            )

        figure.suptitle(
            f"Probability comparison across experiments — {ratio:.0%} retained",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        figure.supylabel("Pooled mean value", x=0.008, fontsize=10)
        sample_count = int(rows[0]["total"])
        figure.text(
            0.5,
            0.012,
            f"Pooled across n = {sample_count} samples; No CoT is the fixed, "
            "ratio-independent baseline.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#4D5663",
        )
        figure.tight_layout(rect=(0.02, 0.055, 0.998, 0.94), w_pad=1.6)

        filename = (
            f"case_study_probability_ratio_{int(round(ratio * 100))}pct"
        )
        for file_format in formats:
            output_path = output_dir / f"{filename}.{file_format}"
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
            if file_format == "png":
                save_kwargs["dpi"] = dpi
            figure.savefig(output_path, **save_kwargs)
            output_paths.append(output_path)
        plt.close(figure)
    return output_paths


def main() -> None:
    args = parse_args()
    if len(set(args.ratios)) != len(args.ratios):
        raise ValueError(f"Duplicate ratios are not allowed: {args.ratios}")
    if any(not 0.0 <= ratio <= 1.0 for ratio in args.ratios):
        raise ValueError(f"Ratios must be within [0, 1]: {args.ratios}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    result_root = Path(args.result_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else result_root / "analysis" / args.data_name / args.model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_probability_rows(
        result_root=result_root,
        data_name=args.data_name,
        model_name=args.model_name,
        ratios=args.ratios,
    )
    source_table_path = output_dir / "case_study_probability_summary.csv"
    write_source_table(source_table_path, rows)

    figure_paths: list[Path] = []
    if not args.skip_figures:
        figure_paths = plot_probability_figures(
            rows=rows,
            ratios=args.ratios,
            output_dir=output_dir,
            formats=args.figure_formats,
            dpi=args.dpi,
        )

    print(
        json.dumps(
            {
                "source_table": str(source_table_path),
                "row_count": len(rows),
                "ratios": args.ratios,
                "figures": [str(path) for path in figure_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
