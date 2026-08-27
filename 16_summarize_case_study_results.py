"""Build the six-condition case-study tables and accuracy figures.

The table is the source data for three 5 x 4 figure grids.  No-CoT is a
ratio-independent baseline and is repeated visually in each ratio figure, but
appears only once per subset in the exported table.

The script also reads the aggregate summaries and draws one grouped bar chart
using pooled micro accuracy (total correct predictions / total samples).  In
that chart, No-CoT is repeated at each ratio as a fixed visual baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    "accuracy",
    "average_truth_probability_raw",
    "average_truth_probability_normalized",
    "average_option_top1_probability",
    "average_option_top1_top2_margin",
)

TABLE_COLUMNS = (
    "experiment_name",
    "subset",
    "maintain_ratio",
    *METRIC_COLUMNS,
)

MICRO_TABLE_COLUMNS = (
    "experiment_name",
    "maintain_ratio",
    "correct_count",
    "total",
    "micro_accuracy",
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

CANONICAL_SUBSET_ORDER = (
    "global_facts",
    "high_school_mathematics",
    "college_mathematics",
    "high_school_computer_science",
    "college_computer_science",
    "high_school_biology",
    "college_biology",
    "professional_law",
    "college_physics",
    "machine_learning",
    "sociology",
    "us_foreign_policy",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "business_ethics",
    "clinical_knowledge",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate No-CoT and five Top-k leakage experiments, then draw "
            "one 5x4 accuracy grid for each maintain ratio."
        )
    )
    parser.add_argument(
        "--result_root", default="./final_results/case_study"
    )
    parser.add_argument("--data_name", default="mmlu_redux")
    parser.add_argument("--model_name", default="Qwen3-8B")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3])
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


def metric_values(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in METRIC_COLUMNS:
        if column not in summary:
            raise KeyError(f"Missing {column!r} in {path}")
        value = summary[column]
        if value is not None and not isinstance(value, (int, float)):
            raise TypeError(
                f"Expected numeric or null {column!r} in {path}, got {value!r}"
            )
        values[column] = value
    return values


def ordered_subsets(discovered: set[str]) -> list[str]:
    canonical = [name for name in CANONICAL_SUBSET_ORDER if name in discovered]
    extras = sorted(discovered - set(canonical))
    return canonical + extras


def collect_table_rows(
    result_root: Path,
    data_name: str,
    model_name: str,
    ratios: list[float],
) -> tuple[list[dict[str, Any]], list[str]]:
    no_cot_root = result_root / "no_cot" / data_name / model_name
    topk_root = result_root / "topk_leakage" / data_name / model_name
    if not no_cot_root.is_dir():
        raise FileNotFoundError(f"No-CoT result directory not found: {no_cot_root}")
    if not topk_root.is_dir():
        raise FileNotFoundError(f"Top-k result directory not found: {topk_root}")

    no_cot_summaries = {
        path.parent.name: path
        for path in no_cot_root.glob("*/summary.json")
    }
    if not no_cot_summaries:
        raise FileNotFoundError(f"No subset summaries found under {no_cot_root}")
    subsets = ordered_subsets(set(no_cot_summaries))

    rows: list[dict[str, Any]] = []
    for subset in subsets:
        path = no_cot_summaries[subset]
        summary = load_json(path)
        if summary.get("subset") != subset:
            raise ValueError(
                f"No-CoT subset mismatch in {path}: {summary.get('subset')!r}"
            )
        rows.append(
            {
                "experiment_name": EXPERIMENTS["no_cot"],
                "subset": subset,
                "maintain_ratio": None,
                **metric_values(summary, path),
            }
        )

    for condition in TOPK_CONDITIONS:
        for subset in subsets:
            for ratio in ratios:
                path = (
                    topk_root
                    / subset
                    / condition
                    / f"top_percentage{format_ratio(ratio)}_summary.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Missing Top-k summary: {path}")
                summary = load_json(path)
                expected = (condition, subset, ratio)
                actual = (
                    summary.get("condition"),
                    summary.get("subset"),
                    summary.get("ratio"),
                )
                if actual != expected:
                    raise ValueError(
                        f"Summary metadata mismatch in {path}: "
                        f"expected {expected}, got {actual}"
                    )
                rows.append(
                    {
                        "experiment_name": EXPERIMENTS[condition],
                        "subset": subset,
                        "maintain_ratio": ratio,
                        **metric_values(summary, path),
                    }
                )

    expected_rows = len(subsets) * (1 + len(TOPK_CONDITIONS) * len(ratios))
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, produced {len(rows)}")
    return rows, subsets


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row[column] is None else row[column]
                    for column in TABLE_COLUMNS
                }
            )


def read_table(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
            raise ValueError(
                f"Unexpected table columns in {path}: {reader.fieldnames}"
            )
        for source in reader:
            row: dict[str, Any] = {
                "experiment_name": source["experiment_name"],
                "subset": source["subset"],
                "maintain_ratio": (
                    None
                    if source["maintain_ratio"] == ""
                    else float(source["maintain_ratio"])
                ),
            }
            for column in METRIC_COLUMNS:
                row[column] = (
                    None if source[column] == "" else float(source[column])
                )
            rows.append(row)
    return rows


def validate_table(
    rows: list[dict[str, Any]], subsets: list[str], ratios: list[float]
) -> None:
    keys: set[tuple[str, str, float | None]] = set()
    for row in rows:
        key = (
            row["experiment_name"],
            row["subset"],
            row["maintain_ratio"],
        )
        if key in keys:
            raise ValueError(f"Duplicate table row: {key}")
        keys.add(key)
        accuracy = row["accuracy"]
        if accuracy is None or not 0 <= accuracy <= 1:
            raise ValueError(f"Invalid accuracy for {key}: {accuracy}")
        for column in METRIC_COLUMNS[1:]:
            value = row[column]
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"Invalid probability metric {column} for {key}: {value}")

    for subset in subsets:
        if (EXPERIMENTS["no_cot"], subset, None) not in keys:
            raise ValueError(f"Missing No-CoT table row for {subset}")
        for condition in TOPK_CONDITIONS:
            for ratio in ratios:
                key = (EXPERIMENTS[condition], subset, ratio)
                if key not in keys:
                    raise ValueError(f"Missing table row: {key}")


def validated_micro_values(
    summary: dict[str, Any], path: Path
) -> tuple[int, int, float]:
    for key in ("correct_count", "total", "accuracy"):
        if key not in summary:
            raise KeyError(f"Missing {key!r} in {path}")
    correct_count = summary["correct_count"]
    total = summary["total"]
    reported_accuracy = summary["accuracy"]
    if not isinstance(correct_count, int) or isinstance(correct_count, bool):
        raise TypeError(f"Expected integer correct_count in {path}")
    if not isinstance(total, int) or isinstance(total, bool):
        raise TypeError(f"Expected integer total in {path}")
    if not isinstance(reported_accuracy, (int, float)):
        raise TypeError(f"Expected numeric accuracy in {path}")
    if total <= 0 or not 0 <= correct_count <= total:
        raise ValueError(
            f"Invalid pooled counts in {path}: {correct_count}/{total}"
        )
    micro_accuracy = correct_count / total
    if not math.isclose(
        reported_accuracy,
        micro_accuracy,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Aggregate accuracy mismatch in {path}: reported "
            f"{reported_accuracy}, recomputed {micro_accuracy}"
        )
    return correct_count, total, micro_accuracy


def collect_micro_accuracy_rows(
    result_root: Path,
    data_name: str,
    model_name: str,
    ratios: list[float],
    subset_count: int,
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
    if no_cot_summary.get("aggregation") != "micro":
        raise ValueError(f"Expected micro aggregation in {no_cot_path}")
    if no_cot_summary.get("subset_count") != subset_count:
        raise ValueError(
            f"Subset count mismatch in {no_cot_path}: expected {subset_count}, "
            f"got {no_cot_summary.get('subset_count')!r}"
        )
    no_cot_values = validated_micro_values(no_cot_summary, no_cot_path)

    topk_root = result_root / "topk_leakage" / data_name / model_name
    topk_summaries: dict[str, tuple[Path, dict[str, Any]]] = {}
    for condition in TOPK_CONDITIONS:
        path = topk_root / f"{condition}_aggregate_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Top-k aggregate summary: {path}")
        summary = load_json(path)
        expected_metadata = (condition, data_name, model_name, "micro")
        actual_metadata = (
            summary.get("condition"),
            summary.get("data_name"),
            summary.get("model_name"),
            summary.get("aggregation"),
        )
        if actual_metadata != expected_metadata:
            raise ValueError(
                f"Aggregate metadata mismatch in {path}: expected "
                f"{expected_metadata}, got {actual_metadata}"
            )
        if not isinstance(summary.get("by_ratio"), dict):
            raise TypeError(f"Expected by_ratio object in {path}")
        topk_summaries[condition] = (path, summary)

    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        no_cot_correct, no_cot_total, no_cot_accuracy = no_cot_values
        rows.append(
            {
                "experiment_name": EXPERIMENTS["no_cot"],
                "maintain_ratio": ratio,
                "correct_count": no_cot_correct,
                "total": no_cot_total,
                "micro_accuracy": no_cot_accuracy,
            }
        )
        ratio_key = format_ratio(ratio)
        for condition in TOPK_CONDITIONS:
            path, summary = topk_summaries[condition]
            ratio_summary = summary["by_ratio"].get(ratio_key)
            if not isinstance(ratio_summary, dict):
                raise KeyError(f"Missing ratio {ratio_key!r} in {path}")
            if ratio_summary.get("subset_count") != subset_count:
                raise ValueError(
                    f"Subset count mismatch for ratio {ratio_key} in {path}: "
                    f"expected {subset_count}, got "
                    f"{ratio_summary.get('subset_count')!r}"
                )
            correct_count, total, micro_accuracy = validated_micro_values(
                ratio_summary, path
            )
            rows.append(
                {
                    "experiment_name": EXPERIMENTS[condition],
                    "maintain_ratio": ratio,
                    "correct_count": correct_count,
                    "total": total,
                    "micro_accuracy": micro_accuracy,
                }
            )

    expected_rows = len(ratios) * len(EXPERIMENTS)
    if len(rows) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} micro-accuracy rows, produced {len(rows)}"
        )
    return rows


def write_micro_accuracy_table(
    path: Path, rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MICRO_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def display_subset_name(subset: str) -> str:
    return "\n".join(textwrap.wrap(subset.replace("_", " "), width=27))


def plot_accuracy_grids(
    rows: list[dict[str, Any]],
    subsets: list[str],
    ratios: list[float],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
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
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    experiment_names = list(EXPERIMENTS.values())
    by_key = {
        (row["experiment_name"], row["subset"], row["maintain_ratio"]): row
        for row in rows
    }
    output_paths: list[Path] = []
    for ratio in ratios:
        fig, axes = plt.subplots(
            5,
            4,
            figsize=(14.5, 17.0),
            sharey=True,
            constrained_layout=False,
        )
        flat_axes = axes.ravel()
        for panel_index, subset in enumerate(subsets):
            ax = flat_axes[panel_index]
            panel_rows = []
            for experiment_name in experiment_names:
                row_ratio = None if experiment_name == "No CoT" else ratio
                panel_rows.append(by_key[(experiment_name, subset, row_ratio)])
            values = [row["accuracy"] for row in panel_rows]
            bars = ax.bar(
                range(len(experiment_names)),
                values,
                width=0.72,
                color=[EXPERIMENT_COLORS[name] for name in experiment_names],
                edgecolor="#2F3540",
                linewidth=0.55,
                zorder=3,
            )
            for bar, experiment_name in zip(bars, experiment_names):
                bar.set_hatch(EXPERIMENT_HATCHES[experiment_name])

            ax.bar_label(
                bars,
                labels=[f"{value:.2f}" for value in values],
                padding=2,
                fontsize=5.5,
                color="#30343B",
            )
            ax.set_title(display_subset_name(subset), fontsize=8.3, pad=7)
            ax.set_ylim(0, 1.10)
            ax.set_xticks([])
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#D9DEE7", linewidth=0.6, alpha=0.8)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(axis="y", labelsize=6.5, length=2.5)
            if panel_index % 4 != 0:
                ax.tick_params(labelleft=False)

            if all(row["average_truth_probability_raw"] is None for row in panel_rows):
                ax.text(
                    0.5,
                    0.48,
                    "n = 0",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#9B2C2C",
                    fontweight="bold",
                )

        for panel_index in range(len(subsets), len(flat_axes)):
            flat_axes[panel_index].set_visible(False)

        legend_handles = [
            Patch(
                facecolor=EXPERIMENT_COLORS[name],
                edgecolor="#2F3540",
                linewidth=0.55,
                hatch=EXPERIMENT_HATCHES[name],
                label=name,
            )
            for name in experiment_names
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.968),
            ncol=3,
            fontsize=8.2,
            columnspacing=1.6,
            handlelength=2.5,
        )
        fig.suptitle(
            f"Case-study accuracy by subset — maintain ratio {ratio:.0%}",
            fontsize=13,
            fontweight="bold",
            y=0.992,
        )
        fig.supylabel("Accuracy", x=0.012, fontsize=10)
        fig.text(
            0.5,
            0.008,
            "Bars show subset-level point estimates; No CoT is the fixed baseline.",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4D5663",
        )
        fig.tight_layout(rect=(0.025, 0.022, 0.998, 0.942), h_pad=2.2, w_pad=1.1)

        filename = f"case_study_accuracy_ratio_{int(round(ratio * 100))}pct"
        for file_format in formats:
            output_path = output_dir / f"{filename}.{file_format}"
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
            if file_format == "png":
                save_kwargs["dpi"] = dpi
            fig.savefig(output_path, **save_kwargs)
            output_paths.append(output_path)
        plt.close(fig)
    return output_paths


def plot_micro_accuracy(
    rows: list[dict[str, Any]],
    ratios: list[float],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
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
            "font.size": 9,
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
    figure, axes = plt.subplots(
        1,
        len(ratios),
        figsize=(12.6, 4.9),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for panel_index, ratio in enumerate(ratios):
        axis = flat_axes[panel_index]
        values = [
            by_key[(experiment_name, ratio)]["micro_accuracy"]
            for experiment_name in experiment_names
        ]
        bars = axis.bar(
            range(len(experiment_names)),
            values,
            width=0.72,
            color=[EXPERIMENT_COLORS[name] for name in experiment_names],
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
            fontsize=7.2,
            rotation=90,
            color="#30343B",
        )
        axis.set_title(
            f"{ratio:.0%} retained",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        axis.set_xticks([])
        axis.set_ylim(0, 1.14)
        axis.set_yticks([value / 10 for value in range(0, 11, 2)])
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#D9DEE7", linewidth=0.7, alpha=0.85)
        if panel_index == 0:
            axis.set_ylabel("Micro accuracy")

    figure.suptitle(
        "Micro accuracy across all subsets",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )

    legend_handles = [
        Patch(
            facecolor=EXPERIMENT_COLORS[name],
            edgecolor="#2F3540",
            linewidth=0.6,
            hatch=EXPERIMENT_HATCHES[name],
            label=name,
        )
        for name in experiment_names
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        fontsize=8.3,
        columnspacing=1.5,
        handlelength=2.5,
    )
    figure.text(
        0.5,
        0.012,
        "Micro accuracy = total correct / total samples across subsets; "
        "No CoT is repeated as the fixed baseline.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4D5663",
    )
    figure.tight_layout(rect=(0.025, 0.055, 0.995, 0.80), w_pad=1.5)

    output_paths: list[Path] = []
    filename = "case_study_micro_accuracy_all_subsets"
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
    if any(not 0 <= ratio <= 1 for ratio in args.ratios):
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

    rows, subsets = collect_table_rows(
        result_root=result_root,
        data_name=args.data_name,
        model_name=args.model_name,
        ratios=args.ratios,
    )
    table_path = output_dir / "case_study_experiment_summary.csv"
    write_table(table_path, rows)

    table_rows = read_table(table_path)
    validate_table(table_rows, subsets, args.ratios)

    micro_rows = collect_micro_accuracy_rows(
        result_root=result_root,
        data_name=args.data_name,
        model_name=args.model_name,
        ratios=args.ratios,
        subset_count=len(subsets),
    )
    micro_table_path = output_dir / "case_study_micro_accuracy.csv"
    write_micro_accuracy_table(micro_table_path, micro_rows)

    figure_paths: list[Path] = []
    if not args.skip_figures:
        figure_paths.extend(plot_accuracy_grids(
            rows=table_rows,
            subsets=subsets,
            ratios=args.ratios,
            output_dir=output_dir,
            formats=args.figure_formats,
            dpi=args.dpi,
        ))
        figure_paths.extend(plot_micro_accuracy(
            rows=micro_rows,
            ratios=args.ratios,
            output_dir=output_dir,
            formats=args.figure_formats,
            dpi=args.dpi,
        ))

    empty_metric_rows = sum(
        row["average_truth_probability_raw"] is None for row in table_rows
    )
    print(
        json.dumps(
            {
                "table": str(table_path),
                "micro_accuracy_table": str(micro_table_path),
                "row_count": len(table_rows),
                "micro_accuracy_row_count": len(micro_rows),
                "subset_count": len(subsets),
                "ratios": args.ratios,
                "rows_with_missing_probability_metrics": empty_metric_rows,
                "figures": [str(path) for path in figure_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
