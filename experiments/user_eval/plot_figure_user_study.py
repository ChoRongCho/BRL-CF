"""Stage 4: plot user-study figures from final CSV files.

Usage:
    python3 experiments/user_eval/plot_figure_user_study.py
    python3 experiments/user_eval/plot_figure_user_study.py --metric "조작성공률 (%)"

Inputs:
    experiments/user_eval/data/figure_user_study/user_study_metric_summary.csv
    experiments/user_eval/data/figure_user_study/user_study_significance_pairs.csv

Outputs:
    experiments/user_eval/figure/user_study/00_YYYYMMDD_HHMMSS/*.png
    experiments/user_eval/figure/user_study/00_YYYYMMDD_HHMMSS/*.pdf
    experiments/user_eval/figure/user_study/00_YYYYMMDD_HHMMSS/figure_tables.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "figure_user_study"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figure" / "user_study"

CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
CONDITION_LABELS = {
    "C1": "All",
    "C2": "No",
    "C3": "Ours1",
    "C4": "Ours2",
    "C5": "KnowNo",
}
NO_PLOT_METRICS = {"조작성공률 (%)", "조작시간 (s)"}
METRIC_LABELS = {
    "조작성공률 (%)": "Task Success Rate (%)",
    "조작시간 (s)": "Operation Time (s)",
    "SAGAT1": "SAGAT Level 1",
    "SAGAT2": "SAGAT Level 2",
    "SAGAT3": "SAGAT Level 3",
    "Fatigue": "Fatigue",
    "NASA-RTLX": "NASA-RTLX",
    "SART": "SART",
}
DOMAIN_LABELS = {"Waste": "Waste", "Tomato": "Tomato"}
BAR_COLORS = {
    "Waste": "#f2d675",
    "Tomato": "#b8a1d9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot user-study figures from read_csv_user_study.py output.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metric", default="", help="Optional metric filter.")
    parser.add_argument("--domain", default="", choices=("", "Waste", "Tomato"), help="Optional domain filter.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def make_output_dir(root: Path) -> Path:
    output_dir = root / datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def metric_slug(metric: str) -> str:
    return (
        metric.lower()
        .replace(" (%)", "")
        .replace(" (s)", "")
        .replace("조작성공률", "task_success")
        .replace("조작시간", "operation_time")
        .replace("-", "_")
    )


def plotted_conditions_for(metric: str) -> tuple[str, ...]:
    if metric in NO_PLOT_METRICS:
        return tuple(condition for condition in CONDITIONS if condition != "C2")
    return CONDITIONS


def annotate_significance(
    ax: plt.Axes,
    pairs: list[dict[str, str]],
    values: list[float],
    errors: list[float],
    plotted_conditions: tuple[str, ...],
) -> None:
    plotted_set = set(plotted_conditions)
    significant = [
        row
        for row in pairs
        if row.get("significant_p_lt_05") == "yes"
        and row.get("stars")
        and row.get("condition_a") in plotted_set
        and row.get("condition_b") in plotted_set
    ]
    if not significant:
        return

    finite_tops = [
        value + (0.0 if math.isnan(error) else error)
        for value, error in zip(values, errors)
        if not math.isnan(value)
    ]
    if not finite_tops:
        return
    y_min, y_max = ax.get_ylim()
    data_span = max(y_max - y_min, 1.0)
    base = max(finite_tops) + data_span * 0.06
    height = data_span * 0.025
    step = data_span * 0.075

    condition_to_x = {condition: index for index, condition in enumerate(plotted_conditions)}
    for level, row in enumerate(significant[:6]):
        left = condition_to_x.get(row["condition_a"])
        right = condition_to_x.get(row["condition_b"])
        if left is None or right is None:
            continue
        y = base + level * step
        ax.plot([left, left, right, right], [y, y + height, y + height, y], color="#2f3b52", linewidth=1.5)
        ax.text((left + right) / 2, y + height, row["stars"], ha="center", va="bottom", fontsize=17, color="#2f3b52")

    ax.set_ylim(y_min, base + len(significant[:6]) * step + data_span * 0.08)


def plot_one(
    output_dir: Path,
    figure_group: str,
    rows: list[dict[str, str]],
    pairs: list[dict[str, str]],
) -> list[dict[str, str]]:
    domain = rows[0]["domain"]
    metric = rows[0]["metric"]
    by_condition = {row["condition"]: row for row in rows}
    plotted_conditions = plotted_conditions_for(metric)

    values = [as_float(by_condition.get(condition, {}).get("mean", "")) for condition in plotted_conditions]
    errors = [as_float(by_condition.get(condition, {}).get("se", "")) for condition in plotted_conditions]
    ns = [by_condition.get(condition, {}).get("n", "0") for condition in plotted_conditions]
    labels = [CONDITION_LABELS[condition] for condition in plotted_conditions]

    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    x_positions = list(range(len(plotted_conditions)))
    color = BAR_COLORS.get(domain, "#8fb8de")
    bars = ax.bar(
        x_positions,
        values,
        yerr=[0.0 if math.isnan(error) else error for error in errors],
        width=0.72,
        color=color,
        edgecolor="#1f2937",
        linewidth=1.8,
        capsize=6,
    )

    for bar, value in zip(bars, values):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#2f3b52",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=24)
    ax.set_title(f"{DOMAIN_LABELS.get(domain, domain)} - {METRIC_LABELS.get(metric, metric)}", fontsize=28, pad=18)
    ax.grid(axis="y", color="#d7dee8", linewidth=1.5, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa8bd")
    ax.spines["bottom"].set_color("#9aa8bd")
    ax.spines["left"].set_linewidth(1.6)
    ax.spines["bottom"].set_linewidth(1.6)

    annotate_significance(ax, pairs, values, errors, plotted_conditions)
    fig.tight_layout()

    output_path = output_dir / f"{figure_group}.png"
    pdf_output_path = output_dir / f"{figure_group}.pdf"
    fig.savefig(output_path, dpi=200)
    fig.savefig(pdf_output_path)
    plt.close(fig)

    table_rows: list[dict[str, str]] = []
    for condition, value, error, n in zip(plotted_conditions, values, errors, ns):
        source = by_condition.get(condition, {})
        table_rows.append(
            {
                "figure": output_path.name,
                "figure_group": figure_group,
                "domain": domain,
                "metric": metric,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "n": n,
                "mean": "" if math.isnan(value) else f"{value:.6g}",
                "se": "" if math.isnan(error) else f"{error:.6g}",
                "sd": source.get("sd", ""),
                "median": source.get("median", ""),
                "min": source.get("min", ""),
                "max": source.get("max", ""),
            }
        )
    return table_rows


def main() -> None:
    args = parse_args()
    summary_rows = read_rows(args.data_dir / "user_study_metric_summary.csv")
    pair_rows = read_rows(args.data_dir / "user_study_significance_pairs.csv")

    if args.metric:
        summary_rows = [row for row in summary_rows if row["metric"] == args.metric]
        pair_rows = [row for row in pair_rows if row["metric"] == args.metric]
    if args.domain:
        summary_rows = [row for row in summary_rows if row["domain"] == args.domain]
        pair_rows = [row for row in pair_rows if row["domain"] == args.domain]

    groups = sorted({row["figure_group"] for row in summary_rows})
    output_dir = make_output_dir(args.output_root)
    table_rows: list[dict[str, str]] = []

    for group in groups:
        rows = [row for row in summary_rows if row["figure_group"] == group]
        rows.sort(key=lambda row: CONDITIONS.index(row["condition"]))
        pairs = [row for row in pair_rows if row["figure_group"] == group]
        table_rows.extend(plot_one(output_dir, group, rows, pairs))

    table_path = output_dir / "figure_tables.csv"
    with table_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "figure",
            "figure_group",
            "domain",
            "metric",
            "condition",
            "condition_label",
            "n",
            "mean",
            "se",
            "sd",
            "median",
            "min",
            "max",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"output_dir: {output_dir}")
    print(f"figures: {len(groups)}")
    print(f"figure_table: {table_path}")


if __name__ == "__main__":
    main()
