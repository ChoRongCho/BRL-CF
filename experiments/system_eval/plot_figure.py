"""Stage 3: plot condition-comparison figures from a wide CSV.

Usage:
    python3 experiments/system_eval/plot_figure.py --csv experiments/system_eval/data/policy_compare_total.csv
    python3 experiments/system_eval/plot_figure.py --csv all
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
CONDITIONS = ("all", "no", "ours", "random", "knowno_gpt4", "knowno_gpt35turbo")
DOMAINS = ("waste", "tomato")
CONDITION_LABELS = {
    "all": "All",
    "no": "No",
    "ours": "Ours",
    "random": "Random",
    "knowno_gpt4": "KnowNo GPT-4",
    "knowno_gpt35turbo": "KnowNo GPT-3.5",
}
DOMAIN_LABELS = {
    "waste": "Waste",
    "tomato": "Tomato",
}
DOMAIN_COLORS = {
    "waste": "#f2d675",
    "tomato": "#b8a1d9",
}
DOMAIN_ALIASES = {"wastesorting": "waste", "tomato": "tomato"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot condition comparison figures from read_csv_experiment.py output.")
    parser.add_argument("--csv", required=True, help="Wide CSV from read_csv_experiment.py, or 'all' for every CSV in data/.")
    parser.add_argument("--output-root", default="", help="Root figure directory. Defaults to <script_dir>/figure.")
    parser.add_argument("--metric", default="", help="Optional metric filter. By default, all metrics are plotted.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def mean_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return mean(valid) if valid else math.nan


def stdev_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return stdev(valid) if len(valid) >= 2 else math.nan


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def rows_for_outcome(rows: list[dict[str, str]], outcome: str) -> list[dict[str, str]]:
    if outcome == "success":
        return [row for row in rows if is_success(row)]
    if outcome == "failure":
        return [row for row in rows if not is_success(row)]
    return rows


def condition_for_raw_row(row: dict[str, str]) -> str:
    if row.get("experiment") in {"knowno", "when"}:
        return row.get("policy", "")
    return ""


def collect_raw_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for condition in CONDITIONS:
        for domain in DOMAINS:
            grouped[(condition, domain)] = []

    for row in rows:
        condition = condition_for_raw_row(row)
        domain = DOMAIN_ALIASES.get(row.get("domain", ""))
        if condition not in CONDITIONS or domain is None:
            continue
        grouped[(condition, domain)].append(row)
    return grouped


def raw_metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "success_rate":
        return [1.0 if is_success(row) else 0.0 for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row.get("planning_length", "")) for row in rows if is_success(row)]
    if metric == "average_step_failure_only":
        return [as_float(row.get("planning_length", "")) for row in rows if not is_success(row)]
    if metric == "average_question_success_only":
        return [as_float(row.get("question_count", "")) for row in rows if is_success(row)]
    if metric == "average_question_failure_only":
        return [as_float(row.get("question_count", "")) for row in rows if not is_success(row)]
    if metric == "average_reward":
        return [as_float(row.get("reward", "")) for row in rows]
    if metric == "average_reward_success_only":
        return [as_float(row.get("reward", "")) for row in rows if is_success(row)]
    if metric == "average_reward_failure_only":
        return [as_float(row.get("reward", "")) for row in rows if not is_success(row)]
    if metric == "elapsed_time":
        return [as_float(row.get("elapsed_seconds", "")) for row in rows]
    if metric == "elapsed_time_success_only":
        return [as_float(row.get("elapsed_seconds", "")) for row in rows if is_success(row)]
    if metric == "elapsed_time_failure_only":
        return [as_float(row.get("elapsed_seconds", "")) for row in rows if not is_success(row)]
    if metric.startswith("prediction_set_size_when_asked"):
        if metric.endswith("_success_only"):
            rows = rows_for_outcome(rows, "success")
        elif metric.endswith("_failure_only"):
            rows = rows_for_outcome(rows, "failure")
        weighted: list[float] = []
        for row in rows:
            question_count = int(float(row.get("question_count") or 0))
            value = as_float(row.get("average_prediction_set_size_when_asked", ""))
            if question_count > 0 and not math.isnan(value):
                weighted.extend([value] * question_count)
        return weighted
    return []


def raw_query_probability_values(rows: list[dict[str, str]]) -> list[float]:
    query_steps = 0.0
    planning_steps = 0.0
    for row in rows:
        query_step_count = as_float(row.get("query_step_count", ""))
        planning_length = as_float(row.get("planning_length", ""))
        if math.isnan(query_step_count) or math.isnan(planning_length) or planning_length <= 0:
            continue
        query_steps += query_step_count
        planning_steps += planning_length
    if planning_steps <= 0:
        return []
    return [query_steps / planning_steps]


def raw_values_for_table(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "query_probability_per_step":
        return raw_query_probability_values(rows)
    if metric == "query_probability_per_step_success_only":
        return raw_query_probability_values(rows_for_outcome(rows, "success"))
    if metric == "query_probability_per_step_failure_only":
        return raw_query_probability_values(rows_for_outcome(rows, "failure"))
    return raw_metric_values(rows, metric)


def load_raw_grouped_rows() -> dict[tuple[str, str], list[dict[str, str]]]:
    raw_path = SCRIPT_DIR / "data" / "raw_runs" / "domain_compare" / "raw_runs.csv"
    if not raw_path.exists():
        return {}
    return collect_raw_rows(read_rows(raw_path))


def make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def resolve_csv_paths(csv_arg: str) -> list[Path]:
    if csv_arg == "all":
        return sorted((SCRIPT_DIR / "data").glob("*.csv"))
    return [Path(csv_arg)]


def row_has_enough_data(row: dict[str, str]) -> bool:
    values = [
        as_float(row.get(f"{condition}_{domain}", ""))
        for condition in CONDITIONS
        for domain in DOMAINS
    ]
    return sum(not math.isnan(value) for value in values) >= 2


def figure_name(metric: str, prefix: str = "") -> str:
    return f"{prefix}_{metric}" if prefix else metric


def plot_metric(row: dict[str, str], output_dir: Path, prefix: str = "") -> str:
    metric = row["metric"]
    metric_label = row.get("metric_label") or metric
    positions = list(range(len(CONDITIONS)))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for domain_index, domain in enumerate(DOMAINS):
        offset = (domain_index - 0.5) * bar_width
        values = [as_float(row.get(f"{condition}_{domain}", "")) for condition in CONDITIONS]
        plot_values = [0.0 if math.isnan(value) else value for value in values]
        bars = ax.bar(
            [position + offset for position in positions],
            plot_values,
            width=bar_width,
            color=DOMAIN_COLORS[domain],
            edgecolor="#1e293b",
            linewidth=0.8,
            label=DOMAIN_LABELS[domain],
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
                fontsize=8,
                color="#334155",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([CONDITION_LABELS[condition] for condition in CONDITIONS])
    ax.set_xlabel("condition")
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.legend(frameon=False)
    style_axis(ax)
    filename = figure_name(metric, prefix)
    save_figure(fig, output_dir / filename)
    return filename


def build_table_rows(
    plot_row: dict[str, str],
    raw_grouped: dict[tuple[str, str], list[dict[str, str]]],
    figure: str,
) -> list[dict[str, str]]:
    metric = plot_row["metric"]
    metric_label = plot_row.get("metric_label") or metric
    rows: list[dict[str, str]] = []
    for condition in CONDITIONS:
        for domain in DOMAINS:
            values = raw_values_for_table(raw_grouped.get((condition, domain), []), metric)
            valid = [value for value in values if not math.isnan(value)]
            plotted_value = as_float(plot_row.get(f"{condition}_{domain}", ""))
            rows.append({
                "figure": figure,
                "metric": metric,
                "metric_label": metric_label,
                "condition": condition,
                "condition_label": CONDITION_LABELS[condition],
                "domain": domain,
                "domain_label": DOMAIN_LABELS[domain],
                "n": str(len(valid)),
                "mean": fmt(mean_or_nan(valid)),
                "min": fmt(min(valid) if valid else math.nan),
                "max": fmt(max(valid) if valid else math.nan),
                "std": fmt(stdev_or_nan(valid)),
                "plotted_value": fmt(plotted_value),
            })
    return rows


def write_figure_table(output_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = [
        "figure",
        "metric",
        "metric_label",
        "condition",
        "condition_label",
        "domain",
        "domain_label",
        "n",
        "mean",
        "min",
        "max",
        "std",
        "plotted_value",
    ]
    path = output_dir / "figure_tables.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")


def save_figure(fig, output_stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_paths = resolve_csv_paths(args.csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found for --csv {args.csv}")

    output_root = Path(args.output_root) if args.output_root else SCRIPT_DIR / "figure"
    output_dir = make_run_dir(output_root)
    use_prefix = len(csv_paths) > 1
    raw_grouped = load_raw_grouped_rows()
    table_rows: list[dict[str, str]] = []

    for csv_path in csv_paths:
        rows = read_rows(csv_path)
        if args.metric:
            rows = [row for row in rows if row["metric"] == args.metric]
        rows = [row for row in rows if row_has_enough_data(row)]
        prefix = csv_path.stem if use_prefix else ""
        for row in rows:
            figure = plot_metric(row, output_dir, prefix)
            table_rows.extend(build_table_rows(row, raw_grouped, figure))

    write_figure_table(output_dir, table_rows)


if __name__ == "__main__":
    main()
