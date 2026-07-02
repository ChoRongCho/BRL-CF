"""Stage 3 for KnowNo qhat comparison: plot figures and table CSV.

Usage:
    python3 experiments/system_eval/plot_knowno_figure.py \
        --csv experiments/system_eval/data/knowno/knowno_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "data" / "knowno" / "knowno_compare.csv"
DEFAULT_RAW_CSV = SCRIPT_DIR / "data" / "knowno" / "raw_runs.csv"

CONDITIONS = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt35turbo_raw98",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "gpt4_raw98",
    "ours",
)
DOMAINS = ("tomato", "wastesorting")
DOMAIN_LABELS = {
    "tomato": "Tomato",
    "wastesorting": "Waste Sorting",
}
CONDITION_LABELS = {
    "gpt4_raw98": "GPT-4o\nraw98",
    "gpt35turbo_raw98": "GPT-3.5\nraw98",
    "ours": "Ours",
}
QHAT_LABELS = {
    "tomato": {
        "gpt35turbo_75": "GPT-3.5\n75%\n(0.8938)",
        "gpt35turbo_85": "GPT-3.5\n85%\n(0.9082)",
        "gpt35turbo_95": "GPT-3.5\n95%\n(0.9243)",
        "gpt4_75": "GPT-4o\n75%\n(0.7322)",
        "gpt4_85": "GPT-4o\n85%\n(0.7779)",
        "gpt4_95": "GPT-4o\n95%\n(0.8404)",
    },
    "wastesorting": {
        "gpt35turbo_75": "GPT-3.5\n75%\n(0.8512)",
        "gpt35turbo_85": "GPT-3.5\n85%\n(0.8851)",
        "gpt35turbo_95": "GPT-3.5\n95%\n(0.9028)",
        "gpt4_75": "GPT-4o\n75%\n(0.7084)",
        "gpt4_85": "GPT-4o\n85%\n(0.7369)",
        "gpt4_95": "GPT-4o\n95%\n(0.8704)",
    },
}

# Matches the visual settings used by plot_figure.py, adjusted for one-series bars.
PLOT_STYLE = {
    "figure_size": (7.6, 4.25),
    "figure_facecolor": "#f8fafc",
    "axis_facecolor": "#ffffff",
    "bar_colors": {
        "gpt35turbo": "#3b82c4",
        "gpt4": "#e76f51",
        "ours": "#2a9d8f",
    },
    "bar_edge_color": "#1e293b",
    "bar_edge_width": 0.8,
    "bar_width": 0.56,
    "value_label_fontsize": 11,
    "value_label_color": "#334155",
    "x_tick_fontsize": 10,
    "y_tick_fontsize": 13,
    "y_label_fontsize": 15,
    "grid_color": "#cbd5e1",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.8,
    "spine_color": "#94a3b8",
    "save_dpi": 200,
    "save_formats": (".png", ".pdf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KnowNo qhat comparison figures.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV from read_csv_knowno_experiment.py.")
    parser.add_argument("--raw-csv", default=str(DEFAULT_RAW_CSV), help="Raw run CSV for figure_tables.csv statistics.")
    parser.add_argument("--output-root", default="", help="Root figure directory. Defaults to <script_dir>/figure/knowno.")
    parser.add_argument("--output-dir", default="", help="Exact output directory. Overrides --output-root and --test.")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition to plot. Can be passed multiple times. Defaults to every condition in the CSV.",
    )
    parser.add_argument("--simple-labels", action="store_true", help="Hide calibrated qhat values in x-axis labels.")
    parser.add_argument("--metric", default="", help="Optional metric filter.")
    parser.add_argument("--test", nargs="?", const=True, default=False, type=parse_bool, help="Save under 00_test.")
    return parser.parse_args()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str | None) -> float:
    if value in {"", None}:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def mean_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return mean(valid) if valid else math.nan


def stdev_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return stdev(valid) if len(valid) >= 2 else math.nan


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def use_percent_scale(metric: str) -> bool:
    return metric.endswith("_rate")


def plot_value(value: float, metric: str) -> float:
    if math.isnan(value):
        return value
    return value * 100.0 if use_percent_scale(metric) else value


def plot_label(metric_label: str, metric: str) -> str:
    label = (
        metric_label
        .replace(" (Success Only)", "")
        .replace(" (Failure Only)", "")
        .replace(" (All)", "")
    )
    return f"{label} (%)" if use_percent_scale(metric) else label


def value_label(value: float, metric: str) -> str:
    if math.isnan(value):
        return ""
    if metric.startswith("query_probability_per_step"):
        label = f"{value:.2f}"
        return label[1:] if 0.0 <= value < 1.0 else label
    return f"{value:.1f}"


def condition_group(condition: str) -> str:
    if condition.startswith("ours"):
        return "ours"
    return "gpt35turbo" if condition.startswith("gpt35") else "gpt4"


def condition_label(domain: str, condition: str, simple: bool = False) -> str:
    if simple:
        if condition.startswith("gpt35turbo_"):
            return f"GPT-3.5\n{condition.rsplit('_', 1)[1]}%"
        if condition.startswith("gpt4_"):
            return f"GPT-4o\n{condition.rsplit('_', 1)[1]}%"
    if condition in QHAT_LABELS.get(domain, {}):
        return QHAT_LABELS[domain][condition]
    return CONDITION_LABELS[condition]


def conditions_from_row(row: dict[str, str], domain: str) -> tuple[str, ...]:
    return tuple(condition for condition in CONDITIONS if f"{domain}_{condition}" in row)


def selected_conditions(row: dict[str, str], domain: str, requested: list[str]) -> tuple[str, ...]:
    if not requested:
        return conditions_from_row(row, domain)
    missing = [condition for condition in requested if f"{domain}_{condition}" not in row]
    if missing:
        raise ValueError(f"Missing columns for {domain}: {', '.join(missing)}")
    return tuple(requested)


def row_has_enough_data(row: dict[str, str], domain: str, conditions: tuple[str, ...]) -> bool:
    values = [as_float(row.get(f"{domain}_{condition}")) for condition in conditions]
    return sum(not math.isnan(value) for value in values) >= 2


def make_run_dir(output_root: Path, test: bool = False) -> Path:
    if test:
        output_dir = output_root / "00_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def style_axis(ax) -> None:
    ax.grid(
        axis="y",
        color=PLOT_STYLE["grid_color"],
        alpha=PLOT_STYLE["grid_alpha"],
        linewidth=PLOT_STYLE["grid_linewidth"],
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT_STYLE["spine_color"])
    ax.spines["bottom"].set_color(PLOT_STYLE["spine_color"])


def save_figure(fig, output_stem: Path) -> None:
    for suffix in PLOT_STYLE["save_formats"]:
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=PLOT_STYLE["save_dpi"], bbox_inches="tight")
        print(path)
    plt.close(fig)


def plot_metric(
    row: dict[str, str],
    domain: str,
    conditions: tuple[str, ...],
    output_dir: Path,
    simple_labels: bool = False,
) -> str:
    metric = row["metric"]
    metric_label = row.get("metric_label") or metric
    values = [plot_value(as_float(row.get(f"{domain}_{condition}")), metric) for condition in conditions]
    plot_values = [0.0 if math.isnan(value) else value for value in values]
    colors = [PLOT_STYLE["bar_colors"][condition_group(condition)] for condition in conditions]

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"], constrained_layout=True)
    fig.patch.set_facecolor(PLOT_STYLE["figure_facecolor"])
    ax.set_facecolor(PLOT_STYLE["axis_facecolor"])
    bars = ax.bar(
        range(len(conditions)),
        plot_values,
        width=PLOT_STYLE["bar_width"],
        color=colors,
        edgecolor=PLOT_STYLE["bar_edge_color"],
        linewidth=PLOT_STYLE["bar_edge_width"],
    )
    for bar, value in zip(bars, values):
        if math.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_label(value, metric),
            ha="center",
            va="bottom",
            fontsize=PLOT_STYLE["value_label_fontsize"],
            color=PLOT_STYLE["value_label_color"],
        )

    ax.set_xticks(list(range(len(conditions))))
    ax.set_xticklabels(
        [condition_label(domain, condition, simple_labels) for condition in conditions],
        fontsize=PLOT_STYLE["x_tick_fontsize"],
    )
    ax.set_ylabel(plot_label(metric_label, metric), fontsize=PLOT_STYLE["y_label_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_STYLE["y_tick_fontsize"])
    if metric == "success_rate":
        ax.set_ylim(0, 100)
    style_axis(ax)
    filename = f"{domain}_{metric}"
    save_figure(fig, output_dir / filename)
    return filename


def raw_metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "success_rate":
        return [1.0 if is_success(row) else 0.0 for row in rows]
    if metric == "average_step":
        return [as_float(row.get("planning_length")) for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row.get("planning_length")) for row in rows if is_success(row)]
    if metric == "average_step_failure_only":
        return [as_float(row.get("planning_length")) for row in rows if not is_success(row)]
    if metric == "average_question":
        return [as_float(row.get("question_count")) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row.get("question_count")) for row in rows if is_success(row)]
    if metric == "average_question_failure_only":
        return [as_float(row.get("question_count")) for row in rows if not is_success(row)]
    if metric == "elapsed_time":
        return [as_float(row.get("elapsed_seconds")) for row in rows]
    if metric == "token_overall":
        return [as_float(row.get("token_overall")) for row in rows]
    if metric == "prediction_set_size_when_asked":
        weighted: list[float] = []
        for row in rows:
            question_count = int(float(row.get("question_count") or 0))
            value = as_float(row.get("average_prediction_set_size_when_asked"))
            if question_count > 0 and not math.isnan(value):
                weighted.extend([value] * question_count)
        return weighted
    return []


def raw_query_probability_values(rows: list[dict[str, str]]) -> list[float]:
    query_steps = 0.0
    planning_steps = 0.0
    for row in rows:
        query_step_count = as_float(row.get("query_step_count"))
        planning_length = as_float(row.get("planning_length"))
        if math.isnan(query_step_count) or math.isnan(planning_length) or planning_length <= 0:
            continue
        query_steps += query_step_count
        planning_steps += planning_length
    return [query_steps / planning_steps] if planning_steps > 0 else []


def raw_values_for_table(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "query_probability_per_step":
        return raw_query_probability_values(rows)
    return raw_metric_values(rows, metric)


def load_raw_grouped(path: Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in read_rows(path):
        grouped.setdefault((row.get("domain", ""), row.get("condition", "")), []).append(row)
    return grouped


def build_table_rows(
    plot_row: dict[str, str],
    raw_grouped: dict[tuple[str, str], list[dict[str, str]]],
    figure: str,
    domain: str,
    conditions: tuple[str, ...],
    simple_labels: bool = False,
) -> list[dict[str, str]]:
    metric = plot_row["metric"]
    metric_label = plot_row.get("metric_label") or metric
    rows: list[dict[str, str]] = []
    for condition in conditions:
        values = raw_values_for_table(raw_grouped.get((domain, condition), []), metric)
        valid = [value for value in values if not math.isnan(value)]
        plotted_value = as_float(plot_row.get(f"{domain}_{condition}"))
        rows.append({
            "figure": figure,
            "metric": metric,
            "metric_label": metric_label,
            "domain": domain,
            "domain_label": DOMAIN_LABELS[domain],
            "condition": condition,
            "condition_label": condition_label(domain, condition, simple_labels).replace("\n", " "),
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
        "domain",
        "domain_label",
        "condition",
        "condition_label",
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


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.csv))
    if args.metric:
        rows = [row for row in rows if row["metric"] == args.metric]
    if not rows:
        raise ValueError("No rows to plot.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_root = Path(args.output_root) if args.output_root else SCRIPT_DIR / "figure" / "knowno"
        output_dir = make_run_dir(output_root, args.test)
    raw_grouped = load_raw_grouped(Path(args.raw_csv))
    table_rows: list[dict[str, str]] = []

    for domain in DOMAINS:
        conditions = selected_conditions(rows[0], domain, args.condition)
        domain_rows = [row for row in rows if row_has_enough_data(row, domain, conditions)]
        for row in domain_rows:
            figure = plot_metric(row, domain, conditions, output_dir, args.simple_labels)
            table_rows.extend(build_table_rows(row, raw_grouped, figure, domain, conditions, args.simple_labels))

    write_figure_table(output_dir, table_rows)


if __name__ == "__main__":
    main()
