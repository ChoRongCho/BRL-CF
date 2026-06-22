"""Plot threshold sweep results from normalized raw run CSV.

Usage:
    python3 experiments/system_eval/analysis_experiment.py
    python3 experiments/system_eval/plot_threshold_figure.py
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "raw_runs" / "domain_compare" / "raw_runs.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figure" / "threshold"

# Edit this block to change figure design.
PLOT_STYLE = {
    "figure_size": (5.5, 4),
    "figure_facecolor": "#f8fafc",
    "axis_facecolor": "#ffffff",
    "domain_colors": {
        "tomato": "#b8a1d9",
        "wastesorting": "#f2d675",
        # "all": "#64748b",
    },
    "fallback_line_color": "#334155",
    "line_width": 3.0,
    "marker": "s",
    "marker_size": 8,
    "x_tick_fontsize": 15,
    "y_tick_fontsize": 13,
    "y_label_fontsize": 15,
    "legend_fontsize": 14,
    "legend_location": "lower center",
    "legend_bbox_to_anchor": (0.5, 1.02),    "legend_edge_color": "#1e293b",
    "grid_color": "#cbd5e1",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.8,
    "spine_color": "#94a3b8",
    "save_dpi": 200,
    "save_formats": (".png", ".pdf"),
}

DOMAIN_LABELS = {
    "tomato": "Tomato",
    "wastesorting": "Waste",
    "all": "All",
}
METRICS = {
    "success_rate": "Success Rate",
    "average_step": "Average Step",
    "average_step_success_only": "Average Step",
    "average_question": "Average Query Number",
    "average_question_success_only": "Average Query Number",
    "query_probability_per_step": "Query Probability per Step",
    "average_reward": "Average Reward",
    "elapsed_time": "Elapsed Time",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot threshold sweep figures from raw threshold runs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="raw_runs/domain_compare/raw_runs.csv path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory where figures are written.")
    parser.add_argument("--metric", default="", help="Optional metric filter. By default, all metrics are plotted.")
    parser.add_argument("--include-all", action="store_true", help="Also plot the combined all-domain line.")
    parser.add_argument("--test", nargs="?", const=True, default=False, type=parse_bool, help="Save figures under 00_test.")
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


def as_float(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def valid_values(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def mean_or_nan(values: list[float]) -> float:
    values = valid_values(values)
    return mean(values) if values else math.nan


def stdev_or_nan(values: list[float]) -> float:
    values = valid_values(values)
    return stdev(values) if len(values) >= 2 else math.nan


def fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "success_rate":
        return [1.0 if is_success(row) else 0.0 for row in rows]
    if metric == "average_step":
        return [as_float(row.get("planning_length")) for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row.get("planning_length")) for row in rows if is_success(row)]
    if metric == "average_question":
        return [as_float(row.get("question_count")) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row.get("question_count")) for row in rows if is_success(row)]
    if metric == "query_probability_per_step":
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
    if metric == "average_reward":
        return [as_float(row.get("reward")) for row in rows]
    if metric == "elapsed_time":
        return [as_float(row.get("elapsed_seconds")) for row in rows]
    raise ValueError(f"unknown metric: {metric}")


def threshold_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("experiment") == "threshold" and not math.isnan(as_float(row.get("threshold")))
    ]


def grouped_rows(rows: list[dict[str, str]], include_all: bool) -> dict[tuple[str, float], list[dict[str, str]]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        threshold = as_float(row.get("threshold"))
        domain = row.get("domain", "")
        grouped[(domain, threshold)].append(row)
        if include_all:
            grouped[("all", threshold)].append(row)
    return grouped


def make_run_dir(output_root: Path, test: bool = False) -> Path:
    if test:
        output_dir = output_root / "00_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    timestamp = datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_summary(rows: list[dict[str, str]], include_all: bool) -> list[dict[str, str]]:
    grouped = grouped_rows(rows, include_all)
    summary: list[dict[str, str]] = []
    for metric, label in METRICS.items():
        for (domain, threshold), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
            values = valid_values(metric_values(group, metric))
            summary.append({
                "metric": metric,
                "metric_label": label,
                "domain": domain,
                "domain_label": DOMAIN_LABELS.get(domain, domain),
                "threshold": f"{threshold:.1f}",
                "n": str(len(group)),
                "valid_n": str(len(values)),
                "mean": fmt(mean_or_nan(values)),
                "std": fmt(stdev_or_nan(values)),
                "min": fmt(min(values) if values else math.nan),
                "max": fmt(max(values) if values else math.nan),
            })
    return summary


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["metric", "metric_label", "domain", "domain_label", "threshold", "n", "valid_n", "mean", "std", "min", "max"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


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


def use_percent_scale(metric: str) -> bool:
    return metric.endswith("_rate")


def plot_value(value: float, metric: str) -> float:
    if math.isnan(value):
        return value
    return value * 100.0 if use_percent_scale(metric) else value


def plot_label(metric: str) -> str:
    label = METRICS[metric]
    return f"{label} (%)" if use_percent_scale(metric) else label


def plot_metric(summary: list[dict[str, str]], metric: str, output_dir: Path) -> None:
    metric_rows = [row for row in summary if row["metric"] == metric and row["mean"]]
    if not metric_rows:
        return

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"], constrained_layout=True)
    fig.patch.set_facecolor(PLOT_STYLE["figure_facecolor"])
    ax.set_facecolor(PLOT_STYLE["axis_facecolor"])

    domains = [domain for domain in ("wastesorting", "tomato", "all") if any(row["domain"] == domain for row in metric_rows)]
    for domain in domains:
        rows = sorted((row for row in metric_rows if row["domain"] == domain), key=lambda row: as_float(row["threshold"]))
        xs = [as_float(row["threshold"]) for row in rows]
        ys = [plot_value(as_float(row["mean"]), metric) for row in rows]
        ax.plot(
            xs,
            ys,
            marker=PLOT_STYLE["marker"],
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            color=PLOT_STYLE["domain_colors"].get(domain, PLOT_STYLE["fallback_line_color"]),
            label=DOMAIN_LABELS.get(domain, domain),
        )

    ax.set_xticks([index / 10 for index in range(11)])
    ax.set_xticklabels([f".{index}" if index < 10 else "1.0" for index in range(11)])
    ax.set_xlabel("")
    ax.set_ylabel(plot_label(metric), fontsize=PLOT_STYLE["y_label_fontsize"])
    ax.tick_params(axis="x", labelsize=PLOT_STYLE["x_tick_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_STYLE["y_tick_fontsize"])
    ax.legend(
        loc=PLOT_STYLE["legend_location"],
        bbox_to_anchor=PLOT_STYLE["legend_bbox_to_anchor"],
        ncol=len(domains),
        fontsize=PLOT_STYLE["legend_fontsize"],
        frameon=True,
        fancybox=False,
        edgecolor=PLOT_STYLE["legend_edge_color"],
        borderaxespad=0.0,
    )
    style_axis(ax)

    for suffix in PLOT_STYLE["save_formats"]:
        path = output_dir / f"{metric}{suffix}"
        fig.savefig(path, dpi=PLOT_STYLE["save_dpi"], bbox_inches="tight")
        print(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.test = True
    
    metrics = [args.metric] if args.metric else list(METRICS)
    unknown_metrics = [metric for metric in metrics if metric not in METRICS]
    if unknown_metrics:
        raise ValueError(f"Unknown metric(s): {', '.join(unknown_metrics)}")

    rows = threshold_rows(read_rows(Path(args.input)))
    if not rows:
        raise ValueError(f"No threshold rows found in {args.input}")

    output_dir = make_run_dir(Path(args.output_root), args.test)
    summary = build_summary(rows, include_all=args.include_all)
    write_summary(output_dir / "threshold_summary.csv", summary)
    for metric in metrics:
        plot_metric(summary, metric, output_dir)


if __name__ == "__main__":
    main()
