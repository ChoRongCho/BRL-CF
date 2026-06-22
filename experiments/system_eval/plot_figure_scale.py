"""Plot scalability comparison figures from scale_compare.csv.

Usage:
    python3 experiments/system_eval/plot_figure_scale.py
    python3 experiments/system_eval/plot_figure_scale.py --csv experiments/system_eval/data/scale_compare.csv
    python3 experiments/system_eval/plot_figure_scale.py --metric success_rate
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
DEFAULT_CSV = SCRIPT_DIR / "data" / "scale_compare.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figure" / "scale"
CONDITIONS = ("original", "scaled")
DOMAINS = ("waste", "tomato")

# Edit this block to change figure design.
PLOT_STYLE = {
    "figure_size": (7.2, 5.2),
    "figure_facecolor": "#f8fafc",
    "axis_facecolor": "#ffffff",
    "domain_colors": {
        "waste": "#f2d675",
        "tomato": "#b8a1d9",
    },
    "bar_edge_color": "#1e293b",
    "bar_edge_width": 0.8,
    "bar_width": 0.32,
    "group_gap": 0.82,
    "value_label_fontsize": 12,
    "value_label_color": "#334155",
    "x_tick_fontsize": 16,
    "y_tick_fontsize": 14,
    "y_label_fontsize": 16,
    "legend_fontsize": 14,
    "legend_location": "upper right",
    "grid_color": "#cbd5e1",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.8,
    "spine_color": "#94a3b8",
    "save_dpi": 200,
    "save_formats": (".png", ".pdf"),
}

CONDITION_LABELS = {
    "original": "Original",
    "scaled": "Scaled",
}
DOMAIN_LABELS = {
    "waste": "Waste",
    "tomato": "Tomato",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scalability comparison figures.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="scale_compare.csv path.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory where figures are written.")
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


def use_percent_scale(metric: str) -> bool:
    return metric.endswith("_rate")


def plot_value(value: float, metric: str) -> float:
    if math.isnan(value):
        return value
    return value * 100.0 if use_percent_scale(metric) else value


def plot_label(metric_label: str, metric: str) -> str:
    return f"{metric_label} (%)" if use_percent_scale(metric) else metric_label


def make_run_dir(output_root: Path) -> Path:
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


def row_has_data(row: dict[str, str]) -> bool:
    values = [
        as_float(row.get(f"{condition}_{domain}", ""))
        for condition in CONDITIONS
        for domain in DOMAINS
    ]
    return any(not math.isnan(value) for value in values)


def write_scale_compare(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = ["metric", "metric_label"]
    for condition in CONDITIONS:
        for domain in DOMAINS:
            fields.append(f"{condition}_{domain}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    print(path)


def plot_metric(row: dict[str, str], output_dir: Path) -> None:
    metric = row["metric"]
    metric_label = row.get("metric_label") or metric
    group_gap = PLOT_STYLE["group_gap"]
    positions = [index * group_gap for index in range(len(CONDITIONS))]
    bar_width = PLOT_STYLE["bar_width"]

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"], constrained_layout=True)
    fig.patch.set_facecolor(PLOT_STYLE["figure_facecolor"])
    ax.set_facecolor(PLOT_STYLE["axis_facecolor"])

    for domain_index, domain in enumerate(DOMAINS):
        offset = (domain_index - 0.5) * bar_width
        values = [plot_value(as_float(row.get(f"{condition}_{domain}", "")), metric) for condition in CONDITIONS]
        plot_values = [0.0 if math.isnan(value) else value for value in values]
        bars = ax.bar(
            [position + offset for position in positions],
            plot_values,
            width=bar_width,
            color=PLOT_STYLE["domain_colors"][domain],
            edgecolor=PLOT_STYLE["bar_edge_color"],
            linewidth=PLOT_STYLE["bar_edge_width"],
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
                fontsize=PLOT_STYLE["value_label_fontsize"],
                color=PLOT_STYLE["value_label_color"],
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [CONDITION_LABELS[condition] for condition in CONDITIONS],
        fontsize=PLOT_STYLE["x_tick_fontsize"],
    )
    ax.set_xlabel("")
    ax.set_ylabel(plot_label(metric_label, metric), fontsize=PLOT_STYLE["y_label_fontsize"])
    ax.tick_params(axis="y", labelsize=PLOT_STYLE["y_tick_fontsize"])
    ax.legend(
        loc=PLOT_STYLE["legend_location"],
        ncol=len(DOMAINS),
        fontsize=PLOT_STYLE["legend_fontsize"],
        frameon=True,
        fancybox=False,
        edgecolor=PLOT_STYLE["bar_edge_color"],
    )
    style_axis(ax)

    for suffix in PLOT_STYLE["save_formats"]:
        path = output_dir / f"{metric}{suffix}"
        fig.savefig(path, dpi=PLOT_STYLE["save_dpi"])
        print(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.csv))
    if args.metric:
        rows = [row for row in rows if row["metric"] == args.metric]
    rows = [row for row in rows if row_has_data(row)]
    output_dir = make_run_dir(Path(args.output_root))
    write_scale_compare(output_dir / "scale_compare.csv", rows)
    for row in rows:
        plot_metric(row, output_dir)


if __name__ == "__main__":
    main()
