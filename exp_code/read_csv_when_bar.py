from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

DOMAINS = ("tomato", "wastesorting")
STRATEGIES = ("all", "no", "ours", "random")
BAR_COLORS = ["#2563eb", "#64748b", "#f97316", "#16a34a"]
METRICS = [
    ("average_question.csv", "average_question", "Average Query Number", True),
    ("average_reward.csv", "average_reward", "Average Cumulated Reward", True),
    ("average_step.csv", "average_step", "Average Step", True),
    ("elapsed_time.csv", "elapsed_time", "Elapsed Time", True),
    ("success_rate.csv", "success_rate", "Average Success Rate", True),
]


def mean_sem(values: list[float]) -> tuple[float, float]:
    valid_values = [value for value in values if not math.isnan(value)]
    if not valid_values:
        raise ValueError("no valid values")
    if len(valid_values) == 1:
        return mean(valid_values), 0.0
    return (
        mean(valid_values),
        stdev(valid_values) / math.sqrt(len(valid_values)),
    )


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for path in output_dir.glob(pattern):
            path.unlink()


def read_all_run_values(csv_path: Path) -> list[float]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    values: list[float] = []
    for row in rows[1:]:
        if not row or not row[0].isdigit():
            continue
        for raw_value in row[1:]:
            if raw_value != "":
                values.append(float(raw_value))
    return values


def plot_metric(
    domain: str,
    metric_slug: str,
    metric_label: str,
    strategies: list[str],
    means: list[float],
    stds: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    colors = BAR_COLORS[:len(strategies)]
    bars = ax.bar(strategies, means, color=colors, edgecolor="#1e293b", linewidth=0.8)
    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
        )
    ax.set_xlabel("strategy")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{domain} - question policy comparison - {metric_label}")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    output_stem = output_dir / f"when_strategy_{metric_slug}"
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=200)
        print(output_path)
    plt.close(fig)


def plot_domain(domain: str) -> None:
    base_dir = Path("logs") / domain / "scene_metrics"
    output_dir = Path("figures") / domain / "plots_step50_when"
    clean_output_dir(output_dir)

    enabled_metrics = [
        (file_name, metric_slug, label)
        for file_name, metric_slug, label, enabled in METRICS
        if enabled
    ]
    if not enabled_metrics:
        raise ValueError("At least one metric must be enabled.")

    available_strategies = [
        strategy
        for strategy in STRATEGIES
        if (base_dir / f"step_50_thres_{strategy}").is_dir()
    ]
    if not available_strategies:
        print(f"no strategy CSV directories found for {domain}")
        return

    for file_name, metric_slug, label in enabled_metrics:
        means: list[float] = []
        stds: list[float] = []
        for strategy in available_strategies:
            csv_path = base_dir / f"step_50_thres_{strategy}" / file_name
            avg_value, std_value = mean_sem(read_all_run_values(csv_path))
            means.append(avg_value)
            stds.append(std_value)

        plot_metric(
            domain,
            metric_slug,
            label,
            available_strategies,
            means,
            stds,
            output_dir,
        )


def main() -> None:
    for domain in DOMAINS:
        plot_domain(domain)


if __name__ == "__main__":
    main()
