from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt

DOMAINS = ("tomato", "wastesorting")
DOMAIN_LABELS = {
    "tomato": "Tomato",
    "wastesorting": "Waste",
}
DOMAIN_COLORS = {
    "tomato": "#b8a1d9",
    "wastesorting": "#f2d675",
}
STRATEGIES = ("all", "no", "ours", "random")
STRATEGY_LABELS = {
    "all": "All",
    "no": "No",
    "ours": "Ours",
    "random": "Random",
}
METRICS = [
    ("average_step.csv", "average_step", "Average Step"),
    ("success_rate.csv", "success_rate", "Average Success Rate"),
    ("average_question.csv", "average_question", "Average Query Number"),
]


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


def read_metric_mean(domain: str, strategy: str, metric_file: str) -> float:
    csv_path = (
        Path("logs")
        / domain
        / "scene_metrics"
        / f"step_50_thres_{strategy}"
        / metric_file
    )
    values = read_all_run_values(csv_path)
    if not values:
        raise ValueError(f"{csv_path} has no valid run values")
    return mean(values)


def plot_metric(metric_file: str, metric_slug: str, metric_label: str, output_dir: Path) -> None:
    x_positions = list(range(len(STRATEGIES)))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for domain_index, domain in enumerate(DOMAINS):
        offset = (domain_index - 0.5) * bar_width
        values = [
            read_metric_mean(domain, strategy, metric_file)
            for strategy in STRATEGIES
        ]
        positions = [position + offset for position in x_positions]
        bars = ax.bar(
            positions,
            values,
            width=bar_width,
            label=DOMAIN_LABELS[domain],
            color=DOMAIN_COLORS[domain],
            edgecolor="#1e293b",
            linewidth=0.8,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([STRATEGY_LABELS[strategy] for strategy in STRATEGIES])
    ax.set_xlabel("strategy")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Question Policy Comparison - {metric_label}")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    output_stem = output_dir / f"when_strategy_{metric_slug}_domain_compare"
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=200)
        print(output_path)
    plt.close(fig)


def main() -> None:
    output_dir = Path("figures") / "domain_compare" / "plots_step50_when"
    clean_output_dir(output_dir)

    for metric_file, metric_slug, metric_label in METRICS:
        plot_metric(metric_file, metric_slug, metric_label, output_dir)


if __name__ == "__main__":
    main()
