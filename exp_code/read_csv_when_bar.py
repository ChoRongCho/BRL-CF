from __future__ import annotations

import csv
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt

DOMAIN = "wastesorting"
BASE_DIR = Path(f"logs/{DOMAIN}/scene_metrics")
OUTPUT_DIR = BASE_DIR / "plots_step50_when"
STRATEGIES = ["all", "no", "random", "ours"]

# Set enabled=False, or remove a line, to choose which metrics are drawn.
METRICS = [
    ("average_question.csv", "Average Query Number", True),
    ("average_reward.csv", "Average Cumulated Reward", True),
    ("average_step.csv", "Average Step", True),
    ("elapsed_time.csv", "Elapsed Time", True),
    ("success_rate.csv", "Average Success Rate", True),
]


def read_last_row_average(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    values = [float(value) for value in rows[-1][1:]]
    return sum(values) / len(values)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    enabled_metrics = [
        (file_name, label)
        for file_name, label, enabled in METRICS
        if enabled
    ]
    if not enabled_metrics:
        raise ValueError("At least one metric must be enabled.")

    cols = min(3, len(enabled_metrics))
    rows = ceil(len(enabled_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), constrained_layout=True)
    axes = [axes] if len(enabled_metrics) == 1 else list(axes.flat)

    for ax, (file_name, label) in zip(axes, enabled_metrics):
        values = []
        for strategy in STRATEGIES:
            csv_path = BASE_DIR / f"step_50_thres_{strategy}" / file_name
            values.append(read_last_row_average(csv_path))

        ax.bar(STRATEGIES, values)
        ax.set_xlabel("strategy")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)

    for ax in list(axes)[len(enabled_metrics):]:
        ax.set_visible(False)

    output_path = OUTPUT_DIR / "when_strategy_metrics.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
