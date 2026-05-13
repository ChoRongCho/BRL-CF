from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

DOMAIN = "wastesorting" # "tomato", "wastesorting"
BASE_DIR = Path(f"logs/{DOMAIN}/scene_metrics")
OUTPUT_DIR = BASE_DIR / "plots_step50"
METRICS = [
    ("average_question.csv", "Average Query Number", 60),
    ("average_reward.csv", "Average Cumulated Reward", 30),
    ("average_step.csv", "Average Step", 50),
    ("success_rate.csv", "Average Success Rate", 1),
]
SCENE_NAMES = [f"Scenario {i}" for i in range(1, 6)] + ["Total Scenario"]


def threshold_value(path: Path) -> float:
    return float(path.name.removeprefix("step_50_thres_").replace("-", "."))


def read_aggregate_values(csv_path: Path) -> list[float]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    return [float(value) for value in rows[-1][1:6]]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_dirs = sorted(
        (path for path in BASE_DIR.iterdir() if path.is_dir() and path.name.startswith("step_50_thres_")),
        key=threshold_value,
    )
    thresholds = [threshold_value(path) for path in threshold_dirs]

    data = {
        label: [[] for _ in SCENE_NAMES]
        for _, label, _ in METRICS
    }

    for threshold_dir in threshold_dirs:
        for file_name, label, _ in METRICS:
            scene_values = read_aggregate_values(threshold_dir / file_name)
            total_value = sum(scene_values) / len(scene_values)
            for index, value in enumerate([*scene_values, total_value]):
                data[label][index].append(value)

    for scene_index, scene_name in enumerate(SCENE_NAMES):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        fig.suptitle(scene_name)

        for ax, (_, label, y_max) in zip(axes.flat, METRICS):
            ax.plot(thresholds, data[label][scene_index], marker="o")
            ax.set_xlabel("threshold")
            ax.set_ylabel(label)
            ax.set_ylim(0, y_max)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        output_path = OUTPUT_DIR / f"{scene_name.replace(' ', '_')}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(output_path)


if __name__ == "__main__":
    main()
