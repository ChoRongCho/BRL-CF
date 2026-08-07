from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

DOMAINS = ("tomato", "wastesorting")
THRESHOLD_DIR_PATTERN = re.compile(r"step_50_thres_(\d-\d)$")
LINE_COLOR = "#2563eb"
MARKER_FACE = "#f97316"
METRICS = [
    ("average_question.csv", "average_question", "Average Query Number", 60),
    ("average_reward.csv", "average_reward", "Average Cumulated Reward", 30),
    ("average_step.csv", "average_step", "Average Step", 50),
    ("success_rate.csv", "success_rate", "Average Success Rate", 1),
]
SCENE_NAMES = [f"Scenario {i}" for i in range(1, 6)] + ["Total Scenario"]


def threshold_value(path: Path) -> float:
    match = THRESHOLD_DIR_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a threshold metric directory: {path}")
    return float(match.group(1).replace("-", "."))


def threshold_dirs(base_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in base_dir.iterdir()
            if path.is_dir() and THRESHOLD_DIR_PATTERN.fullmatch(path.name)
        ),
        key=threshold_value,
    )


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for path in output_dir.glob(pattern):
            path.unlink()


def mean_sem(values: list[float]) -> tuple[float, float]:
    valid_values = [value for value in values if not math.isnan(value)]
    if not valid_values:
        return math.nan, 0.0
    if len(valid_values) == 1:
        return mean(valid_values), 0.0
    return (
        mean(valid_values),
        stdev(valid_values) / math.sqrt(len(valid_values)),
    )


def read_run_values(csv_path: Path) -> list[list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    run_rows = [row for row in rows[1:] if row and row[0].isdigit()]
    values_by_scene = [[] for _ in range(5)]
    total_values: list[float] = []

    for row in run_rows:
        for scene_index in range(5):
            raw_value = row[scene_index + 1] if scene_index + 1 < len(row) else ""
            if raw_value == "":
                values_by_scene[scene_index].append(math.nan)
                continue
            value = float(raw_value)
            values_by_scene[scene_index].append(value)
            total_values.append(value)

    values_by_scene.append(total_values)
    return values_by_scene


def plot_metric_for_scene(
    thresholds: list[float],
    means: list[float],
    _errors: list[float],
    domain: str,
    scene_name: str,
    metric_slug: str,
    metric_label: str,
    y_max: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    ax.plot(
        thresholds,
        means,
        color=LINE_COLOR,
        linewidth=2.4,
        marker="o",
        markersize=6,
        markerfacecolor=MARKER_FACE,
        markeredgecolor="#1e293b",
        markeredgewidth=0.8,
    )
    ax.set_xlabel("threshold")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, y_max)
    ax.set_xticks(thresholds)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if scene_name == "Total Scenario":
        ax.set_title(f"{domain} - {metric_label}")
    else:
        ax.set_title(f"{domain} - {scene_name} - {metric_label}")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.45, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    output_stem = output_dir / f"{scene_name.replace(' ', '_')}_{metric_slug}"
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=200)
        print(output_path)
    plt.close(fig)


def plot_domain(domain: str) -> None:
    base_dir = Path("logs") / domain / "scene_metrics"
    output_dir = Path("figures") / domain / "plots_step50"
    clean_output_dir(output_dir)

    dirs = threshold_dirs(base_dir)
    if not dirs:
        print(f"no threshold CSV directories found for {domain}")
        return

    thresholds = [threshold_value(path) for path in dirs]
    stats = {
        metric_slug: {
            scene_name: {"means": [], "stds": []}
            for scene_name in SCENE_NAMES
        }
        for _, metric_slug, _, _ in METRICS
    }

    for threshold_dir in dirs:
        for file_name, metric_slug, _, _ in METRICS:
            values_by_scene = read_run_values(threshold_dir / file_name)
            for scene_name, values in zip(SCENE_NAMES, values_by_scene):
                avg_value, error_value = mean_sem(values)
                stats[metric_slug][scene_name]["means"].append(avg_value)
                stats[metric_slug][scene_name]["stds"].append(error_value)

    for scene_name in SCENE_NAMES:
        for _, metric_slug, metric_label, y_max in METRICS:
            plot_metric_for_scene(
                thresholds,
                stats[metric_slug][scene_name]["means"],
                stats[metric_slug][scene_name]["stds"],
                domain,
                scene_name,
                metric_slug,
                metric_label,
                y_max,
                output_dir,
            )


def main() -> None:
    for domain in DOMAINS:
        plot_domain(domain)


if __name__ == "__main__":
    main()
