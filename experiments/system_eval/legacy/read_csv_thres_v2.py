from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from statistics import mean

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

DOMAINS = ("tomato", "wastesorting")
DOMAIN_LABELS = {
    "tomato": "Tomato",
    "wastesorting": "Waste",
}
DOMAIN_COLORS = {
    "tomato": "#b8a1d9",
    "wastesorting": "#f2d675",
}
DOMAIN_MARKERS = {
    "tomato": "o",
    "wastesorting": "s",
}
THRESHOLD_DIR_PATTERN = re.compile(r"step_50_thres_(\d-\d)$")
METRICS = [
    ("average_question.csv", "average_question", "Average Query Number", 50),
    ("average_step.csv", "average_step", "Average Step", 50),
    ("success_rate.csv", "success_rate", "Average Success Rate", 1),
]
SCENE_NAMES = [f"Scenario {i}" for i in range(1, 6)] + ["Total Scenario"]


def threshold_value(path: Path) -> float:
    match = THRESHOLD_DIR_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a threshold metric directory: {path}")
    return float(match.group(1).replace("-", "."))


def threshold_dirs(domain: str) -> list[Path]:
    base_dir = Path("logs") / domain / "scene_metrics"
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


def mean_valid(values: list[float]) -> float:
    valid_values = [value for value in values if not math.isnan(value)]
    if not valid_values:
        return math.nan
    return mean(valid_values)


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


def domain_metric_series(
    domain: str,
    metric_file: str,
    scene_index: int,
) -> tuple[list[float], list[float]]:
    dirs = threshold_dirs(domain)
    thresholds: list[float] = []
    means: list[float] = []

    for threshold_dir in dirs:
        csv_path = threshold_dir / metric_file
        if not csv_path.exists():
            continue
        values_by_scene = read_run_values(csv_path)
        thresholds.append(threshold_value(threshold_dir))
        means.append(mean_valid(values_by_scene[scene_index]))

    return thresholds, means


def plot_scene_metric(
    scene_name: str,
    scene_index: int,
    metric_file: str,
    metric_slug: str,
    metric_label: str,
    y_max: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    for domain in DOMAINS:
        thresholds, means = domain_metric_series(domain, metric_file, scene_index)
        ax.plot(
            thresholds,
            means,
            label=DOMAIN_LABELS[domain],
            color=DOMAIN_COLORS[domain],
            linewidth=2.4,
            marker=DOMAIN_MARKERS[domain],
            markersize=6,
            markeredgecolor="#1e293b",
            markeredgewidth=0.8,
        )

    ax.set_xlabel("threshold")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if scene_name == "Total Scenario":
        ax.set_title(metric_label)
    else:
        ax.set_title(f"{scene_name} - {metric_label}")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.grid(axis="x", color="#e2e8f0", alpha=0.45, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    output_stem = output_dir / f"{scene_name.replace(' ', '_')}_{metric_slug}_domain_compare"
    for suffix in (".png", ".pdf"):
        output_path = output_stem.with_suffix(suffix)
        fig.savefig(output_path, dpi=200)
        print(output_path)
    plt.close(fig)


def main() -> None:
    output_dir = Path("figures") / "domain_compare" / "plots_step50"
    clean_output_dir(output_dir)

    for scene_index, scene_name in enumerate(SCENE_NAMES):
        for metric_file, metric_slug, metric_label, y_max in METRICS:
            plot_scene_metric(
                scene_name,
                scene_index,
                metric_file,
                metric_slug,
                metric_label,
                y_max,
                output_dir,
            )


if __name__ == "__main__":
    main()
