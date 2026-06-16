from __future__ import annotations

import csv
import os
import re
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
STRATEGIES = ("all", "no", "ours", "random", "knowno")
STRATEGY_LABELS = {
    "all": "All",
    "no": "No",
    "ours": "Ours",
    "random": "Random",
    "knowno": "KnowNo",
}
METRICS = [
    ("average_step.csv", "average_step", "Average Step (Success Only)"),
    ("success_rate.csv", "success_rate", "Average Success Rate"),
    ("average_question.csv", "average_question", "Average Query Number"),
    ("query_probability_per_step", "query_probability_per_step", "Query Probability per Step"),
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


def read_successful_step_values(domain: str, strategy: str) -> list[float]:
    base_dir = Path("logs") / domain / "scene_metrics" / f"step_50_thres_{strategy}"
    with (base_dir / "average_step.csv").open("r", encoding="utf-8", newline="") as file:
        step_rows = list(csv.reader(file))
    with (base_dir / "success_rate.csv").open("r", encoding="utf-8", newline="") as file:
        success_rows = list(csv.reader(file))

    values: list[float] = []
    for step_row, success_row in zip(step_rows[1:], success_rows[1:]):
        if not step_row or not success_row or not step_row[0].isdigit() or not success_row[0].isdigit():
            continue
        for step_value, success_value in zip(step_row[1:], success_row[1:]):
            if step_value != "" and success_value != "" and float(success_value) >= 0.5:
                values.append(float(step_value))
    return values


def knowno_domain(domain: str) -> str:
    return "waste" if domain == "wastesorting" else domain


def read_knowno_run_values(domain: str, metric_file: str) -> list[float]:
    csv_path = (
        Path("logs")
        / knowno_domain(domain)
        / "scene_metrics"
        / "step_50_knowno"
        / "runs.csv"
    )
    metric_columns = {
        "average_step.csv": "planning_length",
        "success_rate.csv": "success",
        "average_question.csv": "question_count",
    }
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    values: list[float] = []
    for row in rows:
        if metric_file in {"average_step.csv", "average_question.csv"} and row.get("success", "").strip().lower() != "true":
            continue

        if metric_file == "query_probability_per_step":
            steps = float(row.get("planning_length", "") or 0.0)
            questions = float(row.get("question_count", "") or 0.0)
            if steps:
                values.append(questions / steps)
            continue

        column = metric_columns[metric_file]
        if column == "success":
            values.append(1.0 if row[column].strip().lower() == "true" else 0.0)
        elif row.get(column, "") != "":
            values.append(float(row[column]))
    return values


def parse_plan_steps(text: str) -> int:
    match = re.search(r"^steps:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else 0


def parse_question_steps(text: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(r"^Q\d+:\s*step=(\d+),", text, re.MULTILINE)
    }


def read_query_probability_values(domain: str, strategy: str) -> list[float]:
    values: list[float] = []
    for scene_dir in sorted((Path("logs") / domain).glob("scene_*_step50")):
        log_dir = scene_dir / f"when_{strategy}_rand_0-3"
        if not log_dir.is_dir():
            continue
        for log_path in sorted(log_dir.glob("*.txt")):
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            steps = parse_plan_steps(text)
            if steps:
                values.append(len(parse_question_steps(text)) / steps)
    return values


def read_metric_mean(domain: str, strategy: str, metric_file: str) -> float:
    if strategy == "knowno":
        values = read_knowno_run_values(domain, metric_file)
        if not values:
            raise ValueError(f"knowno {domain} has no valid {metric_file} values")
        return mean(values)

    if metric_file == "query_probability_per_step":
        values = read_query_probability_values(domain, strategy)
        if not values:
            raise ValueError(f"{domain} {strategy} has no valid query probability values")
        return mean(values)

    if metric_file == "average_step.csv":
        values = read_successful_step_values(domain, strategy)
        if not values:
            return 0.0
        return mean(values)

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
