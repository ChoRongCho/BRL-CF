"""Compare KnowNo baseline against question-policy experiment summaries.

Run from the repository root:

    python3 exp_code/compare_knowno_with_policies.py --domain tomato

Default inputs for a domain are:
    logs/<domain>/scene_metrics/step_50_thres_all
    logs/<domain>/scene_metrics/step_50_thres_no
    logs/<domain>/scene_metrics/step_50_thres_ours
    logs/<domain>/scene_metrics/step_50_thres_random
    logs/<domain>/scene_metrics/step_50_knowno

Outputs:
    logs/<domain>/scene_metrics/step_50_compare_knowno/comparison_summary.csv
    logs/<domain>/scene_metrics/step_50_compare_knowno/comparison_runs.csv
    figures/<domain>/plots_step50_compare_knowno/*.png
    figures/<domain>/plots_step50_compare_knowno/*.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCENES = tuple(f"scene_{index:02d}" for index in range(1, 6))
POLICIES = ("all", "no", "ours", "random", "knowno")
POLICY_LABELS = {
    "all": "All",
    "no": "No",
    "ours": "Ours",
    "random": "Random",
    "knowno": "KnowNo",
}
POLICY_COLORS = {
    "all": "#2563eb",
    "no": "#64748b",
    "ours": "#f97316",
    "random": "#16a34a",
    "knowno": "#dc2626",
}
PLOT_METRICS = [
    ("success_probability", "Success Probability"),
    ("overall_planning_length_mean", "Planning Length"),
    ("question_count_mean", "Question Count"),
    ("query_probability_per_step", "Query Probability per Step"),
    ("success_planning_length_mean", "Success Planning Length"),
    ("failure_planning_length_mean", "Failure Planning Length"),
    ("elapsed_seconds_mean", "Elapsed Seconds"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare policy and KnowNo CSV summaries.")
    parser.add_argument("--domain", default="tomato", help="Domain folder under logs/, e.g. tomato or waste.")
    parser.add_argument("--logs-root", default="logs", help="Root logs directory.")
    parser.add_argument("--figures-root", default="figures", help="Root figures directory.")
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="Comma-separated policies among all,no,ours,random,knowno.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def to_float(value: str | int | float | None) -> float:
    if value in {None, ""}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def mean_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return mean(valid) if valid else math.nan


def stdev_or_nan(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return math.nan
    return stdev(valid) if len(valid) > 1 else 0.0


def sem_or_zero(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) <= 1:
        return 0.0
    return stdev(valid) / math.sqrt(len(valid))


def scene_from_header(header: str) -> str | None:
    for scene in SCENES:
        if scene in header:
            return scene
    return None


def read_metric_matrix(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if not rows:
        return {}

    scene_columns: list[tuple[int, str]] = []
    for index, header in enumerate(rows[0]):
        scene = scene_from_header(header)
        if scene:
            scene_columns.append((index, scene))

    values: dict[str, list[float]] = {scene: [] for _, scene in scene_columns}
    for row in rows[1:]:
        if not row or not row[0].isdigit():
            continue
        for index, scene in scene_columns:
            if index < len(row) and row[index] != "":
                values[scene].append(float(row[index]))
    return values


def add_old_policy_runs(
    runs: list[dict[str, Any]],
    domain: str,
    policy: str,
    policy_dir: Path,
) -> None:
    success = read_metric_matrix(policy_dir / "success_rate.csv")
    steps = read_metric_matrix(policy_dir / "average_step.csv")
    questions = read_metric_matrix(policy_dir / "average_question.csv")
    elapsed = read_metric_matrix(policy_dir / "elapsed_time.csv")
    rewards = read_metric_matrix(policy_dir / "average_reward.csv")

    for scene in SCENES:
        count = max(
            len(success.get(scene, [])),
            len(steps.get(scene, [])),
            len(questions.get(scene, [])),
            len(elapsed.get(scene, [])),
            len(rewards.get(scene, [])),
        )
        for index in range(count):
            step_value = value_at(steps, scene, index)
            question_value = value_at(questions, scene, index)
            runs.append(
                {
                    "domain": domain,
                    "policy": policy,
                    "scene": scene,
                    "run_index": index + 1,
                    "success": value_at(success, scene, index) >= 0.5,
                    "planning_length": step_value,
                    "question_count": question_value,
                    "elapsed_seconds": value_at(elapsed, scene, index),
                    "reward": value_at(rewards, scene, index),
                    "query_probability": (
                        question_value / step_value
                        if not math.isnan(question_value) and not math.isnan(step_value) and step_value
                        else math.nan
                    ),
                    "source_file": str(policy_dir),
                }
            )


def add_knowno_runs(
    runs: list[dict[str, Any]],
    domain: str,
    policy: str,
    knowno_dir: Path,
) -> None:
    for index, row in enumerate(read_csv_rows(knowno_dir / "runs.csv"), start=1):
        step_value = to_float(row.get("planning_length"))
        question_value = to_float(row.get("question_count"))
        runs.append(
            {
                "domain": domain,
                "policy": policy,
                "scene": row.get("scene", ""),
                "run_index": index,
                "success": str(row.get("success", "")).lower() == "true",
                "planning_length": step_value,
                "question_count": question_value,
                "elapsed_seconds": to_float(row.get("total_elapsed_seconds")),
                "reward": math.nan,
                "query_probability": (
                    question_value / step_value
                    if not math.isnan(question_value) and not math.isnan(step_value) and step_value
                    else math.nan
                ),
                "source_file": row.get("file", str(knowno_dir)),
            }
        )


def value_at(values: dict[str, list[float]], scene: str, index: int) -> float:
    scene_values = values.get(scene, [])
    return scene_values[index] if index < len(scene_values) else math.nan


def policy_scene_metrics_dir(logs_root: Path, domain: str, policy: str) -> Path:
    if policy == "knowno":
        return logs_root / domain / "scene_metrics"
    primary = logs_root / domain / "scene_metrics"
    if (primary / f"step_50_thres_{policy}").exists():
        return primary
    if domain == "waste":
        legacy = logs_root / "wastesorting" / "scene_metrics"
        if (legacy / f"step_50_thres_{policy}").exists():
            return legacy
    return primary


def collect_runs(domain: str, logs_root: Path, policies: list[str]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for policy in policies:
        scene_metrics_dir = policy_scene_metrics_dir(logs_root, domain, policy)
        policy_dir = (
            scene_metrics_dir / "step_50_knowno"
            if policy == "knowno"
            else scene_metrics_dir / f"step_50_thres_{policy}"
        )
        if not policy_dir.exists():
            print(f"missing policy directory, skipping: {policy_dir}")
            continue
        if policy == "knowno":
            add_knowno_runs(runs, domain, policy, policy_dir)
        else:
            add_old_policy_runs(runs, domain, policy, policy_dir)
    return runs


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        scene = row["scene"]
        grouped[(row["domain"], row["policy"], scene)].append(row)
        grouped[(row["domain"], row["policy"], "total")].append(row)

    output: list[dict[str, str]] = []
    keys = sorted(grouped, key=lambda item: (item[0], policy_order(item[1]), scene_order(item[2])))
    for domain, policy, scene in keys:
        rows = grouped[(domain, policy, scene)]
        success_rows = [row for row in rows if row["success"]]
        failure_rows = [row for row in rows if not row["success"]]
        steps = numeric_values(rows, "planning_length")
        questions = numeric_values(rows, "question_count")
        elapsed = numeric_values(rows, "elapsed_seconds")
        rewards = numeric_values(rows, "reward")
        total_steps = sum(value for value in steps if not math.isnan(value))
        total_questions = sum(value for value in questions if not math.isnan(value))
        output.append(
            {
                "domain": domain,
                "policy": policy,
                "scene": scene,
                "run_count": str(len(rows)),
                "success_count": str(len(success_rows)),
                "failure_count": str(len(failure_rows)),
                "success_probability": fmt(len(success_rows) / len(rows)) if rows else "",
                "success_planning_length_mean": fmt(mean_or_nan(numeric_values(success_rows, "planning_length"))),
                "success_planning_length_std": fmt(stdev_or_nan(numeric_values(success_rows, "planning_length"))),
                "failure_planning_length_mean": fmt(mean_or_nan(numeric_values(failure_rows, "planning_length"))),
                "failure_planning_length_std": fmt(stdev_or_nan(numeric_values(failure_rows, "planning_length"))),
                "overall_planning_length_mean": fmt(mean_or_nan(steps)),
                "overall_planning_length_std": fmt(stdev_or_nan(steps)),
                "question_count_mean": fmt(mean_or_nan(questions)),
                "question_count_std": fmt(stdev_or_nan(questions)),
                "query_probability_per_step": fmt(total_questions / total_steps) if total_steps else "",
                "elapsed_seconds_mean": fmt(mean_or_nan(elapsed)),
                "reward_mean": fmt(mean_or_nan(rewards)),
            }
        )
    return output


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]


def policy_order(policy: str) -> int:
    return POLICIES.index(policy) if policy in POLICIES else len(POLICIES)


def scene_order(scene: str) -> int:
    if scene == "total":
        return 99
    return SCENES.index(scene) if scene in SCENES else 98


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def clean_plot_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for file_path in path.glob(pattern):
            file_path.unlink()


def plot_metric(
    domain: str,
    scene: str,
    metric_key: str,
    metric_label: str,
    policies: list[str],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    means: list[float] = []
    errors: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    for policy in policies:
        values = [
            metric_value(row, metric_key)
            for row in rows
            if row["policy"] == policy and (scene == "total" or row["scene"] == scene)
        ]
        valid = [value for value in values if not math.isnan(value)]
        if not valid:
            continue
        means.append(mean(valid))
        errors.append(sem_or_zero(valid))
        labels.append(POLICY_LABELS.get(policy, policy))
        colors.append(POLICY_COLORS.get(policy, "#334155"))

    if not means:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(labels, means, yerr=errors, capsize=4, color=colors, edgecolor="#1e293b", linewidth=0.8)
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
    ax.set_xlabel("Policy")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{domain} {scene} - {metric_label}")
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    output_stem = output_dir / f"{scene}_{metric_key}"
    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), dpi=200)
    plt.close(fig)


def metric_value(row: dict[str, Any], metric_key: str) -> float:
    if metric_key == "success_probability":
        return 1.0 if row["success"] else 0.0
    if metric_key == "overall_planning_length_mean":
        return float(row["planning_length"])
    if metric_key == "question_count_mean":
        return float(row["question_count"])
    if metric_key == "query_probability_per_step":
        return float(row["query_probability"])
    if metric_key == "success_planning_length_mean":
        return float(row["planning_length"]) if row["success"] else math.nan
    if metric_key == "failure_planning_length_mean":
        return float(row["planning_length"]) if not row["success"] else math.nan
    if metric_key == "elapsed_seconds_mean":
        return float(row["elapsed_seconds"])
    return math.nan


def make_plots(domain: str, policies: list[str], runs: list[dict[str, Any]], output_dir: Path) -> None:
    clean_plot_dir(output_dir)
    for scene in (*SCENES, "total"):
        for metric_key, metric_label in PLOT_METRICS:
            plot_metric(domain, scene, metric_key, metric_label, policies, runs, output_dir)


def main() -> None:
    args = parse_args()
    policies = [policy.strip() for policy in args.policies.split(",") if policy.strip()]
    logs_root = Path(args.logs_root)
    scene_metrics_dir = logs_root / args.domain / "scene_metrics"
    output_dir = scene_metrics_dir / "step_50_compare_knowno"
    plot_dir = Path(args.figures_root) / args.domain / "plots_step50_compare_knowno"

    runs = collect_runs(args.domain, logs_root, policies)
    summary = aggregate_runs(runs)

    run_fields = [
        "domain",
        "policy",
        "scene",
        "run_index",
        "success",
        "planning_length",
        "question_count",
        "query_probability",
        "elapsed_seconds",
        "reward",
        "source_file",
    ]
    summary_fields = [
        "domain",
        "policy",
        "scene",
        "run_count",
        "success_count",
        "failure_count",
        "success_probability",
        "success_planning_length_mean",
        "success_planning_length_std",
        "failure_planning_length_mean",
        "failure_planning_length_std",
        "overall_planning_length_mean",
        "overall_planning_length_std",
        "question_count_mean",
        "question_count_std",
        "query_probability_per_step",
        "elapsed_seconds_mean",
        "reward_mean",
    ]
    write_csv(output_dir / "comparison_runs.csv", runs, run_fields)
    write_csv(output_dir / "comparison_summary.csv", summary, summary_fields)
    make_plots(args.domain, policies, runs, plot_dir)

    print(f"Parsed {len(runs)} run rows")
    print(f"Wrote {output_dir / 'comparison_summary.csv'}")
    print(f"Wrote {output_dir / 'comparison_runs.csv'}")
    print(f"Wrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
