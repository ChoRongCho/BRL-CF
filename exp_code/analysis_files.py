"""Aggregate text experiment logs into CSV metric tables.

This script reads log files produced by ``utils.logger.logger_exp`` from the
configured scene folders, extracts plan-level metrics, and writes one CSV per
metric. It also aggregates the ``[Action Schema Summary]`` table from every log
so query cost can be measured as expected questions per executed action.
"""

from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

PlanResult = dict[str, Any]

# DOMAIN = "wastesorting"
DOMAIN = "tomato"
SUMMARY_KEYS = ("success", "steps", "cumulated_reward", "total_questions")

# Output CSV name -> (field parsed from each log, aggregate row label).
METRIC_FILES = {
    "success_rate": ("success", "success rate"),
    "average_step": ("steps", "average step"),
    "average_reward": ("cumulated_reward", "average reward"),
    "average_question": ("total_questions", "average question"),
    "elapsed_time": ("total_time", "average elapsed time"),
}

# Choose the log group to analyze. TARGET supports: all, no, ours, random.
THRES = "0-8"
TARGET = "random"
FOLDERS = [
    f"logs/{DOMAIN}/scene_01_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_02_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_03_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_04_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_05_step50/when_{TARGET}_rand_0-3",
    # f"logs/{DOMAIN}/test_scene_01_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/test_scene_02_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/test_scene_03_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/test_scene_04_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/test_scene_05_step50/thres_{THRES}",
]

OUTPUT_DIR = f"logs/{DOMAIN}/scene_metrics/step_50_thres_{TARGET}"


def _coerce_value(key: str, value: str) -> bool | int | float | str:
    value = value.strip()
    if value.endswith("s"):
        value = value[:-1]
    if key == "success":
        return value.lower() == "true"
    if key in {"steps", "total_questions"}:
        return int(value)
    if key in {"cumulated_reward", "total_time"}:
        return float(value)
    return value


def _parse_action_schema_row(line: str) -> tuple[str, int, int] | None:
    """Parse one row from the fixed-width Action Schema Summary log table."""
    columns = line.split()
    if len(columns) < 3:
        return None
    schema, action_count, question_count = columns[:3]
    if schema == "action_schema" or set(schema) == {"-"}:
        return None
    if not action_count.isdigit() or not question_count.isdigit():
        return None
    return schema, int(action_count), int(question_count)


def read_plan_result(path: Path) -> PlanResult | None:
    """Read the fields needed for analysis from one text log file."""
    section = ""
    result: PlanResult = {
        "action_schema_summary": {},
    }

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for raw_line in file:
                line = raw_line.strip()
                if line == "[Plan Summary]":
                    section = "summary"
                    continue
                if line == "[Timing]":
                    section = "timing"
                    continue
                if line == "[Action Schema Summary]":
                    section = "action_schema_summary"
                    continue
                if not section:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = ""
                    continue
                if not line:
                    continue
                if section == "action_schema_summary":
                    parsed_row = _parse_action_schema_row(line)
                    if parsed_row is None:
                        continue
                    schema, action_count, question_count = parsed_row
                    result["action_schema_summary"][schema] = {
                        "action_count": action_count,
                        "question_count": question_count,
                    }
                    continue
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                if section == "summary" and key in SUMMARY_KEYS:
                    result[key] = _coerce_value(key, value)
                elif section == "timing" and key == "total_time":
                    result[key] = _coerce_value(key, value)
    except OSError:
        return None

    required_keys = (*SUMMARY_KEYS, "total_time")
    return result if all(key in result for key in required_keys) else None


def _list_files(folder: Path) -> list[Path]:
    try:
        return sorted(path for path in folder.iterdir() if path.is_file())
    except OSError:
        return []


def _mean(values: Iterable[int | float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _metric_values(
    results: list[PlanResult],
    key: str,
) -> list[float]:
    if key == "success":
        return [1.0 if result[key] else 0.0 for result in results]
    return [float(result[key]) for result in results]


def _confidence_interval(
    results: list[PlanResult],
    key: str,
    confidence_z: float = 1.96,
) -> str:
    values = _metric_values(results, key)
    count = len(values)
    if count == 0:
        return ""

    mean = _mean(values)
    if count == 1:
        lower = upper = mean
    else:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        margin = confidence_z * math.sqrt(variance) / math.sqrt(count)
        lower = mean - margin
        upper = mean + margin

    if key == "success":
        lower = max(0.0, lower)
        upper = min(1.0, upper)
    return f"{lower:.4f}~{upper:.4f}"


def read_folder_results(
    folders: Iterable[str | Path],
) -> tuple[list[Path], dict[Path, list[PlanResult | None]]]:
    folder_paths = [Path(folder) for folder in folders]
    files_by_folder = {folder: _list_files(folder) for folder in folder_paths}
    jobs = [(folder, index, path) for folder, files in files_by_folder.items() for index, path in enumerate(files)]
    results_by_folder: dict[Path, list[PlanResult | None]] = {
        folder: [None] * len(files)
        for folder, files in files_by_folder.items()
    }

    if jobs:
        with ThreadPoolExecutor() as executor:
            parsed = executor.map(lambda job: (job[0], job[1], read_plan_result(job[2])), jobs)
            for folder, index, result in parsed:
                results_by_folder[folder][index] = result

    return folder_paths, results_by_folder


def _metric_value(
    result: PlanResult | None,
    key: str,
) -> str:
    if result is None:
        return ""
    if key == "success":
        return "1" if result[key] else "0"
    return str(result[key])


def _aggregate_value(
    results: list[PlanResult],
    key: str,
) -> str:
    if not results:
        return ""
    return f"{_mean(_metric_values(results, key)):.4f}"


def build_metric_table(
    folder_paths: list[Path],
    results_by_folder: dict[Path, list[PlanResult | None]],
    metric_key: str,
    aggregate_label: str,
) -> tuple[list[str], list[list[str]]]:
    max_count = max((len(results) for results in results_by_folder.values()), default=0)
    header = ["run", *(str(folder) for folder in folder_paths)]
    rows = [
        [
            str(index + 1),
            *(
                _metric_value(results_by_folder[folder][index], metric_key)
                if index < len(results_by_folder[folder])
                else ""
                for folder in folder_paths
            ),
        ]
        for index in range(max_count)
    ]
    rows.append(
        [
            "95% confidence interval",
            *(
                _confidence_interval(
                    [result for result in results_by_folder[folder] if result is not None],
                    metric_key,
                )
                for folder in folder_paths
            ),
        ]
    )
    rows.append(
        [
            aggregate_label,
            *(
                _aggregate_value(
                    [result for result in results_by_folder[folder] if result is not None],
                    metric_key,
                )
                for folder in folder_paths
            ),
        ]
    )
    return header, rows


def aggregate_action_schema_summary(
    results: Iterable[PlanResult | None],
) -> dict[str, dict[str, int | float]]:
    """Sum action/query counts and recompute expected queries per action."""
    summary: dict[str, dict[str, int | float]] = {}

    for result in results:
        if result is None:
            continue
        for schema, stats in result.get("action_schema_summary", {}).items():
            schema_stats = summary.setdefault(schema, {
                "action_count": 0,
                "question_count": 0,
            })
            schema_stats["action_count"] += int(stats.get("action_count", 0))
            schema_stats["question_count"] += int(stats.get("question_count", 0))

    total_actions = 0
    total_questions = 0
    for stats in summary.values():
        action_count = int(stats["action_count"])
        question_count = int(stats["question_count"])
        total_actions += action_count
        total_questions += question_count
        stats["expected_questions_per_action"] = (
            question_count / action_count
            if action_count > 0
            else 0.0
        )

    summary["ALL"] = {
        "action_count": total_actions,
        "question_count": total_questions,
        "expected_questions_per_action": (
            total_questions / total_actions
            if total_actions > 0
            else 0.0
        ),
    }
    return dict(sorted(summary.items(), key=lambda item: (item[0] == "ALL", item[0])))


def build_action_schema_table(
    folder_paths: list[Path],
    results_by_folder: dict[Path, list[PlanResult | None]],
) -> tuple[list[str], list[list[str]]]:
    """Build a CSV table with per-folder totals and one final TOTAL block."""
    header = [
        "folder",
        "action_schema",
        "action_count",
        "question_count",
        "expected_questions_per_action",
    ]
    rows: list[list[str]] = []
    all_results: list[PlanResult | None] = []

    for folder in folder_paths:
        folder_results = results_by_folder[folder]
        all_results.extend(folder_results)
        summary = aggregate_action_schema_summary(folder_results)
        for schema, stats in summary.items():
            rows.append([
                str(folder),
                schema,
                str(stats["action_count"]),
                str(stats["question_count"]),
                f"{stats['expected_questions_per_action']:.4f}",
            ])

    total_summary = aggregate_action_schema_summary(all_results)
    for schema, stats in total_summary.items():
        rows.append([
            "TOTAL",
            schema,
            str(stats["action_count"]),
            str(stats["question_count"]),
            f"{stats['expected_questions_per_action']:.4f}",
        ])

    return header, rows


def write_csv_table(path: str | Path, header: list[str], rows: list[list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    folder_paths, results_by_folder = read_folder_results(FOLDERS)
    output_dir = Path(OUTPUT_DIR)

    for file_name, (metric_key, aggregate_label) in METRIC_FILES.items():
        header, rows = build_metric_table(
            folder_paths,
            results_by_folder,
            metric_key,
            aggregate_label,
        )
        output_path = output_dir / f"{file_name}.csv"
        write_csv_table(output_path, header, rows)
        print(output_path)

    header, rows = build_action_schema_table(folder_paths, results_by_folder)
    output_path = output_dir / "action_schema_summary.csv"
    write_csv_table(output_path, header, rows)
    print(output_path)


if __name__ == "__main__":
    main()
