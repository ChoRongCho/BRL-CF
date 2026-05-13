from __future__ import annotations

import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

DOMAIN = "wastesorting"
SUMMARY_KEYS = ("success", "steps", "cumulated_reward", "total_questions")
METRIC_FILES = {
    "success_rate": ("success", "success rate"),
    "average_step": ("steps", "average step"),
    "average_reward": ("cumulated_reward", "average reward"),
    "average_question": ("total_questions", "average question"),
    "elapsed_time": ("total_time", "average elapsed time"),
}

# for thres in ['']:
THRES = '0-9'
TARGET = "random" # all, no, ours, random
FOLDERS = [
    f"logs/{DOMAIN}/scene_01_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_02_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_03_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_04_step50/when_{TARGET}_rand_0-3",
    f"logs/{DOMAIN}/scene_05_step50/when_{TARGET}_rand_0-3",
    # f"logs/{DOMAIN}/scene_01_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/scene_02_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/scene_03_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/scene_04_step50/thres_{THRES}",
    # f"logs/{DOMAIN}/scene_05_step50/thres_{THRES}",
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


def read_plan_result(path: Path) -> dict[str, bool | int | float | str] | None:
    section = ""
    result: dict[str, bool | int | float | str] = {}

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
                if not section:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = ""
                    continue
                if not line:
                    continue
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                if section == "summary" and key in SUMMARY_KEYS:
                    result[key] = _coerce_value(key, value)
                elif section == "timing" and key == "total_time":
                    result[key] = _coerce_value(key, value)
                    break
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
    results: list[dict[str, bool | int | float | str]],
    key: str,
) -> list[float]:
    if key == "success":
        return [1.0 if result[key] else 0.0 for result in results]
    return [float(result[key]) for result in results]


def _confidence_interval(
    results: list[dict[str, bool | int | float | str]],
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
) -> tuple[list[Path], dict[Path, list[dict[str, bool | int | float | str] | None]]]:
    folder_paths = [Path(folder) for folder in folders]
    files_by_folder = {folder: _list_files(folder) for folder in folder_paths}
    jobs = [(folder, index, path) for folder, files in files_by_folder.items() for index, path in enumerate(files)]
    results_by_folder: dict[Path, list[dict[str, bool | int | float | str] | None]] = {
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
    result: dict[str, bool | int | float | str] | None,
    key: str,
) -> str:
    if result is None:
        return ""
    if key == "success":
        return "1" if result[key] else "0"
    return str(result[key])


def _aggregate_value(
    results: list[dict[str, bool | int | float | str]],
    key: str,
) -> str:
    if not results:
        return ""
    return f"{_mean(_metric_values(results, key)):.4f}"


def build_metric_table(
    folder_paths: list[Path],
    results_by_folder: dict[Path, list[dict[str, bool | int | float | str] | None]],
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


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def format_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    escaped_header = [_escape_markdown_cell(value) for value in header]
    escaped_rows = [[_escape_markdown_cell(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(escaped_header) + " |",
        "| " + " | ".join("---" for _ in escaped_header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return "\n".join(lines)


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


if __name__ == "__main__":
    main()
