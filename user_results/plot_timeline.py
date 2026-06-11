from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE_DIR = Path(__file__).resolve().parent
TIME_CONSUME_DIR = BASE_DIR / "02_time_consume"
TIMELINE_CSV = TIME_CONSUME_DIR / "timeline.csv"
VIDEO_DURATION_CSV = TIME_CONSUME_DIR / "video_duration_template.csv"
DETAIL_CSV = TIME_CONSUME_DIR / "timeline_duration_records.csv"
SUMMARY_CSV = TIME_CONSUME_DIR / "timeline_duration_summary.csv"
FIGURE_DIR = TIME_CONSUME_DIR / "timeline_figures"

DOMAINS = {
    "R": "분리수거",
    "T": "토마토",
}
CONDITIONS = {
    "1": "ALL",
    "2": "No",
    "3": "Ours1",
    "4": "Ours2",
    "5": "KnowNo",
}
CONDITION_ORDER = ["ALL", "No", "Ours1", "Ours2", "KnowNo"]
SCENARIO_ORDER = [1, 2, 3, 4, 5]
TIME_TYPE_LABELS = {
    "operation": "순수 조작 시간",
    "survey": "설문 시간",
}

HEADER_PATTERN = re.compile(r"([RT])(\d)\(s(\d)\)")
NATURAL_DURATION_PATTERN = re.compile(
    r"(?:(?P<hours>\d+)\s*(?:hours?|시간))?\s*"
    r"(?:(?P<minutes>\d+)\s*(?:minutes?|mins?|분))?\s*"
    r"(?:(?P<seconds>\d+)\s*(?:seconds?|secs?|초))?",
    re.IGNORECASE,
)


def configure_korean_font() -> None:
    candidates = [
        "NanumGothic",
        "NanumBarunGothic",
        "Noto Sans CJK KR",
        "Noto Sans CJK JP",
        "Noto Sans KR",
        "Baekmuk Gulim",
        "UnDotum",
        "UnBatang",
        "Malgun Gothic",
        "AppleGothic",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_duration_to_seconds(value: str) -> int:
    value = value.strip()
    if not value:
        return 0
    if ":" in value:
        parts = [int(part) for part in value.split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            raise ValueError(f"unsupported duration format: {value}")
        return hours * 3600 + minutes * 60 + seconds

    match = NATURAL_DURATION_PATTERN.fullmatch(value)
    if not match or not any(match.groupdict().values()):
        raise ValueError(f"unsupported duration format: {value}")
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def format_seconds(seconds: float) -> str:
    rounded = int(round(seconds))
    hours = rounded // 3600
    minutes = (rounded % 3600) // 60
    secs = rounded % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def load_video_durations(path: Path) -> dict[tuple[str, int], int]:
    if not path.exists():
        return {}

    durations: dict[tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            domain = (row.get("domain") or "").strip()
            scenario_text = (row.get("scenario") or "").strip()
            if not domain or not scenario_text:
                continue
            seconds_text = (row.get("video_seconds") or "").strip()
            hms_text = (row.get("video_hms") or "").strip()
            if seconds_text:
                video_seconds = int(float(seconds_text))
            elif hms_text:
                video_seconds = parse_duration_to_seconds(hms_text)
            else:
                video_seconds = 0
            durations[(domain, int(scenario_text))] = video_seconds
    return durations


def parse_timeline(path: Path, video_durations: dict[tuple[str, int], int]) -> list[dict[str, str | int]]:
    rows = read_rows(path)
    records: list[dict[str, str | int]] = []

    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row):
            match = HEADER_PATTERN.search(cell.strip())
            if match is None:
                continue

            domain_code, condition_code, scenario_text = match.groups()
            domain = DOMAINS[domain_code]
            condition = CONDITIONS[condition_code]
            scenario = int(scenario_text)

            totals: dict[str, int] = {}
            for offset in range(1, 8):
                if row_index + offset >= len(rows):
                    break
                next_row = rows[row_index + offset]
                label = next_row[col_index + 1].strip() if col_index + 1 < len(next_row) else ""
                duration = next_row[col_index + 4].strip() if col_index + 4 < len(next_row) else ""
                if label == "총 조작 시간":
                    totals["operation"] = parse_duration_to_seconds(duration)
                elif label == "총 설문 시간":
                    totals["survey"] = parse_duration_to_seconds(duration)

            for time_type, raw_seconds in totals.items():
                video_seconds = video_durations.get((domain, scenario), 0) if time_type == "operation" else 0
                seconds = max(0, raw_seconds - video_seconds)
                records.append({
                    "domain": domain,
                    "domain_code": domain_code,
                    "condition": condition,
                    "condition_code": condition_code,
                    "scenario": scenario,
                    "time_type": time_type,
                    "raw_seconds": raw_seconds,
                    "raw_hms": format_seconds(raw_seconds),
                    "video_seconds": video_seconds,
                    "video_hms": format_seconds(video_seconds) if video_seconds else "",
                    "seconds": seconds,
                    "time_hms": format_seconds(seconds),
                    "source_row": row_index + 1,
                    "source_column": col_index + 1,
                })

    return records


def write_detail_csv(records: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "domain",
        "domain_code",
        "condition",
        "condition_code",
        "scenario",
        "time_type",
        "raw_seconds",
        "raw_hms",
        "video_seconds",
        "video_hms",
        "seconds",
        "time_hms",
        "source_row",
        "source_column",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def summarize_records(records: list[dict[str, str | int]]) -> list[dict[str, str | int | float]]:
    grouped: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    for record in records:
        seconds = int(record["seconds"])
        if seconds < 0:
            continue
        key = (
            str(record["domain"]),
            str(record["condition"]),
            str(record["time_type"]),
            int(record["scenario"]),
        )
        grouped[key].append(seconds)

    summary: list[dict[str, str | int | float]] = []
    for domain in DOMAINS.values():
        for time_type in TIME_TYPE_LABELS:
            for condition in CONDITION_ORDER:
                for scenario in SCENARIO_ORDER:
                    values = grouped.get((domain, condition, time_type, scenario), [])
                    if not values:
                        summary.append({
                            "domain": domain,
                            "condition": condition,
                            "scenario": scenario,
                            "time_type": time_type,
                            "n": 0,
                            "mean_seconds": "",
                            "std_seconds": "",
                            "mean_minutes": "",
                            "std_minutes": "",
                            "mean_hms": "",
                            "std_hms": "",
                        })
                        continue
                    avg_seconds = mean(values)
                    std_seconds = stdev(values) if len(values) > 1 else 0.0
                    summary.append({
                        "domain": domain,
                        "condition": condition,
                        "scenario": scenario,
                        "time_type": time_type,
                        "n": len(values),
                        "mean_seconds": f"{avg_seconds:.3f}",
                        "std_seconds": f"{std_seconds:.3f}",
                        "mean_minutes": f"{avg_seconds / 60:.3f}",
                        "std_minutes": f"{std_seconds / 60:.3f}",
                        "mean_hms": format_seconds(avg_seconds),
                        "std_hms": format_seconds(std_seconds),
                    })
    return summary


def write_summary_csv(summary: list[dict[str, str | int | float]], path: Path) -> None:
    fieldnames = [
        "domain",
        "condition",
        "scenario",
        "time_type",
        "n",
        "mean_seconds",
        "std_seconds",
        "mean_minutes",
        "std_minutes",
        "mean_hms",
        "std_hms",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_detail_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_summary_lookup(summary_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, int], tuple[float, float]]:
    lookup: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for row in summary_rows:
        if not row["mean_seconds"]:
            continue
        key = (
            row["domain"],
            row["condition"],
            row["time_type"],
            int(row["scenario"]),
        )
        lookup[key] = (float(row["mean_seconds"]), float(row["std_seconds"]))
    return lookup


def condition_total_stats(
    detail_rows: list[dict[str, str]],
    domain: str,
    condition: str,
    time_type: str,
) -> tuple[float, float]:
    values = [
        float(row["seconds"])
        for row in detail_rows
        if row["domain"] == domain
        and row["condition"] == condition
        and row["time_type"] == time_type
        and float(row["seconds"]) > 0
    ]
    if not values:
        return math.nan, 0.0
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def plot_domain_time_type(
    summary_rows: list[dict[str, str]],
    detail_rows: list[dict[str, str]],
    domain: str,
    time_type: str,
    output_path: Path,
) -> None:
    lookup = build_summary_lookup(summary_rows)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"{TIME_TYPE_LABELS[time_type]} 모음-{domain}", fontsize=16)

    scenario_labels = [f"시나리오 {scenario}" for scenario in SCENARIO_ORDER]
    total_means: list[float] = []
    total_errors: list[float] = []

    for ax, condition in zip(axes.flat[:5], CONDITION_ORDER):
        stats = [
            lookup.get((domain, condition, time_type, scenario), (math.nan, 0.0))
            for scenario in SCENARIO_ORDER
        ]
        values = [item[0] for item in stats]
        errors = [item[1] for item in stats]
        total_mean, total_error = condition_total_stats(
            detail_rows,
            domain,
            condition,
            time_type,
        )
        total_means.append(total_mean)
        total_errors.append(total_error)

        ax.bar(scenario_labels, values)
        ax.set_title(condition)
        ax.set_xlabel("시나리오")
        ax.set_ylabel("시간 (초)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    total_ax = axes.flat[5]
    total_ax.bar(CONDITION_ORDER, total_means, yerr=total_errors, capsize=4)
    total_ax.set_title("Total")
    total_ax.set_xlabel("조건")
    total_ax.set_ylabel("평균 시간 (초)")
    total_ax.tick_params(axis="x", rotation=30)
    total_ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(output_path)


def plot_all(summary_csv: Path) -> None:
    configure_korean_font()
    summary_rows = read_summary_csv(summary_csv)
    detail_rows = read_detail_csv(DETAIL_CSV)
    for domain in DOMAINS.values():
        for time_type in TIME_TYPE_LABELS:
            file_name = f"{time_type}_{domain}.png"
            plot_domain_time_type(
                summary_rows,
                detail_rows,
                domain,
                time_type,
                FIGURE_DIR / file_name,
            )


def main() -> None:
    video_durations = load_video_durations(VIDEO_DURATION_CSV)
    records = parse_timeline(TIMELINE_CSV, video_durations)
    write_detail_csv(records, DETAIL_CSV)
    summary = summarize_records(records)
    write_summary_csv(summary, SUMMARY_CSV)
    plot_all(SUMMARY_CSV)
    print(DETAIL_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
