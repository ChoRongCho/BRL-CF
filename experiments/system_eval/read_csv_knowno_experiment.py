"""Stage 2 for KnowNo qhat comparison: aggregate run-level CSV for plotting.

Input:
    experiments/system_eval/data/knowno/raw_runs.csv

Output:
    experiments/system_eval/data/knowno/knowno_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "knowno" / "raw_runs.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "knowno" / "knowno_compare.csv"

PLOT_CONDITIONS = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt35turbo_raw98",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "gpt4_raw98",
    "ours",
)
CALIBRATED_CONDITIONS = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "ours",
)
DOMAINS = ("tomato", "wastesorting")
METRICS = (
    "success_rate",
    "average_step",
    "average_step_success_only",
    "average_step_failure_only",
    "average_question",
    "average_question_success_only",
    "average_question_failure_only",
    "query_probability_per_step",
    "elapsed_time",
    "prediction_set_size_when_asked",
    "token_overall",
)
METRIC_LABELS = {
    "success_rate": "Success Rate",
    "average_step": "Average Step",
    "average_step_success_only": "Average Step",
    "average_step_failure_only": "Average Step",
    "average_question": "Average Query Number",
    "average_question_success_only": "Average Query Number",
    "average_question_failure_only": "Average Query Number",
    "query_probability_per_step": "Query Probability per Step",
    "elapsed_time": "Elapsed Time",
    "prediction_set_size_when_asked": "Prediction Set Size When Asked",
    "token_overall": "Token Usage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate KnowNo qhat raw runs into a plotting CSV.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--exclude-raw98", action="store_true", help="Use only calibrated 75/85/95 conditions plus Ours.")
    parser.add_argument("--exclude-ours", action="store_true", help="Do not include Ours tau=0.8 in the output.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def mean_or_blank(values: list[float]) -> str:
    valid = [value for value in values if not math.isnan(value)]
    return f"{mean(valid):.6f}" if valid else ""


def rows_for_outcome(rows: list[dict[str, str]], outcome: str) -> list[dict[str, str]]:
    if outcome == "success":
        return [row for row in rows if is_success(row)]
    if outcome == "failure":
        return [row for row in rows if not is_success(row)]
    return rows


def query_probability_or_blank(rows: list[dict[str, str]]) -> str:
    query_steps = 0.0
    planning_steps = 0.0
    for row in rows:
        query_step_count = as_float(row.get("query_step_count"))
        planning_length = as_float(row.get("planning_length"))
        if math.isnan(query_step_count) or math.isnan(planning_length) or planning_length <= 0:
            continue
        query_steps += query_step_count
        planning_steps += planning_length
    if planning_steps <= 0:
        return ""
    return f"{query_steps / planning_steps:.6f}"


def metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "success_rate":
        return [1.0 if is_success(row) else 0.0 for row in rows]
    if metric == "average_step":
        return [as_float(row.get("planning_length")) for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row.get("planning_length")) for row in rows if is_success(row)]
    if metric == "average_step_failure_only":
        return [as_float(row.get("planning_length")) for row in rows if not is_success(row)]
    if metric == "average_question":
        return [as_float(row.get("question_count")) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row.get("question_count")) for row in rows if is_success(row)]
    if metric == "average_question_failure_only":
        return [as_float(row.get("question_count")) for row in rows if not is_success(row)]
    if metric == "elapsed_time":
        return [as_float(row.get("elapsed_seconds")) for row in rows]
    if metric == "token_overall":
        return [as_float(row.get("token_overall")) for row in rows]
    if metric == "prediction_set_size_when_asked":
        weighted: list[float] = []
        for row in rows:
            question_count = int(float(row.get("question_count") or 0))
            value = as_float(row.get("average_prediction_set_size_when_asked"))
            if question_count > 0 and not math.isnan(value):
                weighted.extend([value] * question_count)
        return weighted
    raise ValueError(f"unknown metric: {metric}")


def metric_result(rows: list[dict[str, str]], metric: str) -> str:
    if metric == "query_probability_per_step":
        return query_probability_or_blank(rows)
    return mean_or_blank(metric_values(rows, metric))


def collect_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("domain", ""), row.get("condition", ""))].append(row)
    return grouped


def output_fields(conditions: tuple[str, ...]) -> list[str]:
    return ["metric", "metric_label", *(f"{domain}_{condition}" for domain in DOMAINS for condition in conditions)]


def build_output(rows: list[dict[str, str]], conditions: tuple[str, ...]) -> list[dict[str, str]]:
    grouped = collect_rows(rows)
    output: list[dict[str, str]] = []
    for metric in METRICS:
        row = {"metric": metric, "metric_label": METRIC_LABELS[metric]}
        for domain in DOMAINS:
            for condition in conditions:
                row[f"{domain}_{condition}"] = metric_result(grouped[(domain, condition)], metric)
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    conditions = CALIBRATED_CONDITIONS if args.exclude_raw98 else PLOT_CONDITIONS
    if args.exclude_ours:
        conditions = tuple(condition for condition in conditions if condition != "ours")
    rows = build_output(read_rows(Path(args.input)), conditions)
    output_path = Path(args.output)
    write_csv(output_path, rows, output_fields(conditions))
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
