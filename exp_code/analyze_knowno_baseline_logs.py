"""Aggregate KnowNo baseline tomato/waste logs into CSV summaries.

Run from the repository root:

    python3 exp_code/analyze_knowno_baseline_logs.py

By default this reads:
    scripts/baseline/logs/tomato
    scripts/baseline/logs/waste

and writes:
    logs/tomato/scene_metrics/step_50_knowno/summary.csv
    logs/tomato/scene_metrics/step_50_knowno/runs.csv
    logs/waste/scene_metrics/step_50_knowno/summary.csv
    logs/waste/scene_metrics/step_50_knowno/runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DOMAINS = ("tomato", "waste")
SCENES = tuple(f"scene_{index:02d}" for index in range(1, 6))

SUMMARY_FIELDS = {
    "Success": "success",
    "Stop reason": "stop_reason",
    "Planning length": "planning_length",
    "Planning iterations": "planning_iterations",
    "Question count": "question_count",
    "Average candidate count when asked": "average_candidate_count_when_asked",
    "Average prediction set size when asked": "average_prediction_set_size_when_asked",
    "Autonomous action count": "autonomous_action_count",
    "Fallback in prediction count": "fallback_in_prediction_count",
    "Action failure count": "action_failure_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KnowNo baseline logs.")
    parser.add_argument(
        "--logs-root",
        default="scripts/baseline/logs",
        help="Root folder containing tomato/ and waste/ log folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Root directory where <domain>/scene_metrics/step_50_knowno outputs will be written.",
    )
    parser.add_argument(
        "--include-unscened",
        action="store_true",
        help="Also parse txt logs directly under each domain folder as scene=unscened.",
    )
    return parser.parse_args()


def extract_json_after_label(text: str, label: str) -> dict[str, Any]:
    marker = f"{label}:\n"
    start = text.rfind(marker)
    if start == -1:
        return {}

    index = text.find("{", start + len(marker))
    if index == -1:
        return {}

    depth = 0
    in_string = False
    escaped = False
    for pos in range(index, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[index : pos + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_text_summary(text: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    marker = "\n====== Summary ======"
    start = text.rfind(marker)
    if start == -1:
        return summary
    for raw_line in text[start:].splitlines():
        if ":" not in raw_line:
            continue
        label, value = raw_line.split(":", 1)
        key = SUMMARY_FIELDS.get(label.strip())
        if key:
            summary[key] = parse_scalar(value)
    return summary


def parse_line_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_timestamp(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").isoformat(sep=" ")
    except ValueError:
        return match.group(1)


def parse_log(path: Path, domain: str, scene: str) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    summary = extract_json_after_label(text, "Summary") or parse_text_summary(text)
    if not summary:
        return None

    metadata = extract_json_after_label(text, "Run metadata")
    token_usage = summary.get("token_usage") or extract_json_after_label(text, "Token usage totals")
    row = {
        "domain": domain,
        "scene": scene,
        "file": str(path),
        "timestamp": parse_timestamp(path),
        "model": metadata.get("model", ""),
        "prompt_version": metadata.get("prompt_version") or parse_line_value(text, "Prompt version"),
        "seed": metadata.get("seed", ""),
        "qhat": parse_line_value(text, "qhat"),
        "success": bool(summary.get("success", False)),
        "stop_reason": summary.get("stop_reason", ""),
        "planning_length": int(summary.get("planning_length", 0) or 0),
        "planning_iterations": int(summary.get("planning_iterations", 0) or 0),
        "question_count": int(summary.get("question_count", 0) or 0),
        "autonomous_action_count": int(summary.get("autonomous_action_count", 0) or 0),
        "fallback_in_prediction_count": int(summary.get("fallback_in_prediction_count", 0) or 0),
        "action_failure_count": int(summary.get("action_failure_count", 0) or 0),
        "average_candidate_count": float(summary.get("average_candidate_count", 0.0) or 0.0),
        "average_candidate_count_when_asked": float(
            summary.get("average_candidate_count_when_asked", 0.0) or 0.0
        ),
        "average_prediction_set_size": float(summary.get("average_prediction_set_size", 0.0) or 0.0),
        "average_prediction_set_size_when_asked": float(
            summary.get("average_prediction_set_size_when_asked", 0.0) or 0.0
        ),
        "total_elapsed_seconds": float(summary.get("total_elapsed_seconds", 0.0) or 0.0),
        "token_generation": int((token_usage or {}).get("generation", 0) or 0),
        "token_scoring": int((token_usage or {}).get("scoring", 0) or 0),
        "token_overall": int((token_usage or {}).get("overall", 0) or 0),
    }
    return row


def scene_from_path(path: Path, domain_root: Path) -> str:
    try:
        relative = path.relative_to(domain_root)
    except ValueError:
        return "unknown"
    for part in relative.parts:
        if re.fullmatch(r"scene_\d{2}", part):
            return part
    return "unscened"


def collect_runs(logs_root: Path, include_unscened: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        domain_root = logs_root / domain
        if not domain_root.exists():
            continue
        for path in sorted(domain_root.rglob("*.txt")):
            scene = scene_from_path(path, domain_root)
            if scene == "unscened" and not include_unscened:
                continue
            parsed = parse_log(path, domain, scene)
            if parsed is not None:
                rows.append(parsed)
    return rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in {"", None}]


def fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def mean_or_blank(values: list[float]) -> str:
    return fmt(mean(values)) if values else ""


def stdev_or_blank(values: list[float]) -> str:
    return fmt(stdev(values)) if len(values) > 1 else ("0.000000" if values else "")


def weighted_mean_or_blank(rows: list[dict[str, Any]], value_key: str, weight_key: str) -> str:
    weighted_sum = 0.0
    weight_sum = 0.0
    for row in rows:
        weight = float(row.get(weight_key, 0) or 0)
        if weight <= 0:
            continue
        weighted_sum += float(row.get(value_key, 0.0) or 0.0) * weight
        weight_sum += weight
    return fmt(weighted_sum / weight_sum) if weight_sum else ""


def most_common(counter: Counter[str], limit: int = 3) -> str:
    return "; ".join(f"{value} ({count})" for value, count in counter.most_common(limit))


def aggregate_rows(runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        grouped[(row["domain"], row["scene"])].append(row)
        grouped[(row["domain"], "total")].append(row)

    output: list[dict[str, str]] = []
    for domain in DOMAINS:
        for scene in (*SCENES, "total"):
            rows = grouped.get((domain, scene), [])
            if not rows:
                output.append(empty_aggregate(domain, scene))
                continue

            success_rows = [row for row in rows if row["success"]]
            failure_rows = [row for row in rows if not row["success"]]
            planning_lengths = numeric_values(rows, "planning_length")
            planning_iterations = numeric_values(rows, "planning_iterations")
            question_counts = numeric_values(rows, "question_count")
            elapsed = numeric_values(rows, "total_elapsed_seconds")
            token_overall = numeric_values(rows, "token_overall")
            total_steps = sum(planning_lengths)
            total_questions = sum(question_counts)

            output.append(
                {
                    "domain": domain,
                    "scene": scene,
                    "run_count": str(len(rows)),
                    "parsed_log_count": str(len(rows)),
                    "success_count": str(len(success_rows)),
                    "failure_count": str(len(failure_rows)),
                    "success_probability": fmt(len(success_rows) / len(rows)),
                    "success_planning_length_mean": mean_or_blank(numeric_values(success_rows, "planning_length")),
                    "success_planning_length_std": stdev_or_blank(numeric_values(success_rows, "planning_length")),
                    "failure_planning_length_mean": mean_or_blank(numeric_values(failure_rows, "planning_length")),
                    "failure_planning_length_std": stdev_or_blank(numeric_values(failure_rows, "planning_length")),
                    "overall_planning_length_mean": mean_or_blank(planning_lengths),
                    "overall_planning_length_std": stdev_or_blank(planning_lengths),
                    "planning_iterations_mean": mean_or_blank(planning_iterations),
                    "question_count_mean": mean_or_blank(question_counts),
                    "question_count_std": stdev_or_blank(question_counts),
                    "query_probability_per_step": fmt(total_questions / total_steps) if total_steps else "",
                    "fallback_in_prediction_mean": mean_or_blank(numeric_values(rows, "fallback_in_prediction_count")),
                    "action_failure_mean": mean_or_blank(numeric_values(rows, "action_failure_count")),
                    "avg_candidate_count_mean": mean_or_blank(numeric_values(rows, "average_candidate_count")),
                    "avg_candidate_count_when_asked_mean": weighted_mean_or_blank(
                        rows,
                        "average_candidate_count_when_asked",
                        "question_count",
                    ),
                    "avg_prediction_set_size_mean": mean_or_blank(
                        numeric_values(rows, "average_prediction_set_size")
                    ),
                    "avg_prediction_set_size_when_asked_mean": weighted_mean_or_blank(
                        rows,
                        "average_prediction_set_size_when_asked",
                        "question_count",
                    ),
                    "elapsed_seconds_mean": mean_or_blank(elapsed),
                    "token_overall_mean": mean_or_blank(token_overall),
                    "prompt_versions": most_common(Counter(str(row["prompt_version"]) for row in rows if row["prompt_version"])),
                    "models": most_common(Counter(str(row["model"]) for row in rows if row["model"])),
                    "qhat_values": most_common(Counter(str(row["qhat"]) for row in rows if row["qhat"])),
                    "top_stop_reasons": most_common(Counter(str(row["stop_reason"]) for row in rows)),
                    "first_timestamp": min((row["timestamp"] for row in rows if row["timestamp"]), default=""),
                    "last_timestamp": max((row["timestamp"] for row in rows if row["timestamp"]), default=""),
                }
            )
    return output


def empty_aggregate(domain: str, scene: str) -> dict[str, str]:
    return {
        "domain": domain,
        "scene": scene,
        "run_count": "0",
        "parsed_log_count": "0",
        "success_count": "0",
        "failure_count": "0",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    output_dir = Path(args.output_dir)
    runs = collect_runs(logs_root, args.include_unscened)
    aggregates = aggregate_rows(runs)

    run_fields = [
        "domain",
        "scene",
        "file",
        "timestamp",
        "model",
        "prompt_version",
        "seed",
        "qhat",
        "success",
        "stop_reason",
        "planning_length",
        "planning_iterations",
        "question_count",
        "autonomous_action_count",
        "fallback_in_prediction_count",
        "action_failure_count",
        "average_candidate_count",
        "average_candidate_count_when_asked",
        "average_prediction_set_size",
        "average_prediction_set_size_when_asked",
        "total_elapsed_seconds",
        "token_generation",
        "token_scoring",
        "token_overall",
    ]
    aggregate_fields = [
        "domain",
        "scene",
        "run_count",
        "parsed_log_count",
        "success_count",
        "failure_count",
        "success_probability",
        "success_planning_length_mean",
        "success_planning_length_std",
        "failure_planning_length_mean",
        "failure_planning_length_std",
        "overall_planning_length_mean",
        "overall_planning_length_std",
        "planning_iterations_mean",
        "question_count_mean",
        "question_count_std",
        "query_probability_per_step",
        "fallback_in_prediction_mean",
        "action_failure_mean",
        "avg_candidate_count_mean",
        "avg_candidate_count_when_asked_mean",
        "avg_prediction_set_size_mean",
        "avg_prediction_set_size_when_asked_mean",
        "elapsed_seconds_mean",
        "token_overall_mean",
        "prompt_versions",
        "models",
        "qhat_values",
        "top_stop_reasons",
        "first_timestamp",
        "last_timestamp",
    ]

    print(f"Parsed {len(runs)} logs")
    for domain in DOMAINS:
        domain_output_dir = output_dir / domain / "scene_metrics" / "step_50_knowno"
        domain_summary_path = domain_output_dir / "summary.csv"
        domain_runs_path = domain_output_dir / "runs.csv"
        domain_aggregates = [row for row in aggregates if row["domain"] == domain]
        domain_runs = [row for row in runs if row["domain"] == domain]
        write_csv(domain_summary_path, domain_aggregates, aggregate_fields)
        write_csv(domain_runs_path, domain_runs, run_fields)
        print(f"Wrote {domain_summary_path}")
        print(f"Wrote {domain_runs_path}")


if __name__ == "__main__":
    main()
