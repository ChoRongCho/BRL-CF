"""Stage 1 for KnowNo qhat comparison: parse raw logs into run-level CSV.

By default this reads scene_01 through scene_05 KnowNo folders under:
    experiments_logs/system_log/{tomato,wastesorting}/scene_0*

and writes:
    experiments/system_eval/data/knowno/raw_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "experiments_logs" / "system_log"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "knowno" / "raw_runs.csv"
DEFAULT_OURS_RAW_PATHS = (
    SCRIPT_DIR / "data" / "raw_runs" / "tomato" / "raw_runs.csv",
    SCRIPT_DIR / "data" / "raw_runs" / "wastesorting" / "raw_runs.csv",
)

DOMAIN_TARGET_DIRS = {
    "tomato": {
        "gpt35turbo_raw98": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-98", "GPT-3.5 raw98"),
        "gpt35turbo_75": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-8938", "GPT-3.5 75%"),
        "gpt35turbo_85": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-9082", "GPT-3.5 85%"),
        "gpt35turbo_95": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-9243", "GPT-3.5 95%"),
        "gpt4_raw98": ("when_knowno_gpt4", "temperature_5-0_qhat_0-98", "GPT-4o raw98"),
        "gpt4_75": ("when_knowno_gpt4", "temperature_5-0_qhat_0-7322", "GPT-4o 75%"),
        "gpt4_85": ("when_knowno_gpt4", "temperature_5-0_qhat_0-7779", "GPT-4o 85%"),
        "gpt4_95": ("when_knowno_gpt4", "temperature_5-0_qhat_0-8404", "GPT-4o 95%"),
    },
    "wastesorting": {
        "gpt35turbo_raw98": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-98", "GPT-3.5 raw98"),
        "gpt35turbo_75": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-8512", "GPT-3.5 75%"),
        "gpt35turbo_85": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-8851", "GPT-3.5 85%"),
        "gpt35turbo_95": ("when_knowno_gpt35turbo", "temperature_5-0_qhat_0-9028", "GPT-3.5 95%"),
        "gpt4_raw98": ("when_knowno_gpt4", "temperature_5-0_qhat_0-98", "GPT-4o raw98"),
        "gpt4_75": ("when_knowno_gpt4", "temperature_5-0_qhat_0-7084", "GPT-4o 75%"),
        "gpt4_85": ("when_knowno_gpt4", "temperature_5-0_qhat_0-7369", "GPT-4o 85%"),
        "gpt4_95": ("when_knowno_gpt4", "temperature_5-0_qhat_0-8704", "GPT-4o 95%"),
    },
}
DOMAINS = ("tomato", "wastesorting")
DEFAULT_SCENES = tuple(f"scene_{index:02d}" for index in range(1, 6))
OURS_CONDITION = "ours"
OURS_CONDITION_LABEL = "Ours"

FIELDS = [
    "domain",
    "scene",
    "condition",
    "condition_label",
    "model_group",
    "model",
    "qhat_target",
    "qhat",
    "temperature",
    "run_id",
    "source_file",
    "timestamp",
    "success",
    "planning_length",
    "question_count",
    "query_step_count",
    "query_probability_per_step",
    "elapsed_seconds",
    "prompt_version",
    "seed",
    "stop_reason",
    "autonomous_action_count",
    "fallback_in_prediction_count",
    "action_failure_count",
    "average_candidate_count",
    "average_candidate_count_when_asked",
    "average_prediction_set_size",
    "average_prediction_set_size_when_asked",
    "token_overall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse KnowNo qhat experiment logs.")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT), help="Root folder containing domain/scene/when_knowno_* dirs.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output raw run CSV path.")
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene folder to read. Defaults to scene_01 through scene_05. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ours-raw",
        action="append",
        default=[str(path) for path in DEFAULT_OURS_RAW_PATHS],
        help="Raw run CSV to append Ours rows from. Can be passed multiple times.",
    )
    parser.add_argument(
        "--ours-scene",
        action="append",
        default=[],
        help="Scene filter for Ours rows. Defaults to the same scenes as --scene. Can be passed multiple times.",
    )
    parser.add_argument("--exclude-ours", action="store_true", help="Do not append Ours rows.")
    return parser.parse_args()


def timestamp_from_name(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").isoformat(sep=" ")
    except ValueError:
        return match.group(1)


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


def parse_line_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_json_after_label(text: str, label: str) -> dict[str, Any]:
    marker = f"{label}:\n"
    start = text.find(marker)
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


def parse_text_summary(text: str) -> dict[str, Any]:
    summary_fields = {
        "Success": "success",
        "Stop reason": "stop_reason",
        "Planning length": "planning_length",
        "Question count": "question_count",
        "Autonomous action count": "autonomous_action_count",
        "Fallback in prediction count": "fallback_in_prediction_count",
        "Action failure count": "action_failure_count",
    }
    marker = "====== Result Summary ======"
    start = text.find(marker)
    if start == -1:
        return {}
    result: dict[str, Any] = {}
    for raw_line in text[start:].splitlines():
        if raw_line.strip() == "Summary:":
            break
        if ":" not in raw_line:
            continue
        label, value = raw_line.split(":", 1)
        key = summary_fields.get(label.strip())
        if key:
            result[key] = parse_scalar(value)
    return result


def count_query_steps(text: str) -> int:
    query_steps = 0
    for match in re.finditer(r"^Prediction set:\s*(.+)$", text, re.MULTILINE):
        value = match.group(1).strip()
        if value in {"", "[]"}:
            continue
        if value.startswith("["):
            options = re.findall(r"'([^']+)'|\"([^\"]+)\"", value)
            if len(options) > 1:
                query_steps += 1
            continue
        labels = [part.strip() for part in value.split(",") if part.strip()]
        if len(labels) > 1:
            query_steps += 1
    return query_steps


def parse_temperature(folder: str) -> str:
    match = re.search(r"temperature_([^_]+)", folder)
    return match.group(1).replace("-", ".") if match else ""


def parse_log(
    path: Path,
    domain: str,
    scene: str,
    condition: str,
    condition_label: str,
    run_id: int,
) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return None

    summary = extract_json_after_label(text, "Summary") or parse_text_summary(text)
    if not summary:
        return None

    metadata = extract_json_after_label(text, "Run metadata")
    token_usage = summary.get("token_usage") or extract_json_after_label(text, "Token usage totals")
    planning_length = int(summary.get("planning_length", 0) or 0)
    question_count = int(summary.get("question_count", 0) or 0)
    query_step_count = count_query_steps(text)
    if query_step_count == 0 and question_count > 0:
        query_step_count = min(question_count, planning_length)

    model_dir = path.parent.parent.name
    qhat_dir = path.parent.name
    model_group = "gpt35turbo" if "gpt35" in model_dir else "gpt4"
    qhat_target = condition.rsplit("_", 1)[-1]

    return {
        "domain": domain,
        "scene": scene,
        "condition": condition,
        "condition_label": condition_label,
        "model_group": model_group,
        "model": metadata.get("model", ""),
        "qhat_target": qhat_target,
        "qhat": parse_line_value(text, "qhat"),
        "temperature": parse_temperature(qhat_dir),
        "run_id": run_id,
        "source_file": str(path),
        "timestamp": timestamp_from_name(path),
        "success": bool(summary.get("success", False)),
        "planning_length": planning_length,
        "question_count": question_count,
        "query_step_count": query_step_count,
        "query_probability_per_step": (query_step_count / planning_length) if planning_length else "",
        "elapsed_seconds": float(summary.get("total_elapsed_seconds", 0.0) or 0.0),
        "prompt_version": metadata.get("prompt_version") or parse_line_value(text, "Prompt version"),
        "seed": metadata.get("seed", ""),
        "stop_reason": summary.get("stop_reason", ""),
        "autonomous_action_count": summary.get("autonomous_action_count", ""),
        "fallback_in_prediction_count": summary.get("fallback_in_prediction_count", ""),
        "action_failure_count": summary.get("action_failure_count", ""),
        "average_candidate_count": summary.get("average_candidate_count", ""),
        "average_candidate_count_when_asked": summary.get("average_candidate_count_when_asked", ""),
        "average_prediction_set_size": summary.get("average_prediction_set_size", ""),
        "average_prediction_set_size_when_asked": summary.get("average_prediction_set_size_when_asked", ""),
        "token_overall": int((token_usage or {}).get("overall", 0) or 0),
    }


def collect_rows(log_root: Path, scenes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for scene in scenes:
            scene_root = log_root / domain / scene
            for condition, (model_dir, qhat_dir, condition_label) in DOMAIN_TARGET_DIRS[domain].items():
                directory = scene_root / model_dir / qhat_dir
                run_id = 0
                for path in sorted(directory.glob("*.txt")):
                    parsed = parse_log(path, domain, scene, condition, condition_label, run_id + 1)
                    if parsed is None:
                        continue
                    run_id += 1
                    parsed["run_id"] = run_id
                    rows.append(parsed)
    return rows


def is_ours_row(row: dict[str, str], scenes: set[str]) -> bool:
    return (
        row.get("experiment") == "when"
        and row.get("policy") == "ours"
        and row.get("scene") in scenes
    )


def collect_ours_rows(paths: list[str], scenes: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_id = 0
    scene_set = set(scenes)
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as file:
            for source_row in csv.DictReader(file):
                if not is_ours_row(source_row, scene_set):
                    continue
                run_id += 1
                rows.append({
                    "domain": source_row.get("domain", ""),
                    "scene": source_row.get("scene", ""),
                    "condition": OURS_CONDITION,
                    "condition_label": OURS_CONDITION_LABEL,
                    "model_group": "ours",
                    "model": "ours",
                    "qhat_target": "",
                    "qhat": "",
                    "temperature": "",
                    "run_id": run_id,
                    "source_file": source_row.get("source_file", ""),
                    "timestamp": source_row.get("timestamp", ""),
                    "success": source_row.get("success", ""),
                    "planning_length": source_row.get("planning_length", ""),
                    "question_count": source_row.get("question_count", ""),
                    "query_step_count": source_row.get("query_step_count", ""),
                    "query_probability_per_step": source_row.get("query_probability_per_step", ""),
                    "elapsed_seconds": source_row.get("elapsed_seconds", ""),
                    "prompt_version": source_row.get("prompt_version", ""),
                    "seed": source_row.get("seed", ""),
                    "stop_reason": source_row.get("stop_reason", ""),
                    "autonomous_action_count": source_row.get("autonomous_action_count", ""),
                    "fallback_in_prediction_count": source_row.get("fallback_in_prediction_count", ""),
                    "action_failure_count": source_row.get("action_failure_count", ""),
                    "average_candidate_count": source_row.get("average_candidate_count", ""),
                    "average_candidate_count_when_asked": source_row.get("average_candidate_count_when_asked", ""),
                    "average_prediction_set_size": source_row.get("average_prediction_set_size", ""),
                    "average_prediction_set_size_when_asked": source_row.get(
                        "average_prediction_set_size_when_asked", ""
                    ),
                    "token_overall": source_row.get("token_overall", ""),
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scenes = tuple(args.scene) if args.scene else DEFAULT_SCENES
    ours_scenes = tuple(args.ours_scene) if args.ours_scene else scenes
    rows = collect_rows(Path(args.log_root), scenes)
    if not args.exclude_ours:
        rows.extend(collect_ours_rows(args.ours_raw, ours_scenes))
    output_path = Path(args.output)
    write_csv(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
