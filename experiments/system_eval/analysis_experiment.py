"""Stage 1: parse raw experiment logs into normalized run-level CSV files.

This script reads both original POMDP experiment logs and KnowNo baseline logs.

Outputs:
    <script_dir>/data/raw_runs/tomato/raw_runs.csv
    <script_dir>/data/raw_runs/wastesorting/raw_runs.csv
    <script_dir>/data/raw_runs/domain_compare/raw_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DOMAINS = ("tomato", "wastesorting")
SCENES = tuple(f"scene_{index:02d}" for index in range(1, 16))
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "experiments_logs" / "system_log"

RUN_FIELDS = [
    "domain",
    "experiment",
    "policy",
    "threshold",
    "scene",
    "run_id",
    "source_file",
    "timestamp",
    "success",
    "planning_length",
    "reward",
    "question_count",
    "query_step_count",
    "query_probability_per_step",
    "elapsed_seconds",
    "model",
    "prompt_version",
    "seed",
    "qhat",
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
    parser = argparse.ArgumentParser(description="Parse raw experiment logs into normalized CSV files.")
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT), help="Root folder for original POMDP logs.")
    parser.add_argument("--knowno-root", default="", help="Optional override for KnowNo logs. Defaults to --logs-root.")
    parser.add_argument("--output-root", default="", help="Root folder where raw_runs/*.csv is written.")
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
            return float(text.rstrip("s"))
        return int(text)
    except ValueError:
        return text


def parse_section_key_values(text: str, section_name: str) -> dict[str, Any]:
    marker = f"[{section_name}]"
    start = text.find(marker)
    if start == -1:
        return {}
    next_section = text.find("\n[", start + len(marker))
    block = text[start: next_section if next_section != -1 else len(text)]
    values: dict[str, Any] = {}
    for raw_line in block.splitlines()[1:]:
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() == "total_time":
            values["total_time"] = parse_scalar(value)
        elif key.strip() in {"success", "steps", "cumulated_reward", "total_questions", "end_reason"}:
            values[key.strip()] = parse_scalar(value)
    return values


def parse_question_steps(text: str) -> set[int]:
    return {
        int(match.group(1))
        for match in re.finditer(r"^Q\d+:\s*step=(\d+),", text, re.MULTILINE)
    }


def normalize_stop_reason(value: Any) -> str:
    text = str(value or "").strip()
    if re.search(r"\bdead[-_ ]?end\b", text, re.IGNORECASE):
        return "PLAN FAILURE"
    return text


def scene_from_parts(parts: tuple[str, ...]) -> str:
    for part in parts:
        if re.fullmatch(r"scene_\d{2}(?:_step\d+|_knowno)?", part):
            return part[:8]
    return "unknown"


def parse_original_log(path: Path, domain: str, run_id: int) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    summary = parse_section_key_values(text, "Plan Summary")
    timing = parse_section_key_values(text, "Timing")
    if not summary:
        return None

    parts = path.parts
    parent_name = path.parent.name
    experiment = ""
    policy = ""
    threshold = ""
    if parent_name.startswith("when_"):
        experiment = "when"
        match = re.fullmatch(r"when_(all|no|ours|random)_rand_.+", parent_name)
        if match is None:
            return None
        policy = match.group(1)
    elif parent_name.startswith("thres_"):
        experiment = "threshold"
        threshold = parent_name.removeprefix("thres_").replace("-", ".")
    else:
        return None

    steps = int(summary.get("steps", 0) or 0)
    question_steps = parse_question_steps(text)
    return {
        "domain": domain,
        "experiment": experiment,
        "policy": policy,
        "threshold": threshold,
        "scene": scene_from_parts(parts),
        "run_id": run_id,
        "source_file": str(path),
        "timestamp": timestamp_from_name(path),
        "success": bool(summary.get("success", False)),
        "planning_length": steps,
        "reward": float(summary.get("cumulated_reward", 0.0) or 0.0),
        "question_count": int(summary.get("total_questions", 0) or 0),
        "query_step_count": len(question_steps),
        "query_probability_per_step": (len(question_steps) / steps) if steps else "",
        "elapsed_seconds": float(timing.get("total_time", 0.0) or 0.0),
        "model": "",
        "prompt_version": "",
        "seed": parse_line_value(text, "seed"),
        "qhat": "",
        "stop_reason": normalize_stop_reason(summary.get("end_reason", "")),
        "autonomous_action_count": "",
        "fallback_in_prediction_count": "",
        "action_failure_count": "",
        "average_candidate_count": "",
        "average_candidate_count_when_asked": "",
        "average_prediction_set_size": "",
        "average_prediction_set_size_when_asked": "",
        "token_overall": "",
    }


def parse_line_value(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


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


def parse_knowno_text_summary(text: str) -> dict[str, Any]:
    summary_fields = {
        "Success": "success",
        "Stop reason": "stop_reason",
        "Planning length": "planning_length",
        "Question count": "question_count",
        "Autonomous action count": "autonomous_action_count",
        "Fallback in prediction count": "fallback_in_prediction_count",
        "Action failure count": "action_failure_count",
    }
    marker = "\n====== Summary ======"
    start = text.rfind(marker)
    if start == -1:
        return {}
    result: dict[str, Any] = {}
    for raw_line in text[start:].splitlines():
        if ":" not in raw_line:
            continue
        label, value = raw_line.split(":", 1)
        key = summary_fields.get(label.strip())
        if key:
            result[key] = parse_scalar(value)
    return result


def knowno_policy_from_path(path: Path) -> str:
    parent = path.parent.name
    if parent == "when_knowno_gpt35turbo":
        return "knowno_gpt35turbo"
    if parent == "when_knowno_gpt4":
        return "knowno_gpt4"
    if parent.startswith("when_knowno_"):
        return parent.removeprefix("when_")
    return "knowno"


def count_knowno_query_steps(text: str) -> int:
    query_steps = 0
    for match in re.finditer(r"^Prediction set:\s*(.+)$", text, re.MULTILINE):
        value = match.group(1).strip()
        if value in {"", "[]"}:
            continue
        if value.startswith("["):
            options = re.findall(r"'([^']+)'|\"([^\"]+)\"", value)
            option_count = len(options)
            if option_count > 1:
                query_steps += 1
            continue
        labels = [part.strip() for part in value.split(",") if part.strip()]
        if len(labels) > 1:
            query_steps += 1
    return query_steps


def parse_knowno_log(path: Path, output_domain: str, run_id: int, domain_root: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    summary = extract_json_after_label(text, "Summary") or parse_knowno_text_summary(text)
    if not summary:
        return None

    metadata = extract_json_after_label(text, "Run metadata")
    token_usage = summary.get("token_usage") or extract_json_after_label(text, "Token usage totals")
    scene = scene_from_parts(path.relative_to(domain_root).parts)
    steps = int(summary.get("planning_length", 0) or 0)
    questions = int(summary.get("question_count", 0) or 0)
    query_steps = count_knowno_query_steps(text)
    if query_steps == 0 and questions > 0:
        query_steps = min(questions, steps)
    return {
        "domain": output_domain,
        "experiment": "knowno",
        "policy": knowno_policy_from_path(path),
        "threshold": "",
        "scene": scene,
        "run_id": run_id,
        "source_file": str(path),
        "timestamp": timestamp_from_name(path),
        "success": bool(summary.get("success", False)),
        "planning_length": steps,
        "reward": "",
        "question_count": questions,
        "query_step_count": query_steps,
        "query_probability_per_step": (query_steps / steps) if steps else "",
        "elapsed_seconds": float(summary.get("total_elapsed_seconds", 0.0) or 0.0),
        "model": metadata.get("model", ""),
        "prompt_version": metadata.get("prompt_version") or parse_line_value(text, "Prompt version"),
        "seed": metadata.get("seed", ""),
        "qhat": parse_line_value(text, "qhat"),
        "stop_reason": normalize_stop_reason(summary.get("stop_reason", "")),
        "autonomous_action_count": summary.get("autonomous_action_count", ""),
        "fallback_in_prediction_count": summary.get("fallback_in_prediction_count", ""),
        "action_failure_count": summary.get("action_failure_count", ""),
        "average_candidate_count": summary.get("average_candidate_count", ""),
        "average_candidate_count_when_asked": summary.get("average_candidate_count_when_asked", ""),
        "average_prediction_set_size": summary.get("average_prediction_set_size", ""),
        "average_prediction_set_size_when_asked": summary.get("average_prediction_set_size_when_asked", ""),
        "token_overall": int((token_usage or {}).get("overall", 0) or 0),
    }


def collect_original_runs(logs_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for domain in DOMAINS:
        root = logs_root / domain
        if not root.exists():
            continue
        run_id = 0
        for path in sorted(root.glob("scene_*_step50/*/*.txt")):
            parsed = parse_original_log(path, domain, run_id + 1)
            if parsed is not None:
                run_id += 1
                parsed["run_id"] = run_id
                runs.append(parsed)
    return runs


def collect_knowno_runs(knowno_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for input_domain, output_domain in {"tomato": "tomato", "wastesorting": "wastesorting"}.items():
        root = knowno_root / input_domain
        if not root.exists():
            continue
        run_id = 0
        for path in sorted(root.glob("scene_*_step50/when_knowno_*/*.txt")):
            parsed = parse_knowno_log(path, output_domain, run_id + 1, root)
            if parsed is not None and parsed["scene"] in SCENES:
                run_id += 1
                parsed["run_id"] = run_id
                runs.append(parsed)
    return runs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    knowno_root = Path(args.knowno_root) if args.knowno_root else logs_root
    runs = collect_original_runs(logs_root)
    runs.extend(collect_knowno_runs(knowno_root))

    output_root = Path(args.output_root) if args.output_root else SCRIPT_DIR / "data" / "raw_runs"
    for domain in DOMAINS:
        domain_runs = [row for row in runs if row["domain"] == domain]
        output_path = output_root / domain / "raw_runs.csv"
        write_csv(output_path, domain_runs)
        print(f"Wrote {len(domain_runs)} rows to {output_path}")

    compare_path = output_root / "domain_compare" / "raw_runs.csv"
    write_csv(compare_path, runs)
    print(f"Wrote {len(runs)} rows to {compare_path}")


if __name__ == "__main__":
    main()
