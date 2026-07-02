"""Quantify KnowNo failure causes and auto-answer involvement.

Input:
    experiments/system_eval/data/knowno/raw_runs.csv

Outputs:
    experiments/system_eval/data/knowno/failure_diagnostics.csv
    experiments/system_eval/data/knowno/failure_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "knowno" / "raw_runs.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "knowno" / "failure_diagnostics.csv"
DEFAULT_SUMMARY = SCRIPT_DIR / "data" / "knowno" / "failure_summary.csv"

CONDITION_ORDER = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt35turbo_raw98",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "gpt4_raw98",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KnowNo failure causes from detailed logs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Raw runs CSV from analysis_knowno_experiment.py.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Per-run failure diagnostics CSV.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Aggregated failure summary CSV.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def as_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def extract_json_object(text: str, start_index: int) -> dict[str, Any]:
    index = text.find("{", start_index)
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


def auto_answer_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for match in re.finditer(r"^Step\s+(\d+)\s+auto answer:\s*$", text, re.MULTILINE):
        event = extract_json_object(text, match.end())
        if event:
            event["step"] = int(match.group(1))
            events.append(event)
    return events


def selected_execution_events(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    pattern = re.compile(
        r"^Selected/Executed \((?P<source>[^,]+), option (?P<option>[A-Z])\): "
        r"(?P<selected>.*?) -> (?P<result>.*)$",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        events.append(match.groupdict())
    return events


def reason_group(stop_reason: str) -> str:
    reason = stop_reason.lower()
    if "malformed option generation" in reason:
        return "option_generation_malformed"
    if "max steps reached" in reason:
        return "max_steps"
    if "prediction set only includes fallback" in reason:
        return "prediction_set_fallback_only"
    if "non-executable action selected" in reason:
        return "fallback_executed"
    if reason.startswith("plan failure"):
        return "plan_failure_wrong_disposal_or_place"
    if reason.startswith("invalid "):
        return "invalid_action"
    if not reason:
        return "unknown"
    return "other"


def involvement_group(
    stop_reason: str,
    question_count: int,
    events: list[dict[str, Any]],
    selected_events: list[dict[str, str]],
) -> str:
    group = reason_group(stop_reason)
    if group == "option_generation_malformed":
        return "not_auto_answer_option_generation"
    if group == "max_steps":
        return "not_direct_auto_answer_max_steps"
    if group == "prediction_set_fallback_only":
        return "not_auto_answer_prediction_set"

    last_selected = selected_events[-1] if selected_events else {}
    last_source = last_selected.get("source", "")
    last_result = last_selected.get("result", "")
    if last_source == "prediction set":
        return "not_auto_answer_autonomous_prediction_set"
    if last_source == "user":
        if events and events[-1].get("selected_token") == "E":
            return "auto_answer_selected_fallback"
        if "invalid" in last_result.lower() or group in {"invalid_action", "plan_failure_wrong_disposal_or_place"}:
            return "auto_answer_terminal_action"
        return "auto_answer_involved"

    if question_count == 0:
        return "not_auto_answer_no_question"
    if events and events[-1].get("selected_token") == "E":
        return "auto_answer_selected_fallback"
    if events:
        return "auto_answer_involved_unknown_terminal_source"
    return "unknown_no_detailed_event"


def summarize_auto_events(events: list[dict[str, Any]]) -> dict[str, str]:
    selected_tokens = [str(event.get("selected_token", "")) for event in events]
    rules = [str(event.get("rule", "")) for event in events]
    fallback_count = sum(1 for token in selected_tokens if token == "E")
    no_feasible_count = sum(1 for rule in rules if "no feasible option" in rule.lower())
    return {
        "auto_answer_event_count": str(len(events)),
        "auto_answer_fallback_count": str(fallback_count),
        "auto_answer_no_feasible_count": str(no_feasible_count),
        "last_auto_answer_rule": rules[-1] if rules else "",
        "last_auto_answer_token": selected_tokens[-1] if selected_tokens else "",
    }


def diagnose_row(row: dict[str, str]) -> dict[str, str]:
    path = Path(row["source_file"])
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        text = ""

    events = auto_answer_events(text)
    selected_events = selected_execution_events(text)
    last_selected = selected_events[-1] if selected_events else {}
    stop_reason = row.get("stop_reason", "")
    question_count = as_int(row.get("question_count", "0"))
    auto_summary = summarize_auto_events(events)

    result = {
        "domain": row.get("domain", ""),
        "scene": row.get("scene", ""),
        "condition": row.get("condition", ""),
        "success": row.get("success", ""),
        "source_file": row.get("source_file", ""),
        "stop_reason": stop_reason,
        "reason_group": reason_group(stop_reason),
        "involvement_group": involvement_group(stop_reason, question_count, events, selected_events),
        "question_count": row.get("question_count", ""),
        "autonomous_action_count": row.get("autonomous_action_count", ""),
        "planning_length": row.get("planning_length", ""),
        "last_execution_source": last_selected.get("source", ""),
        "last_execution_option": last_selected.get("option", ""),
        "last_execution_selected": last_selected.get("selected", ""),
        "last_execution_result": last_selected.get("result", ""),
    }
    result.update(auto_summary)
    return result


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition_sort_key(condition: str) -> int:
    try:
        return CONDITION_ORDER.index(condition)
    except ValueError:
        return len(CONDITION_ORDER)


def build_summary(diagnostics: list[dict[str, str]]) -> list[dict[str, str]]:
    total_by_group: Counter[tuple[str, str, str]] = Counter()
    failure_by_group: Counter[tuple[str, str, str, str]] = Counter()
    failure_total_by_condition: Counter[tuple[str, str, str]] = Counter()

    for row in diagnostics:
        if row["condition"] == "ours":
            continue
        base = (row["domain"], row["scene"], row["condition"])
        total_by_group[base] += 1
        if row["success"].lower() == "true":
            continue
        failure_total_by_condition[base] += 1
        failure_by_group[(*base, row["involvement_group"])] += 1

    summary: list[dict[str, str]] = []
    for domain, scene, condition, involvement in sorted(
        failure_by_group,
        key=lambda key: (key[0], key[1], condition_sort_key(key[2]), key[3]),
    ):
        base = (domain, scene, condition)
        count = failure_by_group[(domain, scene, condition, involvement)]
        total = total_by_group[base]
        failures = failure_total_by_condition[base]
        summary.append({
            "domain": domain,
            "scene": scene,
            "condition": condition,
            "involvement_group": involvement,
            "count": str(count),
            "total_runs": str(total),
            "failed_runs": str(failures),
            "pct_of_runs": f"{count / total:.6f}" if total else "",
            "pct_of_failures": f"{count / failures:.6f}" if failures else "",
        })
    return summary


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(Path(args.input)) if row.get("condition") != "ours"]
    diagnostics = [diagnose_row(row) for row in rows]
    write_rows(Path(args.output), diagnostics)
    summary = build_summary(diagnostics)
    write_rows(Path(args.summary), summary)
    print(f"Wrote {len(diagnostics)} diagnostics to {args.output}")
    print(f"Wrote {len(summary)} summary rows to {args.summary}")


if __name__ == "__main__":
    main()
