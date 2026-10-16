#!/usr/bin/env python3
"""Summarize participant folders beside this script without third-party dependencies.

Layout: <root>/pNNN/<domain>/<run>/{planner,pomdp}_experiments/*.txt.
The default root is BRL_EXP/01_BASELINE/00_raw/human.

One row per planner result file. Empty or missing participant folders are skipped.
query_rate = number of steps with at least one question / completed steps.
planning_time = Timing.search_time total (seconds).
interaction_time_per_query = interaction_time / total_questions; blank if zero.
Success is the planner's reported result, not a separate video assessment.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path


COLUMNS = [
    "domain", "condition", "name", "scenario", "success", "steps", "query",
    "query_rate", "total_time", "interaction_time", "planning_time",
    "interaction_time_per_query",
]
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / '01_BASELINE/00_raw/human'


def sections(text):
    parts = re.split(r"^\[([^\]\n]+)\]\s*$", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def value(section, key, default=None):
    match = re.search(r"^" + re.escape(key) + r":\s*(.*?)$", section, re.MULTILINE)
    if match:
        return match.group(1)
    if default is not None:
        return default
    raise ValueError("Missing field: " + key)


def seconds(timing, key):
    raw = value(timing, key)
    match = re.fullmatch(r"(?:total=)?([\d.]+)s(?:,\s*avg=[\d.]+s)?", raw)
    if not match:
        raise ValueError("Invalid timing field: " + key + ": " + raw)
    number = float(match.group(1))
    if not math.isfinite(number) or number < 0:
        raise ValueError("Invalid time: " + raw)
    return number


def recording_metadata(run, domain):
    """Use recording metadata only when it unambiguously identifies the run."""
    pairs = set()
    prefix = "T" if domain == "tomato" else "W"
    for path in sorted(run.glob("recordings/*/recording.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        planner = data.get("planner", "")
        if planner == "pomdp_planner":
            planner = "pomdp"
        scenario = str(data.get("scenario", ""))
        if scenario.isdigit():
            scenario = prefix + scenario
        if planner and scenario:
            pairs.add((planner, scenario))
    if len(pairs) > 1:
        raise ValueError("Multiple recording conditions/scenarios in " + str(run))
    return next(iter(pairs), ("", ""))


def parse_result(path, participant):
    text = path.read_text(encoding="utf-8-sig")
    parts = sections(text)
    meta = parts["Meta"]
    summary = parts["Plan Summary"]
    timing = parts["Timing"]
    domain = value(meta, "domain")
    domain = {"wastesorting": "waste", "waste": "waste", "tomato": "tomato"}[domain]
    run = path.parent.parent
    recorded_condition, recorded_scenario = recording_metadata(run, domain)
    condition = value(meta, "planner", "") or recorded_condition
    if not condition:
        planner_log = run / "planner.log"
        command = planner_log.read_text(encoding="utf-8", errors="replace") if planner_log.exists() else ""
        match = re.search(r"\b(pomdp|knowno)_planner_node\.py\b", command)
        condition = match.group(1) if match else ""
    scenario = value(meta, "scenario", "") or recorded_scenario
    prefix = "T" if domain == "tomato" else "W"
    if scenario.isdigit():
        scenario = prefix + scenario
    if condition not in {"pomdp", "knowno"}:
        raise ValueError("Missing/invalid condition: " + condition)
    if not re.fullmatch(prefix + r"[1-4]", scenario):
        raise ValueError("Missing/invalid scenario: " + scenario)
    if recorded_condition and recorded_condition != condition:
        raise ValueError("Condition differs between result and recording")
    if recorded_scenario and recorded_scenario != scenario:
        raise ValueError("Scenario differs between result and recording")

    success = value(summary, "success").upper()
    if success not in {"TRUE", "FALSE"}:
        raise ValueError("Invalid success: " + success)
    steps = int(value(summary, "steps"))
    query = int(value(summary, "total_questions"))
    if steps < 0 or query < 0:
        raise ValueError("Negative steps/questions")
    questions = parts.get("Questions", "")
    question_steps = set()
    counted_questions = 0
    for match in re.finditer(r"^Step=(\d+):\s*\n(.*?)(?=^Step=|\Z)", questions, re.MULTILINE | re.DOTALL):
        count = len(re.findall(r"^- Q\d+:", match.group(2), re.MULTILINE))
        counted_questions += count
        if count:
            question_steps.add(int(match.group(1)))
    if counted_questions != query:
        raise ValueError("Question summary differs from Questions section")
    if len(re.findall(r"^STEP \d+:", parts.get("Full Plan", ""), re.MULTILINE)) != steps:
        raise ValueError("Step summary differs from Full Plan section")
    interaction = seconds(timing, "interaction_time")
    return {
        "domain": domain, "condition": condition, "name": participant,
        "scenario": scenario, "success": success, "steps": steps, "query": query,
        "query_rate": len(question_steps) / steps if steps else None,
        "total_time": seconds(timing, "total_time"),
        "interaction_time": interaction,
        "planning_time": seconds(timing, "search_time"),
        "interaction_time_per_query": interaction / query if query else None,
    }


def collect_rows(root):
    rows = []
    participants = sorted(
        (path for path in root.iterdir()
         if path.is_dir() and re.fullmatch(r"p[0-9]+", path.name)),
        key=lambda path: (int(path.name[1:]), path.name),
    )
    for participant in participants:
        name = participant.name
        # These directories contain result logs; other text files are excluded.
        paths = sorted(path for path in participant.rglob("*.txt")
                       if path.parent.name in {"planner_experiments", "pomdp_experiments"})
        for path in paths:
            try:
                rows.append(parse_result(path, name))
            except (ValueError, KeyError, OSError) as error:
                raise ValueError("%s: %s" % (path, error)) from error
    return rows


def display(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return ("%.6f" % value).rstrip("0").rstrip(".")
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Directory containing pNNN folders (default: BRL_EXP/01_BASELINE/00_raw/human)")
    parser.add_argument("--output", type=Path, help="CSV path (default: BRL_EXP/01_BASELINE/01_processed/human_summary.csv)")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Human data directory does not exist: " + str(args.root))
    try:
        rows = collect_rows(args.root)
    except ValueError as error:
        parser.exit(1, str(error) + "\n")
    output = args.output or Path(__file__).resolve().parents[1] / '01_BASELINE/01_processed/human_summary.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    formatted = [{key: display(row[key]) for key in COLUMNS} for row in rows]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(formatted)
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join(":---:" for _ in COLUMNS) + " |")
    for row in formatted:
        print("| " + " | ".join(row[key] for key in COLUMNS) + " |")
    print("\nSaved %d rows: %s" % (len(rows), output), file=sys.stderr)


if __name__ == "__main__":
    main()
