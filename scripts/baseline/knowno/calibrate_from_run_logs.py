#!/usr/bin/env python3
"""Build multi-label Waste CP calibration records from paid run logs.

The script performs no model calls. It selects one decision state per episode,
labels every task-correct option in that state, and recalculates qhat from the
combined option probabilities that were recorded during the original run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np


TOKENS = ("A", "B", "C", "D", "E")
NOOPT = "an option not listed here"


def json_blocks(text: str, label_pattern: str) -> dict[int, dict]:
    decoder = json.JSONDecoder()
    blocks = {}
    for match in re.finditer(label_pattern, text, re.MULTILINE):
        step = int(match.group(1))
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        value, _end = decoder.raw_decode(text, start)
        blocks[step] = value
    return blocks


def hidden_labels(text: str) -> dict[str, str]:
    match = re.search(r"^True labels:\s*(.+)$", text, re.MULTILINE)
    if not match:
        raise ValueError("missing True labels")
    return dict(re.findall(r"(waste\d+):\s*(general|plastic|paper|can)", match.group(1)))


def visible_objects(state: dict) -> list[str]:
    remaining = set(state["remaining_objects"])
    occlusions = state.get("occlusions", {})
    return [obj for obj in state["remaining_objects"] if occlusions.get(obj) not in remaining]


def correct_actions(state: dict, labels: dict[str, str]) -> tuple[str, list[str]]:
    held = state.get("held_object")
    if held:
        return "place", [f"place {held} into {labels[held]} bin"]

    observed = state.get("observed_attributes", {})
    picks = [f"pick {obj}" for obj in visible_objects(state) if obj in observed]
    if picks:
        return "pick", picks
    return "detect", ["detect"]


def parse_episode(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    states = json_blocks(text, r"^Step\s+(\d+)\s+start:\s*$")
    decisions = json_blocks(text, r"^Step\s+(\d+)\s+decision data:\s*$")
    common = sorted(set(states) & set(decisions))
    if not common:
        raise ValueError("no complete decision records")

    seed_match = re.search(r"_seed(\d+)", path.name)
    iteration_match = re.search(r"_iter(\d+)", path.name)
    scene_match = re.search(r"scene[_-]?(\d+)", str(path), re.IGNORECASE)
    if not (seed_match and iteration_match and scene_match):
        raise ValueError("could not parse scene/iteration/seed")
    seed = int(seed_match.group(1))
    selected_step = common[random.Random(seed ^ 0x42524C).randrange(len(common))]

    state = states[selected_step]
    decision = decisions[selected_step]
    labels = hidden_labels(text)
    action_type, actions = correct_actions(state, labels)
    options = [str(option).strip().lower().rstrip(".") for option in decision["options"]]
    token_by_option = {option: TOKENS[index] for index, option in enumerate(options)}
    correct_tokens = [token_by_option[action] for action in actions if action in token_by_option]
    if not correct_tokens:
        fallback = str(decision["add_mc_prefix"]).strip().upper()
        if fallback not in TOKENS or NOOPT not in options:
            raise ValueError("correct action absent but valid NoOpt is unavailable")
        correct_tokens = [fallback]

    scores = {str(k).upper(): float(v) for k, v in decision["combined_option_scores"].items()}
    p_true = max(scores.get(token, 0.0) for token in correct_tokens)
    return {
        "source": str(path),
        "scene": int(scene_match.group(1)),
        "iteration": int(iteration_match.group(1)),
        "seed": seed,
        "step": selected_step,
        "state": state,
        "options": options,
        "scores": scores,
        "correct_action_type": action_type,
        "true_actions": actions,
        "true_options": correct_tokens,
        "p_true": p_true,
        "nonconformity_score": 1.0 - p_true,
    }


def qhat(records: list[dict], target_success: float) -> tuple[float, float]:
    n = len(records)
    level = min(1.0, math.ceil((n + 1) * target_success) / n)
    value = np.quantile(
        [record["nonconformity_score"] for record in records],
        level,
        method="higher",
    )
    return float(value), float(level)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline", choices=["knowno", "introplan"], required=True)
    parser.add_argument("--iterations-per-scene", type=int, default=20)
    parser.add_argument("--target-success", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    folder = "when_knowno_gpt4" if args.baseline == "knowno" else "when_introplan"
    files = sorted(root.rglob(f"{folder}/*.txt"))
    records = []
    for path in files:
        iteration_match = re.search(r"_iter(\d+)", path.name)
        if iteration_match and int(iteration_match.group(1)) <= args.iterations_per_scene:
            records.append(parse_episode(path))

    expected = 5 * args.iterations_per_scene
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} records, found {len(records)}")
    if not 0 < args.target_success < 1:
        raise ValueError("target-success must be between 0 and 1")

    value, level = qhat(records, args.target_success)
    action_counts = Counter(record["correct_action_type"] for record in records)
    multi_answer_count = sum(len(record["true_actions"]) > 1 for record in records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "baseline": args.baseline,
        "domain": "wastesorting",
        "source_run_root": str(root),
        "selection": "iterations 1..N per scene; one seed-deterministic decision per episode",
        "n": len(records),
        "target_success": args.target_success,
        "quantile_method": "legacy_higher",
        "q_level": level,
        "qhat": value,
        "threshold": 1.0 - value,
        "action_type_counts": dict(sorted(action_counts.items())),
        "multi_answer_records": multi_answer_count,
        "records": records,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fields = ["scene", "iteration", "seed", "step", "correct_action_type", "true_actions", "true_options", "p_true", "nonconformity_score", "source"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **{key: record[key] for key in fields if key not in {"true_actions", "true_options"}},
                "true_actions": " | ".join(record["true_actions"]),
                "true_options": ",".join(record["true_options"]),
            })

    print("baseline:", args.baseline)
    print("records:", len(records))
    print("action_type_counts:", dict(sorted(action_counts.items())))
    print("multi_answer_records:", multi_answer_count)
    print("q_level:", level)
    print("qhat:", value)
    print("threshold:", 1.0 - value)
    print("output:", output)


if __name__ == "__main__":
    main()
