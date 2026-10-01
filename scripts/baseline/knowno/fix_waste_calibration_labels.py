#!/usr/bin/env python3
"""Replace single-path Waste calibration labels with all valid next actions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TOKENS = ["A", "B", "C", "D", "E"]
WASTE_TYPE_PATTERN = re.compile(r"(waste\d+):\s*(general|plastic|paper|can)")


def valid_actions(record: dict) -> list[str]:
    current = [str(action).strip().lower() for action in record.get("true_actions", [])]
    if not current or not current[0].startswith("pick "):
        return current

    observed = [name for name, _kind in WASTE_TYPE_PATTERN.findall(record.get("context", ""))]
    if not observed:
        raise ValueError(f"pick label has no observed waste in context: {current}")
    return [f"pick {name}" for name in dict.fromkeys(observed)]


def relabel(record: dict) -> None:
    record["true_actions"] = valid_actions(record)
    options = record.get("mc_gen_all") or record.get("options") or []
    normalized_options = [str(option).strip().lower() for option in options]
    correct = [
        TOKENS[index]
        for index, option in enumerate(normalized_options[: len(TOKENS)])
        if option in record["true_actions"]
    ]
    if not correct:
        try:
            correct = [TOKENS[normalized_options.index("an option not listed here")]]
        except ValueError as error:
            raise ValueError("No valid action or NoOpt exists in calibration options") from error
    record["true_options"] = correct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    records = data.get("records", []) if isinstance(data, dict) else data
    for record in records:
        relabel(record)

    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Relabeled {len(records)} Waste calibration records: {output_path}")


if __name__ == "__main__":
    main()
