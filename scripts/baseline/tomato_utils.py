from __future__ import annotations

import random
import re

from scripts import structured_prompts as prompts_v1
from scripts import structured_prompts_v2 as prompts_v2
from scripts.structured_prompts import (
    TOMATO_BACKGROUND,
)


TOMATO_PROPERTIES = ["ripe", "unripe", "rotten"]
LOCATIONS = ["dock_station", "stem_01", "stem_02"]
STEMS = ["stem_01", "stem_02"]
TOMATO_LABELS = {
    "tomato1": "ripe",
    "tomato2": "rotten",
    "tomato3": "ripe",
    "tomato4": "unripe",
}
TOMATO_LOCATIONS = {
    "tomato1": "stem_01",
    "tomato2": "stem_01",
    "tomato3": "stem_02",
    "tomato4": "stem_02",
}


def prompt_module(version: str = "v1"):
    normalized = (version or "v1").lower()
    if normalized in {"v1", "structured"}:
        return prompts_v1
    if normalized in {"v2", "natural", "natural_language"}:
        return prompts_v2
    raise ValueError(f"Unsupported prompt version {version!r}. Choose v1 or v2.")

TOMATO_CALIBRATION_TEMPLATE = """
# Tomato calibration dataset.
# Separate examples with --0000--.
# Options are optional. If omitted, the LLM generates options from Context.

--0000--
Context:
Overall instruction: Harvest all ripe tomatoes and discard rotten tomatoes.
Robot location: stem_01
Active tomatoes: tomato1, tomato2
Tomato states:
tomato1: detected, ripe
tomato2: unknown, unknown
Held tomato: None
Loaded tomatoes: None
Discarded tomatoes: None
Actions already completed:
1. navigate to stem_01
2. detect

True actions:
pick tomato1

Options:
A) pick tomato1
B) detect
C) navigate to stem_02
D) scan
E) an option not listed here

Correct options:
A
""".strip() + "\n"


def parse_label_map(text: str, valid_values: list[str], kind: str) -> dict[str, str]:
    parsed = {}
    if not text:
        return parsed
    for item in text.split(","):
        if ":" not in item:
            raise ValueError(f'{kind} must use "object:value" format.')
        obj, value = [part.strip().lower() for part in item.split(":", 1)]
        if value not in valid_values:
            raise ValueError(f"Invalid {kind} value {value!r}. Choose one of {valid_values}.")
        parsed[obj] = value
    return parsed


def initialize_tomato_world(tomatoes: list[str], label_text: str, location_text: str):
    user_labels = parse_label_map(label_text, TOMATO_PROPERTIES, "labels")
    user_locations = parse_label_map(location_text, LOCATIONS, "locations")
    hidden_properties = {
        tomato: user_labels.get(tomato, TOMATO_LABELS.get(tomato, random.choice(TOMATO_PROPERTIES)))
        for tomato in tomatoes
    }
    hidden_locations = {
        tomato: user_locations.get(tomato, TOMATO_LOCATIONS.get(tomato, random.choice(STEMS)))
        for tomato in tomatoes
    }
    return hidden_properties, hidden_locations


def parse_tomato_action(action: str):
    action = action.lower().strip()
    if "done" in action or "no more" in action or "nothing" in action:
        return "done", None

    scan_match = re.fullmatch(r"scan(?:\s+(.+))?", action)
    if scan_match:
        target = scan_match.group(1)
        return "scan", target.strip() if target else None

    detect_match = re.fullmatch(r"detect (dock_station|stem_01|stem_02)", action)
    if detect_match:
        return "detect", detect_match.group(1)

    navigate_match = re.fullmatch(r"navigate to (dock_station|stem_01|stem_02)", action)
    if navigate_match:
        return "navigate", navigate_match.group(1)

    pick_match = re.fullmatch(r"pick (.+)", action)
    if pick_match:
        return "pick", pick_match.group(1).strip()

    place_match = re.fullmatch(r"place (.+)", action)
    if place_match:
        return "place", place_match.group(1).strip()

    discard_match = re.fullmatch(r"discard (.+)", action)
    if discard_match:
        return "discard", discard_match.group(1).strip()

    return None, None


def build_tomato_calibration_prompt(record: dict) -> str:
    if record.get("mc_gen_prompt"):
        return record["mc_gen_prompt"]
    version = record.get("prompt_version", "v1")
    return prompt_module(version).build_tomato_calibration_prompt_text(record["context"])


def format_tomato_state(
    tomatoes,
    observed_properties,
    scanned_properties,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
):
    lines = []
    for tomato in tomatoes:
        if tomato in loaded_tomatoes:
            status = "loaded"
        elif tomato in discarded_tomatoes:
            status = "discarded"
        elif tomato == held_tomato:
            status = "held"
        elif tomato in observed_properties:
            status = "detected"
        else:
            status = "unknown"
        observed_prop = observed_properties.get(tomato, "unknown")
        scanned_prop = scanned_properties.get(tomato, "unknown")
        lines.append(f"{tomato}: {status}, observed {observed_prop}, scanned {scanned_prop}")
    return "\n".join(lines)


def build_tomato_generation_prompt(
    args,
    robot_location,
    active_tomatoes,
    tomato_state_text,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
    history_text,
    required_next_action_text,
):
    return prompt_module(getattr(args, "prompt_version", "v1")).build_tomato_generation_prompt_text(
        args.instruction,
        robot_location,
        active_tomatoes,
        tomato_state_text,
        held_tomato,
        loaded_tomatoes,
        discarded_tomatoes,
        history_text,
        required_next_action_text,
    )


def build_tomato_score_prompt(
    args,
    robot_location,
    active_tomatoes,
    tomato_state_text,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
    history_text,
    mc_gen_full,
    required_next_action_text,
):
    return prompt_module(getattr(args, "prompt_version", "v1")).build_tomato_score_prompt_text(
        args.instruction,
        robot_location,
        active_tomatoes,
        tomato_state_text,
        held_tomato,
        loaded_tomatoes,
        discarded_tomatoes,
        history_text,
        mc_gen_full,
        required_next_action_text,
    )
