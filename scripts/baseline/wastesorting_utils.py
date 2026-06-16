from __future__ import annotations

import random
import re

from scripts import structured_prompts as prompts_v1
from scripts import structured_prompts_v2 as prompts_v2
from scripts.structured_prompts import (
    WASTE_BACKGROUND,
)


WASTE_ATTRIBUTES = ["general", "plastic", "paper", "can"]
AVAILABLE_BINS = [f"{attribute} bin" for attribute in WASTE_ATTRIBUTES]
WASTE_LABELS = {
    "waste1": "can",
    "waste2": "paper",
    "waste3": "general",
    "waste4": "plastic",
}


def prompt_module(version: str = "v1"):
    normalized = (version or "v1").lower()
    if normalized in {"v1", "structured"}:
        return prompts_v1
    if normalized in {"v2", "natural", "natural_language"}:
        return prompts_v2
    raise ValueError(f"Unsupported prompt version {version!r}. Choose v1 or v2.")

WASTE_CALIBRATION_TEMPLATE = """
# Waste-sorting calibration dataset.
# Separate examples with --0000--.
# Options are optional. If omitted, the LLM generates options from Context.

--0000--
Context:
Overall instruction: Discard all waste.
Objects still on the counter: waste1, waste2
Available bins: general bin, plastic bin, paper bin, can bin
Observed waste attributes: waste1: can
Object currently held by the robot: None
Actions already completed:
None

True actions:
pick waste1

Options:
A) detect
B) pick waste1
C) place waste1 into can bin
D) pick waste2
E) an option not listed here

Correct options:
B
""".strip() + "\n"


def parse_waste_action(action: str):
    action = action.lower().strip()
    if "done" in action or "no more" in action or "nothing" in action:
        return "done", None
    if action == "detect":
        return "detect", None

    pick_match = re.fullmatch(r"pick (.+)", action)
    if pick_match:
        return "pick", pick_match.group(1).strip()

    place_match = re.fullmatch(r"place (.+) into (general|plastic|paper|can) bin", action)
    if place_match:
        return "place", (place_match.group(1).strip(), f"{place_match.group(2)} bin")

    return None, None


def parse_waste_label_overrides(text: str) -> dict[str, str]:
    user_labels = {}
    if not text:
        return user_labels
    for item in text.split(","):
        if ":" not in item:
            raise ValueError('Labels must use "object:label" format, e.g. "waste1:can,waste2:paper".')
        obj, label = [part.strip().lower() for part in item.split(":", 1)]
        if label not in WASTE_ATTRIBUTES:
            raise ValueError(f"Invalid label {label!r}. Choose one of {WASTE_ATTRIBUTES}.")
        user_labels[obj] = label
    return user_labels


def initialize_hidden_attributes(objects: list[str], label_text: str) -> dict[str, str]:
    user_labels = parse_waste_label_overrides(label_text)
    global_labels = {obj: label for obj, label in WASTE_LABELS.items() if obj in objects}
    return {
        obj: user_labels.get(obj, global_labels.get(obj, random.choice(WASTE_ATTRIBUTES)))
        for obj in objects
    }


def build_waste_calibration_prompt(record: dict) -> str:
    if record.get("mc_gen_prompt"):
        return record["mc_gen_prompt"]
    version = record.get("prompt_version", "v1")
    return prompt_module(version).build_waste_calibration_prompt_text(record["context"])


def build_waste_generation_prompt(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    prompt_version: str = "v1",
    occlusion_text: str = "None",
) -> str:
    return prompt_module(prompt_version).build_waste_generation_prompt_text(
        instruction,
        remaining_objects,
        observed_text,
        held_text,
        history_text,
        AVAILABLE_BINS,
        occlusion_text,
    )


def build_waste_score_prompt(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    mc_gen_full: str,
    prompt_version: str = "v1",
    occlusion_text: str = "None",
) -> str:
    return prompt_module(prompt_version).build_waste_score_prompt_text(
        instruction,
        remaining_objects,
        observed_text,
        held_text,
        history_text,
        mc_gen_full,
        AVAILABLE_BINS,
        occlusion_text,
    )
