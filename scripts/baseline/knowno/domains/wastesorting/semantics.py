from __future__ import annotations

import random
import re
import json
from datetime import datetime
from pathlib import Path

from domains.wastesorting import calibration_prompts, planning_prompts
from domains.wastesorting.calibration_prompts import (
    WASTE_BACKGROUND,
    WASTE_CALIBRATION_TEMPLATE,
)


WASTE_ATTRIBUTES = ["general", "plastic", "paper", "can"]
AVAILABLE_BINS = [f"{attribute} bin" for attribute in WASTE_ATTRIBUTES]
WASTE_LABELS = {
    "waste1": "can",
    "waste2": "paper",
    "waste3": "general",
    "waste4": "plastic",
}


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
    return calibration_prompts.build_waste_calibration_prompt_text(record["context"], version)


def build_waste_generation_prompt(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    prompt_version: str = "v1",
    occlusion_text: str = "None",
) -> str:
    return planning_prompts.build_waste_generation_prompt_text(
        instruction,
        remaining_objects,
        observed_text,
        held_text,
        history_text,
        AVAILABLE_BINS,
        occlusion_text,
        prompt_version=prompt_version,
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
    return planning_prompts.build_waste_score_prompt_text(
        instruction,
        remaining_objects,
        observed_text,
        held_text,
        history_text,
        mc_gen_full,
        AVAILABLE_BINS,
        occlusion_text,
        prompt_version=prompt_version,
    )


class RunLogger:
    def __init__(
        self,
        script_path: str,
        log_file: str = "",
        verbose: bool = False,
        prefix: str = "knowno_multistep",
        write_immediately: bool = True,
    ):
        log_dir = Path(script_path).resolve().parent / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = Path(log_file) if log_file else log_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.file = self.path.open("w", encoding="utf-8")
        self.verbose = verbose
        self.write_enabled = write_immediately
        self.buffer = []

    def enable_file_logging(self):
        self.write_enabled = True

    def flush_buffer(self):
        for text in self.buffer:
            self.file.write(text + "\n")
        self.file.flush()
        self.buffer.clear()

    def discard_buffer_from_marker(self, marker: str):
        for index, text in enumerate(self.buffer):
            if marker in text:
                del self.buffer[index:]
                return

    def file_only(self, *values):
        text = " ".join(str(value) for value in values)
        if self.write_enabled:
            self.file.write(text + "\n")
            self.file.flush()
        else:
            self.buffer.append(text)
        if self.verbose:
            print(text)

    def console(self, *values):
        text = " ".join(str(value) for value in values)
        print(text)
        if self.write_enabled:
            self.file.write(text + "\n")
            self.file.flush()
        else:
            self.buffer.append(text)

    def colored(self, text: str, plain_text: str):
        print(text)
        if self.write_enabled:
            self.file.write(plain_text + "\n")
            self.file.flush()
        else:
            self.buffer.append(plain_text)

    def json(self, title: str, data):
        self.file_only(title)
        self.file_only(json.dumps(data, indent=2, sort_keys=True, default=str))

    def close(self):
        self.file.close()
