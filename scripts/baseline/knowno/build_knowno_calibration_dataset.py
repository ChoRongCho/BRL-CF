from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace


BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[2]
DATA_DIR = BASELINE_DIR / "data"
SYSTEM_LOG_ROOT = REPO_ROOT / "experiments_logs" / "system_log"

sys.path.insert(0, str(BASELINE_DIR))

from tomato_utils import build_tomato_generation_prompt, format_tomato_state  # noqa: E402
from wastesorting_utils import build_waste_generation_prompt  # noqa: E402


SELECTED_USER_RE = re.compile(r"Selected/Executed \(user, option ([A-E])\): (.*?) ->")


def decode_json_after(text: str, start: int) -> dict | None:
    json_start = text.find("{", start)
    if json_start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[json_start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def json_blocks_by_step(text: str, label: str) -> dict[int, tuple[int, dict]]:
    blocks = {}
    for match in re.finditer(rf"Step (\d+) {re.escape(label)}:\n", text):
        step = int(match.group(1))
        value = decode_json_after(text, match.end())
        if value is not None:
            blocks[step] = (match.start(), value)
    return blocks


def scalar_after(text: str, label: str, default: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*?)\s*$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def list_after(text: str, label: str, default: list[str]) -> list[str]:
    value = scalar_after(text, label, "")
    if not value:
        return default
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def prompt_context_from_prompt(prompt: str) -> str:
    marker = "\nWe: Overall instruction:"
    index = prompt.find(marker)
    if index >= 0:
        return prompt[index + 1 :].strip()
    return prompt.split("\n\n")[-1].strip()


def option_lines(options: list[str]) -> list[str]:
    tokens = ["A", "B", "C", "D", "E"]
    return [f"{tokens[index]}) {option}" for index, option in enumerate(options[:5])]


def normalize_options_with_fallback_e(options: list[str], selected_action: str) -> tuple[list[str], str]:
    tokens = ["A", "B", "C", "D", "E"]
    normalized = [option.strip().lower() for option in options if option.strip().lower() != "an option not listed here"]
    normalized = list(dict.fromkeys(normalized))[:4]
    while len(normalized) < 4:
        normalized.append("do nothing")
    normalized.append("an option not listed here")
    selected_action = selected_action.strip().lower()
    correct_option = "E"
    for index, option in enumerate(normalized):
        if option == selected_action:
            correct_option = tokens[index]
            break
    return normalized, correct_option


def occlusion_text(occlusions: dict[str, str], remaining_objects: list[str]) -> str:
    remaining = set(remaining_objects)
    active = [
        f"{hidden} is under {blocker} and cannot be detected until {blocker} is placed"
        for hidden, blocker in sorted(occlusions.items())
        if hidden in remaining and blocker in remaining
    ]
    return "; ".join(active) if active else "None"


def selected_steps(text: str, start_blocks: dict[int, tuple[int, dict]]) -> list[tuple[int, str, str]]:
    step_positions = sorted((position, step) for step, (position, _) in start_blocks.items())
    selections = []
    for match in SELECTED_USER_RE.finditer(text):
        step = None
        for position, candidate_step in step_positions:
            if position < match.start():
                step = candidate_step
            else:
                break
        if step is None:
            continue
        selections.append((step, match.group(1), match.group(2).strip().lower()))
    return selections


def tomato_record(text: str, step: int, selected_letter: str, selected_action: str, start: dict, decision: dict) -> dict:
    prompt_version = scalar_after(text, "Prompt version", "v2")
    instruction = scalar_after(text, "Instruction", "Harvest all ripe tomatoes and discard rotten tomatoes.")
    tomatoes = list_after(text, "Tomatoes", ["tomato1", "tomato2", "tomato3", "tomato4"])
    args = SimpleNamespace(instruction=instruction, prompt_version=prompt_version)

    tomato_state_text = format_tomato_state(
        tomatoes,
        start.get("observed_properties", {}),
        start.get("scanned_properties", {}),
        start.get("held_tomato"),
        start.get("loaded_tomatoes", []),
        start.get("discarded_tomatoes", []),
    )
    history_text = "\n".join(f"{index + 1}. {action}" for index, action in enumerate(start.get("action_history", []))) or "None"
    prompt = build_tomato_generation_prompt(
        args,
        start.get("robot_location", "dock_station"),
        start.get("active_tomatoes", tomatoes),
        tomato_state_text,
        start.get("held_tomato"),
        start.get("loaded_tomatoes", []),
        start.get("discarded_tomatoes", []),
        history_text,
        start.get("required_next_action", ""),
    )
    options = decision.get("options", [])
    options, correct_option = normalize_options_with_fallback_e(options, selected_action)
    return {
        "prompt": prompt,
        "context": prompt_context_from_prompt(prompt),
        "true_action": selected_action,
        "options": options,
        "correct_option": correct_option,
        "step": step,
    }


def waste_record(text: str, step: int, selected_letter: str, selected_action: str, start: dict, decision: dict) -> dict:
    prompt_version = scalar_after(text, "Prompt version", "v2")
    instruction = scalar_after(text, "Instruction", "Discard all waste.")
    observed = start.get("observed_attributes", {})
    observed_text = ", ".join(f"{obj}: {attr}" for obj, attr in sorted(observed.items())) if observed else "None"
    held_text = start.get("held_object") or "None"
    history_text = "\n".join(f"{index + 1}. {action}" for index, action in enumerate(start.get("action_history", []))) or "None"
    current_occlusion_text = occlusion_text(start.get("occlusions", {}), start.get("remaining_objects", []))
    prompt = build_waste_generation_prompt(
        instruction,
        start.get("remaining_objects", []),
        observed_text,
        held_text,
        history_text,
        prompt_version,
        current_occlusion_text,
    )
    options = decision.get("options", [])
    options, correct_option = normalize_options_with_fallback_e(options, selected_action)
    return {
        "prompt": prompt,
        "context": prompt_context_from_prompt(prompt),
        "true_action": selected_action,
        "options": options,
        "correct_option": correct_option,
        "step": step,
    }


def records_from_log(path: Path, domain: str) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    start_blocks = json_blocks_by_step(text, "start")
    decision_blocks = json_blocks_by_step(text, "decision data")
    records = []
    for step, selected_letter, selected_action in selected_steps(text, start_blocks):
        if step not in decision_blocks:
            continue
        start = start_blocks[step][1]
        decision = decision_blocks[step][1]
        if domain == "tomato":
            record = tomato_record(text, step, selected_letter, selected_action, start, decision)
        else:
            record = waste_record(text, step, selected_letter, selected_action, start, decision)
        if record["options"] and record["correct_option"] in {"A", "B", "C", "D", "E"}:
            record["source_log"] = str(path.relative_to(REPO_ROOT))
            records.append(record)
    return records


def collect_records(domain: str, model_dir: str, limit: int) -> list[dict]:
    log_paths = sorted((SYSTEM_LOG_ROOT / domain).glob(f"scene_*_step50/{model_dir}/*.txt"))
    records = []
    for path in log_paths:
        records.extend(records_from_log(path, "tomato" if domain == "tomato" else "waste"))
        if len(records) >= limit:
            return records[:limit]
    return records


def write_dataset(domain: str, records: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "waste" if domain == "wastesorting" else domain
    prompt_path = output_dir / f"{file_prefix}-mc-gen-prompt.txt"
    info_path = output_dir / f"{file_prefix}-tasks-info.txt"

    prompt_text = "\n--0000--\n".join(record["prompt"].strip() for record in records).strip() + "\n"
    prompt_path.write_text(prompt_text, encoding="utf-8")

    info_entries = []
    for index, record in enumerate(records):
        lines = [
            str(index),
            "Context:",
            record["context"],
            "",
            "True actions:",
            record["true_action"],
            "",
            "Options:",
            *option_lines(record["options"]),
            "",
            "Correct options:",
            record["correct_option"],
            "",
            "Source:",
            f"{record['source_log']} step {record['step']}",
        ]
        info_entries.append("\n".join(lines))
    info_path.write_text("\n\n".join(info_entries).strip() + "\n", encoding="utf-8")
    return prompt_path, info_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KnowNo calibration prompt/info files from GPT-4o system logs."
    )
    parser.add_argument("--domains", nargs="+", default=["tomato", "wastesorting"], choices=["tomato", "wastesorting"])
    parser.add_argument("--model-dir", default="when_knowno_gpt4")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    for domain in args.domains:
        records = collect_records(domain, args.model_dir, args.size)
        if len(records) < args.size:
            print(f"WARNING: requested {args.size} {domain} records, but only found {len(records)}.")
        prompt_path, info_path = write_dataset(domain, records, output_dir)
        print(f"{domain}: wrote {len(records)} records")
        print(f"  prompt: {prompt_path}")
        print(f"  info:   {info_path}")


if __name__ == "__main__":
    main()
