from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.llm import call_llm
from scripts.prompt import process_mc_raw, temperature_scaling, top_choice_logprobs


TOKENS = ["A", "B", "C", "D", "E"]


def write_template(path: str, template: str) -> None:
    template_path = Path(path).expanduser().resolve()
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(template, encoding="utf-8")
    print("Wrote calibration template:", template_path)


SECTION_TITLES = "Prompt|Context|True actions|Options|Correct options|Source"


def _section(entry: str, title: str) -> str:
    section_titles = SECTION_TITLES
    pattern = rf"(?ims)^{re.escape(title)}:\s*(.*?)(?=^(?:{section_titles}):\s*$|\Z)"
    match = re.search(pattern, entry)
    return match.group(1).strip() if match else ""


def _strip_index(entry: str) -> str:
    lines = entry.strip().splitlines()
    if lines and re.fullmatch(r"\d+", lines[0].strip()):
        return "\n".join(lines[1:]).strip()
    return entry.strip()


def _split_indexed_records(text: str) -> list[str]:
    records = []
    current = []
    for line in text.splitlines():
        if re.fullmatch(r"\d+", line.strip()) and current:
            records.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        records.append("\n".join(current).strip())
    return [record for record in records if record]


def _parse_record(entry: str) -> dict:
    entry = _strip_index(entry)
    prompt = _section(entry, "Prompt")
    context = _section(entry, "Context")
    if not prompt and not context and "You:" in entry:
        prompt = entry
    if prompt and not context:
        context = prompt.split("\n\n")[-1].strip()

    true_actions = [
        line.strip().lower()
        for line in _section(entry, "True actions").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    options = []
    for line in _section(entry, "Options").splitlines():
        match = re.match(r"^\s*[A-Ea-e]\)\s*(.+?)\s*$", line)
        if match:
            options.append(match.group(1).strip().lower())
    true_options = re.findall(r"[A-Ea-e]", _section(entry, "Correct options").upper())

    return {
        "context": context,
        "mc_gen_prompt": prompt,
        "true_actions": true_actions,
        "options": options,
        "true_options": true_options,
    }


def _load_structured_records(text: str) -> list[dict]:
    records = []
    for entry in re.split(r"(?m)^--0+--\s*$", text):
        entry = entry.strip()
        if not entry or entry.startswith("#"):
            continue
        record = _parse_record(entry)
        if record.get("context") or record.get("mc_gen_prompt"):
            records.append(record)
    return records


def _load_prompt_records(text: str) -> list[dict]:
    records = []
    for entry in re.split(r"(?m)^--0+--\s*$", text):
        prompt = entry.strip()
        if not prompt or prompt.startswith("#"):
            continue
        records.append({"mc_gen_prompt": prompt, "context": prompt.split("\n\n")[-1].strip()})
    return records


def _load_info_records(text: str) -> list[dict]:
    records = []
    for entry in _split_indexed_records(text):
        if not entry or entry.startswith("#"):
            continue
        record = _parse_record(entry)
        if record.get("context") or record.get("true_actions") or record.get("true_options"):
            records.append(record)
    return records


def load_calibration_dataset(
    path: str,
    num_calibration_data: int,
    num_test_data: int,
    domain_name: str,
    info_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    dataset_path = Path(path).expanduser().resolve()
    print("Calibration prompt file:", dataset_path)
    text = dataset_path.read_text(encoding="utf-8")
    if dataset_path.suffix.lower() == ".json":
        data = json.loads(text)
        records = data["records"] if isinstance(data, dict) else data
    elif info_path:
        info_dataset_path = Path(info_path).expanduser().resolve()
        print("Calibration info file:", info_dataset_path)
        info_records = _load_info_records(info_dataset_path.read_text(encoding="utf-8"))
        prompt_records = _load_prompt_records(text)
        dataset_size = min(len(info_records), len(prompt_records))
        records = []
        for index in range(dataset_size):
            merged = dict(info_records[index])
            merged["mc_gen_prompt"] = prompt_records[index]["mc_gen_prompt"]
            if not merged.get("context"):
                merged["context"] = prompt_records[index].get("context", "")
            records.append(merged)
    else:
        records = _load_structured_records(text)
        if not any(record.get("true_actions") or record.get("true_options") for record in records):
            records = _load_prompt_records(text)

    requested_size = num_calibration_data + num_test_data
    if requested_size > len(records):
        raise ValueError(
            f"Requested {requested_size} {domain_name} samples, but only {len(records)} samples are available in "
            f"{dataset_path}. Add more --0000-- examples or lower --num-calibration/--num-test."
        )
    calibration_set = records[:num_calibration_data]
    test_set = records[num_calibration_data:requested_size]
    return calibration_set, test_set


def label_true_options(record: dict, mc_gen_all: list[str], add_mc_prefix: str) -> list[str]:
    if record.get("true_options") and (not record.get("_generated_options") or not record.get("true_actions")):
        return [token.strip().upper() for token in record["true_options"]]

    true_actions = [action.strip().lower() for action in record.get("true_actions", [])]
    true_options = []
    for index, option in enumerate(mc_gen_all):
        option_text = option.lower()
        if any(action == option_text or action in option_text or option_text in action for action in true_actions):
            true_options.append(TOKENS[index])
    return true_options or [add_mc_prefix]


def prepare_calibration_choices(
    dataset: list[dict],
    background: str,
    generation_prompt_builder: Callable[[dict], str],
    generate: bool,
    print_first: int = 0,
) -> None:
    for index, record in enumerate(dataset):
        prompt = generation_prompt_builder(record)
        if generate or ("mc_gen_raw" not in record and not record.get("options")):
            _, raw = call_llm(prompt, logit_bias={})
            raw = raw.strip()
            if index < print_first:
                print(raw)
                print("----")
            record["mc_gen_raw"] = raw
            mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(raw)
            record["mc_gen_full"] = mc_gen_full
            record["mc_gen_all"] = mc_gen_all
            record["add_mc_prefix"] = add_mc_prefix
            record["_generated_options"] = True
        elif record.get("mc_gen_raw"):
            mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(record["mc_gen_raw"].strip())
            record["mc_gen_full"] = mc_gen_full
            record["mc_gen_all"] = mc_gen_all
            record["add_mc_prefix"] = add_mc_prefix
            record["_generated_options"] = False
        elif record.get("options"):
            options = [option.strip().lower() for option in record["options"]]
            options = [option for option in options if option != "an option not listed here"]
            options = list(dict.fromkeys(options))[:4]
            while len(options) < 4:
                options.append("do nothing")
            options.append("an option not listed here")
            record["mc_gen_all"] = options
            record["mc_gen_full"] = "\n".join(f"{TOKENS[i]}) {option}" for i, option in enumerate(options))
            record["add_mc_prefix"] = "E"
            record["_generated_options"] = False

        record["true_options"] = label_true_options(record, record["mc_gen_all"], record["add_mc_prefix"])
        cur_scenario_prompt = prompt.split("\n\n")[-1].strip()
        record["mc_score_prompt"] = (
            f"{background}\n\n"
            f"{cur_scenario_prompt}\n"
            f"{record['mc_gen_full']}"
            "\nWe: Which option is correct? Answer with exactly one capital letter from A, B, C, D, or E."
            "\nYou: "
        )


def score_calibration_choices(dataset: list[dict], logprobs_count: int = 20) -> None:
    total = len(dataset)
    for index, record in enumerate(dataset):
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == total:
            print(f"Scoring calibration record {index + 1}/{total}")
        response, _ = call_llm(record["mc_score_prompt"], max_tokens=1, logprobs=logprobs_count)
        _top_tokens, _top_logprobs, top_logprobs_full = top_choice_logprobs(response)
        option_logprobs = {}
        for raw_token, logprob in top_logprobs_full.items():
            token = raw_token.strip().strip("'\"").upper()
            if token in TOKENS:
                option_logprobs[token] = max(float(logprob), option_logprobs.get(token, -np.inf))
        top_tokens = list(option_logprobs.keys())
        top_logprobs = list(option_logprobs.values())
        record["top_logprobs_full"] = top_logprobs_full
        record["top_tokens"] = top_tokens
        record["top_logprobs"] = top_logprobs
        record["usage"] = response.get("usage")


def calibrate_qhat(records: list[dict], target_success: float) -> float:
    scores = []
    for record in records:
        probs = temperature_scaling(record["top_logprobs"], temperature=5)
        true_probs = [
            prob for token, prob in zip(record["top_tokens"], probs)
            if token in record["true_options"]
        ]
        if not true_probs:
            scores.append(1.0)
            continue
        scores.append(1 - np.max(true_probs))
    q_level = min(1.0, np.ceil((len(records) + 1) * target_success) / len(records))
    return float(np.quantile(scores, q_level, method="higher"))


def summarize_calibration(test_set: list[dict], qhat: float, target_success: float) -> None:
    cover = 0
    help_count = 0
    set_sizes = []
    for record in test_set:
        probs = temperature_scaling(record["top_logprobs"], temperature=5)
        preds = [token for token, prob in zip(record["top_tokens"], probs) if prob >= 1 - qhat]
        set_sizes.append(len(preds))
        cover += not set(preds).isdisjoint(record["true_options"])
        help_count += len(preds) != 1 or record["add_mc_prefix"] in preds

    num_test = len(test_set)
    print("============== Summary ==============")
    print("Number of test data:", num_test)
    print("Quantile value qhat:", qhat)
    print("Average prediction set size:", np.mean(set_sizes) if set_sizes else 0.0)
    print("Marginal coverage guarantee:", target_success)
    print("Empirical coverage:", cover / num_test if num_test else 0.0)
    print("Help rate:", help_count / num_test if num_test else 0.0)
    print("Success rate:", cover / num_test if num_test else 0.0)


def run_knowno_calibration(
    path: str,
    num_calibration_data: int,
    num_test_data: int,
    target_success: float,
    domain_name: str,
    background: str,
    generation_prompt_builder: Callable[[dict], str],
    info_path: str | None = None,
) -> float:
    calibration_set, test_set = load_calibration_dataset(
        path,
        num_calibration_data,
        num_test_data,
        domain_name=domain_name,
        info_path=info_path,
    )

    if calibration_set:
        print("Sample calibration prompt for MC generation:")
        print(calibration_set[0]["mc_gen_prompt"] or generation_prompt_builder(calibration_set[0]))

    print("Running calibration set...printing first five results...")
    prepare_calibration_choices(calibration_set, background, generation_prompt_builder, generate=True, print_first=5)
    print("Running test set...printing first five results...")
    prepare_calibration_choices(test_set, background, generation_prompt_builder, generate=True, print_first=5)

    print("Running calibration set...printing first five results...")
    score_calibration_choices(calibration_set)
    print("Running test set...printing first five results...")
    score_calibration_choices(test_set)

    qhat = calibrate_qhat(calibration_set, target_success)
    summarize_calibration(test_set, qhat, target_success)
    return qhat
