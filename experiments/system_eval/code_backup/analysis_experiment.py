"""Stage 1: parse raw experiment logs into normalized run-level CSV files.

This script reads both original POMDP experiment logs and KnowNo baseline logs.

Outputs:
    <script_dir>/data/raw_runs/00_ours/tomato.csv
    <script_dir>/data/raw_runs/00_ours/wastesorting.csv
    <script_dir>/data/raw_runs/00_ours/all_domains.csv
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

EXPERIMENT_ROOTS = (
    ("00_ours", "ours reference"),
    ("01_threshold", "threshold comparison"),
    ("02_when", "when-to-query comparison"),
    ("03_scale", "scale comparison"),
    ("04_answer_mode", "answer-mode comparison"),
    ("05_active_search", "active-search baseline"),
    ("06_knowno", "knowno baseline"),
)
ALL_DOMAINS_NAME = "all_domains"
REFERENCE_BACKED_EXPERIMENTS = {"02_when", "04_answer_mode", "05_active_search", "06_knowno"}

# Edit this block to change which log folders are analyzed.
# Paths are relative to experiments_logs/system_log/{domain}.
STEP50_SCENE_DIRS = tuple(f"scene_{index:02d}_step50" for index in range(1, 6))
ANALYSIS_TARGETS = {
    "tomato": {
        "threshold": (
            "thres_0-0",
            "thres_0-1",
            "thres_0-2",
            "thres_0-3",
            "thres_0-4",
            "thres_0-5",
            "thres_0-6",
            "thres_0-7",
            "thres_0-8",
            "thres_0-9",
            "thres_1-0",
        ),
        "when": (
            "when_all_rand_0-5",
            "when_no_rand_0-5",
            "when_ours_rand_0-5",
            "when_random_rand_0-48",
        ),
        "knowno": (
            "when_knowno_gpt35turbo",
            "when_knowno_gpt4",
        ),
        "scale": (
            "scene_01/scale_ours_step50",
            "scene_02/scale_ours_step50",
            "scene_03/scale_ours_step50",
            "scene_04/scale_ours_step50",
            "scene_05/scale_ours_step50",
            "scene_06/scale_ours_step90",
            "scene_07/scale_ours_step90",
            "scene_08/scale_ours_step90",
            "scene_09/scale_ours_step90",
            "scene_10/scale_ours_step90",
            # "scene_11/scale_ours_step30",
        ),
    },
    "wastesorting": {
        "threshold": (
            "thres_0-0",
            "thres_0-1",
            "thres_0-2",
            "thres_0-3",
            "thres_0-4",
            "thres_0-5",
            "thres_0-6",
            "thres_0-7",
            "thres_0-8",
            "thres_0-9",
            "thres_1-0",
        ),
        "when": (
            "when_all_rand_0-5",
            "when_no_rand_0-5",
            "when_ours_rand_0-5",
            "when_random_rand_0-38",
        ),
        "knowno": (
            "when_knowno_gpt35turbo",
            "when_knowno_gpt4",
        ),
        "scale": (
            "scene_01/scale_ours_step50",
            "scene_02/scale_ours_step50",
            "scene_03/scale_ours_step50",
            "scene_04/scale_ours_step50",
            "scene_05/scale_ours_step50",
            "scene_06/scale_ours_step90",
            "scene_07/scale_ours_step90",
            "scene_08/scale_ours_step90",
            "scene_09/scale_ours_step90",
            "scene_10/scale_ours_step90",
            # "scene_11/scale_ours_step125",
        ),
    },
}

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
    "max_step",
    "max_belief_particles",
    "search_time_total",
    "search_time_avg",
    "execute_time_total",
    "execute_time_avg",
    "update_time_total",
    "update_time_avg",
    "interaction_time_total",
    "interaction_time_avg",
    "pruning_time_total",
    "pruning_time_avg",
    "step_total_time_total",
    "step_total_time_avg",
    "tree_node_count_avg",
    "tree_node_count_max",
    "tree_nodes_expanded_this_step_avg",
    "tree_nodes_expanded_this_step_max",
    "root_action_count_avg",
    "root_action_count_max",
    "max_tree_depth_avg",
    "max_tree_depth_max",
    "belief_frontier_size_avg",
    "belief_frontier_size_max",
    "belief_entropy_avg",
    "belief_entropy_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse raw experiment logs into normalized CSV files.")
    parser.add_argument("--logs-root", default=str(DEFAULT_LOGS_ROOT), help="Root folder containing 00_ours through 06_knowno logs.")
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


def parse_timing_values(text: str) -> dict[str, Any]:
    marker = "[Timing]"
    start = text.find(marker)
    if start == -1:
        return {}
    next_section = text.find("\n[", start + len(marker))
    block = text[start: next_section if next_section != -1 else len(text)]
    values: dict[str, Any] = {}
    for raw_line in block.splitlines()[1:]:
        line = raw_line.strip()
        if not line:
            continue
        total_match = re.match(r"^([a-z_]+):\s*total=([0-9.]+)s,\s*avg=([0-9.]+)s$", line)
        if total_match:
            key = total_match.group(1)
            values[f"{key}_total"] = float(total_match.group(2))
            values[f"{key}_avg"] = float(total_match.group(3))
            continue
        total_time_match = re.match(r"^total_time:\s*([0-9.]+)s$", line)
        if total_time_match:
            values["total_time"] = float(total_time_match.group(1))
    return values


def parse_scale_metrics(text: str) -> dict[str, Any]:
    marker = "[Scale Metrics]"
    start = text.find(marker)
    if start == -1:
        return {}
    next_section = text.find("\n[", start + len(marker))
    block = text[start: next_section if next_section != -1 else len(text)]
    values: dict[str, Any] = {}
    for raw_line in block.splitlines()[1:]:
        line = raw_line.strip()
        match = re.match(r"^([a-z_]+):\s*total=([0-9.]+),\s*avg=([0-9.]+),\s*max=([0-9.]+)$", line)
        if not match:
            continue
        key = match.group(1)
        values[f"{key}_avg"] = float(match.group(3))
        values[f"{key}_max"] = float(match.group(4))
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


def parse_answer_mode_condition(name: str) -> dict[str, str] | None:
    match = re.fullmatch(
        r"(?:(runner_knowno)_)?(?:ours_)?answer_(oracle|human_proxy|noisy_oracle)(?:_noise_([0-9]+-[0-9]+))?_thres_([0-9]+-[0-9]+)(?:_model_[A-Za-z0-9_]+_qhat_[0-9]+-[0-9]+)?",
        name,
    )
    if match is None:
        return None

    runner, answer_mode, noise, threshold = match.groups()
    if answer_mode == "oracle":
        policy = "oracle"
    elif answer_mode == "human_proxy":
        policy = "human_proxy"
    else:
        policy = f"noisy_oracle_{noise.replace('-', '_')}" if noise else "noisy_oracle"
    if runner == "runner_knowno":
        policy = f"knowno_{policy}"

    return {
        "policy": policy,
        "threshold": threshold.replace("-", "."),
    }


def parse_original_log(path: Path, domain: str, run_id: int) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    parts = path.parts
    parent_name = path.parent.name
    if parent_name.startswith("runner_knowno_answer_"):
        return parse_knowno_answer_mode_log(path, text, domain, run_id)

    summary = parse_section_key_values(text, "Plan Summary")
    timing = parse_timing_values(text)
    if not summary:
        return None

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
    elif parent_name.startswith("scale_ours_"):
        experiment = "scalability"
        policy = "ours"
        threshold = parse_line_value(text, "threshold")
    elif parent_name.startswith("answer_") or parent_name.startswith("ours_answer_"):
        condition = parse_answer_mode_condition(parent_name)
        if condition is None:
            return None
        experiment = "answer_mode"
        policy = condition["policy"]
        threshold = condition["threshold"]
    else:
        return None

    steps = int(summary.get("steps", 0) or 0)
    question_steps = parse_question_steps(text)
    scale_metrics = parse_scale_metrics(text)
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
        "max_step": parse_line_value(text, "max_step"),
        "max_belief_particles": parse_line_value(text, "max_belief_particles"),
        "search_time_total": timing.get("search_time_total", ""),
        "search_time_avg": timing.get("search_time_avg", ""),
        "execute_time_total": timing.get("execute_time_total", ""),
        "execute_time_avg": timing.get("execute_time_avg", ""),
        "update_time_total": timing.get("update_time_total", ""),
        "update_time_avg": timing.get("update_time_avg", ""),
        "interaction_time_total": timing.get("interaction_time_total", ""),
        "interaction_time_avg": timing.get("interaction_time_avg", ""),
        "pruning_time_total": timing.get("pruning_time_total", ""),
        "pruning_time_avg": timing.get("pruning_time_avg", ""),
        "step_total_time_total": timing.get("step_total_time_total", ""),
        "step_total_time_avg": timing.get("step_total_time_avg", ""),
        "tree_node_count_avg": scale_metrics.get("tree_node_count_avg", ""),
        "tree_node_count_max": scale_metrics.get("tree_node_count_max", ""),
        "tree_nodes_expanded_this_step_avg": scale_metrics.get("tree_nodes_expanded_this_step_avg", ""),
        "tree_nodes_expanded_this_step_max": scale_metrics.get("tree_nodes_expanded_this_step_max", ""),
        "root_action_count_avg": scale_metrics.get("root_action_count_avg", ""),
        "root_action_count_max": scale_metrics.get("root_action_count_max", ""),
        "max_tree_depth_avg": scale_metrics.get("max_tree_depth_avg", ""),
        "max_tree_depth_max": scale_metrics.get("max_tree_depth_max", ""),
        "belief_frontier_size_avg": scale_metrics.get("belief_frontier_size_avg", ""),
        "belief_frontier_size_max": scale_metrics.get("belief_frontier_size_max", ""),
        "belief_entropy_avg": scale_metrics.get("belief_entropy_avg", ""),
        "belief_entropy_max": scale_metrics.get("belief_entropy_max", ""),
    }


def parse_knowno_answer_mode_log(path: Path, text: str, domain: str, run_id: int) -> dict[str, Any] | None:
    condition = parse_answer_mode_condition(path.parent.name)
    if condition is None:
        return None

    summary = extract_json_after_label(text, "Summary") or parse_knowno_text_summary(text)
    if not summary:
        return None

    metadata = extract_json_after_label(text, "Run metadata")
    token_usage = summary.get("token_usage") or extract_json_after_label(text, "Token usage totals")
    steps = int(summary.get("planning_length", 0) or 0)
    questions = int(summary.get("question_count", 0) or 0)
    query_steps = count_knowno_query_steps(text)
    if query_steps == 0 and questions > 0:
        query_steps = min(questions, steps)

    return {
        "domain": domain,
        "experiment": "answer_mode",
        "policy": condition["policy"],
        "threshold": condition["threshold"],
        "scene": scene_from_parts(path.parts),
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
    for part in reversed(path.parts):
        if part == "when_knowno_gpt35turbo":
            return "knowno_gpt35turbo"
        if part == "when_knowno_gpt4":
            return "knowno_gpt4"
        if part.startswith("when_knowno_"):
            return part.removeprefix("when_")
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


def configured_original_dirs(domain_root: Path, domain: str) -> list[Path]:
    answer_mode_dirs = sorted(domain_root.glob("scene_*_step*/*answer_*"))
    if answer_mode_dirs:
        return [
            directory
            for directory in answer_mode_dirs
            if not directory.name.startswith("answer_oracle")
        ]

    targets = ANALYSIS_TARGETS[domain]
    paths: list[Path] = []

    for scene_dir in STEP50_SCENE_DIRS:
        for condition_dir in targets["threshold"]:
            paths.append(domain_root / scene_dir / condition_dir)
        for condition_dir in targets["when"]:
            paths.append(domain_root / scene_dir / condition_dir)

    paths.extend(domain_root / scale_dir for scale_dir in targets["scale"])
    return paths


def configured_knowno_dirs(domain_root: Path, domain: str) -> list[Path]:
    targets = ANALYSIS_TARGETS[domain]
    return [
        domain_root / scene_dir / condition_dir
        for scene_dir in STEP50_SCENE_DIRS
        for condition_dir in targets["knowno"]
    ]


def collect_txt_files(directories: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        if directory.exists():
            paths.extend(directory.glob("*.txt"))
    return sorted(paths)


def print_target_summary(experiment_root: Path, label: str) -> None:
    if not experiment_root.exists():
        print(f"  {experiment_root.name}: skip missing ({label})")
        return

    if experiment_root.name == "00_ours":
        file_count = len(sorted(experiment_root.glob("*/scene_*_step*/ours_answer_*/*.txt")))
        print(f"  {experiment_root.name}: {file_count} candidate reference files ({label})")
        return

    if experiment_root.name == "06_knowno":
        file_count = len(sorted(experiment_root.glob("*/scene_*/when_knowno_*/*/*.txt")))
        print(f"  {experiment_root.name}: {file_count} candidate KnowNo files ({label})")
        return

    files = 0
    dirs = 0
    for domain in DOMAINS:
        target_dirs = configured_original_dirs(experiment_root / domain, domain)
        dirs += len(target_dirs)
        files += len(collect_txt_files(target_dirs))
    print(f"  {experiment_root.name}: {files} candidate files from {dirs} dirs ({label})")


def collect_original_runs(logs_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for domain in DOMAINS:
        root = logs_root / domain
        if not root.exists():
            continue
        run_id = 0
        for path in collect_txt_files(configured_original_dirs(root, domain)):
            parsed = parse_original_log(path, domain, run_id + 1)
            if parsed is not None:
                run_id += 1
                parsed["run_id"] = run_id
                runs.append(parsed)
    return runs


def collect_ours_reference_runs(reference_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not reference_root.exists():
        return runs

    for domain in DOMAINS:
        root = reference_root / domain
        if not root.exists():
            continue
        directories = sorted(root.glob("scene_*_step*/ours_answer_*"))
        run_id = 0
        for path in collect_txt_files(directories):
            parsed = parse_original_log(path, domain, run_id + 1)
            if parsed is None:
                continue
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
        for path in sorted(root.glob("scene_*/when_knowno_*/*/*.txt")):
            parsed = parse_knowno_log(path, output_domain, run_id + 1, root)
            if parsed is not None and parsed["scene"] in SCENES:
                run_id += 1
                parsed["run_id"] = run_id
                runs.append(parsed)
    return runs


def collect_experiment_runs(system_log_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    print("Analysis targets:")

    for directory_name, label in EXPERIMENT_ROOTS:
        experiment_root = system_log_root / directory_name
        print_target_summary(experiment_root, label)
        if not experiment_root.exists():
            continue

        if directory_name == "00_ours":
            runs.extend(collect_ours_reference_runs(experiment_root))
        elif directory_name == "06_knowno":
            runs.extend(collect_knowno_runs(experiment_root))
        else:
            runs.extend(collect_original_runs(experiment_root))

    return runs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def experiment_dir_for_row(row: dict[str, Any]) -> str:
    source_file = str(row.get("source_file", ""))
    for directory_name, _label in EXPERIMENT_ROOTS:
        marker = f"/{directory_name}/"
        if marker in source_file:
            return directory_name
    return "unknown"


def output_experiment_dirs(rows: list[dict[str, Any]]) -> list[str]:
    present = {experiment_dir_for_row(row) for row in rows}
    return [directory_name for directory_name, _label in EXPERIMENT_ROOTS if directory_name in present]


def rows_for_output_experiment(rows: list[dict[str, Any]], experiment_dir: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if experiment_dir_for_row(row) == experiment_dir]
    if experiment_dir in REFERENCE_BACKED_EXPERIMENTS:
        selected.extend(row for row in rows if experiment_dir_for_row(row) == "00_ours")
    return selected


def write_experiment_outputs(output_root: Path, experiment_dir: str, rows: list[dict[str, Any]]) -> None:
    for domain in DOMAINS:
        domain_runs = [row for row in rows if row["domain"] == domain]
        output_path = output_root / experiment_dir / f"{domain}.csv"
        write_csv(output_path, domain_runs)
        print(f"Wrote {len(domain_runs)} rows to {output_path}")

    all_path = output_root / experiment_dir / f"{ALL_DOMAINS_NAME}.csv"
    write_csv(all_path, rows)
    print(f"Wrote {len(rows)} rows to {all_path}")


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    if args.knowno_root:
        runs = collect_experiment_runs(logs_root)
        runs.extend(collect_knowno_runs(Path(args.knowno_root)))
    else:
        runs = collect_experiment_runs(logs_root)

    output_root = Path(args.output_root) if args.output_root else SCRIPT_DIR / "data" / "raw_runs"
    for experiment_dir in output_experiment_dirs(runs):
        experiment_rows = rows_for_output_experiment(runs, experiment_dir)
        write_experiment_outputs(output_root, experiment_dir, experiment_rows)


if __name__ == "__main__":
    main()
