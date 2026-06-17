from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
BASELINE_DIR = SCRIPTS_DIR / "baseline"
BASELINE_SCRIPT_DIR = BASELINE_DIR / "scripts"
DOMAIN_DIR = SCRIPTS_DIR / "domain"
LOGS_DIR = PROJECT_ROOT / "experiments_logs" / "system_log"

TOMATO_PROPERTIES = {"ripe", "unripe", "rotten"}
WASTE_LABELS = {"general", "plastic", "paper", "can"}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a domain-specific KnowNo baseline experiment from one entrypoint."
    )
    parser.add_argument("--domain", choices=["tomato", "wastesorting", "waste"], required=True)
    parser.add_argument("--scene", default="01", help="Scene number such as 01, 1, 02, ..., 05.")
    parser.add_argument("--settings", default=str(BASELINE_DIR / "llm_setting.json"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--qhat", type=float, default=None)
    parser.add_argument("--score-temperature", "--temperature", dest="score_temperature", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--detect-success-prob", type=float, default=None)
    parser.add_argument("--detect-label-error-prob", type=float, default=None)
    parser.add_argument("--scan-success-prob", type=float, default=None)
    parser.add_argument("--scan-label-error-prob", type=float, default=None)
    parser.add_argument("--navigate-failure-prob", type=float, default=None)
    parser.add_argument("--pick-failure-prob", type=float, default=None)
    parser.add_argument("--place-failure-prob", type=float, default=None)
    parser.add_argument("--discard-failure-prob", type=float, default=None)
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--num-calibration", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument("--target-success", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running the baseline.")
    return parser.parse_known_args()


def scene_id(value: str) -> str:
    text = value.strip().lower().removeprefix("scene_").removeprefix("scene")
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid scene value: {value!r}") from exc
    if number < 1:
        raise ValueError(f"Scene number must be positive: {value!r}")
    return f"{number:02d}"


def read_true_init(scene_path: Path) -> list[str]:
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    facts: list[str] = []
    in_true_init = False
    for raw_line in scene_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"true_init\s*:", line):
            in_true_init = True
            continue
        if in_true_init and re.fullmatch(r"[A-Za-z_][\w-]*\s*:", line):
            break
        if not in_true_init:
            continue

        match = re.match(r'-\s*["\']?([^"\']+)["\']?\s*$', line)
        if match:
            facts.append(normalize_fact(match.group(1)))
    if not facts:
        raise ValueError(f"No true_init facts found in {scene_path}")
    return facts


def normalize_fact(fact: str) -> str:
    return re.sub(r"\s+", "", fact.strip())


def natural_key(name: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)", name)
    if not match:
        return name, 999999, name
    return name[: match.start()], int(match.group(1)), name[match.end() :]


def format_mapping(mapping: dict[str, str]) -> str:
    return ",".join(f"{key}:{mapping[key]}" for key in sorted(mapping, key=natural_key))


def tomato_args_from_scene(scene_path: Path) -> list[str]:
    labels: dict[str, str] = {}
    locations: dict[str, str] = {}

    for fact in read_true_init(scene_path):
        match = re.fullmatch(r"(ripe|unripe|rotten)\(([^)]+)\)", fact)
        if match:
            labels[match.group(2)] = match.group(1)
            continue
        match = re.fullmatch(r"at\(([^,]+),([^)]+)\)", fact)
        if match:
            locations[match.group(1)] = match.group(2)

    if not labels:
        raise ValueError(f"No tomato property facts found in {scene_path}")
    unknown_labels = set(labels.values()) - TOMATO_PROPERTIES
    if unknown_labels:
        raise ValueError(f"Unsupported tomato labels in {scene_path}: {sorted(unknown_labels)}")

    args = ["--labels", format_mapping(labels)]
    if locations:
        args.extend(["--locations", format_mapping(locations)])
    return args


def waste_args_from_scene(scene_path: Path) -> list[str]:
    labels: dict[str, str] = {}
    for fact in read_true_init(scene_path):
        match = re.fullmatch(r"(general|plastic|paper|can)\(([^)]+)\)", fact)
        if match:
            labels[match.group(2)] = match.group(1)

    if not labels:
        raise ValueError(f"No waste label facts found in {scene_path}")
    unknown_labels = set(labels.values()) - WASTE_LABELS
    if unknown_labels:
        raise ValueError(f"Unsupported waste labels in {scene_path}: {sorted(unknown_labels)}")
    return ["--labels", format_mapping(labels)]


def waste_occlusions_for_scene(scene: str) -> str:
    if scene in {"02", "03"}:
        return "waste4:waste3"
    if scene in {"04", "05"}:
        return "waste3:waste1,waste4:waste2"
    return ""


def append_optional(cmd: list[str], flag: str, value) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def model_slug(settings_path: str) -> str:
    try:
        settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    model = str(settings.get("model", "")).lower()
    if "gpt-3.5" in model or "gpt-35" in model:
        return "gpt35turbo"
    if "gpt-4" in model:
        return "gpt4"
    return re.sub(r"[^a-z0-9]+", "", model) or "unknown"


def default_log_file(domain: str, scene: str, settings_path: str) -> Path:
    log_dir = LOGS_DIR / domain / f"scene_{scene}_step50" / f"when_knowno_{model_slug(settings_path)}"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"knowno_{domain}_{timestamp}.txt"


def build_command(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    domain = "wastesorting" if args.domain == "waste" else args.domain
    scene = scene_id(args.scene)
    scene_path = DOMAIN_DIR / domain / f"scene_{scene}.yaml"

    if domain == "tomato":
        planner = BASELINE_SCRIPT_DIR / "knowno_multistep_tomato.py"
        scene_args = tomato_args_from_scene(scene_path)
    else:
        planner = BASELINE_SCRIPT_DIR / "knowno_multistep_wastesorting.py"
        scene_args = waste_args_from_scene(scene_path)
        append_optional(scene_args, "--occlusions", waste_occlusions_for_scene(scene))

    cmd = [sys.executable, str(planner), "--settings", args.settings]
    append_optional(cmd, "--api-key", args.api_key)
    append_optional(cmd, "--prompt-version", args.prompt_version)
    append_optional(cmd, "--qhat", args.qhat)
    append_optional(cmd, "--temperature", args.score_temperature)
    append_optional(cmd, "--max-steps", args.max_steps)
    append_optional(cmd, "--seed", args.seed)
    append_optional(cmd, "--log-file", args.log_file or default_log_file(domain, scene, args.settings))
    append_optional(cmd, "--detect-success-prob", args.detect_success_prob)
    append_optional(cmd, "--detect-label-error-prob", args.detect_label_error_prob)

    if domain == "tomato":
        append_optional(cmd, "--scan-success-prob", args.scan_success_prob)
        append_optional(cmd, "--scan-label-error-prob", args.scan_label_error_prob)
        append_optional(cmd, "--navigate-failure-prob", args.navigate_failure_prob)
        append_optional(cmd, "--pick-failure-prob", args.pick_failure_prob)
        append_optional(cmd, "--place-failure-prob", args.place_failure_prob)
        append_optional(cmd, "--discard-failure-prob", args.discard_failure_prob)

    if args.verbose:
        cmd.append("--verbose")
    if args.run_calibration:
        cmd.append("--run-calibration")
    append_optional(cmd, "--num-calibration", args.num_calibration)
    append_optional(cmd, "--num-test", args.num_test)
    append_optional(cmd, "--target-success", args.target_success)

    cmd.extend(scene_args)
    cmd.extend(passthrough)
    return cmd


def shell_quote(items: list[str]) -> str:
    return " ".join("'" + item.replace("'", "'\"'\"'") + "'" for item in items)


def main() -> None:
    args, passthrough = parse_args()
    cmd = build_command(args, passthrough)
    print("Running:", shell_quote(cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    pythonpath_parts = [str(BASELINE_DIR)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()
