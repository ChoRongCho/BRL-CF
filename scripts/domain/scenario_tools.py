#!/usr/bin/env python3
"""Inspect and validate domain YAML assets.

Usage:
  python3 scripts/domain/scenario_tools.py
  python3 scripts/domain/scenario_tools.py --mode validate
  python3 scripts/domain/scenario_tools.py --mode readme
  python3 scripts/domain/scenario_tools.py --mode summary --domain rover
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR.parent
README_PATH = BASE_DIR / "README.md"
README_START = "<!-- DOMAIN_SUMMARY_START -->"
README_END = "<!-- DOMAIN_SUMMARY_END -->"
REQUIRED_FILES = ("domain_rule.yaml", "robot_skill.yaml", "scene_01.yaml")
NUMERIC_SCENE_KEYS = ("fluents", "true_fluents", "goal_fluents")
NUMERIC_ACTION_KEYS = ("precondition_fluents", "del_effect_fluents", "observation_fluents")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@dataclass(frozen=True)
class DomainSummary:
    name: str
    domain_label: str
    scene_count: int
    type_count: int
    object_count: int
    fact_count: int
    true_init_count: int
    goal_count: int
    action_count: int
    source: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def domain_dirs(selected: str | None = None) -> list[Path]:
    dirs = []
    for path in sorted(BASE_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith("__"):
            continue
        if selected and path.name != selected:
            continue
        if any((path / file_name).exists() for file_name in REQUIRED_FILES):
            dirs.append(path)
    return dirs


def scene_files(domain_dir: Path) -> list[Path]:
    return sorted(domain_dir.glob("scene_*.yaml"))


def count_objects(type_section: dict[str, Any]) -> int:
    total = 0
    for values in (type_section or {}).values():
        if isinstance(values, list):
            total += len(values)
    return total


def summarize_domain(domain_dir: Path) -> DomainSummary:
    scene = load_yaml(domain_dir / "scene_01.yaml")
    rule = load_yaml(domain_dir / "domain_rule.yaml")
    skill = load_yaml(domain_dir / "robot_skill.yaml")
    metadata = rule.get("meta_data", {}) or {}
    types = scene.get("type", {}) or {}
    return DomainSummary(
        name=domain_dir.name,
        domain_label=str(scene.get("domain") or rule.get("domain") or domain_dir.name),
        scene_count=len(scene_files(domain_dir)),
        type_count=len(types),
        object_count=count_objects(types),
        fact_count=len(scene.get("facts", []) or []),
        true_init_count=len(scene.get("true_init", []) or []),
        goal_count=len(scene.get("goal", []) or []),
        action_count=len(skill.get("actions", []) or []),
        source=str(metadata.get("source", "-")),
    )


def markdown_summary(summaries: list[DomainSummary]) -> str:
    lines = [
        README_START,
        "",
        "| Domain | Label | Scenes | Types | Objects | Facts | True Init | Goals | Actions | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| "
            f"`{item.name}` | {item.domain_label} | {item.scene_count} | "
            f"{item.type_count} | {item.object_count} | {item.fact_count} | "
            f"{item.true_init_count} | {item.goal_count} | {item.action_count} | {item.source} |"
        )
    lines.extend(["", README_END])
    return "\n".join(lines)


def print_summary(selected: str | None = None) -> None:
    summaries = [summarize_domain(path) for path in domain_dirs(selected)]
    print(markdown_summary(summaries))


def has_numeric_scene_data(scene: dict[str, Any]) -> list[str]:
    return [key for key in NUMERIC_SCENE_KEYS if scene.get(key)]


def has_numeric_action_data(actions: list[dict[str, Any]]) -> list[str]:
    hits = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = action.get("name", "<unknown>")
        for key in NUMERIC_ACTION_KEYS:
            if action.get(key):
                hits.append(f"{name}.{key}")
    return hits


def validate_domain(domain_dir: Path) -> list[str]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (domain_dir / name).exists()]
    if missing:
        errors.extend(f"missing {name}" for name in missing)
        return errors

    try:
        rule = load_yaml(domain_dir / "domain_rule.yaml")
        skill = load_yaml(domain_dir / "robot_skill.yaml")
        scene = load_yaml(domain_dir / "scene_01.yaml")
    except Exception as exc:
        return [f"YAML parse failed: {exc}"]

    for key in ("domain", "type", "facts", "true_init", "goal"):
        if key not in scene:
            errors.append(f"scene_01.yaml missing key: {key}")

    actions = skill.get("actions", [])
    if not isinstance(actions, list):
        errors.append("robot_skill.yaml actions must be a list")
        actions = []

    numeric_scene_hits = has_numeric_scene_data(scene)
    if numeric_scene_hits:
        errors.append(f"numeric scene fields are not allowed: {', '.join(numeric_scene_hits)}")

    numeric_action_hits = has_numeric_action_data(actions)
    if numeric_action_hits:
        errors.append(f"numeric action fields are not allowed: {', '.join(numeric_action_hits)}")

    try:
        from models.action import get_actions
        from models.state import get_state, get_types
        from utils.asp import DomainRuleBridge, solve_asp

        state, _true_state, _goal = get_state(scene)
        get_actions(actions, state, get_types(scene))

        bridge = DomainRuleBridge()
        bridge.load(domain_dir / "domain_rule.yaml")
        bridge.add_runtime_facts(state.facts)
        worlds = solve_asp(bridge.build_certain_worlds())
        if len(worlds) != 1:
            errors.append(f"ASP certain-world check expected 1 world, got {len(worlds)}")
    except Exception as exc:
        errors.append(f"loader validation failed: {exc}")

    show_entries = rule.get("show", []) or []
    for show in show_entries:
        if isinstance(show, str) and not re.fullmatch(r"#show\s+[A-Za-z_][A-Za-z0-9_]*/\d+", show):
            errors.append(f"invalid show directive: {show}")

    return errors


def validate(selected: str | None = None) -> int:
    failed = False
    for domain_dir in domain_dirs(selected):
        errors = validate_domain(domain_dir)
        if errors:
            failed = True
            print(f"[FAIL] {domain_dir.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            summary = summarize_domain(domain_dir)
            print(
                f"[OK] {domain_dir.name}: "
                f"scenes={summary.scene_count}, actions={summary.action_count}, "
                f"facts={summary.fact_count}, goals={summary.goal_count}"
            )
    return 1 if failed else 0


def update_readme(selected: str | None = None) -> None:
    if selected:
        raise ValueError("--domain cannot be used with --mode readme")

    summaries = [summarize_domain(path) for path in domain_dirs()]
    generated = markdown_summary(summaries)
    current = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""

    if README_START in current and README_END in current:
        before = current.split(README_START, 1)[0].rstrip()
        after = current.split(README_END, 1)[1].lstrip()
        next_text = f"{before}\n\n{generated}\n"
        if after:
            next_text += f"\n{after}"
    else:
        next_text = current.rstrip()
        if next_text:
            next_text += "\n\n"
        next_text += generated + "\n"

    README_PATH.write_text(next_text, encoding="utf-8")
    print(f"updated {README_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect scripts/domain YAML assets")
    parser.add_argument(
        "--mode",
        choices=("summary", "validate", "readme"),
        default="summary",
        help="summary: print markdown table, validate: parse/load domains, readme: update README.md",
    )
    parser.add_argument("--domain", type=str, default=None, help="Limit summary/validation to one domain")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "summary":
        print_summary(args.domain)
        return 0
    if args.mode == "validate":
        return validate(args.domain)
    if args.mode == "readme":
        update_readme(args.domain)
        return 0
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
