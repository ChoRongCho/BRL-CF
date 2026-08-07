"""Configuration shared by the Attr-POMDP entry points."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from baseline.attr_pomdp.scripts.domains import SUPPORTED_DOMAINS
from utils.arguments import parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


@dataclass(frozen=True)
class Defaults:
    """Editable defaults; an explicitly supplied CLI option takes precedence."""

    domain: str = "tomato"
    scene: str = "01"
    feedback_source: str = "oracle"
    query_cost: float = 1.0
    action_limit: int = 20
    query_limit: int = 20
    max_depth: int = 2
    max_step: int = 25
    seed: int | None = None


def parse_configuration(argv: Sequence[str], defaults: Defaults):
    """Parse Attr-POMDP options, then delegate common options to the project parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--domain", choices=SUPPORTED_DOMAINS, default=defaults.domain)
    parser.add_argument("--scene", default=defaults.scene)
    parser.add_argument("--robot_skill", "--robot-skill", help=argparse.SUPPRESS)
    parser.add_argument(
        "--feedback-source",
        choices=("oracle", "vlm", "human"),
        default=defaults.feedback_source,
    )
    parser.add_argument("--query-cost", type=float, default=defaults.query_cost)
    parser.add_argument("--action-limit", type=int, default=defaults.action_limit)
    parser.add_argument("--query-limit", type=int, default=defaults.query_limit)
    parser.add_argument("--max-depth", "--max_depth", type=int, default=defaults.max_depth)
    attr_args, common_argv = parser.parse_known_args(argv)

    try:
        scene = f"{int(attr_args.scene):02d}"
    except ValueError as exc:
        raise SystemExit("--scene must be an integer") from exc

    scene_path = SCRIPTS_DIR / "domain" / attr_args.domain / f"scene_{scene}.yaml"
    if not scene_path.exists():
        raise SystemExit(f"Scene file does not exist: {scene_path}")

    robot_skill_path = (
        SCRIPTS_DIR
        / "baseline"
        / "attr_pomdp"
        / "domains"
        / attr_args.domain
        / "robot_skill.yaml"
    )
    if not robot_skill_path.exists():
        raise SystemExit(f"Attr-POMDP robot-skill file does not exist: {robot_skill_path}")

    aliases = {"--log-dir": "--log_dir", "--max-step": "--max_step"}
    translated = [
        "--domain",
        attr_args.domain,
        "--initial_state",
        str(scene_path),
        "--robot_skill",
        str(robot_skill_path),
        "--max_depth",
        str(attr_args.max_depth),
        "--max_step",
        str(defaults.max_step),
    ]
    if defaults.seed is not None:
        translated.extend(("--seed", str(defaults.seed)))
    translated.extend(aliases.get(token, token) for token in common_argv)

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *translated]
        common_args = parse_args(attr_args.domain)
    finally:
        sys.argv = original_argv

    return common_args, attr_args
