"""KnowNo adapter for the environment settings shared with the POMDP models."""

from __future__ import annotations

import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.env_setting import default_env_setting_path, load_env_setting


def add_env_setting_argument(parser, domain: str) -> None:
    parser.add_argument(
        "--env-setting",
        default=str(default_env_setting_path(domain)),
        help="Shared transition/observation distribution yaml.",
    )


def apply_env_setting(args, domain: str):
    """Fill unspecified KnowNo simulator values from the shared YAML."""
    settings = load_env_setting(args.env_setting)
    if settings.get("domain") not in {None, domain}:
        raise ValueError(
            f"Environment setting domain {settings.get('domain')!r} does not match {domain!r}"
        )

    observation = settings["observation"]
    detect = observation["detect"]
    transition = settings["transition"]["success"]

    if args.detect_success_prob is None:
        args.detect_success_prob = tuple(float(v) for v in detect["sampling_success_range"])
    if args.detect_label_error_prob is None:
        args.detect_label_error_prob = round(
            1.0 - float(detect["classification_success"]), 12
        )
    args.detect_false_positive_prob = float(detect.get("false_positive", 0.0))

    if domain == "tomato":
        # The POMDP scan model always returns one label; `success` is its label accuracy.
        if args.scan_success_prob is None:
            args.scan_success_prob = 1.0
        if args.scan_label_error_prob is None:
            args.scan_label_error_prob = round(
                1.0 - float(observation["scan"]["success"]), 12
            )
        for name in ("navigate", "pick", "place", "discard"):
            attr = f"{name}_failure_prob"
            if getattr(args, attr) is None:
                setattr(args, attr, round(1.0 - float(transition[name]), 12))
    elif domain == "wastesorting":
        for name in ("pick", "place"):
            attr = f"{name}_failure_prob"
            if getattr(args, attr) is None:
                setattr(args, attr, round(1.0 - float(transition[name]), 12))

    args.env_setting_data = settings
    return settings


def draw_probability(value) -> float:
    """Resolve either a scalar probability or a shared [min, max] range."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Probability range must contain two values: {value!r}")
        return random.uniform(float(value[0]), float(value[1]))
    return float(value)
