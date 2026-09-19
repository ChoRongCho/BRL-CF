from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_env_setting_path(domain: str) -> Path:
    """Return the canonical environment setting file for a domain."""
    return PROJECT_ROOT / "scripts" / "domain" / domain / "env_setting.yaml"


def load_env_setting(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate a shared environment setting file."""
    setting_path = Path(path)
    if not setting_path.is_file():
        raise FileNotFoundError(f"Environment setting file not found: {setting_path}")
    with setting_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Environment setting must be a mapping: {setting_path}")
    for section in ("transition", "observation"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"Missing mapping '{section}' in {setting_path}")
    _validate_probabilities(data, setting_path)
    return data


def section(settings: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Return a copied nested mapping, or an empty mapping if it is absent."""
    value: Any = settings
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key, {})
    return deepcopy(value) if isinstance(value, dict) else {}


def probability(value: Any, name: str) -> float:
    """Validate and return one probability."""
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return result


def probability_range(value: Iterable[Any], name: str) -> tuple[float, float]:
    """Validate and return a two-element probability range."""
    values = list(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain [min, max]")
    low = probability(values[0], f"{name}[0]")
    high = probability(values[1], f"{name}[1]")
    if low > high:
        raise ValueError(f"{name} minimum cannot exceed maximum")
    return low, high


def _validate_probabilities(data: Dict[str, Any], path: Path) -> None:
    transition = data["transition"].get("success", {})
    if not isinstance(transition, dict):
        raise ValueError(f"transition.success must be a mapping in {path}")
    for action, value in transition.items():
        probability(value, f"transition.success.{action}")

    observation = data["observation"]
    if observation.get("source", "true_init") not in {"true_init", "runtime_state"}:
        raise ValueError(f"observation.source is invalid in {path}")
    if "default_success" in observation:
        probability(observation["default_success"], "observation.default_success")

    detect = observation.get("detect", {})
    if not isinstance(detect, dict):
        raise ValueError(f"observation.detect must be a mapping in {path}")
    for key in ("likelihood_success", "classification_success", "false_positive"):
        if key in detect:
            probability(detect[key], f"observation.detect.{key}")
    for key in (
        "sampling_success_range",
        "true_detection_confidence_range",
        "false_detection_confidence_range",
    ):
        if key in detect:
            probability_range(detect[key], f"observation.detect.{key}")

    for action in ("scan", "navigate", "pick", "place"):
        action_setting = observation.get(action, {})
        if action_setting and not isinstance(action_setting, dict):
            raise ValueError(f"observation.{action} must be a mapping in {path}")
        if "success" in action_setting:
            probability(action_setting["success"], f"observation.{action}.success")
