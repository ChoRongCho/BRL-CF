# models/actions.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class ActionType(str, Enum):
    TASK = "task"
    OBSERVE = "observe"


@dataclass(frozen=True)
class Action:
    """
    Symbolic action schema loaded from YAML.

    Example YAML:
        actions:
          - name: move
            type: task
            parameters: [from, to]
            cost: 1.0
            preconditions:
              - robot_at_{from}
            add_effects:
              - robot_at_{to}
            del_effects:
              - robot_at_{from}
    """
    name: str
    action_type: ActionType
    parameters: Tuple[str, ...] = field(default_factory=tuple)

    preconditions: Tuple[str, ...] = field(default_factory=tuple)
    add_effects: Tuple[str, ...] = field(default_factory=tuple)
    del_effects: Tuple[str, ...] = field(default_factory=tuple)

    cost: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        if not self.parameters:
            return self.name
        return f"{self.name}({', '.join(self.parameters)})"

    def is_observation_action(self) -> bool:
        return self.action_type == ActionType.OBSERVE

    def is_task_action(self) -> bool:
        return self.action_type == ActionType.TASK

    def __str__(self) -> str:
        return self.signature()


@dataclass
class ActionLibrary:
    actions: List[Action] = field(default_factory=list)

    def add(self, action: Action) -> None:
        self.actions.append(action)

    def extend(self, actions: List[Action]) -> None:
        self.actions.extend(actions)

    def all(self) -> List[Action]:
        return list(self.actions)

    def by_name(self, name: str) -> List[Action]:
        return [a for a in self.actions if a.name == name]

    def observation_actions(self) -> List[Action]:
        return [a for a in self.actions if a.is_observation_action()]

    def task_actions(self) -> List[Action]:
        return [a for a in self.actions if a.is_task_action()]


# utils function of action
def _ensure_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"'{field_name}' must be a list, got {type(value).__name__}")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"All items in '{field_name}' must be strings")
    return value


def _parse_action_type(value: str) -> ActionType:
    try:
        return ActionType(value)
    except ValueError as e:
        valid = [t.value for t in ActionType]
        raise ValueError(f"Invalid action type '{value}'. Valid types: {valid}") from e


def _parse_action(raw: Dict[str, Any]) -> Action:
    if "name" not in raw:
        raise ValueError("Each action must have a 'name'")
    if "type" not in raw:
        raise ValueError(f"Action '{raw.get('name', '<unknown>')}' must have a 'type'")

    name = raw["name"]
    action_type = _parse_action_type(raw["type"])

    if not isinstance(name, str):
        raise TypeError("'name' must be a string")

    parameters = tuple(_ensure_list(raw.get("parameters", []), "parameters"))
    preconditions = tuple(_ensure_list(raw.get("preconditions", []), "preconditions"))
    add_effects = tuple(_ensure_list(raw.get("add_effects", []), "add_effects"))
    del_effects = tuple(_ensure_list(raw.get("del_effects", []), "del_effects"))

    cost = raw.get("cost", 1.0)
    if not isinstance(cost, (int, float)):
        raise TypeError(f"Action '{name}': 'cost' must be numeric")
    cost = float(cost)

    reserved_keys = {
        "name",
        "type",
        "parameters",
        "preconditions",
        "add_effects",
        "del_effects",
        "cost",
    }
    metadata = {k: v for k, v in raw.items() if k not in reserved_keys}

    return Action(
        name=name,
        action_type=action_type,
        parameters=parameters,
        preconditions=preconditions,
        add_effects=add_effects,
        del_effects=del_effects,
        cost=cost,
        metadata=metadata,
    )


def load_actions_from_yaml(path: str | Path) -> ActionLibrary:
    """
    Load action schemas from a YAML file.

    Expected YAML format:
        actions:
          - name: move
            type: task
            parameters: [from, to]
            cost: 1.0
            preconditions:
              - robot_at_{from}
            add_effects:
              - robot_at_{to}
            del_effects:
              - robot_at_{from}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty YAML file: {path}")
    if not isinstance(data, dict):
        raise TypeError("Top-level YAML structure must be a dictionary")
    if "actions" not in data:
        raise ValueError("YAML must contain top-level key: 'actions'")
    if not isinstance(data["actions"], list):
        raise TypeError("'actions' must be a list")

    library = ActionLibrary()
    for idx, raw_action in enumerate(data["actions"]):
        if not isinstance(raw_action, dict):
            raise TypeError(f"actions[{idx}] must be a dictionary")
        action = _parse_action(raw_action)
        library.add(action)

    return library