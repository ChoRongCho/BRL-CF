from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


class TransitionRover:
    def __init__(self, type_map: Dict[str, List[str]], true_state: State):
        self.type_map = type_map
        self.true_state = true_state

        self.navigate_success_rate = 0.95
        self.detect_road_success_rate = 0.90
        self.detect_soil_success_rate = 0.90
        self.sample_soil_success_rate = 0.90
        self.take_image_success_rate = 0.90
        self.communicate_success_rate = 0.95

    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        pred, args = _parse_fact(fact)
        variable_positions = []
        variable_domains = []

        for i, arg in enumerate(args):
            if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", arg):
                continue

            type_symbol = arg[0]
            if type_symbol not in self.type_map:
                raise ValueError(f"Unknown type symbol for variable {arg}")

            variable_positions.append(i)
            variable_domains.append(self.type_map[type_symbol])

        if not variable_positions:
            return [fact.replace(" ", "")]

        expanded = []

        def backtrack(depth: int, current_args: List[str]):
            if depth == len(variable_positions):
                expanded.append(_format_fact(pred, current_args))
                return

            pos = variable_positions[depth]
            for obj in variable_domains[depth]:
                next_args = current_args[:]
                next_args[pos] = obj
                backtrack(depth + 1, next_args)

        backtrack(0, args[:])
        return expanded

    @staticmethod
    def _make_outcome(add_facts: List[str], del_facts: List[str], probability: float) -> TransitionOutcome:
        return TransitionOutcome(
            add_facts=_dedup_facts(add_facts),
            del_facts=_dedup_facts(del_facts),
            probability=probability,
        )

    @staticmethod
    def _get_action_args(action: Action) -> List[str]:
        _, args = _parse_fact(action.name.replace(" ", ""))
        return args

    def _true_has_fact(self, fact: str) -> bool:
        return self.true_state.has_fact(fact.replace(" ", ""))

    def _true_observation_facts(self, action: Action) -> List[str]:
        expanded_obs = []
        for obs in action.observation:
            expanded_obs.extend(self._expand_free_variables_in_fact(obs))
        return [fact for fact in _dedup_facts(expanded_obs) if self._true_has_fact(fact)]

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        action_name = action.name.split("(", 1)[0]
        if action_name in {"detect_road", "detect_soil"}:
            return self._build_detect_outcomes(action, state)
        return outcomes

    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "navigate":
            return self._build_navigate_outcomes(action)

        if action_name == "detect_road":
            return self._build_detect_outcomes(action, success_rate=self.detect_road_success_rate)

        if action_name == "detect_soil":
            return self._build_detect_outcomes(action, success_rate=self.detect_soil_success_rate)

        if action_name == "sample_soil":
            return self._build_sample_soil_outcomes(action)

        if action_name == "take_image":
            return self._build_take_image_outcomes(action)

        if action_name in {"communicate_soil_data", "communicate_image_data"}:
            return self._build_communicate_outcomes(action)

        return [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=1.0,
            )
        ]

    def _build_navigate_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            action,
            success_rate=self.navigate_success_rate,
            failure_add_facts=[],
        )

    def _build_detect_outcomes(
        self,
        action: Action,
        state: State | None = None,
        success_rate: float | None = None,
    ) -> List[TransitionOutcome]:
        if success_rate is None:
            action_name = action.name.split("(", 1)[0]
            success_rate = (
                self.detect_soil_success_rate
                if action_name == "detect_soil"
                else self.detect_road_success_rate
            )

        true_facts = self._true_observation_facts(action)
        if state is not None:
            true_facts = [fact for fact in true_facts if not state.has_fact(fact)]

        if not true_facts:
            return [self._make_outcome([], [], 1.0)]

        return [
            self._make_outcome(
                add_facts=true_facts,
                del_facts=[],
                probability=success_rate,
            ),
            self._make_outcome(
                add_facts=[],
                del_facts=[],
                probability=1.0 - success_rate,
            ),
        ]

    def _build_sample_soil_outcomes(self, action: Action) -> List[TransitionOutcome]:
        failure_add_facts = []
        for fact in action.del_effects:
            if fact.replace(" ", "").startswith("at_soil_sample("):
                failure_add_facts.append(fact)

        return self._build_task_outcomes(
            action,
            success_rate=self.sample_soil_success_rate,
            failure_add_facts=failure_add_facts,
        )

    def _build_take_image_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            action,
            success_rate=self.take_image_success_rate,
            failure_add_facts=[],
        )

    def _build_communicate_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            action,
            success_rate=self.communicate_success_rate,
            failure_add_facts=[],
        )

    def _build_task_outcomes(
        self,
        action: Action,
        success_rate: float,
        failure_add_facts: List[str],
    ) -> List[TransitionOutcome]:
        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=success_rate,
        )

        failure = self._make_outcome(
            add_facts=failure_add_facts,
            del_facts=[],
            probability=1.0 - success_rate,
        )

        if failure.probability <= 0.0:
            return [success]
        return [success, failure]
