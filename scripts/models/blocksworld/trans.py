from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


class TransitionBlocksworld:
    def __init__(self, type_map: Dict[str, List[str]], true_state: State):
        self.type_map = type_map
        self.true_state = true_state

        self.pickup_success_rate = 0.95
        self.putdown_success_rate = 0.95
        self.unstack_success_rate = 0.95
        self.stack_success_rate = 0.95

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

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        return outcomes

    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "pickup":
            return self._build_pickup_outcomes(action)

        if action_name == "putdown":
            return self._build_putdown_outcomes(action)

        if action_name == "unstack":
            return self._build_unstack_outcomes(action)

        if action_name == "stack":
            return self._build_stack_outcomes(action)

        return [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=1.0,
            )
        ]

    def _build_pickup_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(action, self.pickup_success_rate)

    def _build_putdown_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(action, self.putdown_success_rate)

    def _build_unstack_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(action, self.unstack_success_rate)

    def _build_stack_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(action, self.stack_success_rate)

    def _build_task_outcomes(self, action: Action, success_rate: float) -> List[TransitionOutcome]:
        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=success_rate,
        )
        failure = self._make_outcome(
            add_facts=[],
            del_facts=[],
            probability=1.0 - success_rate,
        )

        if failure.probability <= 0.0:
            return [success]
        return [success, failure]
