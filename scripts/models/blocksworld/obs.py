from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import ObservationOutcome


class ObservationBlocksworld:
    def __init__(
        self,
        type_map: Dict[str, List[str]],
        noise: float = 0.15,
        true_state: State | None = None,
        observation_source: str = "true_init",
    ):
        self.type_map = type_map
        self.noise = noise
        self.true_state = true_state
        self.observation_source = observation_source

        self.pickup_observation_success_rate = 0.95
        self.putdown_observation_success_rate = 0.95
        self.unstack_observation_success_rate = 0.95
        self.stack_observation_success_rate = 0.95

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "pickup":
            return self._build_default_distribution(state, action, self.pickup_observation_success_rate)

        if action_name == "putdown":
            return self._build_default_distribution(state, action, self.putdown_observation_success_rate)

        if action_name == "unstack":
            return self._build_default_distribution(state, action, self.unstack_observation_success_rate)

        if action_name == "stack":
            return self._build_default_distribution(state, action, self.stack_observation_success_rate)

        return self._build_default_distribution(state, action, 1.0 - self.noise)

    def get_observation_distribution_for_likelihood(
        self,
        state: State,
        action: Action,
    ) -> List[ObservationOutcome]:
        return self.get_observation_distribution(state, action)

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

    def _build_default_distribution(
        self,
        state: State,
        action: Action,
        success_rate: float,
    ) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        true_facts = [fact for fact in candidates if state.has_fact(fact)]

        if not true_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        if success_rate >= 1.0:
            return [ObservationOutcome(facts=true_facts, probability=1.0)]

        return [
            ObservationOutcome(facts=true_facts, probability=success_rate),
            ObservationOutcome(facts=[], probability=1.0 - success_rate),
        ]
