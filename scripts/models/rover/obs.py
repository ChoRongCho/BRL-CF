from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import ObservationOutcome


class ObservationRover:
    def __init__(
        self,
        type_map: Dict[str, List[str]],
        noise: float = 0.15,
        true_state: State | None = None,
        world=None,
        observation_source: str = "true_init",
    ):
        self.type_map = type_map
        self.noise = noise
        self.world = world
        self.true_state = world.true_state if world is not None else true_state
        self.observation_source = observation_source
        self.use_true_init_observation = observation_source == "true_init"

        self.navigate_observation_success_rate = 0.95
        self.detect_road_observation_success_rate = 0.90
        self.detect_soil_observation_success_rate = 0.90
        self.sample_soil_observation_success_rate = 0.95
        self.take_image_observation_success_rate = 0.95
        self.communicate_observation_success_rate = 0.95

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "detect_road":
            return self._build_detect_distribution(
                state,
                action,
                self.detect_road_observation_success_rate,
            )

        if action_name == "detect_soil":
            return self._build_detect_distribution(
                state,
                action,
                self.detect_soil_observation_success_rate,
            )

        if action_name == "navigate":
            return self._build_default_distribution(
                state,
                action,
                self.navigate_observation_success_rate,
            )

        if action_name == "sample_soil":
            return self._build_default_distribution(
                state,
                action,
                self.sample_soil_observation_success_rate,
            )

        if action_name == "take_image":
            return self._build_default_distribution(
                state,
                action,
                self.take_image_observation_success_rate,
            )

        if action_name in {"communicate_soil_data", "communicate_image_data"}:
            return self._build_default_distribution(
                state,
                action,
                self.communicate_observation_success_rate,
            )

        return self._build_default_distribution(state, action, 1.0 - self.noise)

    def get_observation_distribution_for_likelihood(
        self,
        state: State,
        action: Action,
    ) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]
        if action_name == "detect_road":
            return self._build_detect_distribution(
                state,
                action,
                self.detect_road_observation_success_rate,
                use_true_state=False,
            )
        if action_name == "detect_soil":
            return self._build_detect_distribution(
                state,
                action,
                self.detect_soil_observation_success_rate,
                use_true_state=False,
            )
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

    def _observation_truth_state(self, state: State, use_true_state: bool) -> State:
        if use_true_state and self.use_true_init_observation:
            return self.true_state
        return state

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

    def _build_detect_distribution(
        self,
        state: State,
        action: Action,
        success_rate: float,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        if not candidates:
            return [ObservationOutcome(facts=[], probability=1.0)]

        gt_state = self._observation_truth_state(state, use_true_state)
        true_facts = [fact for fact in candidates if gt_state.has_fact(fact)]

        if not true_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(
                facts=true_facts,
                probability=success_rate,
            ),
            ObservationOutcome(
                facts=[],
                probability=1.0 - success_rate,
            ),
        ]
