from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import ObservationOutcome


class ObservationWatering:
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
        self.use_true_init_observation = observation_source == "true_init"

        self.move_observation_success_rate = 0.95
        self.find_observation_success_rate = 0.90
        self.pick_observation_success_rate = 0.95
        self.load_water_observation_success_rate = 0.95
        self.pour_water_observation_success_rate = 0.95

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "find_basket":
            return self._build_find_basket_distribution(state, action)

        if action_name == "find_plant":
            return self._build_find_plant_distribution(state, action)

        if action_name == "move":
            return self._build_default_distribution(
                state,
                action,
                self.move_observation_success_rate,
            )

        if action_name == "pick_basket":
            return self._build_default_distribution(
                state,
                action,
                self.pick_observation_success_rate,
            )

        if action_name == "load_water":
            return self._build_default_distribution(
                state,
                action,
                self.load_water_observation_success_rate,
            )

        if action_name == "pour_water":
            return self._build_default_distribution(
                state,
                action,
                self.pour_water_observation_success_rate,
            )

        return self._build_default_distribution(state, action, 1.0 - self.noise)

    def get_observation_distribution_for_likelihood(
        self,
        state: State,
        action: Action,
    ) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "find_basket":
            return self._build_find_basket_distribution(state, action, use_true_state=False)

        if action_name == "find_plant":
            return self._build_find_plant_distribution(state, action, use_true_state=False)

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

    @staticmethod
    def _get_action_args(action: Action) -> List[str]:
        _, args = _parse_fact(action.name.replace(" ", ""))
        return args

    def _observation_truth_state(self, state: State, use_true_state: bool) -> State:
        if use_true_state and self.use_true_init_observation:
            return self.true_state
        return state

    @staticmethod
    def _is_holding_object(state: State, obj: str) -> bool:
        for fact in state.facts:
            pred, args = _parse_fact(fact)
            if pred == "holding" and len(args) >= 2 and args[1] == obj:
                return True
        return False

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

    def _build_find_basket_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        args = self._get_action_args(action)
        if len(args) < 3:
            return [ObservationOutcome(facts=[], probability=1.0)]

        _, container, room = args[:3]
        if self._is_holding_object(state, container):
            return [ObservationOutcome(facts=[], probability=1.0)]

        return self._build_find_location_distribution(
            state,
            action,
            obj=container,
            room=room,
            use_true_state=use_true_state,
        )

    def _build_find_plant_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        args = self._get_action_args(action)
        if len(args) < 3:
            return [ObservationOutcome(facts=[], probability=1.0)]

        _, plant, room = args[:3]
        return self._build_find_location_distribution(
            state,
            action,
            obj=plant,
            room=room,
            use_true_state=use_true_state,
        )

    def _build_find_location_distribution(
        self,
        state: State,
        action: Action,
        obj: str,
        room: str,
        use_true_state: bool,
    ) -> List[ObservationOutcome]:
        target_fact = f"at({obj},{room})"
        candidates = self.build_candidates(action)
        if target_fact not in candidates:
            return [ObservationOutcome(facts=[], probability=1.0)]

        gt_state = self._observation_truth_state(state, use_true_state)
        true_facts = [target_fact] if gt_state.has_fact(target_fact) else []

        if not true_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(
                facts=true_facts,
                probability=self.find_observation_success_rate,
            ),
            ObservationOutcome(
                facts=[],
                probability=1.0 - self.find_observation_success_rate,
            ),
        ]
