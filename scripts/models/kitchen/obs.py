from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import ObservationOutcome


class ObservationKitchen:
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

        self.open_fridge_observation_success_rate = 0.90
        self.inspect_observation_success_rate = 0.90
        self.pick_observation_success_rate = 0.95
        self.wash_observation_success_rate = 0.95
        self.place_observation_success_rate = 0.95
        self.boil_observation_success_rate = 0.95
        self.serve_observation_success_rate = 0.95

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "open_fridge":
            return self._build_open_fridge_distribution(state, action)

        if action_name in {"inspect_ingredient", "inspect_pot", "inspect_dish"}:
            return self._build_inspect_distribution(state, action)

        if action_name in {"pick_ingredient", "pick_ingredient_from_pot", "pick_pot", "pick_dish"}:
            return self._build_default_distribution(state, action, self.pick_observation_success_rate)

        if action_name in {"wash_ingredient", "wash_pot", "wash_dish"}:
            return self._build_default_distribution(state, action, self.wash_observation_success_rate)

        if action_name in {"place_pot", "place_dish", "place_ingredient_in_pot", "discard_ingredient"}:
            return self._build_default_distribution(state, action, self.place_observation_success_rate)

        if action_name in {"boil_tomato_soup", "boil_onion_soup", "boil_tomato_oniton_soup"}:
            return self._build_default_distribution(state, action, self.boil_observation_success_rate)

        if action_name in {"serve_tomato_soup", "serve_onion_soup", "serve_tomato_oniton_soup"}:
            return self._build_default_distribution(state, action, self.serve_observation_success_rate)

        return self._build_default_distribution(state, action, 1.0 - self.noise)

    def get_observation_distribution_for_likelihood(
        self,
        state: State,
        action: Action,
    ) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "open_fridge":
            return self._build_open_fridge_distribution(state, action, use_true_state=False)

        if action_name in {"inspect_ingredient", "inspect_pot", "inspect_dish"}:
            return self._build_inspect_distribution(state, action, use_true_state=False)

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

    @staticmethod
    def _known_cleanliness_fact(state: State, obj: str) -> str | None:
        if state.has_fact(f"clean({obj})"):
            return f"clean({obj})"
        if state.has_fact(f"dirty({obj})"):
            return f"dirty({obj})"
        return None

    @staticmethod
    def _cleanliness_fact_for_state(state: State, obj: str) -> str | None:
        if state.has_fact(f"clean({obj})"):
            return f"clean({obj})"
        if state.has_fact(f"dirty({obj})"):
            return f"dirty({obj})"
        return None

    def _build_open_fridge_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        args = self._get_action_args(action)
        if len(args) < 3:
            return [ObservationOutcome(facts=[], probability=1.0)]

        _, ingredient, fridge = args[:3]
        target_fact = f"in_fridge({ingredient},{fridge})"
        if self._is_holding_object(state, ingredient):
            return [ObservationOutcome(facts=[], probability=1.0)]

        ref_state = self._observation_truth_state(state, use_true_state)
        if not ref_state.has_fact(target_fact):
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(
                facts=[target_fact],
                probability=self.open_fridge_observation_success_rate,
            ),
            ObservationOutcome(
                facts=[],
                probability=1.0 - self.open_fridge_observation_success_rate,
            ),
        ]

    def _build_inspect_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        args = self._get_action_args(action)
        if len(args) < 2:
            return [ObservationOutcome(facts=[], probability=1.0)]

        obj = args[1]
        label = self._known_cleanliness_fact(state, obj)
        if label is None:
            ref_state = self._observation_truth_state(state, use_true_state)
            label = self._cleanliness_fact_for_state(ref_state, obj)

        if label is None:
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(
                facts=[label],
                probability=self.inspect_observation_success_rate,
            ),
            ObservationOutcome(
                facts=[],
                probability=1.0 - self.inspect_observation_success_rate,
            ),
        ]

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
