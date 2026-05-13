from __future__ import annotations

from itertools import product
from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import ObservationOutcome


Choice = Dict


class ObservationWastesorting:
    CATEGORY_PREDICATES = ("plastic", "can", "paper", "general")
    PLACE_ACTIONS = {
        "place_gw_bin",
        "place_paper_bin",
        "place_can_bin",
        "place_plastic_bin",
    }

    def __init__(
        self,
        type_map: Dict[str, List[str]],
        noise: float = 0.05,
        true_state: State | None = None,
        observation_source: str = "true_init",
    ):
        self.type_map = type_map
        self.noise = noise
        self.true_state = true_state
        self.observation_source = observation_source
        self.use_true_init_observation = observation_source == "true_init"

        # # original
        # self.detect_observed_success_rate = 0.9
        # self.detect_classification_success_rate = 0.8
        # self.pick_observation_success_rate = 0.98
        # self.place_observation_success_rate = 0.98
        
        self.detect_observed_success_rate = 0.9
        self.detect_classification_success_rate = 0.8
        self.pick_observation_success_rate = 0.95
        self.place_observation_success_rate = 0.95

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]

        if action_name == "detect_waste":
            return self._build_detect_distribution(state, action)

        if action_name == "pick":
            return self._build_default_distribution(state, action, self.pick_observation_success_rate)

        if action_name in self.PLACE_ACTIONS:
            return self._build_default_distribution(state, action, self.place_observation_success_rate)

        return self._build_default_distribution(state, action, 1.0 - self.noise)

    def get_observation_distribution_for_likelihood(self, state: State, action: Action) -> List[ObservationOutcome]:
        if action.name.replace(" ", "").split("(", 1)[0] == "detect_waste":
            return self._build_detect_distribution(state, action, use_true_state=False)
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

    def _observation_truth_state(self, state: State) -> State:
        if self.use_true_init_observation:
            return self.true_state
        return state

    @staticmethod
    def _category_label_for_state(state: State, waste: str) -> str | None:
        for pred in ObservationWastesorting.CATEGORY_PREDICATES:
            fact = f"{pred}({waste})"
            if state.has_fact(fact):
                return fact
        return None

    def _category_predicates_from_observation(self, action: Action) -> List[str]:
        predicates = []
        seen = set()
        for fact in action.observation:
            pred, args = _parse_fact(fact)
            if pred not in self.CATEGORY_PREDICATES or not args:
                continue
            if pred in seen:
                continue
            predicates.append(pred)
            seen.add(pred)
        return predicates or list(self.CATEGORY_PREDICATES)

    def _detectable_wastes_from_action(self, action: Action) -> List[str]:
        detected_facts = []
        for fact in action.observation:
            pred, _ = _parse_fact(fact)
            if pred == "detected":
                detected_facts.extend(self._expand_free_variables_in_fact(fact))

        wastes = []
        seen = set()
        for fact in _dedup_facts(detected_facts):
            _, args = _parse_fact(fact)
            if not args or args[0] in seen:
                continue
            wastes.append(args[0])
            seen.add(args[0])
        return wastes

    @staticmethod
    def _waste_is_unavailable(runtime_state: State, waste: str) -> bool:
        for fact in runtime_state.facts:
            pred, args = _parse_fact(fact)
            if pred == "in_bin" and args and args[0] == waste:
                return True
            if pred == "holding" and len(args) >= 2 and args[1] == waste:
                return True
        return False

    def _has_detectable_waste(self, runtime_state: State, gt_state: State, waste: str) -> bool:
        if self._waste_is_unavailable(runtime_state, waste):
            return False
        if self.use_true_init_observation:
            return self._category_label_for_state(gt_state, waste) is not None
        return runtime_state.has_fact(f"waste({waste})")

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

    @staticmethod
    def _detect_choice(facts: List[str], probability: float) -> Choice:
        return {
            "facts": list(dict.fromkeys(fact.replace(" ", "") for fact in facts)),
            "probability": probability,
        }

    def _build_detect_waste_label_choices(
        self,
        gt_state: State,
        waste: str,
        category_predicates: List[str],
    ) -> List[Choice]:
        p_correct_class = self.detect_classification_success_rate
        labels = [f"{pred}({waste})" for pred in category_predicates]

        true_label = self._category_label_for_state(gt_state, waste)
        if true_label not in labels:
            true_label = labels[0]

        wrong_labels = [label for label in labels if label != true_label]
        wrong_prob = (1.0 - p_correct_class) / len(wrong_labels)

        return (
            [self._detect_choice([f"detected({waste})", true_label], p_correct_class)]
            + [self._detect_choice([f"detected({waste})", label], wrong_prob) for label in wrong_labels]
        )

    @staticmethod
    def _merge_detect_choices(per_waste_choices: List[List[Choice]]) -> List[ObservationOutcome]:
        outcome_map = {}

        for combo in product(*per_waste_choices):
            facts = []
            probability = 1.0
            for choice in combo:
                facts.extend(choice["facts"])
                probability *= choice["probability"]

            facts = _dedup_facts(facts)
            key = tuple(sorted(facts))
            outcome_map[key] = outcome_map.get(key, 0.0) + probability

        total = sum(outcome_map.values())
        if total <= 0.0:
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(facts=list(facts_key), probability=probability / total)
            for facts_key, probability in outcome_map.items()
        ]

    def _build_detect_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        gt_state = self._observation_truth_state(state) if use_true_state else state
        candidate_wastes = self._detectable_wastes_from_action(action)
        detectable_wastes = [
            waste
            for waste in candidate_wastes
            if self._has_detectable_waste(state, gt_state, waste)
        ]
        if not detectable_wastes:
            return [ObservationOutcome(facts=[], probability=1.0)]

        category_predicates = self._category_predicates_from_observation(action)
        per_waste_choices = [
            self._build_detect_waste_label_choices(gt_state, waste, category_predicates)
            for waste in detectable_wastes
        ]
        detected_outcomes = self._merge_detect_choices(per_waste_choices)
        p_detect = self.detect_observed_success_rate

        return (
            [ObservationOutcome(facts=[], probability=1.0 - p_detect)]
            + [
                ObservationOutcome(
                    facts=outcome.facts,
                    probability=outcome.probability * p_detect,
                )
                for outcome in detected_outcomes
            ]
        )
