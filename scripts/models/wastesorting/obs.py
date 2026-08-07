# models/wastesorting/obs.py

from __future__ import annotations

from itertools import product
from typing import Dict, List
import random
import re

from models.action import Action
from models.observation import Observation, ObservationOutcome
from models.state import State
from utils.utils import _dedup_facts, _format_fact, _parse_fact


Choice = Dict


DETECT_FALSE_POSITIVE_RATE = 0.01
# (low, high, mode): triangular confidence for correct detector observations.
DETECT_CONFIDENCE_RANGE = (0.70, 0.99, 0.90)
# (low, high, mode): triangular confidence for pick/place gripper observations.
GRIPPER_CONFIDENCE_RANGE = (0.85, 0.99, 0.95)
# (low, high): uniform confidence for false positives or wrong labels.
UNCERTAIN_CONFIDENCE_RANGE = (0.60, 0.70)
MIN_LIKELIHOOD = 1e-6
MAX_LIKELIHOOD = 0.999999


class ObservationWastesorting:
    """
    Waste-sorting observation model with detector-style confidence.

    The symbolic observation still contains facts only. The confidence of each
    observed fact is carried separately in Observation.fact_confidences.
    """

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
        world=None,
        observation_source: str = "true_init",
    ):
        self.type_map = type_map
        self.noise = noise
        self.world = world
        self.true_state = world.true_state if world is not None else true_state
        self.observation_source = observation_source
        self.use_true_init_observation = observation_source == "true_init"

        self.detect_false_positive_rate = DETECT_FALSE_POSITIVE_RATE
        self.correct_confidence_low, self.correct_confidence_high, self.correct_confidence_mode = (
            DETECT_CONFIDENCE_RANGE
        )
        self.uncertain_confidence_low, self.uncertain_confidence_high = UNCERTAIN_CONFIDENCE_RANGE

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    # ------------------------------------------------------------------
    # Public entry points used by ObservationModel
    # ------------------------------------------------------------------

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = self._action_name(action)

        if action_name == "detect_waste":
            return self._build_detect_distribution(state, action)

        return self._build_default_distribution(state, action)

    def get_observation_distribution_for_likelihood(self, state: State, action: Action) -> List[ObservationOutcome]:
        if self._action_name(action) == "detect_waste":
            return self._build_detect_distribution(state, action, use_true_state=False)
        return self.get_observation_distribution(state, action)

    def likelihood(self, observation: Observation, state: State, action: Action) -> float | None:
        if self._action_name(action) == "detect_waste":
            return self._detect_likelihood(observation, state, action)
        return self._confidence_likelihood(observation, state, action)

    # ------------------------------------------------------------------
    # Confidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_name(action: Action) -> str:
        return action.name.replace(" ", "").split("(", 1)[0]

    def _detect_confidence(self) -> float:
        low, high, mode = DETECT_CONFIDENCE_RANGE
        return random.triangular(low, high, mode)

    def _gripper_confidence(self) -> float:
        low, high, mode = GRIPPER_CONFIDENCE_RANGE
        return random.triangular(low, high, mode)

    def _uncertain_confidence(self) -> float:
        low, high = UNCERTAIN_CONFIDENCE_RANGE
        return random.uniform(low, high)

    @staticmethod
    def _valid_likelihood(value: float) -> float:
        return min(MAX_LIKELIHOOD, max(MIN_LIKELIHOOD, float(value)))

    @staticmethod
    def _normalize_fact(fact: str) -> str:
        return str(fact).replace(" ", "")

    def _confidence_map(self, facts: List[str], confidence: float) -> Dict[str, float]:
        return {
            self._normalize_fact(fact): round(float(confidence), 4)
            for fact in facts
        }

    def _make_outcome(
        self,
        facts: List[str],
        probability: float,
        confidence: float | None = None,
        fact_confidences: Dict[str, float] | None = None,
    ) -> ObservationOutcome:
        normalized_facts = list(dict.fromkeys(self._normalize_fact(fact) for fact in facts))
        confidences = {}
        if confidence is not None:
            confidences.update(self._confidence_map(normalized_facts, confidence))
        if fact_confidences:
            confidences.update({
                self._normalize_fact(fact): round(float(value), 4)
                for fact, value in fact_confidences.items()
            })

        return ObservationOutcome(
            facts=normalized_facts,
            probability=float(probability),
            fact_confidences=confidences,
        )

    # ------------------------------------------------------------------
    # Generic fact expansion helpers
    # ------------------------------------------------------------------

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
            return [self._normalize_fact(fact)]

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

    # ------------------------------------------------------------------
    # State interpretation helpers
    # ------------------------------------------------------------------

    def _observation_truth_state(self, state: State) -> State:
        if self.use_true_init_observation:
            return self.true_state
        return state

    @staticmethod
    def _category_label_for_state(state: State, waste: str) -> str | None:
        for predicate in ObservationWastesorting.CATEGORY_PREDICATES:
            fact = f"{predicate}({waste})"
            if state.has_fact(fact):
                return fact
        return None

    def _category_predicates_from_observation(self, action: Action) -> List[str]:
        predicates = []
        seen = set()
        for fact in action.observation:
            predicate, args = _parse_fact(fact)
            if predicate not in self.CATEGORY_PREDICATES or not args:
                continue
            if predicate in seen:
                continue
            predicates.append(predicate)
            seen.add(predicate)
        return predicates or list(self.CATEGORY_PREDICATES)

    def _detectable_wastes_from_action(self, action: Action) -> List[str]:
        detected_facts = []
        for fact in action.observation:
            predicate, _ = _parse_fact(fact)
            if predicate == "detected":
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
            predicate, args = _parse_fact(fact)
            if predicate == "in_bin" and args and args[0] == waste:
                return True
            if predicate == "holding" and len(args) >= 2 and args[1] == waste:
                return True
        return False

    @staticmethod
    def _occluding_pairs(gt_state: State) -> List[tuple[str, str]]:
        pairs = []
        for fact in gt_state.facts:
            predicate, args = _parse_fact(fact)
            if predicate == "on" and len(args) >= 2:
                pairs.append((args[0], args[1]))
        return pairs

    def _waste_is_occluded(self, runtime_state: State, gt_state: State, waste: str) -> bool:
        for top_waste, bottom_waste in self._occluding_pairs(gt_state):
            if bottom_waste != waste:
                continue

            top_cleared = (
                self._waste_is_unavailable(runtime_state, top_waste)
                or self._waste_is_unavailable(gt_state, top_waste)
            )
            if not top_cleared:
                return True

        return False

    def _has_detectable_waste(self, runtime_state: State, gt_state: State, waste: str) -> bool:
        if self._waste_is_unavailable(runtime_state, waste):
            return False
        if self._waste_is_occluded(runtime_state, gt_state, waste):
            return False
        if self.use_true_init_observation:
            return self._category_label_for_state(gt_state, waste) is not None
        return runtime_state.has_fact(f"waste({waste})")

    # ------------------------------------------------------------------
    # Default observation model
    # ------------------------------------------------------------------

    def _build_default_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        true_facts = [fact for fact in candidates if state.has_fact(fact)]

        if not true_facts:
            return [self._make_outcome([], 1.0)]

        confidence = self._gripper_confidence()
        return [
            self._make_outcome(true_facts, confidence, confidence=confidence),
            self._make_outcome([], 1.0 - confidence),
        ]

    # ------------------------------------------------------------------
    # Detect observation model
    # ------------------------------------------------------------------

    def _detect_choice(
        self,
        facts: List[str],
        probability: float,
        fact_confidences: Dict[str, float] | None = None,
    ) -> Choice:
        return {
            "facts": list(dict.fromkeys(self._normalize_fact(fact) for fact in facts)),
            "probability": float(probability),
            "fact_confidences": fact_confidences or {},
        }

    def _label_confidences(
        self,
        detected_fact: str,
        label_fact: str,
        confidence: float,
    ) -> Dict[str, float]:
        confidence = round(float(confidence), 4)
        return {
            self._normalize_fact(detected_fact): confidence,
            self._normalize_fact(label_fact): confidence,
        }

    def _build_false_positive_detect_choices(
        self,
        waste: str,
        category_predicates: List[str],
    ) -> List[Choice]:
        detected_fact = f"detected({waste})"
        false_positive_prob = self.detect_false_positive_rate / len(category_predicates)
        choices = [self._detect_choice([], 1.0 - self.detect_false_positive_rate)]

        for predicate in category_predicates:
            label_fact = f"{predicate}({waste})"
            confidence = self._uncertain_confidence()
            choices.append(
                self._detect_choice(
                    [detected_fact, label_fact],
                    false_positive_prob,
                    self._label_confidences(detected_fact, label_fact, confidence),
                )
            )

        return choices

    def _build_detect_waste_label_choices(
        self,
        gt_state: State,
        waste: str,
        category_predicates: List[str],
    ) -> List[Choice]:
        labels = [f"{predicate}({waste})" for predicate in category_predicates]
        detected_fact = f"detected({waste})"
        true_label = self._category_label_for_state(gt_state, waste)
        if true_label not in labels:
            true_label = labels[0]

        true_confidence = self._detect_confidence()
        choices = [
            self._detect_choice(
                [detected_fact, true_label],
                true_confidence,
                self._label_confidences(detected_fact, true_label, true_confidence),
            )
        ]

        wrong_labels = [label for label in labels if label != true_label]
        if wrong_labels:
            wrong_prob = (1.0 - true_confidence) / len(wrong_labels)
            for label in wrong_labels:
                wrong_confidence = self._uncertain_confidence()
                choices.append(
                    self._detect_choice(
                        [detected_fact, label],
                        wrong_prob,
                        self._label_confidences(detected_fact, label, wrong_confidence),
                    )
                )

        return choices

    @staticmethod
    def _merge_detect_choices(per_waste_choices: List[List[Choice]]) -> List[ObservationOutcome]:
        outcome_map = {}

        for combo in product(*per_waste_choices):
            facts = []
            fact_confidences = {}
            probability = 1.0

            for choice in combo:
                facts.extend(choice["facts"])
                fact_confidences.update(choice.get("fact_confidences", {}))
                probability *= choice["probability"]

            facts = _dedup_facts(facts)
            confidence_key = tuple(
                sorted(
                    (fact, round(float(confidence), 4))
                    for fact, confidence in fact_confidences.items()
                )
            )
            map_key = (tuple(sorted(facts)), confidence_key)
            outcome_map[map_key] = outcome_map.get(map_key, 0.0) + probability

        total = sum(outcome_map.values())
        if total <= 0.0:
            return [ObservationOutcome(facts=[], probability=1.0)]

        outcomes = []
        for (facts_key, confidence_key), probability in outcome_map.items():
            outcomes.append(
                ObservationOutcome(
                    facts=list(facts_key),
                    probability=probability / total,
                    fact_confidences={
                        fact: float(confidence)
                        for fact, confidence in confidence_key
                    },
                )
            )

        return outcomes

    def _build_detect_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        gt_state = self._observation_truth_state(state) if use_true_state else state
        candidate_wastes = self._detectable_wastes_from_action(action)
        category_predicates = self._category_predicates_from_observation(action)

        if not candidate_wastes:
            return [self._make_outcome([], 1.0)]

        per_waste_choices = []
        for waste in candidate_wastes:
            if self._has_detectable_waste(state, gt_state, waste):
                per_waste_choices.append(
                    self._build_detect_waste_label_choices(gt_state, waste, category_predicates)
                )
            else:
                per_waste_choices.append(
                    self._build_false_positive_detect_choices(waste, category_predicates)
                )

        return self._merge_detect_choices(per_waste_choices)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    @staticmethod
    def _observed_category_label(obs_set: set[str], waste: str) -> str | None:
        for predicate in ObservationWastesorting.CATEGORY_PREDICATES:
            label = f"{predicate}({waste})"
            if label in obs_set:
                return label
        return None

    def _candidate_detect_label(self, state: State, waste: str) -> str | None:
        if not self._has_detectable_waste(state, state, waste):
            return None
        return self._category_label_for_state(state, waste)

    def _detect_likelihood(self, observation: Observation, state: State, action: Action) -> float:
        candidate_wastes = self._detectable_wastes_from_action(action)
        if not candidate_wastes:
            return 1.0 if not observation.state.facts else self.noise

        obs_set = set(observation.state.facts)
        probability = 1.0

        for waste in candidate_wastes:
            local_label = self._observed_category_label(obs_set, waste)
            candidate_label = self._candidate_detect_label(state, waste)

            if local_label is None:
                if candidate_label is not None:
                    probability *= max(MIN_LIKELIHOOD, 1.0 - self.correct_confidence_mode)
                else:
                    probability *= max(MIN_LIKELIHOOD, 1.0 - self.detect_false_positive_rate)
                continue

            confidence = self._valid_likelihood(
                observation.confidence(local_label, default=self.correct_confidence_mode)
            )
            if candidate_label == local_label:
                probability *= confidence
            else:
                probability *= 1.0 - confidence

        return max(0.0, float(probability))

    def _confidence_likelihood(self, observation: Observation, state: State, action: Action) -> float:
        candidates = set(self.build_candidates(action))
        observed_facts = set(observation.state.facts)
        if not observed_facts:
            true_candidates = [fact for fact in candidates if state.has_fact(fact)]
            if not true_candidates:
                return 1.0
            return max(MIN_LIKELIHOOD, 1.0 - self.correct_confidence_mode)

        probability = 1.0
        for fact in observed_facts:
            confidence = self._valid_likelihood(
                observation.confidence(fact, default=self.correct_confidence_mode)
            )
            if state.has_fact(fact):
                probability *= confidence
            else:
                probability *= 1.0 - confidence

        return max(0.0, float(probability))
