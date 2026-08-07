# models/tomato/obs.py

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
# (low, high, mode): triangular confidence for scan-style observations.
SCAN_CONFIDENCE_RANGE = (0.85, 0.99, 0.95)
# (low, high, mode): triangular confidence for pick/place gripper observations.
GRIPPER_CONFIDENCE_RANGE = (0.85, 0.99, 0.95)
# (low, high, mode): triangular confidence for navigation observations.
NAVIGATION_CONFIDENCE_RANGE = (0.85, 0.99, 0.95)
# (low, high): uniform confidence for false positives or wrong labels.
UNCERTAIN_CONFIDENCE_RANGE = (0.60, 0.70)
MIN_LIKELIHOOD = 1e-6
MAX_LIKELIHOOD = 0.999999


class ObservationTomato:
    """
    Tomato observation model with detector-style confidence for every action.

    The symbolic observation still contains facts only. The confidence of each
    observed fact is carried separately in Observation.fact_confidences.
    """

    def __init__(
        self,
        type_map: Dict[str, List[str]],
        noise: float = 0.1,
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

    def build_fluent_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation_fluents:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = self._action_name(action)

        if action_name == "detect":
            return self._build_detect_distribution(state, action)

        if action_name == "pick_n_scan":
            return self._build_pick_n_scan_distribution(state, action)

        if action_name == "scan":
            return self._build_scan_distribution(state, action)

        if action_name == "navigate":
            return self._build_navigate_distribution(state, action)

        return self._build_default_distribution(state, action)

    def get_observation_distribution_for_likelihood(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = self._action_name(action)

        if action_name == "detect":
            return self._build_detect_distribution(state, action, use_true_state=False)

        if action_name == "pick_n_scan":
            return self._build_pick_n_scan_distribution(state, action, use_true_state=False)

        if action_name == "scan":
            return self._build_scan_distribution(state, action, use_true_state=False)

        return self.get_observation_distribution(state, action)

    def likelihood(self, observation: Observation, state: State, action: Action) -> float | None:
        action_name = self._action_name(action)

        if action_name == "detect":
            return self._detect_likelihood(observation, state, action)

        if action_name in {"scan", "pick_n_scan"}:
            return self._scan_likelihood(observation, state, action)

        return self._confidence_likelihood(observation, state, action)

    # ------------------------------------------------------------------
    # Confidence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_name(action: Action) -> str:
        return action.name.split("(", 1)[0]

    def _detect_confidence(self) -> float:
        low, high, mode = DETECT_CONFIDENCE_RANGE
        return random.triangular(low, high, mode)

    def _scan_confidence(self) -> float:
        low, high, mode = SCAN_CONFIDENCE_RANGE
        return random.triangular(low, high, mode)

    def _gripper_confidence(self) -> float:
        low, high, mode = GRIPPER_CONFIDENCE_RANGE
        return random.triangular(low, high, mode)

    def _navigation_confidence(self) -> float:
        low, high, mode = NAVIGATION_CONFIDENCE_RANGE
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
        fluents: Dict[str, Dict[str, float]] | None = None,
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
            fluents=fluents or {},
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
    def _parse_fluent_candidate(expr: str):
        predicate, args = _parse_fact(expr.replace(" ", ""))
        if not args:
            return None, None
        return args[0], predicate

    @staticmethod
    def _get_action_args(action: Action) -> List[str]:
        _, args = _parse_fact(action.name.replace(" ", ""))
        return args

    @staticmethod
    def _scan_labels(tomato: str) -> List[str]:
        return [f"ripe({tomato})", f"rotten({tomato})"]

    @staticmethod
    def _detect_labels(tomato: str) -> List[str]:
        return [f"ripe({tomato})", f"unripe({tomato})"]

    # ------------------------------------------------------------------
    # State interpretation helpers
    # ------------------------------------------------------------------

    def _tomato_ground_truth_state(self, state: State) -> State:
        if self.use_true_init_observation:
            return self.true_state
        return state

    @staticmethod
    def _ripeness_label_for_state(state: State, tomato: str, *, detect_mode: bool = False) -> str | None:
        if state.has_fact(f"unripe({tomato})"):
            return f"unripe({tomato})"
        if state.has_fact(f"rotten({tomato})"):
            return f"ripe({tomato})" if detect_mode else f"rotten({tomato})"
        if state.has_fact(f"ripe({tomato})"):
            return f"ripe({tomato})"
        return None

    @staticmethod
    def _tomato_is_no_longer_at_stem(runtime_state: State, tomato: str) -> bool:
        for fact in runtime_state.facts:
            if fact == f"loaded({tomato})" or fact.startswith(f"loaded({tomato},"):
                return True
            if fact == f"discarded({tomato})" or fact.startswith(f"discarded({tomato},"):
                return True
            if fact.startswith("holding(") and fact.endswith(f",{tomato})"):
                return True
            if fact == f"holded({tomato})" or fact.startswith(f"holded({tomato},"):
                return True
        return False

    def _has_detectable_tomato_at(
        self,
        runtime_state: State,
        gt_state: State,
        tomato: str,
        at_fact: str,
        observed_fact: str,
    ) -> bool:
        if self.use_true_init_observation:
            if not gt_state.has_fact(at_fact):
                return False
            return not self._tomato_is_no_longer_at_stem(runtime_state, tomato)

        return runtime_state.has_fact(observed_fact) and runtime_state.has_fact(at_fact)

    # ------------------------------------------------------------------
    # Default / navigate / scan observation models
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

    def _build_navigate_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = [fact for fact in self.build_candidates(action) if fact.startswith("located(")]
        true_facts = [fact for fact in candidates if state.has_fact(fact)]

        if not true_facts:
            return [self._make_outcome([], 1.0)]

        wrong_facts = [fact for fact in candidates if fact not in true_facts]
        if not wrong_facts:
            confidence = self._navigation_confidence()
            return [self._make_outcome(true_facts, 1.0, confidence=confidence)]

        confidence = self._navigation_confidence()
        outcomes = [self._make_outcome(true_facts, confidence, confidence=confidence)]
        alternative_prob = (1.0 - confidence) / (len(wrong_facts) + 1)

        for fact in wrong_facts:
            wrong_confidence = self._uncertain_confidence()
            outcomes.append(self._make_outcome([fact], alternative_prob, confidence=wrong_confidence))
        outcomes.append(self._make_outcome([], alternative_prob))

        return outcomes

    def _build_scan_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        gt_state = self._tomato_ground_truth_state(state) if use_true_state else state
        args = self._get_action_args(action)
        if len(args) < 2:
            return [self._make_outcome([], 1.0)]

        tomato = args[1]
        all_labels = self._scan_labels(tomato)
        true_label = self._ripeness_label_for_state(gt_state, tomato, detect_mode=False)

        if true_label not in all_labels:
            true_label = f"ripe({tomato})"

        true_confidence = self._scan_confidence()
        outcomes = [self._make_outcome([true_label], true_confidence, confidence=true_confidence)]

        wrong_labels = [label for label in all_labels if label != true_label]
        if wrong_labels:
            wrong_prob = (1.0 - true_confidence) / len(wrong_labels)
            for label in wrong_labels:
                wrong_confidence = self._uncertain_confidence()
                outcomes.append(self._make_outcome([label], wrong_prob, confidence=wrong_confidence))

        return outcomes

    def _build_pick_n_scan_distribution(
        self,
        state: State,
        action: Action,
        use_true_state: bool = True,
    ) -> List[ObservationOutcome]:
        return self._build_scan_distribution(state, action, use_true_state=use_true_state)

    # ------------------------------------------------------------------
    # Detect observation model
    # ------------------------------------------------------------------

    def _build_detect_tomato_entries(self, action: Action) -> List[Dict[str, str]]:
        args = self._get_action_args(action)
        if len(args) < 2:
            return []

        target_location = args[1]
        entries = []
        for tomato in self.type_map.get("T", []):
            entries.append({
                "tomato": tomato,
                "location": target_location,
                "at_fact": f"at({tomato},{target_location})",
                "observed_fact": f"observed({tomato})",
            })
        return entries

    def _get_observed_fluents(self, tomato: str, action: Action) -> Dict[str, Dict[str, float]]:
        if self.true_state is None or tomato not in self.true_state.fluents:
            return {}

        fluent_values = {}
        for candidate in self.build_fluent_candidates(action):
            obj, key = self._parse_fluent_candidate(candidate)
            if obj == tomato and key in self.true_state.fluents[tomato]:
                fluent_values[key] = self.true_state.fluents[tomato][key]

        if not fluent_values:
            return {}
        return {tomato: fluent_values}

    def _detect_choice(
        self,
        facts: List[str],
        probability: float,
        fluents: Dict[str, Dict[str, float]] | None = None,
        fact_confidences: Dict[str, float] | None = None,
    ) -> Choice:
        return {
            "facts": list(dict.fromkeys(self._normalize_fact(fact) for fact in facts)),
            "fluents": fluents or {},
            "probability": float(probability),
            "fact_confidences": fact_confidences or {},
        }

    @staticmethod
    def _observed_label_facts(tomato: str) -> tuple[List[str], List[str]]:
        observed_fact = f"observed({tomato})"
        return (
            [observed_fact, f"ripe({tomato})"],
            [observed_fact, f"unripe({tomato})"],
        )

    def _label_confidences(
        self,
        observed_fact: str,
        label_fact: str,
        confidence: float,
    ) -> Dict[str, float]:
        confidence = round(float(confidence), 4)
        return {
            self._normalize_fact(observed_fact): confidence,
            self._normalize_fact(label_fact): confidence,
        }

    def _build_false_positive_detect_choices(
        self,
        observed_fact: str,
        observed_ripe: List[str],
        observed_unripe: List[str],
        fluents: Dict[str, Dict[str, float]],
        tomato: str,
    ) -> List[Choice]:
        false_positive_prob = self.detect_false_positive_rate / 2.0
        ripe_confidence = random.uniform(
            self.uncertain_confidence_low,
            self.uncertain_confidence_high,
        )
        unripe_confidence = random.uniform(
            self.uncertain_confidence_low,
            self.uncertain_confidence_high,
        )

        return [
            self._detect_choice([], 1.0 - self.detect_false_positive_rate),
            self._detect_choice(
                observed_ripe,
                false_positive_prob,
                fluents,
                self._label_confidences(observed_fact, f"ripe({tomato})", ripe_confidence),
            ),
            self._detect_choice(
                observed_unripe,
                false_positive_prob,
                fluents,
                self._label_confidences(observed_fact, f"unripe({tomato})", unripe_confidence),
            ),
        ]

    def _true_and_wrong_detect_facts(
        self,
        gt_state: State,
        tomato: str,
        observed_ripe: List[str],
        observed_unripe: List[str],
    ) -> tuple[str, List[str], str, List[str]]:
        true_label = self._ripeness_label_for_state(gt_state, tomato, detect_mode=True)
        if true_label == f"unripe({tomato})":
            return true_label, observed_unripe, f"ripe({tomato})", observed_ripe

        true_label = f"ripe({tomato})"
        return true_label, observed_ripe, f"unripe({tomato})", observed_unripe

    def _build_detect_tomato_choices(
        self,
        runtime_state: State,
        gt_state: State,
        action: Action,
        entry: Dict[str, str],
    ) -> List[Choice]:
        tomato = entry["tomato"]
        observed_fact = entry["observed_fact"]
        at_fact = entry["at_fact"]
        fluents = self._get_observed_fluents(tomato, action)
        observed_ripe, observed_unripe = self._observed_label_facts(tomato)

        exists_at_target = self._has_detectable_tomato_at(
            runtime_state,
            gt_state,
            tomato,
            at_fact,
            observed_fact,
        )

        if not exists_at_target:
            return self._build_false_positive_detect_choices(
                observed_fact,
                observed_ripe,
                observed_unripe,
                fluents,
                tomato,
            )

        true_label, true_facts, wrong_label, wrong_facts = self._true_and_wrong_detect_facts(
            gt_state,
            tomato,
            observed_ripe,
            observed_unripe,
        )

        true_confidence = self._detect_confidence()
        wrong_confidence = self._uncertain_confidence()

        return [
            self._detect_choice(
                true_facts,
                true_confidence,
                fluents,
                self._label_confidences(observed_fact, true_label, true_confidence),
            ),
            self._detect_choice(
                wrong_facts,
                1.0 - true_confidence,
                fluents,
                self._label_confidences(observed_fact, wrong_label, wrong_confidence),
            ),
        ]

    @staticmethod
    def _merge_detect_choices(per_tomato_choices: List[List[Choice]]) -> List[ObservationOutcome]:
        outcome_map = {}

        for combo in product(*per_tomato_choices):
            facts = []
            fluents = {}
            fact_confidences = {}
            probability = 1.0

            for choice in combo:
                facts.extend(choice["facts"])
                for obj, values in choice["fluents"].items():
                    fluents.setdefault(obj, {}).update(values)
                fact_confidences.update(choice.get("fact_confidences", {}))
                probability *= choice["probability"]

            facts = _dedup_facts(facts)
            fluent_key = tuple(
                sorted(
                    (obj, key, float(value))
                    for obj, values in fluents.items()
                    for key, value in values.items()
                )
            )
            confidence_key = tuple(
                sorted(
                    (fact, round(float(confidence), 4))
                    for fact, confidence in fact_confidences.items()
                )
            )
            map_key = (tuple(sorted(facts)), fluent_key, confidence_key)
            outcome_map[map_key] = outcome_map.get(map_key, 0.0) + probability

        total = sum(outcome_map.values())
        if total <= 0.0:
            return [ObservationOutcome(facts=[], probability=1.0)]

        outcomes = []
        for (facts_key, fluent_key, confidence_key), probability in outcome_map.items():
            fluents = {}
            for obj, key, value in fluent_key:
                fluents.setdefault(obj, {})[key] = value

            outcomes.append(
                ObservationOutcome(
                    facts=list(facts_key),
                    fluents=fluents,
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
        gt_state = self._tomato_ground_truth_state(state) if use_true_state else state
        tomato_entries = self._build_detect_tomato_entries(action)
        if not tomato_entries:
            return [ObservationOutcome(facts=[], probability=1.0)]

        per_tomato_choices = [
            self._build_detect_tomato_choices(state, gt_state, action, entry)
            for entry in tomato_entries
        ]
        return self._merge_detect_choices(per_tomato_choices)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def _scan_likelihood(self, observation: Observation, state: State, action: Action) -> float:
        args = self._get_action_args(action)
        if len(args) < 2:
            return 1.0 if not observation.state.facts else self.noise

        tomato = args[1]
        labels = set(self._scan_labels(tomato))
        observed_labels = labels & set(observation.state.facts)
        if not observed_labels:
            return self.noise

        observed_label = next(iter(observed_labels))
        confidence = self._valid_likelihood(
            observation.confidence(observed_label, default=self.correct_confidence_mode)
        )

        true_label = self._ripeness_label_for_state(state, tomato, detect_mode=False)
        if true_label not in labels:
            true_label = f"ripe({tomato})"

        return confidence if observed_label == true_label else 1.0 - confidence

    @staticmethod
    def _observed_detect_label(obs_set: set[str], tomato: str) -> str | None:
        for label in (f"ripe({tomato})", f"unripe({tomato})"):
            if label in obs_set:
                return label
        return None

    def _candidate_detect_label(self, state: State, entry: Dict[str, str]) -> str | None:
        tomato = entry["tomato"]
        ripe_fact, unripe_fact = self._detect_labels(tomato)
        if not self._has_detectable_tomato_at(
            state,
            state,
            tomato,
            entry["at_fact"],
            entry["observed_fact"],
        ):
            return None

        candidate_label = self._ripeness_label_for_state(state, tomato, detect_mode=True)
        if candidate_label not in {ripe_fact, unripe_fact}:
            return ripe_fact
        return candidate_label

    def _detect_likelihood(self, observation: Observation, state: State, action: Action) -> float:
        tomato_entries = self._build_detect_tomato_entries(action)
        if not tomato_entries:
            return 1.0 if not observation.state.facts and not observation.state.fluents else self.noise

        obs_set = set(observation.state.facts)
        probability = 1.0

        for entry in tomato_entries:
            tomato = entry["tomato"]
            local_label = self._observed_detect_label(obs_set, tomato)
            candidate_label = self._candidate_detect_label(state, entry)

            if local_label is None:
                if candidate_label is not None:
                    probability *= max(1e-6, 1.0 - self.correct_confidence_mode)
                else:
                    probability *= max(1e-6, 1.0 - self.detect_false_positive_rate)
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
            return max(1e-6, 1.0 - self.correct_confidence_mode)

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
