# models/tomato/obs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import Observation, ObservationOutcome


Choice = Dict


class ObservationTomato:
    """
    Observation model for the tomato domain.

    Two execution modes are supported:
    - real_robot: observations are generated from the hypothesized runtime state.
    - true_init: detect/scan observations use initial_state.yaml:true_init as GT.

    The detect model is factorized by tomato. For each tomato, the local
    observation space has three states:
    - not observed
    - observed and classified as ripe
    - observed and classified as unripe

    The full detect distribution is the Cartesian product of these local
    distributions, followed by normalization.
    """

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

        self.detect_observed_success_rate = 0.99
        self.detect_classification_success_rate = 0.90
        self.scan_success_rate = 0.70
        self.navigate_success_rate = 0.95

        if self.use_true_init_observation and self.true_state is None:
            raise ValueError("observation_source=true_init requires initial_state.yaml true_init")

    # ------------------------------------------------------------------
    # Public entry points used by ObservationModel
    # ------------------------------------------------------------------

    def build_candidates(self, action: Action) -> List[str]:
        """Expand all symbolic observation facts in an action."""
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def build_fluent_candidates(self, action: Action) -> List[str]:
        """Expand all symbolic observation fluent expressions in an action."""
        expanded = []
        for obs in action.observation_fluents:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return _dedup_facts(expanded)

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        """Dispatch to an action-specific observation model."""
        action_name = action.name.split("(")[0]

        if action_name == "detect":
            return self._build_detect_distribution(state, action)
        
        if action_name == "pick_n_scan":
            return self._build_pick_n_scan_distribution(state, action)
        
        if action_name == "scan":
            return self._build_scan_distribution(state, action)
        
        if action_name == "navigate":
            return self._build_navigate_distribution(state, action)
        
        else:
            return self._build_default_distribution(state, action)

    # ------------------------------------------------------------------
    # Generic fact expansion helpers
    # ------------------------------------------------------------------
    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        """
        Expand facts with typed variables using the domain type map.

        Example:
            observed(T) with T=[tomato1,tomato2]
            -> observed(tomato1), observed(tomato2)
        """
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

    # ------------------------------------------------------------------
    # State interpretation helpers
    # ------------------------------------------------------------------

    def _tomato_ground_truth_state(self, state: State) -> State:
        """
        Return the state that should define tomato GT for observations.

        In true_init simulation mode, tomato position/ripeness comes from
        initial_state.yaml:true_init. In real_robot mode, the runtime state is
        treated as the observation reference.
        """
        if self.use_true_init_observation:
            return self.true_state
        return state

    @staticmethod
    def _ripeness_label_for_state(state: State, tomato: str, *, detect_mode: bool = False) -> str | None:
        """
        Return the tomato ripeness label in a state.

        Detection uses a camera-style coarse label: rotten tomatoes are seen as
        visually ripe. Scan keeps rotten as a separate label.
        """
        if state.has_fact(f"unripe({tomato})"):
            return f"unripe({tomato})"
        if state.has_fact(f"rotten({tomato})"):
            return f"ripe({tomato})" if detect_mode else f"rotten({tomato})"
        if state.has_fact(f"ripe({tomato})"):
            return f"ripe({tomato})"
        return None

    @staticmethod
    def _tomato_is_no_longer_at_stem(runtime_state: State, tomato: str) -> bool:
        """
        Detect should not rediscover tomatoes that have already moved out of a stem.
        """
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
        """
        Decide whether a tomato exists at the detect target.

        In true_init mode, existence comes from GT at(T,S), but runtime facts can
        remove the tomato from the stem after pick/place/discard. In real_robot
        mode, an already hypothesized observed+at tomato is treated as the
        reference.
        """
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
            return [ObservationOutcome(facts=[], probability=1.0)]

        return [
            ObservationOutcome(facts=true_facts, probability=1.0 - self.noise),
            ObservationOutcome(facts=[], probability=self.noise),
        ]

    def _build_navigate_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = [fact for fact in self.build_candidates(action) if fact.startswith("located(")]
        true_facts = [fact for fact in candidates if state.has_fact(fact)]

        if not true_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        wrong_facts = [fact for fact in candidates if fact not in true_facts]
        if not wrong_facts:
            return [ObservationOutcome(facts=true_facts, probability=1.0)]

        outcomes = [ObservationOutcome(facts=true_facts, probability=self.navigate_success_rate)]
        alternative_prob = (1.0 - self.navigate_success_rate) / (len(wrong_facts) + 1)

        for fact in wrong_facts:
            outcomes.append(ObservationOutcome(facts=[fact], probability=alternative_prob))
        outcomes.append(ObservationOutcome(facts=[], probability=alternative_prob))

        return outcomes

    def _build_scan_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        """
        Scan observes the true ripe/unripe/rotten label for the held tomato.

        The tomato is taken from the grounded action name instead of
        action.observation. This lets pick_n_scan reuse the same scan model even
        when robot_skill.yaml leaves its observation field empty.
        """
        gt_state = self._tomato_ground_truth_state(state)
        args = self._get_action_args(action)
        if len(args) < 2:
            return [ObservationOutcome(facts=[], probability=1.0)]

        tomato = args[1]
        all_labels = [f"ripe({tomato})", f"unripe({tomato})", f"rotten({tomato})"]
        true_label = self._ripeness_label_for_state(gt_state, tomato, detect_mode=False)

        if true_label not in all_labels:
            return [ObservationOutcome(facts=[], probability=1.0)]

        wrong_labels = [label for label in all_labels if label != true_label]

        correct_prob = self.scan_success_rate
        miss_or_wrong_prob = (1.0 - correct_prob) / (len(wrong_labels) + 1)

        outcomes = [ObservationOutcome(facts=[true_label], probability=correct_prob)]
        for label in wrong_labels:
            outcomes.append(ObservationOutcome(facts=[label], probability=miss_or_wrong_prob))
        outcomes.append(ObservationOutcome(facts=[], probability=miss_or_wrong_prob))

        return outcomes

    def _build_pick_n_scan_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        """
        pick_n_scan observes exactly the scan part of the combined skill.

        Pick is assumed deterministic in the transition model, so the only
        uncertain observation is the tomato ripeness/classification result.
        """
        return self._build_scan_distribution(state, action)

    # ------------------------------------------------------------------
    # Detect observation model
    # ------------------------------------------------------------------

    def _build_detect_tomato_entries(self, action: Action) -> List[Dict[str, str]]:
        """Create one detect entry per tomato for the action target stem."""
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
        """
        Attach true fluents to a detected tomato when observation_fluents exists.
        """
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

    @staticmethod
    def _detect_choice(
        facts: List[str],
        probability: float,
        fluents: Dict[str, Dict[str, float]] | None = None,
    ) -> Choice:
        """Small constructor for one local detect choice."""
        return {
            "facts": list(dict.fromkeys(fact.replace(" ", "") for fact in facts)),
            "fluents": fluents or {},
            "probability": probability,
        }

    def _build_detect_tomato_choices(
        self,
        runtime_state: State,
        gt_state: State,
        action: Action,
        entry: Dict[str, str],
    ) -> List[Choice]:
        """
        Build the local 3-state observation distribution for one tomato.

        Local states:
        - not observed
        - observed & ripe
        - observed & unripe

        The probabilities are unnormalized local likelihoods. They are multiplied
        across tomatoes and normalized after Cartesian composition.
        """
        p_detect = self.detect_observed_success_rate
        p_miss = 1.0 - p_detect
        p_correct_class = self.detect_classification_success_rate
        p_wrong_class = 1.0 - p_correct_class

        tomato = entry["tomato"]
        observed_fact = entry["observed_fact"]
        at_fact = entry["at_fact"]
        fluents = self._get_observed_fluents(tomato, action)

        observed_ripe = [observed_fact, f"ripe({tomato})"]
        observed_unripe = [observed_fact, f"unripe({tomato})"]

        exists_at_target = self._has_detectable_tomato_at(
            runtime_state,
            gt_state,
            tomato,
            at_fact,
            observed_fact,
        )

        if not exists_at_target:
            # GT says no tomato is available at this stem. The most likely local
            # observation is no detection; false positives are given miss mass.
            return [
                self._detect_choice([], p_detect),
                self._detect_choice(observed_ripe, p_miss, fluents),
                self._detect_choice(observed_unripe, p_miss, fluents),
            ]

        true_detect_label = self._ripeness_label_for_state(gt_state, tomato, detect_mode=True)
        if true_detect_label == f"unripe({tomato})":
            ripe_prob = p_detect * p_wrong_class
            unripe_prob = p_detect * p_correct_class
        else:
            # Camera detect cannot distinguish rotten from visually ripe.
            # Unknown labels are conservatively treated as ripe-like.
            ripe_prob = p_detect * p_correct_class
            unripe_prob = p_detect * p_wrong_class

        return [
            self._detect_choice([], p_miss),
            self._detect_choice(observed_ripe, ripe_prob, fluents),
            self._detect_choice(observed_unripe, unripe_prob, fluents),
        ]

    @staticmethod
    def _merge_detect_choices(per_tomato_choices: List[List[Choice]]) -> List[ObservationOutcome]:
        """
        Cartesian-compose local tomato choices into global outcomes.

        Equivalent outcomes are merged, then probabilities are normalized so the
        returned distribution sums to 1.
        """
        outcome_map = {}

        for combo in product(*per_tomato_choices):
            facts = []
            fluents = {}
            probability = 1.0

            for choice in combo:
                facts.extend(choice["facts"])
                for obj, values in choice["fluents"].items():
                    fluents.setdefault(obj, {}).update(values)
                probability *= choice["probability"]

            facts = _dedup_facts(facts)
            fluent_key = tuple(
                sorted(
                    (obj, key, float(value))
                    for obj, values in fluents.items()
                    for key, value in values.items()
                )
            )
            map_key = (tuple(sorted(facts)), fluent_key)
            outcome_map[map_key] = outcome_map.get(map_key, 0.0) + probability

        total = sum(outcome_map.values())
        if total <= 0.0:
            return [ObservationOutcome(facts=[], probability=1.0)]

        outcomes = []
        for (facts_key, fluent_key), probability in outcome_map.items():
            fluents = {}
            for obj, key, value in fluent_key:
                fluents.setdefault(obj, {})[key] = value

            outcomes.append(
                ObservationOutcome(
                    facts=list(facts_key),
                    fluents=fluents,
                    probability=probability / total,
                )
            )

        return outcomes

    def _build_detect_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        """
        Build P(o | state, detect) as a product of independent tomato factors.
        """
        gt_state = self._tomato_ground_truth_state(state)
        tomato_entries = self._build_detect_tomato_entries(action)
        if not tomato_entries:
            return [ObservationOutcome(facts=[], probability=1.0)]

        per_tomato_choices = [
            self._build_detect_tomato_choices(state, gt_state, action, entry)
            for entry in tomato_entries
        ]

        return self._merge_detect_choices(per_tomato_choices)
