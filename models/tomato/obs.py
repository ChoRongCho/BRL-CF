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


class ObservationTomato:
    def __init__(self, type_map: Dict[str, List[str]], noise: float = 0.05, true_state: State | None = None):
        self.type_map = type_map
        self.noise = noise
        self.true_state = true_state

        self.detect_observed_success_rate = 1.00
        self.detect_classification_success_rate = 0.90
        self.scan_success_rate = 0.97
        self.navigate_success_rate = 0.90


    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        pred, args = _parse_fact(fact)

        variable_positions = []
        variable_domains = []

        for i, arg in enumerate(args):
            if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", arg):
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

    # observation.py <- obs.py
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

    def _build_detect_tomato_entries(self, action: Action) -> List[Dict[str, str]]:
        """Build detect candidates for all tomatoes at the action target stem."""
        args = self._get_action_args(action)
        if len(args) < 2:
            return []

        target_location = args[1]
        tomatoes = self.type_map.get("T", [])
        entries = []
        for tomato in tomatoes:
            entries.append({
                "tomato": tomato,
                "location": target_location,
                "at_fact": f"at({tomato},{target_location})",
                "observed_fact": f"observed({tomato})",
            })
        return entries

    def _get_observed_fluents(self, tomato: str, action: Action) -> Dict[str, Dict[str, float]]:
        if self.true_state is None:
            return {}
        if tomato not in self.true_state.fluents:
            return {}

        fluent_values = {}
        for candidate in self.build_fluent_candidates(action):
            obj, key = self._parse_fluent_candidate(candidate)
            if obj != tomato:
                continue
            if key in self.true_state.fluents[tomato]:
                fluent_values[key] = self.true_state.fluents[tomato][key]

        if not fluent_values:
            return {}

        return {tomato: fluent_values}

    # observation.py <- obs.py
    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        action_name = action.name.split("(")[0]

        if action_name == "scan":
            return self._build_scan_distribution(state, action)
        elif action_name == "detect":
            return self._build_detect_distribution(state, action)
        elif action_name == "navigate":
            return self._build_navigate_distribution(state, action)
        else:
            return self._build_default_distribution(state, action)

    def _build_default_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        true_facts = [f for f in candidates if state.has_fact(f)]
        
        if true_facts:
            return [
                ObservationOutcome(facts=true_facts, probability=1.0 - self.noise),
                ObservationOutcome(facts=[], probability=self.noise)
            ]
        return [ObservationOutcome(facts=[], probability=1.0 - self.noise)]

    def _build_navigate_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = [f for f in self.build_candidates(action) if f.startswith("located(")]
        true_facts = [f for f in candidates if state.has_fact(f)]

        if not true_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        correct = true_facts
        wrongs = [f for f in candidates if f not in correct]

        outcomes = [ObservationOutcome(facts=correct, probability=self.navigate_success_rate)]

        remain = 1.0 - self.navigate_success_rate
        if wrongs:
            alt_prob = remain / (len(wrongs) + 1)
            for w in wrongs:
                outcomes.append(ObservationOutcome(facts=[w], probability=alt_prob))
            outcomes.append(ObservationOutcome(facts=[], probability=alt_prob))
        else:
            outcomes[0].probability = 1.0

        return outcomes

    def _build_scan_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        ripeness = [f for f in candidates if f.startswith(("ripe(", "unripe(", "rotten("))]
        true_ripeness = [f for f in ripeness if state.has_fact(f)]

        if not true_ripeness:
            return [ObservationOutcome(facts=[], probability=1.0)]

        true_label = true_ripeness[0]
        _, args = _parse_fact(true_label)
        tomato = args[0]

        all_labels = [f"ripe({tomato})", f"unripe({tomato})", f"rotten({tomato})"]
        wrong_labels = [x for x in all_labels if x != true_label]

        correct_prob = self.scan_success_rate if true_label.startswith(("ripe(", "rotten(")) else 0.95
        outcomes = [ObservationOutcome(facts=[true_label], probability=correct_prob)]

        remain = 1.0 - correct_prob
        alt_prob = remain / (len(wrong_labels) + 1)

        for wrong in wrong_labels:
            outcomes.append(ObservationOutcome(facts=[wrong], probability=alt_prob))

        outcomes.append(ObservationOutcome(facts=[], probability=alt_prob))
        return outcomes

    def _build_detect_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        tomato_entries = self._build_detect_tomato_entries(action)
        if not tomato_entries:
            return [ObservationOutcome(facts=[], probability=1.0)]

        tomato_groups = []
        for entry in tomato_entries:
            tomato = entry["tomato"]
            observed_fluents = self._get_observed_fluents(tomato, action)
            observed_fact = entry["observed_fact"]
            at_fact = entry["at_fact"]

            has_observed = state.has_fact(observed_fact) and state.has_fact(at_fact)
            true_label = None
            if state.has_fact(f"unripe({tomato})"):
                true_label = f"unripe({tomato})"
            elif state.has_fact(f"ripe({tomato})") or state.has_fact(f"rotten({tomato})"):
                # detect cannot distinguish ripe from rotten
                true_label = f"ripe({tomato})"

            if not has_observed:
                tomato_groups.append([{"facts": [], "fluents": {}, "probability": 1.0}])
                continue

            p_detect = self.detect_observed_success_rate
            p_miss = 1.0 - p_detect

            local = [
                {"facts": [], "fluents": {}, "probability": p_miss},
            ]

            if true_label is None:
                local.append({
                    "facts": [at_fact, observed_fact],
                    "fluents": observed_fluents,
                    "probability": p_detect,
                })
            else:
                wrong_label = f"ripe({tomato})" if true_label == f"unripe({tomato})" else f"unripe({tomato})"
                p_label = self.detect_classification_success_rate
                local.append({
                    "facts": [at_fact, observed_fact, true_label],
                    "fluents": observed_fluents,
                    "probability": p_detect * p_label,
                })
                local.append({
                    "facts": [at_fact, observed_fact, wrong_label],
                    "fluents": observed_fluents,
                    "probability": p_detect * (1.0 - p_label),
                })

            tomato_groups.append(local)

        outcome_map = {}
        for combo in product(*tomato_groups):
            flat_facts = []
            flat_fluents = {}
            prob = 1.0
            for part in combo:
                flat_facts.extend(part["facts"])
                flat_fluents.update(part["fluents"])
                prob *= part["probability"]

            flat_facts = _dedup_facts(flat_facts)
            fluent_key = tuple(
                sorted(
                    (obj, key, float(value))
                    for obj, values in flat_fluents.items()
                    for key, value in values.items()
                )
            )
            key = (tuple(sorted(flat_facts)), fluent_key)
            outcome_map[key] = outcome_map.get(key, 0.0) + prob

        outcomes = []
        for (facts_key, fluent_key), prob in outcome_map.items():
            fluents = {}
            for obj, key, value in fluent_key:
                fluents.setdefault(obj, {})[key] = value
            outcomes.append(ObservationOutcome(facts=list(facts_key), fluents=fluents, probability=prob))

        return outcomes
