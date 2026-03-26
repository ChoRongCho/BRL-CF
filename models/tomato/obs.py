# models/tomato/obs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import re

from models.state import State
from models.action import Action
from models.observation import Observation, ObservationOutcome


class ObservationTomato:
    def __init__(self, type_map: Dict[str, List[str]], noise: float = 0.05):
        self.type_map = type_map
        self.noise = noise

        self.detect_success_rate = 0.85
        self.scan_success_rate = 0.95
        self.navigate_success_rate = 0.90

    @staticmethod
    def _format_fact(pred: str, args: List[str]) -> str:
        return f"{pred}({','.join(args)})"

    @staticmethod
    def _parse_fact(fact: str) -> Tuple[str, List[str]]:
        fact = fact.replace(" ", "")
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", fact)
        if not m:
            raise ValueError(f"Invalid fact format: {fact}")
        pred = m.group(1)
        args = [x.strip() for x in m.group(2).split(",")]
        return pred, args

    @staticmethod
    def _dedup_facts(facts: List[str]) -> List[str]:
        return list(dict.fromkeys(f.replace(" ", "") for f in facts))

    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        pred, args = self._parse_fact(fact)

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
                expanded.append(self._format_fact(pred, current_args))
                return
            pos = variable_positions[depth]
            for obj in variable_domains[depth]:
                next_args = current_args[:]
                next_args[pos] = obj
                backtrack(depth + 1, next_args)

        backtrack(0, args[:])
        return expanded

    def build_candidates(self, action: Action) -> List[str]:
        expanded = []
        for obs in action.observation:
            expanded.extend(self._expand_free_variables_in_fact(obs))
        return self._dedup_facts(expanded)

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
        _, args = self._parse_fact(true_label)
        tomato = args[0]

        all_labels = [f"ripe({tomato})", f"unripe({tomato})", f"rotten({tomato})"]
        wrong_labels = [x for x in all_labels if x != true_label]

        outcomes = [ObservationOutcome(facts=[true_label], probability=self.scan_success_rate)]

        remain = 1.0 - self.scan_success_rate
        alt_prob = remain / (len(wrong_labels) + 1)

        for wrong in wrong_labels:
            outcomes.append(ObservationOutcome(facts=[wrong], probability=alt_prob))

        outcomes.append(ObservationOutcome(facts=[], probability=alt_prob))
        return outcomes

    def _build_detect_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        candidates = self.build_candidates(action)
        at_facts = [f for f in candidates if f.startswith("at(") and state.has_fact(f)]

        if not at_facts:
            return [ObservationOutcome(facts=[], probability=1.0)]

        tomato_groups = []
        for at_fact in at_facts:
            _, args = self._parse_fact(at_fact)
            tomato = args[0]

            # labels = [f"ripe({tomato})", f"unripe({tomato})", f"rotten({tomato})"]
            labels = [f"ripe({tomato})", f"unripe({tomato})"]
            true_label = None
            for lbl in labels:
                if state.has_fact(lbl):
                    true_label = lbl
                    break

            if true_label is None:
                tomato_groups.append([[at_fact], []])
            else:
                wrong_labels = [x for x in labels if x != true_label]
                # local = [
                #     [],
                #     [at_fact, true_label],
                #     [at_fact, wrong_labels[0]],
                #     [at_fact, wrong_labels[1]],
                # ]
                local = [
                    [],
                    [at_fact, true_label],
                    [at_fact, wrong_labels[0]],
                    # [at_fact, wrong_labels[1]],
                ]
                tomato_groups.append(local)

        success_facts = []
        for local in tomato_groups:
            success_facts.extend(local[1])
        success_facts = self._dedup_facts(success_facts)

        outcomes = [ObservationOutcome(facts=success_facts, probability=self.detect_success_rate)]

        alternatives = []
        for combo in product(*tomato_groups):
            flat = []
            for part in combo:
                flat.extend(part)
            flat = self._dedup_facts(flat)
            if set(flat) == set(success_facts):
                continue
            alternatives.append(flat)

        unique = []
        seen = set()
        for alt in alternatives:
            key = tuple(sorted(alt))
            if key not in seen:
                seen.add(key)
                unique.append(alt)

        if unique:
            alt_prob = (1.0 - self.detect_success_rate) / len(unique)
            for alt in unique:
                outcomes.append(ObservationOutcome(facts=alt, probability=alt_prob))
        else:
            outcomes[0].probability = 1.0

        return outcomes