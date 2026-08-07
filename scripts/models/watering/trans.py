from __future__ import annotations

from itertools import product
from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


class TransitionWatering:
    def __init__(self, type_map: Dict[str, List[str]]):
        self.type_map = type_map

        self.move_success_rate = 0.95
        self.find_success_rate = 0.99
        self.load_water_success_rate = 0.98
        self.pour_water_success_rate = 0.98

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
    def _make_outcome(add_facts: List[str], del_facts: List[str], probability: float) -> TransitionOutcome:
        return TransitionOutcome(
            add_facts=_dedup_facts(add_facts),
            del_facts=_dedup_facts(del_facts),
            probability=probability,
        )

    @staticmethod
    def _get_action_args(action: Action) -> List[str]:
        _, args = _parse_fact(action.name.replace(" ", ""))
        return args

    def _all_at_facts_for_object(self, obj: str) -> List[str]:
        return [f"at({obj},{room})" for room in self.type_map.get("R", [])]

    def _known_at_facts_for_object(self, state: State | None, obj: str) -> List[str]:
        if state is None:
            return []
        return [fact for fact in self._all_at_facts_for_object(obj) if state.has_fact(fact)]

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        action_name = action.name.split("(", 1)[0]

        if action_name == "find_plant":
            return self._build_find_plant_outcomes(action, state)

        return outcomes

    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "move":
            return self._build_move_outcomes(action)

        if action_name == "find_plant":
            return self._build_find_plant_outcomes(action)

        if action_name == "load_water":
            return self._build_load_water_outcomes(action)

        if action_name == "pour_water":
            return self._build_pour_water_outcomes(action)

        return [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=1.0,
            )
        ]

    def _build_move_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            action,
            success_rate=self.move_success_rate,
            failure_add_facts=[],
        )

    def _build_find_plant_outcomes(
        self,
        action: Action,
        state: State | None = None,
    ) -> List[TransitionOutcome]:
        args = self._get_action_args(action)
        if len(args) < 2:
            return [self._make_outcome([], [], 1.0)]

        _, room = args[:2]
        scanned_fact = f"scanned({room})"
        outcomes = self._build_find_all_plants_outcomes_for_room(room, state)
        return [
            self._make_outcome(
                add_facts=outcome.add_facts + [scanned_fact],
                del_facts=outcome.del_facts,
                probability=outcome.probability,
            )
            for outcome in outcomes
        ]

    def _build_find_all_plants_outcomes_for_room(
        self,
        room: str,
        state: State | None,
    ) -> List[TransitionOutcome]:
        per_plant_outcomes = []

        for plant in self.type_map.get("P", []):
            target_fact = f"at({plant},{room})"
            per_plant_outcomes.append(
                self._build_find_location_outcomes_for_state(plant, target_fact, state)
            )

        return self._merge_independent_outcome_groups(per_plant_outcomes)

    def _merge_independent_outcome_groups(
        self,
        outcome_groups: List[List[TransitionOutcome]],
    ) -> List[TransitionOutcome]:
        if not outcome_groups:
            return [self._make_outcome([], [], 1.0)]

        outcome_map = {}
        for combo in product(*outcome_groups):
            add_facts = []
            del_facts = []
            probability = 1.0

            for outcome in combo:
                add_facts.extend(outcome.add_facts)
                del_facts.extend(outcome.del_facts)
                probability *= outcome.probability

            add_facts = _dedup_facts(add_facts)
            del_facts = _dedup_facts(del_facts)
            key = (tuple(sorted(add_facts)), tuple(sorted(del_facts)))
            outcome_map[key] = outcome_map.get(key, 0.0) + probability

        return [
            self._make_outcome(
                add_facts=list(add_key),
                del_facts=list(del_key),
                probability=probability,
            )
            for (add_key, del_key), probability in outcome_map.items()
        ]

    def _build_find_location_outcomes(self, obj: str, target_fact: str) -> List[TransitionOutcome]:
        return self._build_find_location_outcomes_for_state(obj, target_fact, state=None)

    def _build_find_location_outcomes_for_state(
        self,
        obj: str,
        target_fact: str,
        state: State | None,
    ) -> List[TransitionOutcome]:
        if state is not None and state.has_fact(target_fact):
            return [self._make_outcome([], [], 1.0)]

        known_locations = self._known_at_facts_for_object(state, obj)
        if known_locations and target_fact not in known_locations:
            return [self._make_outcome([], [], 1.0)]

        other_locations = [
            fact
            for fact in self._all_at_facts_for_object(obj)
            if fact != target_fact
        ]

        return [
            self._make_outcome(
                add_facts=[target_fact],
                del_facts=other_locations,
                probability=self.find_success_rate,
            ),
            self._make_outcome(
                add_facts=[],
                del_facts=[],
                probability=1.0 - self.find_success_rate,
            ),
        ]

    def _build_load_water_outcomes(self, action: Action) -> List[TransitionOutcome]:
        args = self._get_action_args(action)
        failure_add_facts = []
        if len(args) >= 1:
            failure_add_facts.append(f"water_empty({args[0]})")

        return self._build_task_outcomes(
            action,
            success_rate=self.load_water_success_rate,
            failure_add_facts=failure_add_facts,
        )

    def _build_pour_water_outcomes(self, action: Action) -> List[TransitionOutcome]:
        args = self._get_action_args(action)
        failure_add_facts = []
        if len(args) >= 1:
            failure_add_facts.append(f"water_loaded({args[0]})")

        return self._build_task_outcomes(
            action,
            success_rate=self.pour_water_success_rate,
            failure_add_facts=failure_add_facts,
        )

    def _build_task_outcomes(
        self,
        action: Action,
        success_rate: float,
        failure_add_facts: List[str],
    ) -> List[TransitionOutcome]:
        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=success_rate,
        )

        failure = self._make_outcome(
            add_facts=failure_add_facts,
            del_facts=[],
            probability=1.0 - success_rate,
        )

        if failure.probability <= 0.0:
            return [success]
        return [success, failure]
