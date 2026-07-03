from __future__ import annotations

from typing import Dict, List
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


class TransitionKitchen:
    def __init__(self, type_map: Dict[str, List[str]]):
        self.type_map = type_map

        self.open_fridge_success_rate = 0.90
        self.inspect_success_rate = 0.90
        self.pick_success_rate = 0.90
        self.wash_success_rate = 0.95
        self.place_success_rate = 0.95
        self.boil_success_rate = 0.95
        self.serve_success_rate = 0.95

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

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        action_name = action.name.split("(", 1)[0]

        if action_name == "open_fridge":
            return self._build_open_fridge_outcomes(action, state)

        if action_name in {"inspect_ingredient", "inspect_pot", "inspect_dish"}:
            return self._build_inspect_outcomes(action, state)

        return outcomes

    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "open_fridge":
            return self._build_open_fridge_outcomes(action)

        if action_name in {"inspect_ingredient", "inspect_pot", "inspect_dish"}:
            return self._build_inspect_outcomes(action)

        if action_name in {"pick_ingredient", "pick_ingredient_from_pot", "pick_pot", "pick_dish"}:
            return self._build_pick_outcomes(action)

        if action_name in {"wash_ingredient", "wash_pot", "wash_dish"}:
            return self._build_wash_outcomes(action)

        if action_name in {"place_pot", "place_dish", "place_ingredient_in_pot", "discard_ingredient"}:
            return self._build_place_outcomes(action)

        if action_name in {"boil_tomato_soup", "boil_onion_soup", "boil_tomato_oniton_soup"}:
            return self._build_boil_outcomes(action)

        if action_name in {"serve_tomato_soup", "serve_onion_soup", "serve_tomato_oniton_soup"}:
            return self._build_serve_outcomes(action)

        return [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=1.0,
            )
        ]

    def _build_open_fridge_outcomes(
        self,
        action: Action,
        state: State | None = None,
    ) -> List[TransitionOutcome]:
        args = self._get_action_args(action)
        if len(args) < 3:
            return [self._make_outcome([], [], 1.0)]

        _, ingredient, fridge = args[:3]
        target_fact = f"in_fridge({ingredient},{fridge})"

        if state is not None:
            if self._is_holding_object(state, ingredient):
                return [self._make_outcome([], [], 1.0)]
            if state.has_fact(target_fact):
                return [self._make_outcome([], [], 1.0)]

        known_fridge_facts = self._known_in_fridge_facts_for_ingredient(state, ingredient)
        if known_fridge_facts and target_fact not in known_fridge_facts:
            return [self._make_outcome([], [], 1.0)]

        return [
            self._make_outcome(
                add_facts=[target_fact],
                del_facts=self._all_in_fridge_facts_for_ingredient(ingredient, exclude=target_fact),
                probability=self.open_fridge_success_rate,
            ),
            self._make_outcome([], [], 1.0 - self.open_fridge_success_rate),
        ]

    def _build_inspect_outcomes(
        self,
        action: Action,
        state: State | None = None,
    ) -> List[TransitionOutcome]:
        args = self._get_action_args(action)
        if len(args) < 2:
            return [self._make_outcome([], [], 1.0)]

        obj = args[1]
        label = self._known_cleanliness_fact(state, obj) if state is not None else None

        if label is not None:
            return [self._make_outcome([], [], 1.0)]

        del_facts = [f"clean({obj})", f"dirty({obj})"]
        reveal_prob = self.inspect_success_rate / 2.0
        return [
            self._make_outcome(
                add_facts=[f"clean({obj})"],
                del_facts=del_facts,
                probability=reveal_prob,
            ),
            self._make_outcome(
                add_facts=[f"dirty({obj})"],
                del_facts=del_facts,
                probability=reveal_prob,
            ),
            self._make_outcome([], [], 1.0 - self.inspect_success_rate),
        ]

    def _known_in_fridge_facts_for_ingredient(self, state: State | None, ingredient: str) -> List[str]:
        if state is None:
            return []
        return [
            fact
            for fact in self._all_in_fridge_facts_for_ingredient(ingredient)
            if state.has_fact(fact)
        ]

    def _all_in_fridge_facts_for_ingredient(self, ingredient: str, exclude: str | None = None) -> List[str]:
        facts = [f"in_fridge({ingredient},{fridge})" for fridge in self.type_map.get("F", [])]
        if exclude is not None:
            facts = [fact for fact in facts if fact != exclude]
        return facts

    def _build_pick_outcomes(self, action: Action) -> List[TransitionOutcome]:
        action_name = action.name.replace(" ", "").split("(", 1)[0]
        add_facts = list(action.add_effects)
        del_facts = list(action.del_effects)

        if action_name == "pick_ingredient":
            args = self._get_action_args(action)
            if len(args) >= 3:
                _, ingredient, fridge = args[:3]
                del_facts.append(f"in_fridge({ingredient},{fridge})")

        return self._build_task_outcomes(
            add_facts=add_facts,
            del_facts=del_facts,
            success_rate=self.pick_success_rate,
            failure_add_facts=self._facts_to_preserve_on_failure(del_facts),
        )

    def _build_wash_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            success_rate=self.wash_success_rate,
            failure_add_facts=self._facts_to_preserve_on_failure(action.del_effects),
        )

    def _build_place_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            success_rate=self.place_success_rate,
            failure_add_facts=self._facts_to_preserve_on_failure(action.del_effects),
        )

    def _build_boil_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            success_rate=self.boil_success_rate,
            failure_add_facts=self._facts_to_preserve_on_failure(action.del_effects),
        )

    def _build_serve_outcomes(self, action: Action) -> List[TransitionOutcome]:
        return self._build_task_outcomes(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            success_rate=self.serve_success_rate,
            failure_add_facts=self._facts_to_preserve_on_failure(action.del_effects),
        )

    @staticmethod
    def _facts_to_preserve_on_failure(del_facts: List[str]) -> List[str]:
        return [fact.replace(" ", "") for fact in del_facts]

    def _build_task_outcomes(
        self,
        add_facts: List[str],
        del_facts: List[str],
        success_rate: float,
        failure_add_facts: List[str],
    ) -> List[TransitionOutcome]:
        success = self._make_outcome(
            add_facts=add_facts,
            del_facts=del_facts,
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
