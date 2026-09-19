# models/wastesorting/trans.py
# Only for "wastesorting" domain.
# another: models/blocksworld/trans.py and models/tomato/trans.py


from __future__ import annotations
from itertools import product
from typing import List, Dict, Tuple

import re
from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


    

class TransitionWastesorting:
    REWARD_STATE_OBJECT = "__reward_state__"
    DETECT_STREAK_FLUENT = "detect_streak"

    CATEGORY_PREDICATES = ("plastic", "can", "paper", "general")
    PLACE_ACTIONS = {
        "place_gw_bin",
        "place_paper_bin",
        "place_can_bin",
        "place_plastic_bin",
    }

    def __init__(self, type_map: Dict[str, List[str]], settings: Dict | None = None):
        self.type_map = type_map
        success = (settings or {}).get("success", {})
        self.detect_observed_success_rate = float(success.get("detect_observed", 0.90))
        self.detect_classification_success_rate = float(success.get("detect_classification", 0.50))
        self.pick_success_rate = float(success.get("pick", 0.90))
        self.place_success_rate = float(success.get("place", 0.90))
        
    
    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        """
        예: 'located(brl_robot,L3)' -> ['located(brl_robot,dockstation)', 'located(brl_robot,stem1)', ...]
        """
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
        
    @staticmethod
    def _make_outcome(add_facts: List[str], del_facts: List[str], probability: float) -> TransitionOutcome:
        return TransitionOutcome(
            add_facts=list(dict.fromkeys(f.replace(" ", "") for f in add_facts)),
            del_facts=list(dict.fromkeys(f.replace(" ", "") for f in del_facts)),
            probability=probability
        )

    def _extract_holding_facts(self, action: Action) -> List[str]:
        return [f.replace(" ", "") for f in action.observation if f.replace(" ", "").startswith("holding(")]

    def _extract_located_facts_from_observation(self, action: Action) -> List[str]:
        expanded_obs = []
        for obs in action.observation:
            expanded_obs.extend(self._expand_free_variables_in_fact(obs))
        expanded_obs = _dedup_facts(expanded_obs)
        return [f for f in expanded_obs if f.startswith("located(")]

    def _detect_facts_from_action(self, action: Action) -> List[str]:
        detected_facts = []
        sources = (action.add_effects, action.observation)
        for facts in sources:
            for fact in facts:
                pred, _ = _parse_fact(fact)
                if pred != "detected":
                    continue
                detected_facts.extend(self._expand_free_variables_in_fact(fact))
            if detected_facts:
                break
        return _dedup_facts(detected_facts)

    def _build_detect_waste_entries(
        self,
        action: Action,
        blocked_wastes: set[str] | None = None,
    ) -> List[Dict[str, str]]:
        blocked_wastes = blocked_wastes or set()
        wastes = []
        seen = set()
        for fact in self._detect_facts_from_action(action):
            _, args = _parse_fact(fact)
            if not args or args[0] in seen or args[0] in blocked_wastes:
                continue
            wastes.append(args[0])
            seen.add(args[0])

        return [
            {
                "waste": waste,
                "detected_fact": f"detected({waste})",
            }
            for waste in wastes
        ]

    @staticmethod
    def _category_label_for_state(state: State, waste: str) -> str | None:
        for pred in TransitionWastesorting.CATEGORY_PREDICATES:
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

    @staticmethod
    def _waste_status_from_state(state: State) -> Tuple[set[str], set[str]]:
        detected_wastes = set()
        unavailable_wastes = set()

        for fact in state.facts:
            pred, args = _parse_fact(fact)
            if not args:
                continue

            if pred == "detected":
                detected_wastes.add(args[0])
            elif pred == "in_bin":
                unavailable_wastes.add(args[0])
            elif pred == "holding" and len(args) >= 2:
                unavailable_wastes.add(args[1])

        return detected_wastes, unavailable_wastes

    @staticmethod
    def _effect_key(state: State, add_facts: List[str], del_facts: List[str]) -> Tuple[str, ...]:
        facts = set(fact.replace(" ", "") for fact in state.facts)
        for fact in del_facts:
            facts.discard(fact.replace(" ", ""))
        for fact in add_facts:
            facts.add(fact.replace(" ", ""))
        return tuple(sorted(facts))

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        is_detect = action.name.startswith("detect_waste(")
        if is_detect:
            _, unavailable_wastes = self._waste_status_from_state(state)
            outcomes = self._build_detect_outcomes(
                action,
                blocked_wastes=unavailable_wastes,
            )

        current_streak = int(state.get_fluent(
            self.REWARD_STATE_OBJECT,
            self.DETECT_STREAK_FLUENT,
            0,
        ))
        next_streak = min(3, current_streak + 1) if is_detect else 0

        # Store the counter in each simulated state. Keeping it on the reward
        # object would leak history between independent POMCP branches.
        streak_outcomes = []
        for outcome in outcomes:
            fluent_effects = {
                obj: dict(values)
                for obj, values in outcome.fluent_effects.items()
            }
            fluent_effects.setdefault(self.REWARD_STATE_OBJECT, {})[
                self.DETECT_STREAK_FLUENT
            ] = float(next_streak)
            streak_outcomes.append(TransitionOutcome(
                add_facts=list(outcome.add_facts),
                del_facts=list(outcome.del_facts),
                probability=outcome.probability,
                fluent_effects=fluent_effects,
            ))
        return streak_outcomes
    
    
    
    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "detect_waste":
            return self._build_detect_outcomes(action)

        elif action_name == "pick":
            return self._build_pick_outcomes(action)

        elif action_name in self.PLACE_ACTIONS:
            return self._build_place_outcomes(action)

        else:
            return [
                self._make_outcome(
                    add_facts=action.add_effects,
                    del_facts=action.del_effects,
                    probability=1.0
                )
            ]

    def _build_detect_label_choices(
        self,
        waste: str,
        category_predicates: List[str],
    ) -> List[Tuple[List[str], List[str], float]]:
        category_del_facts = [f"{pred}({waste})" for pred in category_predicates]
        labels = [f"{pred}({waste})" for pred in category_predicates]
        probability = 1.0 / (len(labels) + 1)

        return [([], [], probability)] + [
            (
                [f"detected({waste})", label],
                category_del_facts,
                probability,
            )
            for label in labels
        ]

    def _build_detect_outcomes(
        self,
        action: Action,
        blocked_wastes: set[str] | None = None,
    ) -> List[TransitionOutcome]:
        waste_entries = self._build_detect_waste_entries(action, blocked_wastes)
        if not waste_entries:
            return [self._make_outcome(add_facts=[], del_facts=[], probability=1.0)]

        category_predicates = self._category_predicates_from_observation(action)
        per_waste_choices = [
            self._build_detect_label_choices(entry["waste"], category_predicates)
            for entry in waste_entries
        ]

        outcome_map = {}
        for combo in product(*per_waste_choices):
            add_facts = []
            del_facts = []
            prob = 1.0

            for facts_part, del_part, part_prob in combo:
                add_facts.extend(facts_part)
                del_facts.extend(del_part)
                prob *= part_prob

            add_facts = _dedup_facts(add_facts)
            del_facts = _dedup_facts(del_facts)
            key = (tuple(sorted(add_facts)), tuple(sorted(del_facts)))
            outcome_map[key] = outcome_map.get(key, 0.0) + prob

        outcomes = [
            self._make_outcome(
                add_facts=list(add_key),
                del_facts=list(del_key),
                probability=prob,
            )
            for (add_key, del_key), prob in outcome_map.items()
        ]
        return outcomes

    def _build_pick_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.pick_success_rate
        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=prob,
        )

        failure_add = []
        if any("handempty(" in d.replace(" ", "") for d in action.del_effects):
            failure_add.extend(
                [d.replace(" ", "") for d in action.del_effects if "handempty(" in d.replace(" ", "")]
            )

        failure = self._make_outcome(
            add_facts=failure_add,
            del_facts=[],
            probability=1.0 - prob,
        )

        return [success, failure]

    def _build_place_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.place_success_rate
        holding_facts = self._extract_holding_facts(action)

        outcomes = [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=prob,
            )
        ]

        if holding_facts:
            outcomes.append(
                self._make_outcome(
                    add_facts=holding_facts,
                    del_facts=[],
                    probability=1.0 - prob,
                )
            )
        else:
            outcomes.append(
                self._make_outcome(
                    add_facts=[],
                    del_facts=[],
                    probability=1.0 - prob,
                )
            )

        return outcomes
