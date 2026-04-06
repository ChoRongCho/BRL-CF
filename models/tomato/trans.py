# models/tomato/trans.py

from __future__ import annotations
from itertools import product
from typing import List, Dict, Tuple

import re
from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.transition import TransitionOutcome


class TransitionTomato:
    def __init__(self, type_map: Dict[str, List[str]], true_state: State):
        self.type_map = type_map
        self.true_state = true_state

        self.navigate_success_rate = 0.90
        self.prepare_nav_success_rate = 1.0
        self.detect_success_rate = 0.90
        self.pick_success_rate = 0.88
        self.scan_success_rate = 0.97
        self.place_success_rate = 0.99
        self.discard_success_rate = 0.99


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

    @staticmethod
    def _fluent_effect_to_state_update(effect) -> Dict[str, Dict[str, float]]:
        if not isinstance(effect, dict):
            return {}

        updates: Dict[str, Dict[str, float]] = {}
        for expr, value in effect.items():
            predicate, args = _parse_fact(str(expr).replace(" ", ""))
            if not args:
                continue

            obj = args[0]
            if obj not in updates:
                updates[obj] = {}
            updates[obj][predicate] = float(value)

        return updates

    @classmethod
    def _fluent_effects_to_state_update(cls, effects) -> Dict[str, Dict[str, float]]:
        updates: Dict[str, Dict[str, float]] = {}

        for effect in effects or []:
            for obj, values in cls._fluent_effect_to_state_update(effect).items():
                if obj not in updates:
                    updates[obj] = {}
                updates[obj].update(values)

        return updates

    @staticmethod
    def _make_outcome(
        add_facts: List[str],
        del_facts: List[str],
        probability: float,
        fluent_effects: Dict[str, Dict[str, float]] | None = None,
    ) -> TransitionOutcome:
        return TransitionOutcome(
            add_facts=list(dict.fromkeys(f.replace(" ", "") for f in add_facts)),
            del_facts=list(dict.fromkeys(f.replace(" ", "") for f in del_facts)),
            probability=probability,
            fluent_effects=fluent_effects or {},
        )

    def _true_has_fact(self, fact: str) -> bool:
        return self.true_state.has_fact(fact.replace(" ", ""))

    def _extract_holding_facts(self, action: Action) -> List[str]:
        return [f.replace(" ", "") for f in action.observation if f.replace(" ", "").startswith("holding(")]

    def _extract_located_facts_from_observation(self, action: Action) -> List[str]:
        expanded_obs = []
        for obs in action.observation:
            expanded_obs.extend(self._expand_free_variables_in_fact(obs))
        expanded_obs = _dedup_facts(expanded_obs)
        return [f for f in expanded_obs if f.startswith("located(")]

    def _extract_true_facts_from_observation(self, action: Action) -> List[str]:
        expanded_obs = []
        for obs in action.observation:
            expanded_obs.extend(self._expand_free_variables_in_fact(obs))
        expanded_obs = _dedup_facts(expanded_obs)
        
        true_facts = []
        for fact in expanded_obs:
            if self._true_has_fact(fact):
                true_facts.append(fact)
        return true_facts

    def handle_exeception(self, state: State, action: Action, outcomes: List[TransitionOutcome]):
        current_facts = set(state.facts)
        
        if not action.name.startswith("detect("):
            return outcomes
        
        observed_tomatoes = set()
        for fact in current_facts:
            if fact.startswith("observed(") and fact.endswith(")"):
                tomato = fact[len("observed("):-1].strip()
                observed_tomatoes.add(tomato)
        
        
        if not observed_tomatoes:
            return outcomes
        
        merged = {}
        for outcome in outcomes:
            filtered_add_facts = []

            for fact in outcome.add_facts:
                remove_fact = False

                for tomato in observed_tomatoes:
                    if fact == f"observed({tomato})":
                        remove_fact = True
                        break
                    if fact == f"ripe({tomato})":
                        remove_fact = True
                        break
                    if fact == f"unripe({tomato})":
                        remove_fact = True
                        break
                    if fact == f"rotten({tomato})":
                        remove_fact = True
                        break
                    if fact.startswith(f"at({tomato},") and fact.endswith(")"):
                        remove_fact = True
                        break

                if not remove_fact:
                    filtered_add_facts.append(fact)

            filtered_add_facts = _dedup_facts(filtered_add_facts)
            key = (tuple(sorted(filtered_add_facts)), tuple(sorted(outcome.del_facts)))

            if key not in merged:
                merged[key] = TransitionOutcome(
                    add_facts=filtered_add_facts,
                    del_facts=outcome.del_facts,
                    probability=outcome.probability,
                    fluent_effects=outcome.fluent_effects,
                )
            else:
                merged[key].probability += outcome.probability
                
        return list(merged.values())

    
    
    
    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        if action_name == "navigate":
            return self._build_navigate_outcomes(action)

        elif action_name == "prepare_nav":
            return self._build_prepare_nav_outcomes(action)

        elif action_name == "detect":
            return self._build_detect_outcomes(action)

        elif action_name == "pick":
            return self._build_pick_outcomes(action)

        elif action_name == "scan":
            return self._build_scan_outcomes(action)

        elif action_name == "place":
            return self._build_place_outcomes(action)

        elif action_name == "discard":
            return self._build_discard_outcomes(action)

        else:
            return [
                self._make_outcome(
                    add_facts=action.add_effects,
                    del_facts=action.del_effects,
                    probability=1.0
                )
            ]

    def _build_navigate_outcomes(self, action: Action) -> List[TransitionOutcome]:
        nominal_target = action.add_effects[0].replace(" ", "")
        location_candidates = self._extract_located_facts_from_observation(action)
        alternatives = [f for f in location_candidates if f != nominal_target]

        outcomes = []
        prob = self.navigate_success_rate

        outcomes.append(
            self._make_outcome(
                add_facts=[nominal_target],
                del_facts=action.del_effects,
                probability=prob
            )
        )

        if alternatives:
            alt_prob = (1.0 - prob) / len(alternatives)
            for alt in alternatives:
                outcomes.append(
                    self._make_outcome(
                        add_facts=[alt],
                        del_facts=action.del_effects,
                        probability=alt_prob
                    )
                )
        else:
            outcomes[0].probability = 1.0

        return outcomes

    def _build_prepare_nav_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.prepare_nav_success_rate

        return [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=prob
            ),
            self._make_outcome(
                add_facts=[],
                del_facts=[],
                probability=1.0 - prob
            )
        ]

    # TODO: Changmin
    def _build_detect_outcomes(self, action: Action) -> List[TransitionOutcome]:
        """
        detect:
        각 tomato에 대해
        - 기존 detection 성공률은 유지한다.
        - 성공한 경우 quality label(ripe/unripe)은 동일 확률로 둔다.
        - 실패한 경우 아무 정보도 못 얻음.
        """

        true_facts = self._extract_true_facts_from_observation(action)
        at_facts = [f for f in true_facts if f.startswith("at(")]

        tomato_entries = []
        for fact in at_facts:
            predicate, arity = _parse_fact(fact)
            if predicate != "at":
                continue

            tomato = arity[0]
            location = arity[1]

            tomato_entries.append({
                "tomato": tomato,
                "location": location,
                "at_fact": fact.replace(" ", ""),
                "observed_fact": f"observed({tomato})",
            })

        # stem에 tomato가 하나도 없으면 detect 결과는 빈 outcome 1개
        if not tomato_entries:
            return [
                self._make_outcome(
                    add_facts=[],
                    del_facts=[],
                    probability=1.0
                )
            ]

        per_tomato_choices = []

        p_detect = 0.95
        p_miss = 0.05
        
        for entry in tomato_entries:
            tomato = entry["tomato"]
            quality_del_facts = [
                f"ripe({tomato})",
                f"unripe({tomato})",
                f"rotten({tomato})",
            ]

            # 가능한 label들
            all_labels = [
                f"ripe({tomato})",
                f"unripe({tomato})",
            ]
            
            local_choices = []
            label_prob = p_detect / len(all_labels)
            
            # 1. Detection success: observed + at + one unbiased quality label.
            for label in all_labels:
                local_choices.append((
                    [entry["observed_fact"], entry["at_fact"], label],
                    quality_del_facts,
                    label_prob
                ))

            # 2. Detection failure.
            local_choices.append((
                [],
                [],
                p_miss
            ))
            per_tomato_choices.append(local_choices)
        
        outcomes = []
        outcome_map = {}

        # 전체 조합 생성
        for combo in product(*per_tomato_choices):
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

            if key in outcome_map:
                outcome_map[key] += prob
            else:
                outcome_map[key] = prob

        for (add_key, del_key), prob in outcome_map.items():
            outcomes.append(
                self._make_outcome(
                    add_facts=list(add_key),
                    del_facts=list(del_key),
                    probability=prob
                )
            )

        return outcomes

    def _build_pick_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.pick_success_rate
        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=prob,
            fluent_effects=self._fluent_effects_to_state_update(action.del_effect_fluents),
        )

        failure_add = []
        if any("handempty(" in d.replace(" ", "") for d in action.del_effects):
            failure_add.extend(
                [d.replace(" ", "") for d in action.del_effects if "handempty(" in d.replace(" ", "")]
            )

        failure = self._make_outcome(
            add_facts=failure_add,
            del_facts=[],
            probability=1.0 - prob
        )

        return [success, failure]

    def _build_scan_outcomes(self, action: Action) -> List[TransitionOutcome]:
        """
        scan은 항상 성공하며 scanned(T)를 반환한다.
        quality label은 ripe/rotten만 동일 확률로 구분한다.
        """
        _, args = _parse_fact(action.name)
        tomato = args[1]
        quality_facts = [
            f"ripe({tomato})",
            f"unripe({tomato})",
            f"rotten({tomato})",
        ]
        scanned_fact = f"scanned({tomato})"

        labels = [
            f"ripe({tomato})",
            f"rotten({tomato})",
        ]

        return [
            self._make_outcome(
                add_facts=[label, scanned_fact],
                del_facts=quality_facts,
                probability=1.0 / len(labels)
            )
            for label in labels
        ]

    def _build_place_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.place_success_rate
        holding_facts = self._extract_holding_facts(action)

        outcomes = [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=prob
            )
        ]

        if holding_facts:
            outcomes.append(
                self._make_outcome(
                    add_facts=holding_facts,
                    del_facts=[],
                    probability=1.0 - prob
                )
            )
        else:
            outcomes.append(
                self._make_outcome(
                    add_facts=[],
                    del_facts=[],
                    probability=1.0 - prob
                )
            )

        return outcomes

    def _build_discard_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.discard_success_rate
        holding_facts = self._extract_holding_facts(action)

        outcomes = [
            self._make_outcome(
                add_facts=action.add_effects,
                del_facts=action.del_effects,
                probability=prob
            )
        ]

        if holding_facts:
            outcomes.append(
                self._make_outcome(
                    add_facts=holding_facts,
                    del_facts=[],
                    probability=1.0 - prob
                )
            )
        else:
            outcomes.append(
                self._make_outcome(
                    add_facts=[],
                    del_facts=[],
                    probability=1.0 - prob
                )
            )

        return outcomes
