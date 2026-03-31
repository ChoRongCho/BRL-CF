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
        self.prepare_nav_success_rate = 0.98
        self.detect_success_rate = 0.85
        self.pick_success_rate = 0.75
        self.scan_success_rate = 0.85
        self.place_success_rate = 0.90
        self.discard_success_rate = 0.95


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
    def _make_outcome(add_facts: List[str], del_facts: List[str], probability: float) -> TransitionOutcome:
        return TransitionOutcome(
            add_facts=list(dict.fromkeys(f.replace(" ", "") for f in add_facts)),
            del_facts=list(dict.fromkeys(f.replace(" ", "") for f in del_facts)),
            probability=probability
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
                    probability=outcome.probability
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
        - 0.9 확률로 발견: observed(T), at(T,S), true ripeness
        - 0.1 확률로 미발견: 아무 정보도 못 얻음
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

            true_label = None
            for c in [f"ripe({tomato})", f"unripe({tomato})", f"rotten({tomato})"]:
            # for c in [f"ripe({tomato})", f"unripe({tomato})"]:
                if self._true_has_fact(c):
                    if c.startswith("rotten"):
                        true_label = f"ripe({tomato})"
                    else:
                        true_label = c  
                    break

            if true_label is None:
                continue

            tomato_entries.append({
                "tomato": tomato,
                "location": location,
                "at_fact": fact.replace(" ", ""),
                "observed_fact": f"observed({tomato})",
                "true_label": true_label
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

        # 각 tomato별 local choice:
        # 1) 발견 성공: observed + at + true_label
        # 2) 발견 실패: []
        per_tomato_choices = []

        
        p_true = 0.8
        p_false = 0.15
        p_miss = 0.05
        
        for entry in tomato_entries:
            tomato = entry["tomato"]
            true_label = entry["true_label"]

            # 가능한 label들
            all_labels = [
                f"ripe({tomato})",
                f"unripe({tomato})",
                f"rotten({tomato})"
            ]
            
            wrong_labels = [lbl for lbl in all_labels if lbl != true_label]
            local_choices = []
            
            # 1. True detection
            local_choices.append((
                [entry["observed_fact"], entry["at_fact"], true_label],
                p_true
            ))
            
             # 2. False detection (label만 틀림)
            false_prob_each = p_false / len(wrong_labels)
            for wrong_label in wrong_labels:
                local_choices.append((
                    [entry["observed_fact"], entry["at_fact"], wrong_label],
                    false_prob_each
                ))

            # 3. Miss detection
            local_choices.append((
                [],
                p_miss
            ))
            per_tomato_choices.append(local_choices)
        
        outcomes = []
        outcome_map = {}

        # 전체 조합 생성
        for combo in product(*per_tomato_choices):
            add_facts = []
            prob = 1.0

            for facts_part, part_prob in combo:
                add_facts.extend(facts_part)
                prob *= part_prob

            add_facts = _dedup_facts(add_facts)
            key = tuple(sorted(add_facts))

            if key in outcome_map:
                outcome_map[key] += prob
            else:
                outcome_map[key] = prob

        for add_key, prob in outcome_map.items():
            outcomes.append(
                self._make_outcome(
                    add_facts=list(add_key),
                    del_facts=[],
                    probability=prob
                )
            )

        return outcomes

    def _build_pick_outcomes(self, action: Action) -> List[TransitionOutcome]:
        prob = self.pick_success_rate

        success = self._make_outcome(
            add_facts=action.add_effects,
            del_facts=action.del_effects,
            probability=prob
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
        scan은 observation 후보 중 true_state에 실제 있는 fact를 기반으로 결과를 만든다.
        성공 시 true ripeness를 반환하고,
        실패 alternatives에서는 wrong ripeness 또는 empty observation을 반환한다.
        """
        true_facts = self._extract_true_facts_from_observation(action)
        prob = self.scan_success_rate

        # scan에서 ripeness 관련 fact만 뽑기
        ripeness_facts = []
        for fact in true_facts:
            if fact.startswith("ripe(") or fact.startswith("unripe(") or fact.startswith("rotten("):
                ripeness_facts.append(fact)

        # true ripeness가 없으면 기존 방식 fallback
        if not ripeness_facts:
            success_add = true_facts if true_facts else action.add_effects
            return [
                self._make_outcome(
                    add_facts=success_add,
                    del_facts=action.del_effects,
                    probability=prob
                ),
                self._make_outcome(
                    add_facts=[],
                    del_facts=[],
                    probability=1.0 - prob
                )
            ]

        # 성공 outcome
        success_add = ripeness_facts
        outcomes = [
            self._make_outcome(
                add_facts=success_add,
                del_facts=action.del_effects,
                probability=prob
            )
        ]

        # 실패 alternatives 생성
        alternatives = []

        for true_label in ripeness_facts:
            predicate, arity = _parse_fact(true_label)
            tomato = arity[0]

            all_labels = [
                f"ripe({tomato})",
                f"unripe({tomato})",
                f"rotten({tomato})"
            ]

            wrong_labels = [lbl for lbl in all_labels if lbl != true_label]
            for wrong_label in wrong_labels:
                alternatives.append([wrong_label])

        # 완전 실패: 아무 것도 못 얻음
        alternatives.append([])

        # 중복 제거
        unique_alternatives = []
        seen = set()
        for alt in alternatives:
            key = tuple(sorted(alt))
            if key not in seen:
                seen.add(key)
                unique_alternatives.append(alt)

        alt_prob = (1.0 - prob) / len(unique_alternatives)

        for alt in unique_alternatives:
            outcomes.append(
                self._make_outcome(
                    add_facts=alt,
                    del_facts=[],
                    probability=alt_prob
                )
            )

        return outcomes

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