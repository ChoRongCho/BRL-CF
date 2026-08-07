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
    CATEGORY_PREDICATES = ("plastic", "can", "paper", "general")
    PLACE_ACTIONS = {
        "place_gw_bin",
        "place_paper_bin",
        "place_can_bin",
        "place_plastic_bin",
    }

    def __init__(self, type_map: Dict[str, List[str]]):
        self.type_map = type_map

        # # original
        # self.detect_observed_success_rate = 0.995
        # self.detect_classification_success_rate = 0.50
        # self.pick_success_rate = 0.98
        # self.place_success_rate = 0.95
        
        self.detect_observed_success_rate = 0.80
        self.detect_classification_success_rate = 0.25
        self.pick_success_rate = 0.90
        self.place_success_rate = 0.90
        
    
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

    @staticmethod
    def _category_label_for_state(state: State, waste: str) -> str | None:
        for pred in TransitionWastesorting.CATEGORY_PREDICATES:
            fact = f"{pred}({waste})"
            if state.has_fact(fact):
                return fact
        return None

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
        if not action.name.startswith("detect_waste("):
            return outcomes

        _, unavailable_wastes = self._waste_status_from_state(state)
        return self._build_detect_outcomes(
            action,
            blocked_wastes=unavailable_wastes,
            state=state,
        )
    
    
    
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
        known_label: str | None = None,
    ) -> List[Tuple[List[str], List[str], float]]:
        category_del_facts = [f"{pred}({waste})" for pred in category_predicates]
        labels = [f"{pred}({waste})" for pred in category_predicates]
        if known_label not in labels:
            known_label = None

        if known_label is None:
            label_prob = 1.0 / len(labels)
            return [
                (
                    [f"detected({waste})", label],
                    category_del_facts,
                    label_prob,
                )
                for label in labels
            ]

        wrong_labels = [label for label in labels if label != known_label]
        wrong_prob = (1.0 - self.detect_classification_success_rate) / len(wrong_labels)

        choices = [
            (
                [f"detected({waste})", known_label],
                category_del_facts,
                self.detect_classification_success_rate,
            )
        ]
        choices.extend(
            (
                [f"detected({waste})", label],
                category_del_facts,
                wrong_prob,
            )
            for label in wrong_labels
        )
        return choices

    def _build_detect_outcomes(
        self,
        action: Action,
        blocked_wastes: set[str] | None = None,
        state: State | None = None,
    ) -> List[TransitionOutcome]:
        
        # 1. 관측 대상에서 차단된 폐기물과 중복을 제거하여 처리 순서대로 정리한다.
        blocked_wastes = blocked_wastes or set()
        wastes = []
        seen_wastes = set()
        for fact in self._detect_facts_from_action(action):
            _, args = _parse_fact(fact)
            if not args:
                continue
            waste = args[0]
            if waste in blocked_wastes or waste in seen_wastes:
                continue
            seen_wastes.add(waste)
            wastes.append(waste)

        if not wastes:
            return [self._make_outcome(add_facts=[], del_facts=[], probability=1.0)]
        
        # 2. 각 폐기물의 탐지 성공과 미탐지 결과에 배정할 확률을 계산한다.
        p_detect = self.detect_observed_success_rate
        p_miss = 1.0 - p_detect

        # 3. action이 관측하는 분류 카테고리를 추출하고, 없으면 전체 카테고리를 사용한다.
        category_predicates = []
        seen_categories = set()
        for fact in action.observation:
            predicate, args = _parse_fact(fact)
            if (
                args
                and predicate in self.CATEGORY_PREDICATES
                and predicate not in seen_categories
            ):
                seen_categories.add(predicate)
                category_predicates.append(predicate)
                
        if not category_predicates:
            category_predicates = list(self.CATEGORY_PREDICATES)

        # 4. 각 폐기물에 대해 미탐지와 정답·오분류 레이블을 독립적인 선택지로 생성한다.
        per_waste_choices = []
        for waste in wastes:
            label_choices = self._build_detect_label_choices(
                waste,
                category_predicates,
                known_label=(
                    self._category_label_for_state(state, waste)
                    if state is not None
                    else None
                ),
            )
            per_waste_choices.append(
                [([], [], p_miss)]
                + [
                    (add_facts, del_facts, probability * p_detect)
                    for add_facts, del_facts, probability in label_choices
                ]
            )

        # 5. 폐기물별 분류 결과의 데카르트 곱을 만들고, 동일한 상태 변화의 확률을 합산한다.
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

        # 6. 동일한 상태 변화별로 합산된 확률을 최종 전이 결과로 변환한다.
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
