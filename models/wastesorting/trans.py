# models/wastesorting/trans.py
# Only for "wastesorting" domain.
# another: models/blocksworld/trans.py and models/tomato/trans.py
# 

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import re

from models.state import State
from models.action import Action


@dataclass(slots=True)
class TransitionOutcome:
    add_facts: List[str]
    del_facts: List[str]
    probability: float
    

class TransitionWastesorting:
    def __init__(self):
        self.navigate_success_rate = 0.9
    
    
    @staticmethod
    def _format_fact(pred: str, args: List[str]) -> str:
        return f"{pred}({','.join(args)})"
    
    
    @staticmethod
    def _parse_fact(fact: str) -> Tuple[str, List[str]]:
        """
        'located(changmin,stem1)' -> ('located', ['changmin', 'stem1'])
        """
        fact = fact.replace(" ", "")
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", fact)
        if not m:
            raise ValueError(f"Invalid fact format: {fact}")
        pred = m.group(1)
        args = [x.strip() for x in m.group(2).split(",")]
        return pred, args
    
    
    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        """
        예: 'located(changmin,L3)' -> ['located(changmin,dockstation)', 'located(changmin,stem1)', ...]
        """
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
        
        
    
    def bulid_outcomes(self, action_name: str, action: Action):
        if action_name == "navigate":
            return self._build_navigate_outcomes(action)

        elif action_name == "place":
            return self._build_prepare_nav_outcomes(action)

        elif action_name == "discard":
            return self._build_detect_outcomes(action)
        
        elif action_name == "discard":
            return self._build_pick_outcomes(action)
        
        elif action_name == "discard":
            return self._build_scan_outcomes(action)
        
        elif action_name == "discard":
            return self._build_place_outcomes(action)
        
        elif action_name == "discard":
            return self._build_discard_outcomes(action)
        
        else:
            # 기본값: deterministic
            return []

    
    
    def _build_navigate_outcomes(self, action: Action) -> List[TransitionOutcome]:
        nominal_target = action.add_effects[0]  # ex) located(changmin,stem1)
        
        expanded_obs = []
        for obs in action.observation:
            expanded_obs.extend(self._expand_free_variables_in_fact(obs))
        expanded_obs = list(dict.fromkeys(expanded_obs))  # 중복 제거
        
        # location 관련 후보만 사용
        location_candidates = [f for f in expanded_obs if f.startswith("located(")]

        # nominal target 제외한 나머지 후보
        alternatives = [f for f in location_candidates if f != nominal_target]
        outcomes = []
        # 성공
        
        prob = self.navigate_success_rate # + random.norm
        outcomes.append(
            TransitionOutcome(
                add_facts=[nominal_target],
                del_facts=action.del_effects,
                probability=prob
            )
        )
        
        # 나머지 가능한 위치로 잘못 가는 경우
        if alternatives:
            alt_prob = (1-prob) / len(alternatives)
            for alt in alternatives:
                # 기존 del_effects 중 located(...) 삭제 후 대체 위치 추가
                outcomes.append(
                    TransitionOutcome(
                        add_facts=[alt],
                        del_facts=action.del_effects,
                        probability=alt_prob
                    )
                )
        else:
            outcomes[0].probability = 1.0

        return outcomes
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    def _build_prepare_nav_outcomes(self, action: Action):
        pass
    
    
    def _build_detect_outcomes(self, action: Action):
        pass
    
    def _build_pick_outcomes(self, action: Action):
        pass
    
    def _build_scan_outcomes(self, action: Action):
        pass
    
    def _build_place_outcomes(self, action: Action):
        pass
    
    def _build_discard_outcomes(self, action: Action):
        pass