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
    def __init__(self, type_map: Dict[str, List[str]], true_state: State):
        self.type_map = type_map
        self.true_state = true_state

        # action success rate
        self.detect = 0.90
        self.pick = 0.98
        self.scan_material = 0.85
        self.probe_hardness = 0.75
        self.compress_item = 0.85
        self.place_compressed_in_bin = 0.90
        self.place_compressed_can_in_bin = 0.95
        self.place_direct_in_bin = 0.95
        self.place_direct_can_in_bin = 0.95
        self.place_direct_glass_in_bin = 0.95
        self.place_direct_general_in_bin = 0.95
        self.place_direct_hazardous_in_bin = 0.95
        
    
    
    def _expand_free_variables_in_fact(self, fact: str) -> List[str]:
        """
        예: 'located(changmin,L3)' -> ['located(changmin,dockstation)', 'located(changmin,stem1)', ...]
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
        """
        """
        
        merged = {}
        return list(merged.values())
    
    
    
    def build_outcomes(self, action_name: str, action: Action) -> List[TransitionOutcome]:
        """
        # action list
        self.detect = 0.90
        self.pick = 0.98
        self.scan_material = 0.85
        self.probe_hardness = 0.75
        self.compress_item = 0.85
        self.place_compressed_in_bin = 0.90
        self.place_compressed_can_in_bin = 0.95
        self.place_direct_in_bin = 0.95
        self.place_direct_can_in_bin = 0.95
        self.place_direct_glass_in_bin = 0.95
        self.place_direct_general_in_bin = 0.95
        self.place_direct_hazardous_in_bin = 0.95
        """
        if action_name == "detect":
            return self._build_detect_outcomes(action)

        elif action_name == "pick":
            return self._build_pick_outcomes(action)

        elif action_name == "scan_material":
            return self._build_scan_material_outcomes(action)

        elif action_name == "probe_hardness":
            return self._build_probe_hardness_outcomes(action)

        elif action_name == "compress_item":
            return self._build_compress_item_outcomes(action)

        elif "place" in action_name:
            return self._build_place_outcomes(action)

        else:
            return [
                self._make_outcome(
                    add_facts=action.add_effects,
                    del_facts=action.del_effects,
                    probability=1.0
                )
            ]


    def _build_detect_outcomes(action):
        pass

    def _build_pick_outcomes(action):
        pass

    def _build_scan_material_outcomes(action):
        pass

    def _build_probe_hardness_outcomes(action):
        pass

    def _build_compress_item_outcomes(action):
        pass

    def _build_place_outcomes(action):
        pass





