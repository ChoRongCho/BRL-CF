# models/transition.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
import random
import re

from models.state import State
from models.action import Action


@dataclass(slots=True)
class TransitionOutcome:
    add_facts: List[str]
    del_facts: List[str]
    probability: float
    fluent_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class NextStateOutcome:
    next_state: State
    probability: float


class TransitionModel:
    def __init__(self, domain: str, 
                 actions: List[Action], 
                 obj_type: Dict[str, List[str]]):
        """
        """
        self.domain = domain
        self.actions = actions
        self.obj_type = obj_type

        self.type_map = self._build_type_map()
        self.trans_model = None
        self.transition_table: Dict[str, List[TransitionOutcome]] = {}
        self.load_transition()

    
    # ==========================================================================
    def _build_type_map(self) -> Dict[str, List[str]]:
        type_map = {}
        for type_declare, objects in self.obj_type.items():
            m = re.match(r".*\(([A-Z])\)", type_declare)
            if m:
                type_symbol = m.group(1)
                type_map.setdefault(type_symbol, [])
                for obj in objects:
                    if obj not in type_map[type_symbol]:
                        type_map[type_symbol].append(obj)
        return type_map

    def _apply_outcome(self, state: State, outcome: TransitionOutcome) -> State:
        next_state = state.copy()

        for fact in outcome.del_facts:
            next_state.remove_fact(fact)

        for fact in outcome.add_facts:
            located_match = re.fullmatch(r"located\(([^,]+),([^)]+)\)", fact)
            if located_match:
                robot = located_match.group(1)
                next_state.set_facts([
                    existing
                    for existing in next_state.facts
                    if not existing.startswith(f"located({robot},")
                ])
            next_state.add_fact(fact)

        for obj, values in outcome.fluent_effects.items():
            for key, value in values.items():
                next_state.set_fluent(obj, key, value)

        return next_state

    @staticmethod
    def _state_key(state: State):
        facts_key = tuple(sorted(str(fact) for fact in state.facts))
        fluents_key = tuple(
            sorted(
                (obj, key, float(value))
                for obj, values in state.fluents.items()
                for key, value in values.items()
            )
        )
        return facts_key, fluents_key

    def _merge_duplicate_next_states(
        self,
        action: Action,
        result: List[NextStateOutcome],
    ) -> List[NextStateOutcome]:
        state_groups = {}
        for outcome in result:
            key = self._state_key(outcome.next_state)
            state_groups.setdefault(key, []).append(outcome)

        merged_result = []
        for outcomes in state_groups.values():
            if len(outcomes) == 1:
                merged_result.append(outcomes[0])
                continue

            merged_result.append(
                NextStateOutcome(
                    next_state=outcomes[0].next_state,
                    probability=sum(outcome.probability for outcome in outcomes),
                )
            )

        if len(merged_result) == len(result):
            return result

        return merged_result
    # ==========================================================================

    def load_transition(self):
        """
        Docstring for load_transition
        
        :param self: Description
        :param state: Description
        :type state: State
        """
        for a in self.actions:
            a_name = a.name.split("(")[0]

            if self.domain == "tomato":
                from models.tomato.trans import TransitionTomato
                self.trans_model = TransitionTomato(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            elif self.domain == "blocksworld":
                from models.blocksworld.trans import TransitionBlocksworld
                self.trans_model = TransitionBlocksworld(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            elif self.domain == "wastesorting":
                from models.wastesorting.trans import TransitionWastesorting
                self.trans_model = TransitionWastesorting(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            elif self.domain == "kitchen":
                from models.kitchen.trans import TransitionKitchen
                self.trans_model = TransitionKitchen(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            elif self.domain == "rover":
                from models.rover.trans import TransitionRover
                self.trans_model = TransitionRover(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            elif self.domain == "watering":
                from models.watering.trans import TransitionWatering
                self.trans_model = TransitionWatering(type_map=self.type_map)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)
            else:
                raise ValueError("Domain is wrong")
        


    def sample_next_state(self, state: State, action: Action) -> State:
        """
        """        
        outcomes = self.transition_table[action.name]

        # Exeception Handler
        outcomes = self.trans_model.handle_exeception(state, action, outcomes)
        
        r = random.random()
        cum = 0.0

        for outcome in outcomes:
            cum += outcome.probability
            if r <= cum:
                return self._apply_outcome(state, outcome)

        return self._apply_outcome(state, outcomes[-1])



    def get_next_state_distribution(self, state: State, action: Action) -> List[NextStateOutcome]:
        """
        """
        outcomes = self.transition_table[action.name]
        outcomes = self.trans_model.handle_exeception(state, action, outcomes)
        result = []

        for outcome in outcomes:
            next_state = self._apply_outcome(state, outcome)
            result.append(
                NextStateOutcome(
                    next_state=next_state,
                    probability=outcome.probability
                )
            )

        return self._merge_duplicate_next_states(action, result)
    
    
