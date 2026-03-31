# models/transition.py

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(slots=True)
class NextStateOutcome:
    next_state: State
    probability: float


class TransitionModel:
    """
    T(s' | s, a)
    """

    def __init__(
        self,
        domain: str,
        actions: List[Action],
        obj_type: Dict[str, List[str]],
        true_state: State
    ):
        self.domain = domain
        self.actions = actions
        self.obj_type = obj_type
        self.true_state = true_state

        self.type_map = self._build_type_map()
        self.trans_model = None
        self.transition_table: Dict[str, List[TransitionOutcome]] = {}
        self.load_transition(state=self.true_state)

    def _build_type_map(self) -> Dict[str, List[str]]:
        type_map = {}
        for type_declare, objects in self.obj_type.items():
            m = re.match(r".*\(([A-Z])\)", type_declare)
            if m:
                type_symbol = m.group(1)
                type_map[type_symbol] = objects
        return type_map

    def _apply_outcome(self, state: State, outcome: TransitionOutcome) -> State:
        next_state = state.copy()

        for fact in outcome.del_facts:
            next_state.remove_fact(fact)

        for fact in outcome.add_facts:
            next_state.add_fact(fact)

        return next_state

    def load_transition(self, state: State):
        for a in self.actions:

            a_name = a.name.split("(")[0]

            if self.domain == "tomato":
                from models.tomato.trans import TransitionTomato
                self.trans_model = TransitionTomato(type_map=self.type_map, true_state=state)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)

            elif self.domain == "blocksworld":
                from models.blocksworld.trans import TransitionBlocksworld
                self.trans_model = TransitionBlocksworld(type_map=self.type_map, true_state=state)
                self.transition_table[a.name] = self.trans_model.build_outcomes(a_name, a)

            elif self.domain == "wastesorting":
                from models.wastesorting.trans import TransitionWastesorting
                self.trans_model = TransitionWastesorting(type_map=self.type_map, true_state=state)
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
        result = []

        for outcome in outcomes:
            next_state = self._apply_outcome(state, outcome)
            result.append(
                NextStateOutcome(
                    next_state=next_state,
                    probability=outcome.probability
                )
            )

        return result
    
    