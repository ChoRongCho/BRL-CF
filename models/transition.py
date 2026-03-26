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
        
        # for key, value in self.transition_table.items():
        #     print(key)
        #     for trans_outcome in value:
        #         print("    ", trans_outcome)

    def sample_next_state(self, state: State, action: Action) -> State:
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
    
    def pretty_print(self, sort_keys: bool = True):
        print("\n========== Transition Table ==========\n")

        action_names = list(self.transition_table.keys())
        if sort_keys:
            action_names.sort()

        for action_name in action_names:
            outcomes = self.transition_table[action_name]
            
            # Changmin add
            if "detect" in action_name or "scan" in action_name:
                print(f"[ACTION] {action_name}")
                print(f"  #outcomes: {len(outcomes)}")

                total_prob = sum(o.probability for o in outcomes)

                for i, o in enumerate(outcomes):
                    print(f"    ({i}) p = {o.probability:.4f}")

                    if o.add_facts:
                        print(f"        + add:")
                        for f in o.add_facts:
                            print(f"            {f}")

                    if o.del_facts:
                        print(f"        - del:")
                        for f in o.del_facts:
                            print(f"            {f}")

                # 확률 sanity check
                if abs(total_prob - 1.0) > 1e-6:
                    print("  [WARNING] probabilities do not sum to 1.0")

                print("-" * 50)

        print("\n======================================\n")