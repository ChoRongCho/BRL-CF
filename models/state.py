from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
from utils.asp import PossibleWorld


@dataclass(slots=True)
class State:
    facts: List[str] = field(default_factory=list)

    def merge_state(self, other: State) -> State:
        for f in other.facts:
            self.add_fact(f)
        return self
        
    def get_size(self) -> int:
        return len(self.facts)

    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    def add_fact(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def remove_fact(self, fact: str) -> None:
        if fact in self.facts:
            self.facts.remove(fact)

    def copy(self) -> State:
        return State(facts=self.facts.copy())
    
    def clean_all(self) -> None:
        for fact in self.facts:
            self.remove_fact(fact)

    def convert_world_to_state(self, world: PossibleWorld) -> State:
        self.clean_all()
        self.facts = world.atoms


  


def get_types(init_config: Dict) -> Dict:
    obj_type = init_config.get("type", {}) or {}
    return obj_type



def get_state(init_config: Dict) -> List[State]:
    state, hidden_state, goal = State(), State(), State()
    
    i_state = init_config.get("facts", []) or []
    t_state = init_config.get("true_init", []) or []
    g_state = init_config.get("goal", []) or []
    
    for i_s in i_state:
        state.add_fact(i_s.replace(" ", ""))
    for t_s in t_state:
        hidden_state.add_fact(t_s.replace(" ", ""))
    for g_s in g_state:
        goal.add_fact(g_s.replace(" ", ""))

    return state, hidden_state, goal


def world_2_state(world: PossibleWorld):
    new_state = State()
    new_state.facts = world.atoms
    return new_state