from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import List, Dict
from utils.asp import PossibleWorld


@dataclass(slots=True)
class State:
    """
    Docstring for State
    """
    facts: List[str] = field(default_factory=list)
    fluents: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def merge_state(self, other: State) -> State:
        for f in other.facts:
            self.add_fact(f)
        for obj, values in other.fluents.items():
            self.fluents[obj] = copy.deepcopy(values)
        return self
    
    def replace_state(self, other: State):
        
        self.clean_all()
        self.merge_state(other=other)
        
        return self
        
    def get_size(self) -> int:
        return len(self.facts) + sum(len(values) for values in self.fluents.values())

    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    def add_fact(self, fact: str) -> None:
        if fact not in self.facts:
            self.facts.append(fact)

    def remove_fact(self, fact: str) -> None:
        if fact in self.facts:
            self.facts.remove(fact)

    def copy(self) -> State:
        return State(
            facts=self.facts.copy(),
            fluents=copy.deepcopy(self.fluents),
        )
    
    def clean_all(self) -> None:
        self.facts.clear()
        self.fluents.clear()

    def convert_world_to_state(self, world: PossibleWorld) -> State:
        self.clean_all()
        self.facts = world.atoms

    def set_fluent(self, obj: str, key: str, value: float) -> None:
        if obj not in self.fluents:
            self.fluents[obj] = {}
        self.fluents[obj][key] = float(value)

    def get_fluent(self, obj: str, key: str, default=None):
        return self.fluents.get(obj, {}).get(key, default)



def get_types(init_config: Dict) -> Dict:
    obj_type = init_config.get("type", {}) or {}
    return obj_type



def get_state(init_config: Dict) -> List[State]:
    state, hidden_state, goal = State(), State(), State()
    
    i_state = init_config.get("facts", []) or []
    t_state = init_config.get("true_init", []) or []
    g_state = init_config.get("goal", []) or []
    i_fluents = init_config.get("fluents", {}) or {}
    t_fluents = init_config.get("true_fluents", {}) or {}
    
    for i_s in i_state:
        state.add_fact(i_s.replace(" ", ""))
    for t_s in t_state:
        hidden_state.add_fact(t_s.replace(" ", ""))
    for g_s in g_state:
        goal.add_fact(g_s.replace(" ", ""))

    for obj, values in i_fluents.items():
        for key, value in (values or {}).items():
            state.set_fluent(obj, key, value)

    for obj, values in t_fluents.items():
        for key, value in (values or {}).items():
            hidden_state.set_fluent(obj, key, value)

    return state, hidden_state, goal


def world_2_state(world: PossibleWorld):
    new_state = State()
    new_state.facts = world.atoms
    return new_state
