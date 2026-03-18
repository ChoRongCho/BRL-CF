from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from itertools import product
import random
import re

from models.state import State


@dataclass
class ActionSchema:
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    add_effects: List[str] = field(default_factory=list)
    del_effects: List[str] = field(default_factory=list)
    observation: List[str] = field(default_factory=list)
    cost: float = 1.0
    action_type: str = "task"

    def infer_parameter_types(self) -> Dict[str, str]:
        """
        robot(R), tomato(T), stem(S), location(L) 같은 unary type predicate를 보고
        parameter의 타입을 추론한다.
        """
        type_preds = {"robot", "tomato", "stem", "location"}
        param_types: Dict[str, str] = {}

        all_atoms: List[str] = []
        all_atoms.extend(self.preconditions or [])
        all_atoms.extend(self.add_effects or [])
        all_atoms.extend(self.del_effects or [])
        all_atoms.extend(self.observation or [])

        for atom in all_atoms:
            pred, args = self._parse_fact(atom)
            if pred in type_preds and len(args) == 1:
                var = args[0]
                if var in self.parameters:
                    param_types[var] = pred

        return param_types

    @staticmethod
    def _parse_fact(fact: str) -> Tuple[str, List[str]]:
        fact = fact.strip().rstrip(".")
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", fact)
        if not m:
            return fact, []
        pred = m.group(1)
        arg_str = m.group(2).strip()
        args = [x.strip() for x in arg_str.split(",")] if arg_str else []
        return pred, args

    @staticmethod
    def _substitute(atom: str, binding: Dict[str, str]) -> str:
        result = atom
        for var, obj in binding.items():
            result = re.sub(rf"\b{re.escape(var)}\b", obj, result)
        return result

    def ground(self, binding: Dict[str, str]) -> "Action":
        return Action(
            name=f"{self.name}({', '.join(binding[p] for p in self.parameters)})",
            preconditions=[
                normalize_fact(self._substitute(a, binding))
                for a in (self.preconditions or [])
                if a and a != "null"
            ],
            add_effects=[
                normalize_fact(self._substitute(a, binding))
                for a in (self.add_effects or [])
                if a and a != "null"
            ],
            del_effects=[
                normalize_fact(self._substitute(a, binding))
                for a in (self.del_effects or [])
                if a and a != "null"
            ],
            observation=[
                normalize_fact(self._substitute(a, binding))
                for a in (self.observation or [])
                if a and a != "null"
            ],
            cost=self.cost,
        )


@dataclass(slots=True)
class Action:
    name: str
    preconditions: List[str] = field(default_factory=list)
    add_effects: List[str] = field(default_factory=list)
    del_effects: List[str] = field(default_factory=list)
    observation: List[str] = field(default_factory=list)
    cost: float = 1.0

    def is_applicable(self, state: State) -> bool:
        return all(state.has_fact(normalize_fact(pre)) for pre in self.preconditions)

    def apply_action(self, state: State, is_stochastic: bool = False) -> State:
        if not self.is_applicable(state):
            raise ValueError(f"Action '{self.name}' is not applicable.")

        next_state = state.copy()

        if is_stochastic and random.random() < 0.1:
            return next_state

        for fact in self.del_effects:
            next_state.remove_fact(normalize_fact(fact))

        for fact in self.add_effects:
            next_state.add_fact(normalize_fact(fact))

        return next_state
    

class GroundedActionSet:
    def __init__(self, action_schemas: List[ActionSchema], init_state: State):
        self.action_schemas = action_schemas
        self.initial_facts = init_state
        self.objects = self._collect_objects(init_state)
        self.actions: List[Action] = []

    def _parse_fact(self, fact: str) -> Tuple[str, List[str]]:
        fact = fact.strip().rstrip(".")
        m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", fact)
        if not m:
            return fact, []
        pred = m.group(1)
        arg_str = m.group(2).strip()
        args = [x.strip() for x in arg_str.split(",")] if arg_str else []
        return pred, args

    def _collect_objects(self, init_state: State) -> Dict[str, Set[str]]:
        """
        초기 상태의 unary typing fact들로부터 고정 object universe를 만든다.
        그리고 stem(S) -> location(S) rule도 반영한다.
        """
        objects: Dict[str, Set[str]] = {
            "robot": set(),
            "tomato": set(),
            "stem": set(),
            "location": set(),
        }

        for fact in init_state.facts:
            pred, args = self._parse_fact(fact)
            if pred in objects and len(args) == 1:
                objects[pred].add(args[0])

        objects["location"].update(objects["stem"])
        return objects


    def _is_valid_binding(self, schema: ActionSchema, binding: Dict[str, str]) -> bool:
        """
        간단한 grounding pruning 규칙.
        """
        if schema.name == "navigate":
            if "L1" in binding and "L2" in binding and binding["L1"] == binding["L2"]:
                return False
        return True
    
    def generate_action_set(self) -> List[Action]:
        grounded_actions: List[Action] = []

        for schema in self.action_schemas:
            param_types = schema.infer_parameter_types()

            domains: List[List[str]] = []
            for param in schema.parameters:
                if param not in param_types:
                    raise ValueError(
                        f"[{schema.name}] parameter '{param}' type could not be inferred."
                    )
                obj_type = param_types[param]
                domains.append(sorted(self.objects[obj_type]))

            for values in product(*domains):
                binding = dict(zip(schema.parameters, values))

                if not self._is_valid_binding(schema, binding):
                    continue

                grounded_action = schema.ground(binding)
                grounded_actions.append(grounded_action)

        self.actions = grounded_actions
        return grounded_actions
    


    
def normalize_fact(fact: str) -> str:
    fact = fact.strip().rstrip(".")

    m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$", fact)
    if not m:
        return fact

    pred = m.group(1)
    args = m.group(2)

    args = [a.strip() for a in args.split(",")]
    return f"{pred}({','.join(args)})"



def get_actions(action_dicts, state):
    action_schema = [
        ActionSchema(
            name=a["name"],
            parameters=a.get("parameters", []),
            preconditions=a.get("preconditions", []) or [],
            add_effects=a.get("add_effects", []) or [],
            del_effects=[x for x in (a.get("del_effects", []) or []) if x and x != "null"],
            observation=a.get("observation", []) or [],
            cost=float(a.get("cost", 1.0)),
            action_type=a.get("type", "task"),
        )
        for a in action_dicts
    ]
    grounded_action_set = GroundedActionSet(
        action_schemas=action_schema,
        init_state=state,
    )
    actions = grounded_action_set.generate_action_set()
    return actions