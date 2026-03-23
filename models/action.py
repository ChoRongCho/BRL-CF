from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from itertools import product
import random
import re

from models.state import State


@dataclass
class ActionSchema:
    """
    robot_skill.yaml을 플래닝에 쓸 수 있도록 바꿔주는 브릿지 포맷
    
    """
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    add_effects: List[str] = field(default_factory=list)
    del_effects: List[str] = field(default_factory=list)
    observation: List[str] = field(default_factory=list)
    cost: float = 1.0
    action_type: str = "task"
    
    

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
    

class Grounding:
    def __init__(self, action_schemas: List[ActionSchema], 
                 init_state: State, 
                 obj_type: Dict):
        """

        """
        self.action_schemas = action_schemas
        self.initial_facts = init_state
        self.obj_type = obj_type
        self.actions: List[Action] = []  

    def _is_valid_binding(self, schema: ActionSchema, 
                          param_type_symbols,
                          assignment) -> bool:
        
        valid = True
        for i in range(len(schema.parameters)):
            for j in range(i+1, len(schema.parameters)):
                if param_type_symbols[i] == param_type_symbols[j]:
                    if assignment[i] == assignment[j]:
                        valid = False
                        break
            if not valid:
                break  
        return valid
    
    
    @staticmethod
    def substitute(binding, expr: str) -> str:
        result = expr
        # 긴 변수명부터 치환해야 L1이 L보다 먼저 바뀜
        for var in sorted(binding.keys(), key=len, reverse=True):
            result = re.sub(rf"\b{re.escape(var)}\b", binding[var], result)
        return result.replace(" ", "")
    
    
    def generate_action_set(self) -> List[Action]:
        # 1. mapper 만들기        
        grounded_actions: List[Action] = []
        type_map = {}  # "robot(R)" -> type_symbol = R, objects=["changmin"]
        for type_declare, objects in self.obj_type.items():
            m = re.match(r".*\(([A-Z])\)", type_declare)
            if m:
                # print(m)
                type_symbol = m.group(1)
                type_map[type_symbol] = objects
        
        
        for schema in self.action_schemas:
            # 2. schema parameter별 domain 만들기
            # 예: ["R", "L1", "L2"] -> [["changmin"], ["dockstation", ...], ["dockstation", ...]]
            param_domains = []
            param_type_symbols = []
            for param in schema.parameters:
                type_symbol = param[0]   # # L1, L2도 첫 글자 L 사용
                if type_symbol not in type_map:
                    raise ValueError(f"{param} 에 해당하는 타입이 self.obj_type에 없음")
                param_domains.append(type_map[type_symbol])   
                param_type_symbols.append(type_symbol)
 

            # 3. 가능한 모든 grounding 조합 생성
            for assignment in product(*param_domains):
                binding = dict(zip(schema.parameters, assignment))
                
                if not self._is_valid_binding(schema, param_type_symbols, assignment):
                    continue
                
                grounded_name = f"{schema.name}(" + ", ".join(binding[p] for p in schema.parameters) + ")"
                grounded_action = Action(
                    name=grounded_name,
                    preconditions=[self.substitute(binding, p) for p in schema.preconditions],
                    add_effects=[self.substitute(binding, a) for a in schema.add_effects],
                    del_effects=[self.substitute(binding, d) for d in schema.del_effects],
                    observation=[self.substitute(binding, o) for o in schema.observation],
                    cost=schema.cost
                )
                
                # print(grounded_name, "|", grounded_action.observation)
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



def get_actions(action_dicts, state: State, obj_type: Dict):
    """
    1. 액션 스키마를 만들고
    2. 초기 상태를 통해 grounded action set을 전부 만든 다음
    3. List[Action] 형식으로 전달
    
    :param action_dicts: 액션 스키마가 적혀있는 yaml 파일을 바로 읽은 결과. env._load_config의 출력
    :param state: 초기 상태를 중심으로 grounded actions를 얻어야 함
    """
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
    
    action_grounder = Grounding(
        action_schemas=action_schema,
        init_state=state,
        obj_type=obj_type
    )
    
    actions = action_grounder.generate_action_set()
    return actions

