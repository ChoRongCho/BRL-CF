from __future__ import annotations

from collections import defaultdict
from utils.asp import DomainRuleBridge
from typing import Any, Dict, Tuple
import copy
import yaml


class Environment:
    def __init__(self, args):
        """
        Docstring for __init__
        
        :param self: Description
        :param domain_rule_path: Description
        :type domain_rule_path: str
        :param initial_state_path: Description
        :type initial_state_path: str
        """
        self.args = args
        self.domain_name = self.args.domain
        self.domain_rule_path = self.args.domain_rule
        self.initial_state_path = self.args.initial_state
        self.robot_skill_path = self.args.robot_skill
        
        self.asp_bridge = DomainRuleBridge(self.domain_rule_path)
        self.asp_bridge.load()
        self.domain_rule = self.asp_bridge.build_program()
        self.initial_state = self._load_config(self.initial_state_path).get("facts", []) or []
        
        # self.domain_rule = self.asp_bridge.build_program()
        self.asp_bridge.add_runtime_facts(self.initial_state)
        self.asp_program = self.asp_bridge.build_program()

        # reset
        self.state = copy.deepcopy(self.initial_state)       
        self.done = False
        self.step_count = 0

    @staticmethod
    def _load_config(yaml_path) -> Dict:
        """YAML 파일 로드"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            contents = yaml.safe_load(f) or {}
        return contents
    
    
    # =========================== Print Method ===========================
    def print_domain_rule(self): 
        print(f"======= Rule: {self.domain_name} ========")
        print(self.domain_rule)
        
    def print_initial_state(self): 
        print(f"======= Init: {self.domain_name} ========")
        groups = defaultdict(list)
        for fact in self.initial_state:

            if isinstance(fact, str):
                fact_str = fact
                pred = fact.split("(")[0]

            elif isinstance(fact, dict):
                fact_str = fact.get("fact", str(fact))
                pred = fact_str.split("(")[0]

            else:
                fact_str = str(fact)
                pred = fact_str

            groups[pred].append(fact_str)

        for pred in sorted(groups.keys()):
            print(f"\n[{pred}]")
            for f in sorted(groups[pred]):
                print(f"  {f}")
    # =========================== Print Method ===========================

    

    def reset(self) -> Dict[str, Any]:
        """
        환경 초기화.
        내부 hidden state를 초기 상태로 복원하고,
        초기 observation을 반환.
        """
        self.state = copy.deepcopy(self.initial_state)
        self.done = False
        self.step_count = 0

        observation = self._get_observation()
        return observation

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        action을 받아 내부 state를 transition 시키고,
        observation, reward, done, info 반환.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        self._apply_action(action)
        reward = self._compute_reward(action)
        self.done = self._check_done()
        self.step_count += 1

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, self.done, info

    def _apply_action(self, action: Dict[str, Any]) -> None:
        """
        내부 hidden state transition.
        실제 domain logic는 여기서 구현.
        """
        action_type = action.get("type")

        if action_type == "move":
            robot = action["robot"]
            target = action["to"]
            self.state["robot_state"]["location"][robot] = target

        elif action_type == "pick":
            robot = action["robot"]
            obj = action["object"]
            self.state["robot_state"]["holding"][robot] = obj

        elif action_type == "drop":
            robot = action["robot"]
            self.state["robot_state"]["holding"][robot] = None

        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _get_observation(self) -> Dict[str, Any]:
        """
        hidden state 전체를 주지 말고,
        관측 가능한 일부만 잘라서 반환.
        """
        obs = {
            "robot_location": copy.deepcopy(self.state.get("robot_state", {}).get("location", {})),
            "visible_objects": copy.deepcopy(self.state.get("visible_objects", [])),
        }
        return obs

    def _compute_reward(self, action: Dict[str, Any]) -> float:
        """
        reward 계산.
        우선은 placeholder.
        """
        return -1.0

    def _check_done(self) -> bool:
        """
        종료 조건 검사.
        우선은 placeholder.
        """
        max_steps = self.domain_rule.get("max_steps", 50)
        return self.step_count >= max_steps

    def _get_info(self) -> Dict[str, Any]:
        """
        디버깅용 부가 정보.
        planner에는 안 써도 됨.
        """
        return {
            "step_count": self.step_count,
        }

    def render(self) -> None:
        """디버깅용 현재 상태 출력"""
        print("=== Env State ===")
        print(self.state)