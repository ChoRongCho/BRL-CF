from __future__ import annotations

from collections import defaultdict
from utils.asp import DomainRuleBridge, solve_asp
from typing import Any, Dict, Tuple, List
import copy
import yaml

from models.state import State, get_state, get_types
from models.action import Action, Grounding, ActionSchema, get_actions
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel
from models.reward import RewardModel


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
        self.max_step = self.args.max_step
        
        # Domain rule: initially imported to the DomainRuleBridge
        self.asp_bridge = DomainRuleBridge()
        self.asp_bridge.load(self.domain_rule_path)
        self.domain_rule = self.asp_bridge.build_possible_worlds()
        
        # Get (initial) state, hidden_init_state, and goal state
        init_config = self._load_config(self.initial_state_path) 
        self.state, self.gt_init_state, self.goal = get_state(init_config) # a list of facts
        self.exec_init_state = self.state.copy()
        self.true_state = self.gt_init_state.copy()
        self.obj_type = get_types(init_config)
        
        # generate possible worlds and clear runtime
        _ = self.build_state(runtime_facts=self.state.facts, build_type="certain") 

        # Get all grounded actions
        action_dicts = self._load_config(self.robot_skill_path).get("actions", []) or []
        self.actions = get_actions(action_dicts, self.state, self.obj_type)
                
        # Transition Model
        self.transition_model = TransitionModel(
            domain=self.domain_name,
            actions=self.actions,
            obj_type=self.obj_type,
            true_state=self.true_state
        )
        
        # Observation Model
        self.observation_model = ObservationModel(
            domain=self.domain_name,
            actions=self.actions,
            obj_type=self.obj_type,
            noise=0.05,
            true_state=self.true_state,
        )
        
        # Reward Model TODO
        self.reward_model = RewardModel(
            self.domain_name,
            self.goal,
        )
        
        # reset
        self.done = False
        self.step_count = 0


    @staticmethod
    def _load_config(yaml_path) -> Dict:
        """YAML 파일 로드"""
        with open(yaml_path, "r", encoding="utf-8") as f:
            contents = yaml.safe_load(f) or {}
        return contents
    
    def get_asp_bridge(self) -> DomainRuleBridge:
        return self.asp_bridge
    
    def build_state(self, runtime_facts: List[str], build_type: str ="certain"):
        """
        Update state using domain rule given 'self.domain_rule'
        
        :param runtime_facts: Description
        :type runtime_facts: List[str]
        :param build_type: Description
        :type build_type: str
        """
        self.asp_bridge.add_runtime_facts(runtime_facts)
        
        if build_type == "possible":
            program = self.asp_bridge.build_possible_worlds()
                        
        elif build_type == "certain":
            program = self.asp_bridge.build_certain_worlds()
            worlds = solve_asp(program)
            if not len(worlds)==1:
                raise SystemError(f"The domain rule of {self.domain_name} went wrong.. ")
            self.state.convert_world_to_state(worlds[0])
            
        else: 
            raise ValueError("The build type must be one of 'possible' or 'certain'(default)..")
        
        # reset runtime after building program
        self.asp_bridge.clear_runtime()
        
        return program
            
        
    
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
    # ====================================================================

    def reset(self) -> Dict[str, Any]:
        """
        환경 초기화.
        내부 hidden state를 초기 상태로 복원하고,
        초기 observation을 반환.
        """
        self.done = False
        self.step_count = 0
        self.state = self.exec_init_state.copy()
        self.true_state = self.gt_init_state.copy()
        self._sync_models_with_state()

        observation = copy.deepcopy(self.state)
        return observation


    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        action을 받아 내부 state를 transition 시키고,
        observation, reward, done, info 반환.
        """
        
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        prev_state = copy.deepcopy(self.state)
        self._apply_action(action)
        # _ = self.build_state(runtime_facts=self.state.facts, build_type="certain")
        
        reward = self.reward_model.get_reward(prev_state, action, self.state)
        self.step_count += 1
        self.done = self._check_done()
        
        # get observation        
        if action is None:
            observation =  copy.deepcopy(self.state)
        observation = self.observation_model.sample(self.state, action)
        
        info = self._get_info()
        
        # self.transition_model.load_transition(state=self.state)

        return observation, reward, self.done, info


    def _apply_action(self, action: Dict[str, Any]) -> None:        
        self.state = self.transition_model.sample_next_state(self.state, action)
        self._update_true_state_from_execution(action)
        self._sync_models_with_state()

    def _update_true_state_from_execution(self, action: Action) -> None:
        action_name = action.name.split("(")[0]
        if action_name not in {"navigate", "prepare_nav", "pick", "place", "discard"}:
            return

        dynamic_prefixes = (
            "at(",
            "located(",
            "handempty(",
            "holding(",
            "holded(",
            "loaded(",
            "discarded(",
            "navprepared(",
        )

        self.true_state.facts = [
            fact for fact in self.true_state.facts
            if not fact.startswith(dynamic_prefixes)
        ]
        for fact in self.state.facts:
            if fact.startswith(dynamic_prefixes):
                self.true_state.add_fact(fact)

        for obj, values in self.state.fluents.items():
            for key, value in values.items():
                if float(value) != -1.0:
                    self.true_state.set_fluent(obj, key, value)

    def _sync_models_with_state(self) -> None:
        """
        Keep transition/observation models aligned with the maintained hidden true state.
        """
        self.transition_model.true_state = self.true_state
        self.transition_model.load_transition(state=self.true_state)

        self.observation_model.true_state = self.true_state
        if hasattr(self.observation_model, "domain_model"):
            self.observation_model.domain_model.true_state = self.true_state


    def _check_done(self) -> bool:
        """Goal 달성 또는 max_step 초과 시 episode 종료"""
        # 1. goal 달성 여부
        if self.goal:
            if all(self.state.has_fact(f) for f in self.goal.facts):
                print("[Planner] Goal Done")
                return True
            
        # 2. step limit
        if self.step_count >= self.max_step:
            print("[Planner] MAX STEP Done")
            return True



        return False


    def _get_info(self) -> Dict[str, Any]:
        applicable_actions = [
            action.name for action in self.actions if action.is_applicable(self.state)
        ]

        return {
            "step_count": self.step_count,
            "current_state_size": self.state.get_size(),
            "applicable_actions": applicable_actions,
        }

    def render(self) -> None:
        print("=== Env State ===")
        for f in self.state.facts:
            print("  ", f)
