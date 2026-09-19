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
from models.belief import Belief
from shared.env_setting import load_env_setting

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
        self.env_setting_path = self.args.env_setting
        self.max_step = self.args.max_step
        self.env_setting = load_env_setting(self.env_setting_path)
        configured_domain = self.env_setting.get("domain")
        if configured_domain not in {None, self.domain_name}:
            raise ValueError(
                f"Environment setting domain {configured_domain!r} does not match "
                f"requested domain {self.domain_name!r}"
            )
        
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

        # for f in self.state.facts:
        #     print("[STATE] ", f)
        # for a in self.actions:
        #     print("[ACTION] ", a)
        # asdf
        
        # Transition Model
        self.transition_model = TransitionModel(
            domain=self.domain_name,
            actions=self.actions,
            obj_type=self.obj_type,
            true_state=self.true_state,
            settings=self.env_setting.get("transition", {}),
        )
        
        # Observation Model
        self.observation_model = ObservationModel(
            domain=self.domain_name,
            actions=self.actions,
            obj_type=self.obj_type,
            noise=1.0 - float(self.env_setting["observation"].get("default_success", 0.95)),
            true_state=self.true_state,
            settings=self.env_setting.get("observation", {}),
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
        
        # Execution rewards must describe what happened in the hidden physical
        # world.  The runtime state may contain a transition hypothesis (for
        # example, scan samples fresh/rotten independently for planning), while
        # observations and success/failure are grounded in true_state.
        prev_true_state = self.true_state.copy()
        self._apply_action(action)
        # _ = self.build_state(runtime_facts=self.state.facts, build_type="certain")
        
        reward = self.reward_model.get_reward(
            prev_true_state,
            action,
            self.true_state,
        )
        self.step_count += 1
        
        # get observation        
        if action is None:
            observation = copy.deepcopy(self.state)
        observation = self.observation_model.sample(self.state, action)
        
        info = self._get_info()
        
        # self.transition_model.load_transition(state=self.state)

        return observation, reward, self.done, info


    def _apply_action(self, action: Dict[str, Any]) -> None:
        # Planning applicability is evaluated against the robot's belief, but
        # execution feasibility must be evaluated against the hidden physical
        # state.  Otherwise a belief-compatible pick at the wrong stem can be
        # sampled as a success and copied into true_state.
        self.last_execution_applicable = self._is_physically_executable(action)
        if not self.last_execution_applicable:
            return
        self.state = self.transition_model.sample_next_state(self.state, action)
        self._update_true_state_from_execution(action)
        self._sync_models_with_state()

    def _is_physically_executable(self, action: Action) -> bool:
        """Check only physical preconditions against the executed world.

        Semantic labels such as ripe/fresh or a waste category are deliberately
        excluded: the robot may execute a physically possible but task-wrong
        action and should then receive PLAN FAILURE.  Location, gripper state,
        and occlusion determine whether the physical action can succeed.
        """
        normalized = action.name.replace(" ", "")
        action_name, _, raw_args = normalized.rstrip(")").partition("(")
        args = raw_args.split(",") if raw_args else []
        runtime = self.state
        hidden = self.true_state

        if self.domain_name == "tomato":
            if action_name == "navigate" and len(args) >= 3:
                robot, source, _ = args[:3]
                return (
                    runtime.has_fact(f"located({robot},{source})")
                    and runtime.has_fact(f"handempty({robot})")
                )
            if action_name == "prepare_nav":
                return True
            if action_name == "detect" and len(args) >= 2:
                robot, stem = args[:2]
                return (
                    runtime.has_fact(f"located({robot},{stem})")
                    and runtime.has_fact(f"handempty({robot})")
                )
            if action_name in {"pick", "pick_n_scan"} and len(args) >= 3:
                robot, tomato, stem = args[:3]
                return (
                    runtime.has_fact(f"located({robot},{stem})")
                    and runtime.has_fact(f"handempty({robot})")
                    and hidden.has_fact(f"at({tomato},{stem})")
                )
            if action_name in {"scan", "place", "discard"} and len(args) >= 2:
                robot, tomato = args[:2]
                return (
                    runtime.has_fact(f"holding({robot},{tomato})")
                    or runtime.has_fact(f"holded({tomato},{robot})")
                )
            return True

        if self.domain_name == "wastesorting":
            if action_name == "detect_waste" and args:
                return runtime.has_fact(f"handempty({args[0]})")
            if action_name == "pick" and len(args) >= 2:
                robot, waste = args[:2]
                if not runtime.has_fact(f"handempty({robot})"):
                    return False
                if any(
                    fact.startswith(f"in_bin({waste},")
                    or fact.endswith(f",{waste})") and fact.startswith("holding(")
                    for fact in runtime.facts
                ):
                    return False
                for fact in hidden.facts:
                    if not fact.startswith("occ("):
                        continue
                    top, bottom = fact[4:-1].split(",", 1)
                    if bottom != waste:
                        continue
                    top_cleared = any(
                        current.startswith(f"in_bin({top},")
                        or (
                            current.startswith("holding(")
                            and current.endswith(f",{top})")
                        )
                        for current in runtime.facts
                    )
                    if not top_cleared:
                        return False
                return True
            if action_name.startswith("place_") and len(args) >= 2:
                robot, waste = args[:2]
                return runtime.has_fact(f"holding({robot},{waste})")
            return True

        return True

    def _update_true_state_from_execution(self, action: Action) -> None:
        action_name = action.name.split("(")[0]
        state_changing_actions = {
            "navigate",
            "prepare_nav",
            "pick",
            "pick_n_scan",
            "place",
            "discard",
            "place_gw_bin",
            "place_paper_bin",
            "place_can_bin",
            "place_plastic_bin",
        }

        # Reward-history fluents also advance on observation actions such as
        # detect and scan, even though those actions do not change physical
        # object facts.
        for obj, values in self.state.fluents.items():
            for key, value in values.items():
                if float(value) != -1.0:
                    self.true_state.set_fluent(obj, key, value)

        if action_name not in state_changing_actions:
            return

        dynamic_prefixes = (
            "located(",
            "handempty(",
            "holding(",
            "holded(",
            "loaded(",
            "discarded(",
            "in_bin(",
            "navprepared(",
        )

        self.true_state.set_facts([
            fact for fact in self.true_state.facts
            if not fact.startswith(dynamic_prefixes)
        ])
        for fact in self.state.facts:
            if fact.startswith(dynamic_prefixes):
                self.true_state.add_fact(fact)

        if action_name in {"pick", "pick_n_scan"}:
            _, args = action.name.replace(" ", "").split("(", 1)
            action_args = args.rstrip(")").split(",")
            if len(action_args) >= 3:
                _, tomato, stem = action_args[:3]
                robot = action_args[0]
                pick_succeeded = (
                    self.true_state.has_fact(f"holding({robot},{tomato})")
                    or self.true_state.has_fact(f"holded({tomato},{robot})")
                )
                if pick_succeeded:
                    self.true_state.remove_fact(f"at({tomato},{stem})")

    def _sync_models_with_state(self) -> None:
        """
        Keep transition/observation models aligned with the maintained hidden true state.
        """
        self.transition_model.true_state = self.true_state
        self.transition_model.load_transition(state=self.true_state)

        self.observation_model.true_state = self.true_state
        if hasattr(self.observation_model, "domain_model"):
            self.observation_model.domain_model.true_state = self.true_state

    def check_done(self, belief: Belief):
        """
        Return the terminal result for the current episode.

        Tomato failure conditions are evaluated from true_state because they
        describe physical execution outcomes, not the robot's current belief:
        - a tomato was marked as picked but still exists at a stem,
        - an unripe tomato was picked,
        - a rotten tomato was loaded,
        - a fresh tomato was discarded.

        The Tomato goal mixes physical facts with epistemic facts. Physical
        facts are checked in true_state, while observed/scanned facts are
        checked in belief.knowledge.
        """
        def parse_fact(raw_fact):
            fact = raw_fact.replace(" ", "")
            if not fact.endswith(")"):
                return None, ()

            predicate, sep, args = fact[:-1].partition("(")
            if not sep:
                return None, ()

            return predicate, tuple(args.split(","))

        if self.domain_name == "tomato":
            """
            true_init:
                - "ripe(tomato1)"
                - "rotten(tomato2)"
                - "ripe(tomato3)"
                - "unripe(tomato4)"
                - "at(tomato1, stem_01)"
                - "at(tomato2, stem_01)"
                - "at(tomato3, stem_02)"
                - "at(tomato4, stem_02)"
            """
            true_unripe = set()
            true_rotten = set()
            true_fresh = set()
            at_stem = set()
            picked = set()
            loaded = set()
            discarded = set()

            for raw_fact in self.true_state.facts:
                predicate, args = parse_fact(raw_fact)
                if not args:
                    continue

                if predicate == "unripe":
                    true_unripe.add(args[0])
                elif predicate == "rotten":
                    true_rotten.add(args[0])
                elif predicate == "fresh":
                    true_fresh.add(args[0])
                elif predicate == "at" and len(args) >= 2:
                    at_stem.add(args[0])
                elif predicate == "holding" and len(args) >= 2:
                    picked.add(args[1])
                elif predicate == "holded":
                    picked.add(args[0])
                elif predicate == "discarded":
                    discarded.add(args[0])
                    picked.add(args[0])
                elif predicate == "loaded":
                    loaded.add(args[0])
                    picked.add(args[0])

            if picked & at_stem:
                return "PLAN FAILURE"
            if picked & true_unripe:
                return "PLAN FAILURE"
            if loaded & true_rotten:
                return "PLAN FAILURE"
            if discarded & true_fresh:
                return "PLAN FAILURE"


        elif self.domain_name == "wastesorting":
            goal_bin_by_waste = {}

            if self.goal:
                for goal_fact in self.goal.facts:
                    predicate, args = parse_fact(goal_fact)
                    if predicate == "in_bin" and len(args) >= 2:
                        goal_bin_by_waste[args[0]] = args[1]

            for raw_fact in belief.knowledge.facts:
                predicate, args = parse_fact(raw_fact)
                if not args:
                    continue

                if predicate != "in_bin" or len(args) < 2:
                    continue

                waste, bin_name = args[:2]
                goal_bin = goal_bin_by_waste.get(waste)
                if goal_bin and bin_name != goal_bin:
                    return "PLAN FAILURE"

        if self.goal:
            if self.domain_name == "tomato":
                epistemic_predicates = {"observed", "scanned"}

                def tomato_goal_satisfied(goal_fact):
                    predicate, _ = parse_fact(goal_fact)
                    if predicate in epistemic_predicates:
                        return belief.knowledge.has_fact(goal_fact)
                    return self.true_state.has_fact(goal_fact)

                if all(tomato_goal_satisfied(fact) for fact in self.goal.facts):
                    return "GOAL DONE"
            elif all(belief.knowledge.has_fact(f) for f in self.goal.facts):
                return "GOAL DONE"

        if self.step_count >= self.max_step:
            return "MAX STEP"

        return False


    def _get_info(self) -> Dict[str, Any]:
        applicable_actions = [
            action.name for action in self.actions if action.is_applicable(self.state)
        ]

        return {
            "step_count": self.step_count,
            "current_state_size": self.state.get_size(),
            "applicable_actions": applicable_actions,
            "execution_applicable": getattr(
                self, "last_execution_applicable", True
            ),
        }

    def render(self) -> None:
        print("=== Env State ===")
        for f in self.state.facts:
            print("  ", f)
