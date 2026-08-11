from __future__ import annotations
import numpy as np
import copy
import yaml
from typing import Any, Dict, List, Optional, Iterable, Tuple

from utils.asp import DomainRuleBridge, solve_asp
from models.belief import Belief
from models.state import State
from models.action import Action
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel, TransitionOutcome, NextStateOutcome
from utils.utils import _parse_fact

from collections import defaultdict

class FeedbackManger:
    def __init__(self, args, conf_threshold):
        self.args = args
        self.domain_name = self.args.domain
        self.conf_threshold = conf_threshold
        self.num_of_query = 0
        self.query_log = []
        self.use_llm = False

        if self.args.answer_type == "auto":
            self.is_human_answer = False
        elif self.args.answer_type == "human":
            self.is_human_answer = True
        else:
            raise ValueError(f"Wrong answer type: {self.args.answer_type} | 'auto' or 'human'")
        self._true_init_facts = None
        

    def compute_confidence(self, weights: np.ndarray) -> float:
        if len(weights) == 0 or len(weights) == 1:
            return 1.0

        theta = np.log2(len(weights))
        eps = 1e-5
        h = -np.sum(weights * np.log2(weights + eps))
        
        confidence = 1.0 - (h / theta)
        
        return float(confidence)


    def normalize(self, weights: np.ndarray) -> np.ndarray:
        weights = np.array(weights, dtype=float)
        s = weights.sum()
        if s <= 0:
            return np.ones(len(weights), dtype=float) / len(weights)
        return weights / s


    def entropy(self, weights: np.ndarray) -> float:
        weights = self.normalize(weights)
        eps = 1e-12
        return float(-np.sum(weights * np.log2(weights + eps)))


    def fact_in_state(self, state, fact: str) -> bool:
        return state.has_fact(fact)


    def get_changed_facts(self, knowledge, frontier):
        """
        frontier 내부에서 truth value가 실제로 갈리는 fact 후보를 모은다.
        """
        if not frontier:
            return []

        fact_counts = defaultdict(int)
        for state in frontier:
            for fact in set(state.facts):
                fact_counts[fact] += 1

        num_frontier = len(frontier)
        return sorted(
            fact for fact, count in fact_counts.items()
            if 0 < count < num_frontier
        )


    def expected_entropy_after_asking(self, frontier, weights, fact: str) -> float:
        """
        fact를 질문해서 참/거짓을 알게 된다고 가정할 때의 기대 엔트로피.
        """
        weights = self.normalize(weights)
        
        true_idx = []
        false_idx = []

        for i, state in enumerate(frontier):
            if self.fact_in_state(state, fact):
                true_idx.append(i)
            else:
                false_idx.append(i)

        p_true = weights[true_idx].sum() if len(true_idx) > 0 else 0.0
        p_false = weights[false_idx].sum() if len(false_idx) > 0 else 0.0

        h_true = self.entropy(weights[true_idx]) if p_true > 0 else 0.0
        h_false = self.entropy(weights[false_idx]) if p_false > 0 else 0.0

        return p_true * h_true + p_false * h_false


    def select_best_fact_to_ask(self, belief: Belief):
        """
        frontier의 uncertainty를 가장 많이 줄여줄 fact를 고른다.
        """
        frontier = belief.frontier
        weights = self.normalize(belief.frontier_weights)

        # print("[DEBUG] The length of the frontier: ", len(weights), weights)
        candidate_facts = self.get_changed_facts(belief.knowledge, frontier)
                
        if not candidate_facts:
            return None

        current_entropy = self.entropy(weights)

        best_fact = None
        best_gain = -float("inf")

        for fact in candidate_facts:
            exp_h = self.expected_entropy_after_asking(frontier, weights, fact)
            info_gain = current_entropy - exp_h

            if info_gain > best_gain:
                best_gain = info_gain
                best_fact = fact

        return best_fact


    def apply_fact_answer_to_belief(self, belief: Belief, fact: str, answer: bool):
        """
        질문 결과(answer)에 맞는 frontier만 남긴다.
        answer=True  -> fact가 있는 state만 유지
        answer=False -> fact가 없는 state만 유지
        """
        new_frontier = []
        new_weights = []

        for state, w in zip(belief.frontier, belief.frontier_weights):
            has_fact = self.fact_in_state(state, fact)


            if answer is True and has_fact:
                new_frontier.append(state)
                new_weights.append(w)
            
                
            elif answer is False and not has_fact:
                new_frontier.append(state)
                new_weights.append(w)
                

        if len(new_frontier) == 0:
            return belief

        
        
        belief.frontier = new_frontier
        belief.frontier_weights = self.normalize(np.array(new_weights, dtype=float))
        
        
        return belief
    
    
    def get_new_observation(self, 
                            belief: Belief, 
                            step: int = None, 
                            action_name: str = None,
                            observation_facts=None,
                            oracle_state_facts=None) -> Belief:
        """
        
        """
        
        # ================ compute confidence and human ask ================
        confidence = self.compute_confidence(belief.frontier_weights)
        while confidence < self.conf_threshold:
            original_belief = copy.deepcopy(belief)
            print(f"    [Planner] Current confidence: {confidence}")
            
            # 더 이상 질문할 fact가 없으면 종료
            target_fact = self.select_best_fact_to_ask(belief)
            if target_fact is None:
                break
            
            # 질문
            answer = self.call_feedback(
                target_fact,
                action_name,
                observation_facts=observation_facts,
                oracle_state_facts=oracle_state_facts,
            )
            
            self.num_of_query += 1
            print(f"    [Query] A: {answer}")
            
            # 적용 후 다시 계산            
            belief = self.apply_fact_answer_to_belief(belief, target_fact, answer)
            updated_confidence = self.compute_confidence(belief.frontier_weights)
            self.query_log.append({
                "step": step,
                "action": action_name,
                "observation": list(observation_facts or []),
                "question": target_fact,
                "answer": answer,
                "confidence_before": confidence,
                "confidence_after": updated_confidence,
            })
            confidence = updated_confidence
            print(f"    [Planner] Redeuced {len(original_belief.frontier)} --> {len(belief.frontier)}")    
            
            
        print(f"    [Planner] Updated confidence: {confidence}")    
        # expand knowledge
        if len(belief.frontier) > 0:
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]
            prev_facts = set(belief.knowledge.facts)
            final_facts = set(max_frontier.facts)
            added_facts = [fact for fact in max_frontier.facts if fact not in prev_facts]
            deleted_facts = [fact for fact in belief.knowledge.facts if fact not in final_facts]
            belief.knowledge = max_frontier
            belief.reset_belief()
            print("    [Belief Diff]")
            print(f"      + add ({len(added_facts)}): {', '.join(added_facts) if added_facts else '-'}")
            print(f"      - del ({len(deleted_facts)}): {', '.join(deleted_facts) if deleted_facts else '-'}")
            
        return belief
    
    
    def call_feedback(
        self,
        target_fact,
        action_name,
        observation_facts=None,
        oracle_state_facts=None,
    ):
        
        question = self.refining_query(target_fact, action_name)
               
        if self.is_human_answer:
            if getattr(self.args, "use_interface", False):
                interface_server = getattr(self.args, "interface_server", None)
                if interface_server is None:
                    raise RuntimeError(
                        "The human interface was requested but its server is unavailable"
                    )
                return interface_server.ask(target_fact, question)

            while True:
                answer = input("    [Human] Enter t/f: ").strip().lower()
                if answer == "t":
                    return True
                if answer == "f":
                    return False
                print("    [Human] Invalid input. Please enter 't' or 'f'.")
                
        else:
            return self.query_oracle(
                target_fact,
                action_name,
                observation_facts=observation_facts,
                oracle_state_facts=oracle_state_facts,
            )

    def _get_oracle_facts(self):
        """Return normalized facts declared in scene YAML."""
        if self._true_init_facts is None:
            yaml_file_path = self.args.initial_state
            with open(yaml_file_path, "r", encoding="utf-8") as f:
                init_config = yaml.safe_load(f) or {}
            initial_facts = list(init_config.get("facts", []) or [])
            initial_facts.extend(init_config.get("true_init", []) or [])
            self._true_init_facts = {
                str(fact).replace(" ", "") for fact in initial_facts
            }
        return self._true_init_facts

    def query_oracle(
        self,
        target_fact,
        action_name=None,
        observation_facts=None,
        oracle_state_facts=None,
    ):
        """
        Dispatch an Oracle query to the active domain implementation.

        입력 예: target_fact="fresh(tomato1)", action_name="scan(brl_robot,tomato1)"
        출력 예: 현재 선택된 Tomato scene의 true state 기준 True
        """
        target_fact = str(target_fact).replace(" ", "")
        action = "" if action_name is None else str(action_name).replace(" ", "")
        true_facts = self._get_oracle_facts()

        if self.domain_name == "tomato":
            target_predicate, target_args = _parse_fact(target_fact)
            action_schema, action_args = (
                _parse_fact(action) if action else ("", [])
            )

            if (
                action_schema == "detect"
                and target_predicate in {"observed", "ripe", "unripe"}
                and len(target_args) == 1
                and len(action_args) >= 2
            ):
                tomato = target_args[0]
                stem = action_args[1]
                if f"at({tomato},{stem})" not in true_facts:
                    return False
                return (
                    True
                    if target_predicate == "observed"
                    else target_fact in true_facts
                )

            if (
                action_schema == "scan"
                and target_predicate in {"scanned", "fresh", "rotten"}
                and len(target_args) == 1
                and len(action_args) >= 2
            ):
                if target_args[0] != action_args[1]:
                    return False
                return (
                    True
                    if target_predicate == "scanned"
                    else target_fact in true_facts
                )

            action_success_rates = {
                "navigate": 0.90,
                "prepare_nav": 1.0,
                "pick_n_scan": 1.0,
                "pick": 0.95,
                "place": 0.99,
                "discard": 0.99,
            }
            if action_schema in action_success_rates:
                return bool(
                    np.random.random() < action_success_rates[action_schema]
                )

            return target_fact in true_facts

        elif self.domain_name == "wastesorting":
            target_predicate, target_args = _parse_fact(target_fact)
            action_schema, _ = _parse_fact(action) if action else ("", [])
            observation_set = {
                str(fact).replace(" ", "")
                for fact in (observation_facts or [])
            }
            current_state_set = {
                str(fact).replace(" ", "")
                for fact in (oracle_state_facts or [])
            }

            if action_schema == "detect_waste" and target_predicate == "detected":
                return target_fact in observation_set

            if (
                action_schema == "detect_waste"
                and target_predicate in {"can", "paper", "general", "plastic"}
                and len(target_args) == 1
            ):
                target_waste = target_args[0]
                for fact in true_facts:
                    predicate, args = _parse_fact(fact)
                    if predicate != "occ" or len(args) < 2 or args[1] != target_waste:
                        continue

                    front_waste = args[0]
                    front_cleared = False
                    for current_fact in current_state_set:
                        current_predicate, current_args = _parse_fact(current_fact)
                        if current_predicate == "in_bin" and current_args and current_args[0] == front_waste:
                            front_cleared = True
                            break
                        if (
                            current_predicate == "holding"
                            and len(current_args) >= 2
                            and current_args[1] == front_waste
                        ):
                            front_cleared = True
                            break

                    if not front_cleared:
                        return False

            return target_fact in true_facts

        else:
            raise ValueError(
                f"Oracle answer is not implemented for domain: {self.domain_name}"
            )

    
    # LLM-based refining vs Machine 
    def refining_query(self, target_fact, action_name):
        if self.use_llm:
            try:
                from models.llm_manager import get_llm_manager

                question = get_llm_manager().make_refining_query(
                    self.domain_name,
                    str(target_fact).replace(" ", ""),
                    action_name,
                )
            except Exception as exc:
                print(f"    [Query] LLM refining failed: {exc}")
                question = f"{target_fact} is True?"

            print(f"    [Query] T: {target_fact}=True? | Q: {question}")
        else:
            question = f"Is {target_fact} true?"
            print(f"    [Query] T: {target_fact}=True?")
        return question
