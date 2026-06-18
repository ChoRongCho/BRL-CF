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

from collections import defaultdict

class FeedbackManger:
    def __init__(self, args, conf_threshold):
        self.args = args
        self.domain_name = self.args.domain
        self.conf_threshold = conf_threshold
        self.num_of_query = 0
        self.query_log = []
        self.use_llm = False
        # ablation study
        """
        f_strategy: int = "1: no, 2: all, 3: ours, 4:random"
        q_strategy: int = "1: ours, 2: entropy"
        """
        self.f_strategy = self.args.f_strategy
        self.q_strategy = self.args.q_strategy
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
    
    
    def get_new_observation(self, belief: Belief, step: int = None, action_name: str = None) -> Belief:
        
        
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
            answer = self.query_human(target_fact, action_name)            
            
            
            self.num_of_query += 1
            print(f"    [Query] A: {answer}")
            
            # 적용 후 다시 계산            
            belief = self.apply_fact_answer_to_belief(belief, target_fact, answer)
            updated_confidence = self.compute_confidence(belief.frontier_weights)
            self.query_log.append({
                "step": step,
                "action": action_name,
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
    
    
    def query_human(self, target_fact, action_name):
        
        
        self.refining_query(target_fact, action_name)
        
        
        
        
        
        if self.is_human_answer:
            """
            """
            while True:
                answer = input("    [Human] Enter t/f: ").strip().lower()
                if answer == "t":
                    return True
                if answer == "f":
                    return False
                print("    [Human] Invalid input. Please enter 't' or 'f'.")
                
        else:
            """
            """
            if self._true_init_facts is None:
                yaml_file_path = self.args.initial_state
                with open(yaml_file_path, "r", encoding="utf-8") as f:
                    init_config = yaml.safe_load(f) or {}
                self._true_init_facts = {
                    str(fact).replace(" ", "")
                    for fact in init_config.get("true_init", []) or []
                }

            target_fact = str(target_fact).replace(" ", "")
            action = "" if action_name is None else str(action_name)

            if self.domain_name == "tomato":
                if action.startswith("pick") and target_fact.startswith("at("):
                    # return bool(np.random.random() < 0.1)
                    return False
                
                if action.startswith("detect("):
                    # detect(R,S)
                    stem = action[action.rfind(",") + 1:-1].strip()
                    if target_fact.startswith("ripe("):
                        tomato = target_fact[len("ripe("):-1]
                        q_fact = f"at({tomato},{stem})"
                        if q_fact in self._true_init_facts:
                            if f"rotten({tomato})" in self._true_init_facts:
                                return True
                            else:
                                return target_fact in self._true_init_facts
                        else:
                            return False
                        
                    elif target_fact.startswith("unripe("):
                        tomato = target_fact[len("unripe("):-1]
                        q_fact = f"at({tomato},{stem})"
                        if q_fact in self._true_init_facts:
                            return target_fact in self._true_init_facts
                        else:
                            return False
                        
                    elif target_fact.startswith("at("):
                        return target_fact in self._true_init_facts
                    
                if action.startswith("scan("):
                    return target_fact in self._true_init_facts
                    
                if (action.startswith("place") or action.startswith("discard")) and target_fact.startswith("handempty("):
                    # return bool(np.random.random() < 0.8)
                    return True
                
                return bool(np.random.random() < 0.5)
                

            if self.domain_name == "wastesorting":
                return target_fact in self._true_init_facts

            return bool(np.random.random() < 0.8)

    
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
            print(f"    [Query] T: {target_fact}=True?")
        return
