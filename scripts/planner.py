from scripts.knowledge_base import KnowledgeBase
from scripts import GT_MODEL_CONFIDENCE, ACTIONS
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import math


# =========================
# 2) Planner + Actions
# =========================
@dataclass(frozen=True)
class Action:
    name: str
    preconditions: List[str]

    def is_applicable(self, kb: KnowledgeBase) -> bool:
        return all(kb.has_fact(p) for p in self.preconditions)



class Planner:
    def __init__(self, do_label=True, show_actions=True):
        """
        Planner 초기화

        :param do_label: True이면 action 성공/실패를 확률적으로 라벨링
        :param show_actions: True이면 초기화 시 action 목록 출력

        내부 동작:
        1. ACTIONS dict를 기반으로 Action 객체 리스트 생성
        2. do_label=True이면 precondition 확률 곱으로 gt_vals 계산
        3. show_actions=True이면 현재 action 정보 출력
        """
        
        self.do_label = do_label
        self.show_actions = show_actions
        
        self.actions = self.init_actions()
        self.actions_with_gt = ACTIONS
        
        if self.do_label:
            self.update_gt_vals()
        
        if self.show_actions:
            self.dump_actions()
        
    def init_actions(self) -> List[Action]:
        """
        GLOBAL ACTIONS dict로부터 Action 객체 리스트를 생성.

        각 action:
            name → action 이름
            preconditions → 해당 action의 전제조건 리스트

        반환값:
            List[Action]
        """
        actions = []
        for name, value in ACTIONS.items():
            preconds = value['preconditions']
            actions.append(Action(name=name, preconditions=preconds))
        return actions
                    
                    
    def applicable_actions(self, kb: KnowledgeBase) -> List[Action]:
        """
        현재 KnowledgeBase 상태에서 실행 가능한 action만 필터링.

        전제조건이 모두 KB에 존재하면 applicable.
        """
        return [a for a in self.actions if a.is_applicable(kb)]

    
    def choose_action(self, kb: KnowledgeBase, rng: random.Random) -> Optional[Action]:
        """
        단순 정책:
        현재 실행 가능한 action 중 랜덤 선택.
        """
        candidates = self.applicable_actions(kb)
        return rng.choice(candidates) if candidates else None
    
    
    def generate_random_plan(self, horizon: int, rng: Optional[random.Random] = None):
        """
        길이 horizon의 랜덤 plan 생성.

        동작:
        1. self.actions에서 중복 허용 랜덤 선택 (with replacement)
        2. do_label=False → Action 리스트 반환
        3. do_label=True  → (Action, bool) 리스트 반환

            bool 의미:
                True  → 성공 (O)
                False → 실패 (X)

        성공 확률:
            action별 gt_vals (precondition confidence의 곱)

        즉,
            P(success | action) = ∏ P(predicate_i)
        """
        rng = rng or random.Random()
        if not self.actions or horizon <= 0:
            return []

        plan = rng.choices(self.actions, k=horizon)

        if not self.do_label:
            return plan

        # --- Bernoulli 라벨링 ---
        labeled_plan: List[Tuple[Action, bool]] = []
        for a in plan:
            p = float(self.actions_with_gt[a.name].get("gt_vals"))
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Invalid gt_vals for '{a.name}': {p}")
            lab = True if rng.random() < p else False
            labeled_plan.append((a, lab))
            
        return labeled_plan
    
    
    def dump_actions(self):
        """
        현재 action 목록 출력.

        do_label=True:
            action별 성공확률(gt_vals) 출력

        do_label=False:
            Action 객체 그대로 출력
        """
        if self.do_label:
            for name, prop in self.actions_with_gt.items():
                vals = prop['gt_vals']
                print(f"{name: <20} | "
                      f"p={vals:.3f} | ")            
        else:
            for a in self.actions:
                print(a)

    def dump_plan(self, plan):
        """
        생성된 plan 출력.

        do_label=True:
            각 action에 대해
                - 성공/실패 (✔ / ✘)
                - 확률 p
                - empirical success rate
            를 함께 출력

        do_label=False:
            action 이름과 preconditions 출력
        """
        if self.do_label:
            if not plan:
                print("[Planner] Empty labeled plan")
                return

            print("=========== LABELED PLAN ===========")
            success = 0
            for i, (action, label) in enumerate(plan):

                p = float(self.actions_with_gt[action.name].get("gt_vals", 0.0))
                mark = "✔" if label else "✘"
                text = "O" if label else "X"

                if label:
                    success += 1

                print(
                    f"{i:04d} | "
                    f"{action.name:<20} | "
                    f"{mark} ({text}) | "
                    f"p={p:.3f}"
                )

            total = len(plan)
            rate = success / total if total > 0 else 0.0
            print("------------------------------------")
            print(f"Total: {total}")
            print(f"Success: {success}")
            print(f"Failure: {total - success}")
            print(f"Empirical Success Rate: {rate:.3f}")
            print("====================================")
            
        else:
            print("===== PLAN DUMP =====")
            for i, action in enumerate(plan):
                    print(f"{i:02d}: {action.name} | pre: {action.preconditions}")
            print("=====================")
    
    
    @staticmethod
    def extract_predicate(fact: str) -> str:
        """
        fact 이름에서 predicate 부분만 추출.

        예:
            ripe_tomato1 → ripe
            pose_tomato1_0.1_8.2_-0.1 → pose
            at_stem1_tomato1 → at
        """
        return fact.split("_")[0]
    
    def update_gt_vals(self):
        """
        각 action의 preconditions에 대해
        predicate confidence의 곱을 계산하여
        ACTIONS dict의 gt_vals에 저장.

        수식적으로:
            gt_vals(action)
            = ∏ GT_MODEL_CONFIDENCE[predicate_i]

        독립성 가정(independence assumption) 기반.
        """
        for action in self.actions:
            product = 1.0
            preconditions = action.preconditions
            
            for fact in preconditions:
                predicate = self.extract_predicate(fact=fact)
                product *= GT_MODEL_CONFIDENCE[predicate]
                self.actions_with_gt[action.name]["gt_vals"] = product