from scripts.knowledge_base import KnowledgeBase

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


@dataclass
class Planner:
    actions: List[Action]

    def applicable_actions(self, kb: KnowledgeBase) -> List[Action]:
        return [a for a in self.actions if a.is_applicable(kb)]

    
    def choose_action(self, kb: KnowledgeBase, rng: random.Random) -> Optional[Action]:
        """Simple policy: pick a random applicable action."""
        candidates = self.applicable_actions(kb)
        return rng.choice(candidates) if candidates else None
    
    
    def generate_random_plan(self, horizon=20, rng: Optional[random.Random]=None):
        """
        Random plan of length 'horizon'
        
        :param self: Description
        :param horizon: Description
        :param rng: Description
        :type rng: Optional[random.Random]
        """
        rng = rng or random.Random()
        if not self.actions or horizon <= 0:
            return []
        
        return rng.choices(self.actions, k=horizon)
    
    def dump_plan(
        self,
        plan: List[Action],
        show_preconditions: bool = False
    ) -> None:
        """
        Pretty-print a plan.
        """

        if not plan:
            print("[Planner] Empty plan")
            return

        print("===== PLAN DUMP =====")
        for i, action in enumerate(plan):
            if show_preconditions:
                print(f"{i:02d}: {action.name} | pre: {action.preconditions}")
            else:
                print(f"{i:02d}: {action.name}")
        print("=====================")
    
    
def generate_action():
    actions: List[Action] = []
    
    actions.append(Action(name="pick_tomato1", 
                          preconditions=["ripe_tomato1", "pose_tomato1_0.1_8.2_-0.1", "at_stem1_tomato1"]))
    actions.append(Action(name="pick_tomato2", 
                          preconditions=["ripe_tomato2", "pose_tomato2_0.3_8.1_-0.2", "at_stem1_tomato2"]))
    actions.append(Action(name="pick_tomato3", 
                          preconditions=["unripe_tomato3", "pose_tomato3_0.4_6.2_0.3", "at_stem1_tomato3"]))
    
    actions.append(Action(name="place_tomato1", preconditions=["at_robot1_stem1", "at_stem1_tomato1"]))
    actions.append(Action(name="place_tomato2", preconditions=["at_robot1_stem1", "at_stem1_tomato2"]))
    actions.append(Action(name="place_tomato3", preconditions=["at_robot1_stem1", "at_stem1_tomato3"]))
    
    actions.append(Action(name="action1", preconditions=["predidcate1_dummy1"]))
    actions.append(Action(name="action2", preconditions=["predidcate1_dummy2"]))
    actions.append(Action(name="action3", preconditions=["predidcate2_dummy1"]))
    actions.append(Action(name="action4", preconditions=["predidcate2_dummy3"]))
    actions.append(Action(name="action5", preconditions=["predidcate3_dummy3"]))
    actions.append(Action(name="action6", preconditions=["predidcate1_dummy1", "predidcate1_dummy2"]))
    actions.append(Action(name="action7", preconditions=["predidcate2_dummy1", "predidcate3_dummy3"]))
    
    
    return actions