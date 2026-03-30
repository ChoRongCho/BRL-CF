# models/tomato/rw.py

from __future__ import annotations
from typing import Iterable

from models.state import State
from models.action import Action


class RewardTomato:
    def __init__(self, goal:State=None):
        self.goal = goal or State()

    def _to_fact_set(self, state: State) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)
        

    def _new_facts(self, state: State, next_state: State) -> set[str]:
        s = self._to_fact_set(state)
        ns = self._to_fact_set(next_state)
        return ns - s

    def _goal_satisfied_count(self, state: State) -> int:
        facts = self._to_fact_set(state)
        return sum(1 for g in self.goal.facts if g in facts)

    def _tomato_reward(self, state, next_state):
        
        t_reward = 0
        
        s_facts = self._to_fact_set(state)
        ns_facts = self._to_fact_set(next_state)
        added = ns_facts - s_facts
        
        # positive t_reward
        for fact in added:
            if fact.startswith("loaded("):
                tomato = fact[len("loaded("):-1].split(",")[0].strip()
                if tomato == "t1" or tomato == "t2" or tomato == "t4":  # ripe
                    t_reward += 15.0
                elif tomato == "t5":    # rotten
                    t_reward -= 15.0
                elif tomato == "t3":    # unripe
                    t_reward -= 10.0
                    
        # for fact in added:
            if fact.startswith("discarded("):
                tomato = fact[len("discarded("):-1].split(",")[0].strip()
                if tomato == "t1" or tomato == "t2" or tomato == "t4":  # ripe
                    t_reward -= 15.0
                elif tomato == "t5":    # rotten
                    t_reward += 15.0
                elif tomato == "t3":    # unripe
                    t_reward -= 10.0
        
        return t_reward
    
    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        reward = 0.0

        s_facts = self._to_fact_set(state)
        ns_facts = self._to_fact_set(next_state)
        
        added = ns_facts - s_facts
        deled = s_facts - ns_facts

        # 기본 step cost: 쓸데없는 반복 방지7
        reward -= action.cost

        # 1) 새 observed 보상
        new_observed = [fact for fact in added if fact.startswith("observed(")]
        reward += 7.0 * len(new_observed)
        new_quality = [
            fact for fact in added
            if fact.startswith("ripe(")
            or fact.startswith("unripe(")
            or fact.startswith("rotten(")
        ]

        # 2) 토마토 탐지 보상
        reward += self._tomato_reward(state=state, next_state=next_state)
        
        # 3) detect or scan 액션이 아무 새 정보도 못 얻었으면 패널티
        if action.name.startswith("detect("):
            if len(new_observed) == 0:
                reward -= 3.0

        if action.name.startswith("scan("):
            if len(added) == 0:
                reward -= 2.0
        
        # 4) dockstation 도착 보상
        
        
        return reward