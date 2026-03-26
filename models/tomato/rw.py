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

    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        reward = 0.0

        s_facts = self._to_fact_set(state)
        ns_facts = self._to_fact_set(next_state)
        
        added = ns_facts - s_facts
        deled = s_facts - ns_facts

        # 기본 step cost: 쓸데없는 반복 방지7
        reward -= action.cost

        # 1) 토마토가 새로 observed 되면 보상
        # 예시 predicate: observed(t1), observed(t1, stem1), detected(t1) 등
        for fact in added:
            if fact.startswith("observed("):
                reward += 7.0

        # 2) ripe 토마토를 수확/적재 상태로 만들면 큰 보상
        # goal 예시에 loaded(t1, changmin) 형태가 있으므로 loaded 추가를 강하게 보상
        for fact in added:
            if fact.startswith("loaded("):
                reward += 15.0

        # 3) rotten 토마토를 discard 하면 큰 보상
        for fact in added:
            if fact.startswith("discarded("):
                reward += 15.0

        # 4) unripe는 그대로 두기
        # 즉 unripe를 잘못 집거나 버리면 패널티
        # predicate 이름은 실제 state 표현에 맞춰야 함
        for fact in added:
            if fact.startswith("loaded("):
                tomato = fact[len("loaded("):-1].split(",")[0].strip()
                if f"unripe({tomato})" in s_facts or f"unripe({tomato})" in ns_facts:
                    reward -= 15.0

            if fact.startswith("discarded("):
                tomato = fact[len("discarded("):-1].split(",")[0].strip()
                if f"unripe({tomato})" in s_facts or f"unripe({tomato})" in ns_facts:
                    reward -= 15.0

        # ripe를 discard하면 패널티
        for fact in added:
            if fact.startswith("discarded("):
                tomato = fact[len("discarded("):-1].split(",")[0].strip()
                if f"ripe({tomato})" in s_facts or f"ripe({tomato})" in ns_facts:
                    reward -= 50.0

        # rotten을 loaded하면 패널티
        for fact in added:
            if fact.startswith("loaded("):
                tomato = fact[len("loaded("):-1].split(",")[0].strip()
                if f"rotten({tomato})" in s_facts or f"rotten({tomato})" in ns_facts:
                    reward -= 50.0

        return reward