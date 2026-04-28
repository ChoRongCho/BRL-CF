# models/tomato/rw.py

from __future__ import annotations
from typing import Iterable

from models.state import State
from models.action import Action


class RewardTomato:
    def __init__(self, goal:State=None):
        self.goal = goal or State()
        self.previous_action_name = None
        self.repeat_action_penalty = 20.0

    def _to_fact_set(self, state: State) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    @staticmethod
    def _normalize_action_name(action: Action) -> str:
        """Normalize action string so whitespace does not affect repeat checks."""
        if action is None or not getattr(action, "name", None):
            return ""
        return action.name.replace(" ", "")

    def _repeat_action_penalty(self, action: Action) -> float:
        """
        Penalize immediately repeating the exact same grounded action.

        Example:
            pick(changmin,tomato1,stem_01) -> pick(changmin,tomato1,stem_01)
            receives `repeat_action_penalty`.
        """
        current_action_name = self._normalize_action_name(action)
        if not current_action_name:
            return 0.0
        if self.previous_action_name == current_action_name:
            return self.repeat_action_penalty
        return 0.0

    def _remember_action(self, action: Action) -> None:
        current_action_name = self._normalize_action_name(action)
        if current_action_name:
            self.previous_action_name = current_action_name

    def reset_action_history(self) -> None:
        """Clear remembered action history, e.g. when starting a new episode."""
        self.previous_action_name = None
        

    def _new_facts(self, state: State, next_state: State) -> set[str]:
        s = self._to_fact_set(state)
        ns = self._to_fact_set(next_state)
        return ns - s

    def _goal_satisfied_count(self, state: State) -> int:
        facts = self._to_fact_set(state)
        return sum(1 for g in self.goal.facts if g in facts)


    def _tomato_reward(self, state, next_state, action: Action):
        
        t_reward = 0
        action_name = action.name
        s_facts = self._to_fact_set(state)
        ns_facts = self._to_fact_set(next_state)
        added = ns_facts - s_facts

        # helper: tomato 상태 판별
        def get_tomato_state(tomato, facts):
            if f"rotten({tomato})" in facts:
                return "rotten"
            elif f"unripe({tomato})" in facts:
                return "unripe"
            elif f"ripe({tomato})" in facts:
                return "ripe"
            return None



        if action_name.startswith("pick("):
            
            # pick_n_scan(robot_name,tomato_name,stem_name)
            tomato = action_name[len("pick("):-1].split(",")[1].strip()
            state_type = get_tomato_state(tomato, ns_facts)
            if state_type == "ripe":
                t_reward += 2.0
            elif state_type == "rotten":
                t_reward += 2.0
            elif state_type == "unripe":
                t_reward -= 30.0
                    
                    
        for fact in added:
            # loaded
            
            if fact.startswith("loaded("):
                tomato = fact[len("loaded("):-1].split(",")[0].strip()
                state_type = get_tomato_state(tomato, ns_facts)

                if state_type == "ripe":
                    t_reward += 15.0
                elif state_type == "rotten":
                    t_reward -= 10.0
                elif state_type == "unripe":
                    t_reward -= 10.0

            # discarded
            if fact.startswith("discarded("):
                tomato = fact[len("discarded("):-1].split(",")[0].strip()
                state_type = get_tomato_state(tomato, ns_facts)
                if state_type == "ripe":
                    t_reward -= 15.0
                elif state_type == "rotten":
                    t_reward += 15.0
                elif state_type == "unripe":
                    t_reward -= 10.0

        return t_reward

    
    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        reward = 0.0

        s_facts = self._to_fact_set(state)
        ns_facts = self._to_fact_set(next_state)
        
        added = ns_facts - s_facts
        deled = s_facts - ns_facts

        # 기본 step cost: 쓸데없는 반복 방지7
        reward -= (action.cost+2)

        # 같은 grounded action을 바로 다시 선택하면 강한 패널티
        # reward -= self._repeat_action_penalty(action)

        # 1) 새 observed 보상
        new_observed = [fact for fact in added if fact.startswith("observed(")]
        reward += 7.0 * len(new_observed)
        new_quality = [
            fact for fact in added
            if fact.startswith("ripe(")
            or fact.startswith("unripe(")
            or fact.startswith("rotten(")
        ]

        # 2) 토마토 보상
        reward += self._tomato_reward(state=state, next_state=next_state, action=action)

        # 2-1) scan으로 새 정보를 얻으면 보상 
        new_scanned = [fact for fact in added if fact.startswith("scanned(")]
        reward += 8.0 * sum(
            f"unripe({fact[8:-1]})" not in ns_facts
            for fact in new_scanned
        )

        # 3) detect or scan 액션이 아무 새 정보도 못 얻었으면 패널티
        if action.name.startswith("detect("):
            if len(new_observed) == 0:
                reward -= 8.0

        if action.name.startswith("scan("):
            if len(new_scanned) == 0:
                reward -= 6.0



        self._remember_action(action)
        return reward
