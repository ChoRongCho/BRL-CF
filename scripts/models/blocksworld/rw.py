from __future__ import annotations

from models.state import State
from models.action import Action


class RewardBlocksworld:
    def __init__(self, goal: State | None = None):
        self.goal = goal or State()
        self.goal_reward = 10.0
        self.goal_fact_reward = 1.0

    @staticmethod
    def _to_fact_set(state: State | None) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        current_facts = self._to_fact_set(state)
        next_facts = self._to_fact_set(next_state)
        goal_facts = self._to_fact_set(self.goal)

        newly_satisfied_goal_facts = (next_facts - current_facts) & goal_facts
        progress_reward = self.goal_fact_reward * len(newly_satisfied_goal_facts)

        state_reward = (
            self.goal_reward
            if goal_facts and goal_facts.issubset(next_facts) and not goal_facts.issubset(current_facts)
            else 0.0
        )

        return state_reward + progress_reward
