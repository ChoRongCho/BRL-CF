from __future__ import annotations

from models.state import State
from models.action import Action


class RewardWatering:
    def __init__(self, goal: State | None = None):
        self.goal = goal or State()
        self.goal_reward = 10.0
        self.find_basket_reward = 1.0
        self.water_plant_reward = 2.0

    @staticmethod
    def _to_fact_set(state: State | None) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    def calculate_action_reward(
        self,
        action: Action,
        added: set[str],
        current_facts: set[str],
        next_facts: set[str],
    ) -> float:
        action_name = action.name.replace(" ", "")
        action_type = action_name.split("(", 1)[0]

        if not action_name.endswith(")"):
            return 0.0

        args = action_name[action_name.find("(") + 1:-1].split(",")

        if action_type == "find_basket":
            if len(args) < 3:
                return 0.0
            basket, room = args[1], args[2]
            return self.find_basket_reward if f"at({basket},{room})" in added else 0.0

        if action_type == "pour_water":
            if len(args) < 3:
                return 0.0
            plant = args[2]
            watered_fact = f"watered({plant})"
            if watered_fact in added:
                return self.water_plant_reward

        return 0.0

    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        current_facts = self._to_fact_set(state)
        next_facts = self._to_fact_set(next_state)
        added = next_facts - current_facts
        goal_facts = self._to_fact_set(self.goal)

        state_reward = (
            self.goal_reward
            if goal_facts and goal_facts.issubset(next_facts) and not goal_facts.issubset(current_facts)
            else 0.0
        )
        action_reward = self.calculate_action_reward(
            action=action,
            added=added,
            current_facts=current_facts,
            next_facts=next_facts,
        )

        return state_reward + action_reward
