from __future__ import annotations

from models.state import State
from models.action import Action


class RewardKitchen:
    def __init__(self, goal: State | None = None):
        self.goal = goal or State()
        self.goal_reward = 10.0
        self.clean_reward = 1.0
        self.cooked_reward = 3.0
        self.served_reward = 5.0

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

        if action_type in {"wash_ingredient", "wash_pot", "wash_dish"}:
            return self.clean_reward if any(fact.startswith("clean(") for fact in added) else 0.0

        if action_type in {"boil_tomato_soup", "boil_onion_soup", "boil_tomato_oniton_soup"}:
            cooked = any(
                fact.startswith("cooked_tomato_soup(")
                or fact.startswith("cooked_onion_soup(")
                or fact.startswith("cooked_tomato_oniton_soup(")
                for fact in added
            )
            return self.cooked_reward if cooked else 0.0

        if action_type in {"serve_tomato_soup", "serve_onion_soup", "serve_tomato_oniton_soup"}:
            served = any(
                fact.startswith("served_tomato_soup(")
                or fact.startswith("served_onion_soup(")
                or fact.startswith("served_tomato_oniton_soup(")
                for fact in added
            )
            return self.served_reward if served else 0.0

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
