from __future__ import annotations

from models.state import State
from models.action import Action


class RewardRover:
    def __init__(self, goal: State | None = None):
        self.goal = goal or State()
        self.goal_reward = 10.0
        self.analysis_reward = 2.0
        self.image_reward = 2.0
        self.communication_reward = 3.0

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

        if action_type == "sample_soil":
            return self.analysis_reward if any(fact.startswith("have_soil_analysis(") for fact in added) else 0.0

        if action_type == "take_image":
            return self.image_reward if any(fact.startswith("have_image(") for fact in added) else 0.0

        if action_type in {"communicate_soil_data", "communicate_image_data"}:
            communicated = any(
                fact.startswith("communicated_soil_data(")
                or fact.startswith("communicated_image_data(")
                for fact in added
            )
            return self.communication_reward if communicated else 0.0

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
