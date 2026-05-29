from __future__ import annotations

from models.state import State
from models.action import Action


class RewardWastesorting:
    def __init__(self, goal: State = None):
        self.goal = goal or State()
        self.goal_reward = 10.0

    @staticmethod
    def _to_fact_set(state: State) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    def calculate_state_reward(self, state: State) -> float:
        facts = self._to_fact_set(state)
        goal_facts = self._to_fact_set(self.goal)

        if goal_facts and goal_facts.issubset(facts):
            return self.goal_reward
        return 0.0

    def calculate_action_reward(self, action: Action, 
                                added: set[str], 
                                current_facts: set[str], 
                                next_facts: set[str]) -> float:
        action_name = action.name.replace(" ", "")
        reward_categories = {
            "place_gw_bin": "general",
            "place_can_bin": "can",
            "place_plastic_bin": "plastic",
            "place_paper_bin": "paper",
        }
        action_type = action_name[:action_name.find("(")]
        category = reward_categories.get(action_type)

        if category and action_name.endswith(")"):
            args = action_name[action_name.find("(") + 1:-1].split(",")
            if (
                len(args) >= 3
                and f"{category}({args[1]})" in current_facts
                and f"in_bin({args[1]},{args[2]})" in added
            ):
                return 5.0
        
        return 0.0
    
    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        total_reward = 0.0
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

        total_reward += (state_reward + action_reward)
        
        return total_reward
