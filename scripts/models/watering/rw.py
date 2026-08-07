from __future__ import annotations

from models.state import State
from models.action import Action


class RewardWatering:
    def __init__(self, goal: State | None = None):
        self.goal = goal or State()
        self.goal_reward = 10.0
        self.first_scan_reward = 5.0
        self.find_discovery_reward = 0.0
        self.empty_find_penalty = -0.5
        self.move_penalty = -0.8

    @staticmethod
    def _to_fact_set(state: State | None) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    @staticmethod
    def _get_action_type_and_args(action: Action) -> tuple[str, list[str]]:
        action_name = action.name.replace(" ", "")
        if not action_name.endswith(")") or "(" not in action_name:
            return action_name, []

        action_type = action_name.split("(", 1)[0]
        args = action_name[action_name.find("(") + 1:-1].split(",")
        return action_type, args

    @staticmethod
    def _new_location_facts_for_prefix(added: set[str], obj_prefixes: tuple[str, ...], room: str) -> set[str]:
        return {
            fact
            for fact in added
            if any(fact.startswith(f"at({obj_prefix}") for obj_prefix in obj_prefixes)
            and fact.endswith(f",{room})")
        }

    def calculate_action_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> float:
        
        current_facts = self._to_fact_set(state)
        next_facts = self._to_fact_set(next_state)
        added = next_facts - current_facts
        action_type, args = self._get_action_type_and_args(action)

        if action_type == "move":
            return self.move_penalty

        if action_type == "find_plant":
            if len(args) < 2:
                return 0.0
            room = args[1]
            
            reward = self.first_scan_reward if f"scanned({room})" in added else 0.0
            
            discovered = self._new_location_facts_for_prefix(added, ("plant",), room)
            
            reward += self.find_discovery_reward if discovered else self.empty_find_penalty
            
            return reward

        return 0.0

    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        current_facts = self._to_fact_set(state)
        next_facts = self._to_fact_set(next_state)
        goal_facts = self._to_fact_set(self.goal)

        added = next_facts - current_facts
        newly_satisfied_goal_facts = added & goal_facts
        state_reward = self.goal_reward * len(newly_satisfied_goal_facts)
        action_reward = self.calculate_action_reward(state, action, next_state)

        return state_reward + action_reward
