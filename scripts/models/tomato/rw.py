# models/tomato/rw.py

from __future__ import annotations

from models.state import State
from models.action import Action


class RewardTomato:
    REWARD_STATE_OBJECT = "__reward_state__"
    # STEP_REWARD = -1.0
    CONSECUTIVE_ACTION_LIMIT = 2
    CONSECUTIVE_DETECT_REWARD = -5.0
    CONSECUTIVE_SCAN_REWARD = -5.0
    CONSECUTIVE_NAVIGATE_REWARD = -5.0
    SUCCESSFUL_PLACE_REWARD = 10.0
    SUCCESSFUL_DISCARD_REWARD = 10.0

    def __init__(self, goal:State=None):
        self.goal = goal or State()

    @staticmethod
    def _to_fact_set(state: State) -> set[str]:
        if state is None:
            return set()
        return set(state.facts)

    @staticmethod
    def get_tomato_state(tomato: str, facts: set[str]) -> str | None:
        if f"rotten({tomato})" in facts:
            return "rotten"
        if f"unripe({tomato})" in facts:
            return "unripe"
        if f"ripe({tomato})" in facts:
            return "ripe"
        return None

    def calculate_state_reward(self, state: State) -> float:
        return 0.0

    def calculate_action_reward(self, action: Action, 
                                added: set[str], 
                                current_facts: set[str], 
                                next_facts: set[str],
                                state: State = None) -> float:
        action_name = action.name.replace(" ", "")
        action_type = action_name.split("(", 1)[0]

        args = action_name.split("(", 1)[1].rstrip(")").split(",") if "(" in action_name else []
        streak_fluent = None
        streak_reward = 0.0
        if action_type == "detect" and len(args) >= 2:
            streak_fluent = f"detect_streak_{args[1]}"
            streak_reward = self.CONSECUTIVE_DETECT_REWARD
        elif action_type == "scan" and len(args) >= 2:
            streak_fluent = f"scan_streak_{args[1]}"
            streak_reward = self.CONSECUTIVE_SCAN_REWARD
        elif action_type == "navigate":
            streak_fluent = "navigate_streak"
            streak_reward = self.CONSECUTIVE_NAVIGATE_REWARD

        if streak_fluent is not None:
            current_streak = int(state.get_fluent(
                self.REWARD_STATE_OBJECT,
                streak_fluent,
                0,
            )) if state is not None else 0
            if current_streak >= self.CONSECUTIVE_ACTION_LIMIT - 1:
                return streak_reward

        if action_type == "pick":
            args = action_name[len("pick("):-1].split(",")
            if len(args) < 2:
                return 0.0
            tomato = args[1]
            is_picked = f"holding({args[0]},{tomato})" in added

            if is_picked and self.get_tomato_state(tomato, current_facts) == "unripe":
                return 0.0
            return 0.0
            
        elif action_type == "place":
            args = action_name[len("place("):-1].split(",")
            if len(args) < 2:
                return 0.0
            tomato = args[1]
            is_loaded = f"loaded({tomato},{args[0]})" in added

            if is_loaded and f"fresh({tomato})" in current_facts:
                return self.SUCCESSFUL_PLACE_REWARD

        elif action_type == "discard":
            args = action_name[len("discard("):-1].split(",")
            if len(args) < 2:
                return 0.0
            tomato = args[1]
            is_discarded = f"discarded({tomato})" in added

            if is_discarded and f"rotten({tomato})" in current_facts:
                return self.SUCCESSFUL_DISCARD_REWARD
        return 0.0


    def calculate_reward(self, state: State, action: Action, next_state: State) -> float:
        total_reward = 0.0
        current_facts = self._to_fact_set(state)
        next_facts = self._to_fact_set(next_state)
        added = next_facts - current_facts

        state_reward = self.calculate_state_reward(next_state)
        action_reward = self.calculate_action_reward(
            action=action,
            added=added,
            current_facts=current_facts,
            next_facts=next_facts,
            state=state,
        )

        total_reward += (state_reward + action_reward)
        # Uncomment to apply a negative reward on every executed step.
        # total_reward += self.STEP_REWARD
        
        return total_reward
