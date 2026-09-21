"""Reward model for the Query-as-Action POMDP baseline."""

from __future__ import annotations

from models.action import Action
from models.state import State

from .query_actions import QueryAction


DEFAULT_QUERY_COST = 1.0
DEFAULT_FAILURE_PENALTY = 10.0


def _facts(state: State | None) -> set[str]:
    return set(state.facts) if state is not None else set()


class _TomatoTaskReward:
    REWARD_STATE_OBJECT = "__reward_state__"
    CONSECUTIVE_ACTION_LIMIT = 2
    CONSECUTIVE_DETECT_REWARD = -5.0
    CONSECUTIVE_SCAN_REWARD = -5.0
    CONSECUTIVE_NAVIGATE_REWARD = -5.0
    SUCCESSFUL_PLACE_REWARD = 10.0
    SUCCESSFUL_DISCARD_REWARD = 10.0

    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        current = _facts(state)
        added = _facts(next_state) - current
        action_name = action.name.replace(" ", "")
        action_type = action_name.split("(", 1)[0]
        args = (
            action_name.split("(", 1)[1].rstrip(")").split(",")
            if "(" in action_name else []
        )

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
            streak = int(state.get_fluent(
                self.REWARD_STATE_OBJECT,
                streak_fluent,
                0,
            ))
            if streak >= self.CONSECUTIVE_ACTION_LIMIT - 1:
                return streak_reward

        if action_type == "place" and len(args) >= 2:
            tomato = args[1]
            if (
                f"loaded({tomato},{args[0]})" in added
                and f"fresh({tomato})" in current
            ):
                return self.SUCCESSFUL_PLACE_REWARD
        elif action_type == "discard" and len(args) >= 2:
            tomato = args[1]
            if (
                f"discarded({tomato})" in added
                and f"rotten({tomato})" in current
            ):
                return self.SUCCESSFUL_DISCARD_REWARD
        return 0.0


class _WasteSortingTaskReward:
    REWARD_STATE_OBJECT = "__reward_state__"
    DETECT_STREAK_FLUENT = "detect_streak"
    CONSECUTIVE_DETECT_LIMIT = 3
    CONSECUTIVE_DETECT_PENALTY = -10.0
    CORRECT_PLACEMENT_REWARD = 5.0
    GOAL_REWARD = 10.0

    def __init__(self, goal: State | None) -> None:
        self.goal = goal or State()

    def get_reward(self, state: State, action: Action, next_state: State) -> float:
        current = _facts(state)
        following = _facts(next_state)
        added = following - current
        action_name = action.name.replace(" ", "")
        action_type = action_name.split("(", 1)[0]

        action_reward = 0.0
        if action_type == "detect_waste":
            streak = int(state.get_fluent(
                self.REWARD_STATE_OBJECT,
                self.DETECT_STREAK_FLUENT,
                0,
            ))
            if streak >= self.CONSECUTIVE_DETECT_LIMIT - 1:
                action_reward = self.CONSECUTIVE_DETECT_PENALTY

        categories = {
            "place_gw_bin": "general",
            "place_can_bin": "can",
            "place_plastic_bin": "plastic",
            "place_paper_bin": "paper",
        }
        category = categories.get(action_type)
        if category and action_name.endswith(")"):
            args = action_name[action_name.find("(") + 1:-1].split(",")
            if (
                len(args) >= 3
                and f"{category}({args[1]})" in current
                and f"in_bin({args[1]},{args[2]})" in added
            ):
                action_reward = self.CORRECT_PLACEMENT_REWARD

        goal_facts = _facts(self.goal)
        goal_reward = (
            self.GOAL_REWARD
            if goal_facts
            and goal_facts.issubset(following)
            and not goal_facts.issubset(current)
            else 0.0
        )
        return goal_reward + action_reward


def _build_task_reward(domain_name: str, goal: State | None):
    if domain_name == "tomato":
        return _TomatoTaskReward()
    if domain_name == "wastesorting":
        return _WasteSortingTaskReward(goal)
    raise ValueError(f"Unsupported Query-as-Action reward domain: {domain_name}")


class QueryAsActionRewardModel:
    """Combine query costs, invalid-action penalties, and domain rewards."""

    def __init__(
        self,
        domain_name: str | None = None,
        goal: State | None = None,
        *,
        failure_penalty: float,
        task_reward_model=None,
    ) -> None:
        if failure_penalty < 0.0:
            raise ValueError("failure_penalty must be non-negative")
        if task_reward_model is None:
            if domain_name is None:
                raise ValueError(
                    "domain_name is required when task_reward_model is not supplied"
                )
            task_reward_model = _build_task_reward(domain_name, goal)
        self.task_reward_model = task_reward_model
        self.failure_penalty = float(failure_penalty)

    @staticmethod
    def query_reward(action: QueryAction) -> float:
        """Return the immediate reward for asking one Boolean question."""
        return -float(action.cost)

    def invalid_action_reward(self, action: Action) -> float:
        """Return the terminal penalty for an inapplicable physical action."""
        del action
        return -self.failure_penalty

    def physical_action_reward(
        self,
        state: State,
        action: Action,
        next_state: State,
    ) -> float:
        """Return the physical-action reward defined locally in this module."""
        return float(self.task_reward_model.get_reward(state, action, next_state))
