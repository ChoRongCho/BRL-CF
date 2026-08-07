"""Finite-horizon planner for the grounded-fact adaptation of Attr-POMDP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.action import Action
from models.belief import Belief

from .belief import state_key
from .query import QueryAction, QueryFactory


@dataclass
class PlanResult:
    action: Action | QueryAction | None
    value: float
    action_values: dict[str, float]


class AttrPOMDPPlanner:
    """Jointly evaluate task actions and binary grounded-fact questions.

    Task actions use the repository transition/observation/reward models.
    Query actions are no-op world transitions with oracle yes/no observations:

        Q(b, query) = -cost + gamma * sum_o P(o|b,q) V(b_q,o)
    """

    def __init__(
        self,
        args: Any,
        env: Any,
        belief_model: Any,
        query_factory: QueryFactory,
        provider: Any,
        *,
        action_limit: int = 20,
        query_limit: int = 20,
    ) -> None:
        self.env = env
        self.belief_model = belief_model
        self.query_factory = query_factory
        self.provider = provider
        self.gamma = float(args.gamma)
        self.horizon = max(1, int(args.max_depth))
        self.action_limit = max(1, int(action_limit))
        self.query_limit = max(1, int(query_limit))
        self._cache: dict[tuple, float] = {}
        self._belief_keys: dict[int, tuple] = {}
        self._task_actions_cache: dict[int, list[Action]] = {}
        self._query_actions_cache: dict[tuple, list[QueryAction]] = {}

    def _task_actions(self, belief: Belief) -> list[Action]:
        cache_key = id(belief)
        cached = self._task_actions_cache.get(cache_key)
        if cached is not None:
            return cached

        states, weights = self.belief_model.materialize(belief)
        actions: list[tuple[float, Action]] = []
        for action in self.env.actions:
            if action.name.startswith("ask_"):
                continue
            probability = sum(
                float(weight)
                for state, weight in zip(states, weights)
                if action.is_applicable(state)
            )
            if probability > 0.0:
                actions.append((probability, action))
        actions.sort(key=lambda item: (-item[0], item[1].name))
        result = [action for _, action in actions[: self.action_limit]]
        self._task_actions_cache[cache_key] = result
        return result

    def _query_actions(
        self,
        belief: Belief,
        context_action_name: str | None,
        asked_facts: frozenset[str],
    ) -> list[QueryAction]:
        cache_key = (id(belief), context_action_name, asked_facts)
        cached = self._query_actions_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.query_factory.candidates(
            belief,
            context_action_name,
            asked_facts,
        )[: self.query_limit]
        self._query_actions_cache[cache_key] = result
        return result

    def _task_evaluation(
        self,
        belief: Belief,
        action: Action,
        *,
        include_branches: bool,
    ) -> tuple[float, list[tuple[float, Belief]]]:
        return self.belief_model.evaluate_task(
            belief,
            action,
            self.env.reward_model,
            include_branches=include_branches,
        )

    def _signature(
        self,
        belief: Belief,
        depth: int,
        context_action_name: str | None,
        asked_facts: frozenset[str],
    ) -> tuple:
        belief_id = id(belief)
        weighted_states = self._belief_keys.get(belief_id)
        if weighted_states is None:
            weighted_states = tuple(
                sorted(
                    (state_key(state), round(float(weight), 8))
                    for state, weight in zip(belief.particles, belief.particle_weights)
                    if float(weight) > 1e-12
                )
            )
            self._belief_keys[belief_id] = weighted_states
        return depth, context_action_name, tuple(sorted(asked_facts)), weighted_states

    def _task_value(
        self,
        belief: Belief,
        action: Action,
        depth: int,
        asked_facts: frozenset[str],
    ) -> float:
        reward, branches = self._task_evaluation(
            belief,
            action,
            include_branches=depth > 1,
        )
        future = 0.0
        if depth > 1:
            future = sum(
                probability
                * self._value(child, depth - 1, action.name, asked_facts)
                for probability, child in branches
            )
        return reward + self.gamma * future

    def _query_value(
        self,
        belief: Belief,
        query: QueryAction,
        depth: int,
        asked_facts: frozenset[str],
    ) -> float:
        if depth <= 1:
            return -float(query.cost)
        next_asked = asked_facts | {query.target_fact}
        future = sum(
            probability
            * self._value(
                posterior,
                depth - 1,
                query.context_action_name,
                next_asked,
            )
            for probability, _, posterior in self.belief_model.query_branches(
                belief, query, self.provider
            )
        )
        return -float(query.cost) + self.gamma * future

    def _value(
        self,
        belief: Belief,
        depth: int,
        context_action_name: str | None,
        asked_facts: frozenset[str],
    ) -> float:
        if depth <= 0:
            return 0.0
        key = self._signature(belief, depth, context_action_name, asked_facts)
        if key in self._cache:
            return self._cache[key]

        values = [
            self._task_value(belief, action, depth, asked_facts)
            for action in self._task_actions(belief)
        ]
        values.extend(
            self._query_value(belief, query, depth, asked_facts)
            for query in self._query_actions(belief, context_action_name, asked_facts)
        )
        value = max(values, default=0.0)
        self._cache[key] = value
        return value

    def search(
        self,
        belief: Belief,
        *,
        context_action_name: str | None = None,
        asked_facts: frozenset[str] = frozenset(),
    ) -> PlanResult:
        self._cache.clear()
        self._belief_keys.clear()
        self._task_actions_cache.clear()
        self._query_actions_cache.clear()
        actions: list[Action | QueryAction] = [
            *self._task_actions(belief),
            *self._query_actions(belief, context_action_name, asked_facts),
        ]
        values = {}
        by_name = {}
        for action in actions:
            if isinstance(action, QueryAction):
                value = self._query_value(belief, action, self.horizon, asked_facts)
            else:
                value = self._task_value(belief, action, self.horizon, asked_facts)
            values[action.name] = value
            by_name[action.name] = action

        if not values:
            return PlanResult(None, float("-inf"), {})
        name = max(values, key=lambda candidate: (values[candidate], candidate))
        return PlanResult(by_name[name], values[name], values)
