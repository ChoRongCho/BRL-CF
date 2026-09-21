from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any

import numpy as np

from models.belief_update import BeliefManager
from scripts.baseline.targeted_query_pomdp.planner import QueryAsActionPOMCPPlanner
from scripts.baseline.targeted_query_pomdp.query_actions import QueryAction

from .policy_types import PolicyContext


@contextmanager
def isolated_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def derived_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass(frozen=True)
class RootValueResult:
    physical_q: dict[str, float]
    query_q: dict[str, float]
    visits: dict[str, int]
    simulation_budget: int
    root_mode: str

    def as_dict(self):
        return {
            "physical_q": dict(self.physical_q),
            "query_q": dict(self.query_q),
            "visits": dict(self.visits),
            "simulation_budget": self.simulation_budget,
            "root_mode": self.root_mode,
        }


class RestrictedRootQueryPlanner(QueryAsActionPOMCPPlanner):
    """QaA planner whose root candidate family is controlled by the ablation."""

    root_mode = "compare"

    def _root_candidates(self, belief):
        task = [
            action for action in self.task_actions
            if action.is_applicable(belief.knowledge)
        ]
        queries = list(self.query_actions)
        self.root_query_names = {action.name for action in queries}
        candidates = queries if self.root_mode == "query_only" else task + queries
        self.root_candidate_names = {action.name for action in candidates}
        return candidates


class QueryValueEvaluator:
    """Estimate QaA root values without letting the evaluator execute an action."""

    def __init__(
        self,
        *,
        args,
        env,
        query_cost: float,
        answer_accuracy: float,
        failure_penalty: float,
        base_simulations: int,
    ):
        private_belief_manager = BeliefManager(
            args,
            env.transition_model,
            env.observation_model,
            env.asp_bridge,
        )
        self.planner = RestrictedRootQueryPlanner(
            args,
            env,
            private_belief_manager,
            query_cost=query_cost,
            answer_accuracy=answer_accuracy,
            failure_penalty=failure_penalty,
        )
        self.query_cost = float(query_cost)
        self.base_simulations = int(base_simulations)

    def _query_actions(self, facts: list[str]) -> list[QueryAction]:
        return [
            QueryAction(
                name=f"query_fact({fact})",
                target_fact=fact,
                query_schema="dynamic_fact",
                preconditions=[],
                cost=self.query_cost,
            )
            for fact in facts
        ]

    def evaluate(
        self,
        context: PolicyContext,
        *,
        query_facts: list[str],
        root_mode: str,
    ) -> RootValueResult:
        if root_mode not in {"compare", "query_only"}:
            raise ValueError(f"Unknown root mode: {root_mode}")
        facts = sorted(dict.fromkeys(query_facts))
        if not facts:
            raise ValueError("At least one query fact is required")

        queries = self._query_actions(facts)
        planner = self.planner
        planner.query_actions = queries
        planner.actions = planner.task_actions + queries
        planner.action_map = {action.name: action for action in planner.actions}
        planner.root_mode = root_mode

        physical_count = 0
        if root_mode == "compare":
            physical_count = sum(
                action.is_applicable(context.belief.knowledge)
                for action in planner.task_actions
            )
        candidate_count = len(queries) + physical_count
        # QueryAsActionPOMCPPlanner uses the first root simulation to initialize
        # the observation node with a rollout, before selecting a root action.
        # One additional simulation is therefore required to visit every root
        # candidate at least once.
        budget = max(self.base_simulations, candidate_count + 1)
        original_budget = planner.n_simulations
        planner.n_simulations = budget
        seed = derived_seed(
            context.seed,
            context.step,
            context.query_index,
            root_mode,
            *facts,
        )
        try:
            with isolated_rng(seed):
                planner.search(context.belief)
        finally:
            planner.n_simulations = original_budget

        query_by_name = {action.name: action.target_fact for action in queries}
        physical_q: dict[str, float] = {}
        query_q: dict[str, float] = {}
        visits: dict[str, int] = {}
        for action, node_id in planner.tree.get_action_children(planner.tree.root_id):
            node = planner.tree.get_node(node_id)
            if action.name not in planner.root_candidate_names:
                continue
            if node.visits <= 0:
                continue
            value = float(node.value)
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite root value for {action.name}: {value}")
            visits[action.name] = int(node.visits)
            if action.name in query_by_name:
                query_q[query_by_name[action.name]] = value
            else:
                physical_q[action.name] = value

        missing_queries = sorted(set(facts) - set(query_q))
        if missing_queries:
            raise RuntimeError(f"Unvisited root query actions: {missing_queries}")
        if root_mode == "compare" and physical_count and not physical_q:
            raise RuntimeError("No physical root action received a visit")
        return RootValueResult(
            physical_q=physical_q,
            query_q=query_q,
            visits=visits,
            simulation_budget=budget,
            root_mode=root_mode,
        )
