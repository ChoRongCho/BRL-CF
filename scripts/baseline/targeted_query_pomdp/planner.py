"""Joint task/query POMCP solver for the Query-as-Action baseline."""

from __future__ import annotations

import random
import time

import numpy as np

from planners.pomcp import POMCPPlanner
from planners.tree import POMDPTree

from .query_actions import QueryAction, build_query_actions, fact_is_ambiguous


class QueryAsActionPOMCPPlanner(POMCPPlanner):
    """Plan over physical actions and fixed Boolean query actions together."""

    def __init__(self, args, env, belief_manager, *, query_cost, answer_accuracy):
        super().__init__(args=args, env=env, belief_manager=belief_manager)
        if not 0.5 <= answer_accuracy <= 1.0:
            raise ValueError("answer_accuracy must be in [0.5, 1.0]")
        self.task_actions = list(env.actions)
        self.query_actions = build_query_actions(
            env.domain_name,
            env.obj_type,
            cost=query_cost,
        )
        self.actions = self.task_actions + self.query_actions
        self.action_map = {action.name: action for action in self.actions}
        self.answer_accuracy = float(answer_accuracy)
        self._applicable_action_cache.clear()
        self.root_query_names: set[str] = set()

    def _sample_query_answer(self, state, action: QueryAction) -> bool:
        correct = state.has_fact(action.target_fact)
        if random.random() <= self.answer_accuracy:
            return correct
        return not correct

    def _generate(self, state, action):
        if isinstance(action, QueryAction):
            return state.copy(), self._sample_query_answer(state, action), -action.cost
        next_state = self.transition_model.sample_next_state(state, action)
        observation = self.observation_model.sample(next_state, action)
        reward = self.reward_model.get_reward(state, action, next_state)
        return next_state, observation, reward

    def _root_candidates(self, belief):
        task = [a for a in self.task_actions if a.is_applicable(belief.knowledge)]
        queries = [
            a for a in self.query_actions
            if a.is_applicable(belief.knowledge)
            and fact_is_ambiguous(belief.frontier, a.target_fact)
        ]
        self.root_query_names = {a.name for a in queries}
        return task + queries

    def search(self, belief):
        # The real posterior changes after each query. Rebuilding the root avoids
        # retaining particles from a different posterior history.
        self.tree = POMDPTree()
        self.belief = belief
        root = self.tree.root_id
        self.tree.get_node(root).knowledge = belief.knowledge.copy()
        root_candidates = self._root_candidates(belief)
        if not root_candidates:
            return None

        # BRL initializes the frontier with one empty sentinel State while the
        # actual initial state lives in ``knowledge``. Do not sample that
        # sentinel as a physical world.
        particles = list(belief.frontier)
        if particles and all(not state.facts and not state.fluents for state in particles):
            particles = []
        probabilities = np.asarray(belief.frontier_weights, dtype=float)
        if particles and probabilities.size == len(particles):
            total = float(probabilities.sum())
            probabilities = (
                probabilities / total if total > 0.0
                else np.full(len(particles), 1.0 / len(particles))
            )
        else:
            probabilities = np.array([], dtype=float)

        started = time.time()
        for _ in range(self.n_simulations):
            if particles and probabilities.size:
                index = int(np.random.choice(len(particles), p=probabilities))
                sampled = particles[index].copy()
            else:
                sampled = belief.knowledge.copy()
            self.tree.add_particle(root, sampled, self.max_node_particles)
            self.simulate(sampled, root, 0)
        elapsed = (time.time() - started) / max(1, self.n_simulations)
        print("[Query-as-Action] Average simulation time: ", round(elapsed, 4))

        candidates = [
            (action, node_id)
            for action, node_id in self.tree.get_action_children(root)
            if action.name in {item.name for item in root_candidates}
        ]
        if not candidates:
            return None
        print("[Query-as-Action] applicable values:")
        for action, node_id in candidates:
            print(f"        {action.name}: {self.tree.get_value(node_id)}")
        print()
        return max(candidates, key=lambda item: self.tree.get_value(item[1]))[0]

    def simulate(self, state, history, depth):
        if self._should_stop(depth):
            return 0.0
        applicable = self.get_applicable_actions(state)
        if not applicable:
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0
        if self.tree.is_leaf_node(history):
            self._expand_applicable_actions(history, state)
            value = self.rollout(state, depth)
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, value)
            return value

        action, action_node = self.search_best(history, state, use_ucb=True)
        if action is None:
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0
        next_state, observation, reward = self._generate(state, action)
        observation_node = self.tree.get_observation_node(action_node, observation)
        total = reward + self.gamma * self.simulate(next_state, observation_node, depth + 1)
        self.tree.add_particle(history, state, self.max_node_particles)
        self.tree.increment_visit(history)
        self.tree.increment_visit(action_node)
        self.tree.update_action_value(action_node, total)
        return total

    def rollout(self, state, depth):
        if self._should_stop(depth):
            return 0.0
        # Query actions remain in the explicit search tree. Physical-only
        # rollouts avoid long random chains of unrelated questions.
        applicable = [a for a in self.task_actions if a.is_applicable(state)]
        if not applicable:
            return 0.0
        action = random.choice(applicable)
        next_state, _, reward = self._generate(state, action)
        return reward + self.gamma * self.rollout(next_state, depth + 1)
