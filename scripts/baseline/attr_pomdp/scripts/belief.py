"""Particle belief prediction and Bayesian updates for adapted Attr-POMDP."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from models.belief import Belief
from models.state import State
from models.transition import NextStateOutcome

from .query import QueryAction


def state_key(state: State):
    return (
        tuple(sorted(state.facts)),
        tuple(sorted((o, k, float(v)) for o, values in state.fluents.items() for k, v in values.items())),
    )


class ParticleBeliefModel:
    def __init__(
        self,
        transition_model,
        observation_model,
        max_particles: int = 250,
        max_outcomes_per_state: int = 8,
        max_observation_branches: int = 16,
    ):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.max_particles = max(1, int(max_particles))
        self.max_outcomes_per_state = max(1, int(max_outcomes_per_state))
        self.max_observation_branches = max(1, int(max_observation_branches))

    @staticmethod
    def materialize(belief: Belief):
        # The shared initializer uses State() as a lazy sentinel.
        if len(belief.particles) == 1 and belief.particles[0].get_size() == 0:
            return [belief.knowledge.copy()], np.array([1.0])
        return belief.particles, belief.particle_weights

    def transition_outcomes(self, state: State, action):
        action_name = action.name.replace(" ", "").split("(", 1)[0]
        domain = getattr(self.transition_model, "domain", "")
        if domain == "wastesorting" and action_name == "detect_waste":
            next_state = state.copy()
            for waste in self.transition_model.type_map.get("W", []):
                next_state.add_fact(f"detected({waste})")
            return [NextStateOutcome(next_state, 1.0)]
        if domain == "tomato" and action_name == "detect":
            next_state = state.copy()
            args = action.name[action.name.find("(") + 1:-1].replace(" ", "").split(",")
            stem = args[-1] if args else ""
            for tomato in self.transition_model.type_map.get("T", []):
                if state.has_fact(f"at({tomato},{stem})"):
                    next_state.add_fact(f"observed({tomato})")
            return [NextStateOutcome(next_state, 1.0)]

        outcomes = self.transition_model.get_next_state_distribution(state, action)
        if len(outcomes) <= self.max_outcomes_per_state:
            return outcomes
        return sorted(
            outcomes,
            key=lambda outcome: float(outcome.probability),
            reverse=True,
        )[: self.max_outcomes_per_state]

    def _predict_and_reward(self, belief: Belief, action, reward_model=None):
        states, weights = self.materialize(belief)
        merged = {}
        masses = defaultdict(float)
        expected_reward = 0.0
        for state, prior in zip(states, weights):
            if not action.is_applicable(state):
                continue
            outcomes = self.transition_outcomes(state, action)
            selected_mass = sum(float(outcome.probability) for outcome in outcomes)
            for outcome in outcomes:
                key = state_key(outcome.next_state)
                merged[key] = outcome.next_state
                probability = float(outcome.probability) / selected_mass if selected_mass > 0 else 0.0
                masses[key] += float(prior) * probability
                if reward_model is not None:
                    reward = reward_model.get_reward(state, action, outcome.next_state)
                    expected_reward += float(prior) * probability * float(reward)
        if not merged:
            predicted = Belief(
                belief.knowledge.copy(),
                [belief.knowledge.copy()],
                np.array([1.0]),
            )
            return predicted, expected_reward

        ranked = sorted(merged, key=masses.get, reverse=True)[: self.max_particles]
        result = Belief(merged[ranked[0]].copy(), [merged[k] for k in ranked], np.array([masses[k] for k in ranked]))
        result.sync_knowledge_to_map()
        return result, expected_reward

    def predict(self, belief: Belief, action) -> Belief:
        predicted, _ = self._predict_and_reward(belief, action)
        return predicted

    def update_predicted(self, predicted: Belief, action, observation) -> Belief:
        likelihoods = np.array([
            self.observation_model.likelihood(observation, state, action)
            for state in predicted.particles
        ], dtype=float)
        posterior = predicted.particle_weights * np.maximum(likelihoods, 0.0)
        if float(posterior.sum()) <= 0.0:
            posterior = predicted.particle_weights.copy()
        result = Belief(predicted.knowledge.copy(), [s.copy() for s in predicted.particles], posterior)
        result.sync_knowledge_to_map()
        return result

    def update(self, belief: Belief, action, observation) -> Belief:
        return self.update_predicted(self.predict(belief, action), action, observation)

    def query_update(self, belief: Belief, query: QueryAction, answer: bool, provider) -> Belief:
        """Apply a binary feedback observation without changing world state."""

        likelihoods = np.array(
            [provider.likelihood(answer, query, state) for state in belief.particles],
            dtype=float,
        )
        posterior = belief.particle_weights * np.maximum(likelihoods, 0.0)
        if float(posterior.sum()) <= 0.0:
            raise ValueError(
                f"Oracle answer {answer} for {query.target_fact} has zero likelihood "
                "under the current Attr-POMDP belief."
            )
        result = Belief(
            belief.knowledge.copy(),
            [state.copy() for state in belief.particles],
            posterior,
        )
        if answer:
            result.knowledge.add_fact(query.target_fact)
        return result

    def query_branches(self, belief: Belief, query: QueryAction, provider):
        """Return normalized yes/no branches for a query action."""

        branches = []
        for answer in (True, False):
            likelihoods = np.array(
                [provider.likelihood(answer, query, state) for state in belief.particles],
                dtype=float,
            )
            probability = float(np.dot(belief.particle_weights, likelihoods))
            if probability <= 0.0:
                continue
            posterior = Belief(
                belief.knowledge.copy(),
                [state.copy() for state in belief.particles],
                belief.particle_weights * likelihoods,
            )
            if answer:
                posterior.knowledge.add_fact(query.target_fact)
            branches.append((probability, answer, posterior))

        total = sum(probability for probability, _, _ in branches)
        if total <= 0.0:
            return []
        return [
            (probability / total, answer, posterior)
            for probability, answer, posterior in branches
        ]

    @staticmethod
    def _projected_observation_key(state: State, action) -> tuple[str, ...]:
        predicates = {
            str(pattern).replace(" ", "").split("(", 1)[0]
            for pattern in action.observation
        }
        return tuple(
            sorted(
                fact
                for fact in state.facts
                if fact.split("(", 1)[0] in predicates
            )
        )

    def _observation_branches(self, predicted: Belief, action):
        grouped: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, state in enumerate(predicted.particles):
            grouped[self._projected_observation_key(state, action)].append(index)

        ranked = sorted(
            grouped.items(),
            key=lambda item: -sum(float(predicted.particle_weights[i]) for i in item[1]),
        )[: self.max_observation_branches]
        branches = []
        for _, indices in ranked:
            mask = np.zeros(len(predicted.particles), dtype=float)
            mask[indices] = 1.0
            probability = float(np.dot(predicted.particle_weights, mask))
            if probability <= 0.0:
                continue
            posterior = Belief(
                predicted.knowledge.copy(),
                [state.copy() for state in predicted.particles],
                predicted.particle_weights * mask,
            )
            posterior.sync_knowledge_to_map()
            branches.append((probability, posterior))

        total = sum(probability for probability, _ in branches)
        if total <= 0.0:
            return [(1.0, predicted)]
        return [(probability / total, posterior) for probability, posterior in branches]

    def branches(self, belief: Belief, action):
        """Build deterministic task-observation branches."""
        return self._observation_branches(self.predict(belief, action), action)

    def evaluate_task(self, belief: Belief, action, reward_model, *, include_branches=True):
        """Compute transition branches and expected reward in one shared pass."""
        if not include_branches:
            states, weights = self.materialize(belief)
            reward = 0.0
            for state, prior in zip(states, weights):
                if not action.is_applicable(state):
                    continue
                outcomes = self.transition_outcomes(state, action)
                selected_mass = sum(float(outcome.probability) for outcome in outcomes)
                for outcome in outcomes:
                    probability = (
                        float(outcome.probability) / selected_mass
                        if selected_mass > 0.0
                        else 0.0
                    )
                    reward += (
                        float(prior)
                        * probability
                        * float(reward_model.get_reward(state, action, outcome.next_state))
                    )
            return reward, []

        predicted, reward = self._predict_and_reward(
            belief,
            action,
            reward_model,
        )
        return reward, self._observation_branches(predicted, action)
