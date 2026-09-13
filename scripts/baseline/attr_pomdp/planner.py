"""Finite-horizon Attr-POMDP belief-tree planner.

This is an independent reimplementation of the model in Yang, Lou, and Choi,
"Interactive Robotic Grasping with Attribute-Guided Disambiguation", ICRA 2022.
It is not copied from, or claimed to be, the authors' official source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class QuestionAction:
    """A question and its conditional observation model.

    ``likelihoods[i][j]`` is P(observation_j | hidden candidate_i, action).
    """

    kind: str
    key: Hashable
    observations: tuple[Hashable, ...]
    likelihoods: tuple[tuple[float, ...], ...]
    cost: float


@dataclass(frozen=True)
class Decision:
    kind: str
    value: float
    candidate_index: int | None = None
    question: QuestionAction | None = None


class AttrPOMDPPlanner:
    """Depth-limited exact belief-tree search for small discrete beliefs."""

    def __init__(
        self,
        *,
        depth: int = 3,
        correct_commit_reward: float = 1.0,
        wrong_commit_reward: float = -1.0,
        discount: float = 1.0,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        self.depth = depth
        self.correct_commit_reward = float(correct_commit_reward)
        self.wrong_commit_reward = float(wrong_commit_reward)
        self.discount = float(discount)

    @staticmethod
    def normalize(belief: Sequence[float]) -> np.ndarray:
        values = np.asarray(belief, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("belief must be a non-empty one-dimensional vector")
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("belief must contain finite non-negative values")
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(values), 1.0 / len(values), dtype=float)
        return values / total

    def commit_value(self, belief: np.ndarray, candidate_index: int) -> float:
        probability_correct = float(belief[candidate_index])
        return (
            probability_correct * self.correct_commit_reward
            + (1.0 - probability_correct) * self.wrong_commit_reward
        )

    @staticmethod
    def observation_probability(
        belief: np.ndarray,
        action: QuestionAction,
        observation_index: int,
    ) -> float:
        likelihood = np.asarray(action.likelihoods, dtype=float)[:, observation_index]
        return float(np.dot(belief, likelihood))

    def update_belief(
        self,
        belief: Sequence[float],
        action: QuestionAction,
        observation: Hashable,
    ) -> np.ndarray:
        belief_array = self.normalize(belief)
        try:
            observation_index = action.observations.index(observation)
        except ValueError as exc:
            raise ValueError(f"unknown observation {observation!r}") from exc
        likelihood = np.asarray(action.likelihoods, dtype=float)[:, observation_index]
        posterior = belief_array * likelihood
        total = float(posterior.sum())
        return belief_array.copy() if total <= 0.0 else posterior / total

    def plan(
        self,
        belief: Sequence[float],
        questions: Sequence[QuestionAction],
    ) -> Decision:
        belief_array = self.normalize(belief)
        candidate_count = len(belief_array)
        for action in questions:
            likelihoods = np.asarray(action.likelihoods, dtype=float)
            if likelihoods.shape != (candidate_count, len(action.observations)):
                raise ValueError(
                    f"invalid likelihood shape for {action.key!r}: {likelihoods.shape}"
                )
            if np.any(likelihoods < 0.0):
                raise ValueError("observation likelihoods must be non-negative")
            if not np.allclose(likelihoods.sum(axis=1), 1.0):
                raise ValueError("each observation distribution must sum to one")

        @lru_cache(maxsize=None)
        def value(belief_key: tuple[float, ...], depth: int) -> Decision:
            current = self.normalize(belief_key)
            commit_index = int(np.argmax(current))
            best = Decision(
                kind="commit",
                candidate_index=commit_index,
                value=self.commit_value(current, commit_index),
            )
            if depth == 0:
                return best

            for action in questions:
                expected_future = 0.0
                for obs_index, observation in enumerate(action.observations):
                    probability = self.observation_probability(
                        current,
                        action,
                        obs_index,
                    )
                    if probability <= 0.0:
                        continue
                    posterior = self.update_belief(current, action, observation)
                    posterior_key = tuple(float(f"{item:.14g}") for item in posterior)
                    expected_future += probability * value(
                        posterior_key,
                        depth - 1,
                    ).value
                question_value = -action.cost + self.discount * expected_future
                if question_value > best.value + 1e-12:
                    best = Decision(
                        kind="ask",
                        question=action,
                        value=question_value,
                    )
            return best

        start_key = tuple(float(f"{item:.14g}") for item in belief_array)
        return value(start_key, self.depth)


def binary_attribute_question(
    key: Hashable,
    candidate_has_attribute: Sequence[bool],
    *,
    accuracy: float = 0.99,
    cost: float = 0.1,
) -> QuestionAction:
    """Build the paper's cooperative-user binary observation model."""
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0.5 and 1.0")
    likelihoods = tuple(
        (1.0 - accuracy, accuracy) if has_attribute else (accuracy, 1.0 - accuracy)
        for has_attribute in candidate_has_attribute
    )
    return QuestionAction(
        kind="attribute",
        key=key,
        observations=(False, True),
        likelihoods=likelihoods,
        cost=float(cost),
    )


def pointing_question(
    candidate_index: int,
    candidate_count: int,
    *,
    accuracy: float = 0.99,
    cost: float = 0.3,
) -> QuestionAction:
    """Build AskPoint(x_b), retained for paper-model completeness."""
    attribute_model = binary_attribute_question(
        ("candidate", candidate_index),
        [index == candidate_index for index in range(candidate_count)],
        accuracy=accuracy,
        cost=cost,
    )
    return QuestionAction(
        kind="point",
        key=attribute_model.key,
        observations=attribute_model.observations,
        likelihoods=attribute_model.likelihoods,
        cost=attribute_model.cost,
    )
