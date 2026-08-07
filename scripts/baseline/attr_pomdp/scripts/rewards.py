"""Belief-dependent rewards from Araya-Lopez et al. (NeurIPS 2010)."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np


def probability_vector(weights) -> np.ndarray:
    p = np.asarray(weights, dtype=float)
    if p.size == 0:
        return np.array([1.0])
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    return p / total if total > 0.0 else np.full(p.size, 1.0 / p.size)


def negative_entropy(weights) -> float:
    """Paper Eq. (2): log2(|S|) + sum_s b(s) log2 b(s)."""
    p = probability_vector(weights)
    nz = p[p > 0.0]
    return float(log2(p.size) + np.sum(nz * np.log2(nz)))


@dataclass(frozen=True)
class TangentPWLCEntropy:
    """PWLC lower bound formed by tangent hyperplanes (paper Sec. 4.2)."""

    epsilon: float = 1e-6

    def _bases(self, dimension: int) -> list[np.ndarray]:
        if dimension <= 1:
            return [np.ones(1)]
        uniform = np.full(dimension, 1.0 / dimension)
        bases = [uniform]
        for index in range(dimension):
            point = np.full(dimension, self.epsilon)
            point[index] = 1.0 - self.epsilon * (dimension - 1)
            bases.append(point)
        return bases

    @staticmethod
    def _value(point: np.ndarray) -> float:
        return negative_entropy(point)

    @staticmethod
    def _gradient(point: np.ndarray) -> np.ndarray:
        return (np.log(point) + 1.0) / np.log(2.0)

    def __call__(self, weights) -> float:
        p = probability_vector(weights)
        tangents = [
            self._value(base) + float(np.dot(p - base, self._gradient(base)))
            for base in self._bases(p.size)
        ]
        # Convex functions lie above their tangent planes.
        return float(max(tangents))


def entropy_reward(weights, mode: str, *, omit_cardinality_constant: bool = False) -> float:
    p = probability_vector(weights)
    if mode == "exact_entropy":
        value = negative_entropy(p)
    elif mode == "pwlc_entropy":
        value = TangentPWLCEntropy()(p)
    else:
        raise ValueError(f"Unknown rho reward: {mode}")
    # log|S| is constant in the paper's fixed state space. Particle support is
    # not |S| and changes by action, so never let log(support size) bias search.
    return value - log2(p.size) if omit_cardinality_constant else value
