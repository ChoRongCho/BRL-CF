from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from models.state import State







@dataclass
class Belief:
    knowledge: State
    particles: List[State]
    particle_weights: np.ndarray

    def __post_init__(self):
        if not isinstance(self.particles, list):
            self.particles = list(self.particles)

        if not isinstance(self.particle_weights, np.ndarray):
            self.particle_weights = np.array(self.particle_weights, dtype=float)
        else:
            self.particle_weights = self.particle_weights.astype(float)

        if len(self.particles) != len(self.particle_weights):
            raise ValueError(
                f"Belief mismatch: len(particles)={len(self.particles)} "
                f"!= len(weights)={len(self.particle_weights)}"
            )

        if self.is_empty():
            self.particle_weights = np.array([], dtype=float)
            return

        self.normalize()






    @property
    def frontier(self) -> List[State]:
        return self.particles

    @frontier.setter
    def frontier(self, value: List[State]) -> None:
        self.particles = list(value)

    @property
    def frontier_weights(self) -> np.ndarray:
        return self.particle_weights

    @frontier_weights.setter
    def frontier_weights(self, value) -> None:
        self.particle_weights = np.array(value, dtype=float)

    def normalize(self) -> None:
        if len(self.particle_weights) == 0:
            return

        total = float(np.sum(self.particle_weights))
        if total <= 0.0:
            self.particle_weights = np.ones(len(self.particles), dtype=float) / len(self.particles)
        else:
            self.particle_weights = self.particle_weights / total

    def reset_belief(self) -> None:
        self.particles = [self.knowledge.copy()]
        self.particle_weights = np.array([1.0], dtype=float)

    def is_empty(self) -> bool:
        return len(self.particles) == 0


    def sync_knowledge_to_map(self) -> None:
        if self.is_empty():
            return State()

        idx = int(np.argmax(self.particle_weights))
        self.knowledge = self.particles[idx].copy()


    def pretty_print(self, digits: int = 4) -> str:
        if self.is_empty():
            return "Belief(empty)"

        lines = [f"Belief(knowledge={self.knowledge})"]
        for i, (state, w) in enumerate(zip(self.particles, self.particle_weights)):
            lines.append(f"  [{i}] p={w:.{digits}f} | {state}")
        return "\n".join(lines)

    def topk(self, k: int = 5) -> List[Tuple[State, float]]:
        if self.is_empty():
            return []

        pairs = list(zip(self.particles, self.particle_weights))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]
