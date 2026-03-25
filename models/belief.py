from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from models.state import State



@dataclass
class Belief:
    knowledge: State
    frontier: List[State]                   # reachable states (support)
    frontier_weights: np.ndarray                     # 각 state의 확률 (normalize됨)

    def __post_init__(self):  # @dataclass의 __init__
        if not isinstance(self.frontier, list):
            self.frontier = list(self.frontier)

        if not isinstance(self.frontier_weights, np.ndarray):
            self.frontier_weights = np.array(self.frontier_weights, dtype=float)
        else:
            self.frontier_weights = self.frontier_weights.astype(float)

        if len(self.frontier) != len(self.frontier_weights):
            raise ValueError(
                f"Belief mismatch: len(frontier)={len(self.frontier)} "
                f"!= len(weights)={len(self.frontier_weights)}"
            )

        if len(self.frontier) == 0:
            self.weights = np.array([], dtype=float)
            return

        self.normalize()

    def normalize(self) -> None:
        """
        # 모든 확률이 0이 되면 균등분포로 fallback
        
        :param self: Description
        """
        if len(self.frontier_weights) == 0:
            return

        total = float(np.sum(self.frontier_weights))
        if total <= 0.0:
            self.frontier_weights = np.ones(len(self.frontier), dtype=float) / len(self.frontier)
        else:
            self.frontier_weights = self.frontier_weights / total

    def reset_belief(self):
        self.frontier = [State()]
        self.frontier_weights = np.array([1.0])
    
    
    def is_empty(self) -> bool:
        return len(self.frontier) == 0


    def pretty_print(self, digits: int = 4) -> str:
        if self.is_empty():
            return "Belief(empty)"

        lines = ["Belief:"]
        for i, (state, w) in enumerate(zip(self.frontier, self.frontier_weights)):
            lines.append(f"  [{i}] p={w:.{digits}f} | {state}")
        return "\n".join(lines)


    def topk(self, k: int = 5) -> List[Tuple[State, float]]:
        if self.is_empty():
            return []

        pairs = list(zip(self.frontier, self.frontier_weights))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]

