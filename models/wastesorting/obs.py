from __future__ import annotations

from typing import Dict, List
import random

from models.state import State
from models.action import Action
from models.observation import Observation


class ObservationWastesorting:
    def __init__(self, type_map: Dict[str, List[str]], noise: float = 0.15):
        self.type_map = type_map
        self.noise = noise

    def build_candidates(self, action: Action) -> List[str]:
        return action.observation.copy() if action.observation else []

    def sample(self, state: State, action: Action) -> Observation:
        return Observation(State())

    def likelihood(self, observation: Observation, state: State, action: Action) -> float:
        return 1.0