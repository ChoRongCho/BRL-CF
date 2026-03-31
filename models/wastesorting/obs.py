# models/wastesorting/obs.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from itertools import product
import re

from utils.utils import _dedup_facts, _parse_fact, _format_fact
from models.state import State
from models.action import Action
from models.observation import Observation, ObservationOutcome



class ObservationWastesorting:
    def __init__(self, type_map: Dict[str, List[str]], noise: float = 0.05):
        self.type_map = type_map
        self.noise = noise

        self.detect_success_rate = 0.85
        self.scan_success_rate = 0.95
        self.navigate_success_rate = 0.90


    






    def build_candidates(self, action: Action) -> List[str]:
        return action.observation.copy() if action.observation else []

    def sample(self, state: State, action: Action) -> Observation:
        return Observation(State())
