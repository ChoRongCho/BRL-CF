from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from utils.asp import solve_asp
from models.state import State
from models.observation import Observation
from models.transition import TransitionOutcome
import numpy as np



@dataclass
class Belief:
    certain: State
    frontier: List[State]
    weight: np.ndarray[float]
    

class BeliefModel:
    def __init__(self):
        
        self.belief: Belief = None
        
    
    
    def update_belief(self, b_prev: Belief, obs, action):
        
        
        
        
        
        
        pass
    
    