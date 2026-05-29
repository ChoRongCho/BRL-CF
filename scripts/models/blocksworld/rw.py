# models/blocksworld/rw.py

from __future__ import annotations
from itertools import product
from typing import List, Dict, Tuple
import re

from models.state import State
from models.action import Action
from models.transition import TransitionOutcome
from models.reward import Reward


class RewardBlocksworld:
    def __init__(self):
        pass
    
    
    def calculate_reward(state, action, next_state):
        return 1