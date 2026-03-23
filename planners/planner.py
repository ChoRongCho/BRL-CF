from __future__ import annotations

from collections import defaultdict
from utils.asp import DomainRuleBridge
from typing import Any, Dict, Tuple
import copy
import yaml

from models.action import Action, ActionSchema, Grounding


class Planner:
    def __init__(self, actions):
        pass
    
    def sample_action(self, belief):
        return "Dummy"