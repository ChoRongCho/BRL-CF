from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import re

import clingo
import yaml

from utils.asp import solve_asp

class Belief:
    def __init__(self):
        # self.domain_rule = domain_rule
        self.possible_worlds = 0
        
    
    
    def generate_possible_worlds(self, program):
        worlds = solve_asp(program=program)
        self.possible_worlds = len(worlds)
        
        return worlds[0]
    
    