from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from utils.asp import solve_asp


class Belief:
    """
    1. asp program을 읽고, possible worlds 를 생성
    2. 
    
    """
    
    def __init__(self, args):
        
        self.args = args
        self.possible_worlds = 0
        
    
    
    def generate_possible_worlds(self, program):
        worlds = solve_asp(program=program)
        self.possible_worlds = len(worlds)
        
        return worlds[0]
    
    # def     