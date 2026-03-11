# models/belief_state.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List
import math
import random



class BaseBeliefState(ABC):
    """
    Abstract interface for belief representations.

    Planner and model code should only rely on this interface,
    not on a specific representation.
    """

    @abstractmethod
    def prob(self, fact: str) -> float:
        """Return belief probability of a symbolic fact."""
        pass

    @abstractmethod
    def set_prob(self, fact: str, p: float) -> None:
        """Set probability of a fact."""
        pass

    @abstractmethod
    def facts(self) -> Iterable[str]:
        """Return all tracked facts."""
        pass

    @abstractmethod
    def copy(self) -> "BaseBeliefState":
        """Return a copy of the belief state."""
        pass

    @abstractmethod
    def sample_state(self) -> List[str]:
        """
        Sample a possible world state from the belief.
        Returns a list of facts that are true in that sample.
        """
        pass
    
    
    
    
@dataclass
class FactorizedBeliefState(BaseBeliefState):
    """
    Factorized belief representation.

    Each fact is treated as an independent Bernoulli variable.

    Example:
        {
            "tomato_exists_stem1": 0.8,
            "ripe_stem1": 0.5,
            "robot_at_stem1": 1.0
        }
    """

    fact_probs: Dict[str, float] = field(default_factory=dict)

    def prob(self, fact: str) -> float:
        return float(self.fact_probs.get(fact, 0.0))

    def set_prob(self, fact: str, p: float) -> None:
        p = max(0.0, min(1.0, float(p)))
        self.fact_probs[fact] = p

    def facts(self) -> Iterable[str]:
        return self.fact_probs.keys()

    def copy(self) -> "FactorizedBeliefState":
        return FactorizedBeliefState(dict(self.fact_probs))

    def sample_state(self) -> List[str]:
        """
        Sample a world state assuming independent facts.
        """
        sampled = []
        for fact, p in self.fact_probs.items():
            if random.random() < p:
                sampled.append(fact)
        return sampled

    def entropy(self, fact: str) -> float:
        """
        Shannon entropy of a fact.
        Useful for measuring uncertainty.
        """
        p = self.prob(fact)
        if p in (0.0, 1.0):
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    def summary(self) -> Dict[str, float]:
        """Return a copy of belief probabilities."""
        return dict(self.fact_probs)
    
    
    
    
    
@dataclass
class ParticleBeliefState(BaseBeliefState):
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    