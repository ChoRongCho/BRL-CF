from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

from models.state import State
from models.action import Action


@dataclass(slots=True)
class TransitionOutcome:
    next_state: State
    probability: float


# TODO
class TransitionModel:
    def __init__(self, domain, actions: List[Action]):
        self.domain = domain
        self.actions = actions
        
        self.load_transition()
        
        if self.domain == "tomato":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "blocksworld":  # TODO
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "wastesorting":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        else:
            raise ValueError("Domain is worng")


    def load_transition(self):
        
        if self.domain == "tomato":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "blocksworld":  # TODO
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "wastesorting":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        else:
            raise ValueError("Domain is worng")
    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_next_state_distribution(self, state: State, action: Action) -> List[TransitionOutcome]:
        """핵심: P(s' | s, a) 분포를 반환"""
        if not action.is_applicable(state):
            return [TransitionOutcome(state.copy(), 1.0)]

        success_state = state.copy()
        for fact in action.del_effects:
            success_state.remove_fact(fact)
        for fact in action.add_effects:
            success_state.add_fact(fact)

        if not self.stochastic:
            return [TransitionOutcome(success_state, 1.0)]

        success_prob = 0.9
        for prefix, prob in self.success_probs.items():
            if action.name.startswith(prefix):
                success_prob = prob
                break

        fail_state = self.generate_failed_state(action, state)

        # return transition distribution
        if self._key(success_state) == self._key(fail_state):
            return [TransitionOutcome(success_state, 1.0)]

        return [
            TransitionOutcome(success_state, success_prob),
            TransitionOutcome(fail_state, 1.0 - success_prob),
        ]

    def sample_next_state(self, state: State, action: Action) -> State:
        """실제 시뮬레이션용 샘플링"""
        dist = self.get_next_state_distribution(state, action)

        if len(dist) == 1:
            return dist[0].next_state

        # 이거 역할은 뭐지? , state가 2개 이상일때 동작하기 위한 general code
        r = random.random()
        cumulative = 0.0
        for outcome in dist:
            cumulative += outcome.probability
            if r <= cumulative:
                return outcome.next_state

        return dist[-1].next_state

    def _key(self, state: State) -> Tuple[str, ...]:
        return tuple(sorted(state.facts))
    
    def generate_failed_state(self, action: Action, state: State):
        """
        Assume actions is applicable
        
        :param self: Description
        :param action: Description
        :type action: Action
        """
        if self.domain == "tomato":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "blocksworld":  # TODO
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        elif self.domain == "wastesorting":
            self.success_probs = {
                "navigate": 0.95,
                "pick": 0.85,
                "place": 1.00,
                "discard": 1.00,
                "prepare_nav": 1.0,
                "detect": 1.0,
                "scan": 1.0,
            }
        else:
            raise ValueError("Domain is worng")
        
        return state.copy()  # temporaly
