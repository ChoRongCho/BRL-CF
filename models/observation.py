from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random

from models.state import State
from models.action import Action


@dataclass(slots=True)
class Observation:
    facts: State

    def is_empty(self) -> bool:
        return len(self.facts) == 0

    def __repr__(self) -> str:
        return f"Observation({self.facts})"

# TODO
class ObservationModel:
    def __init__(self, actions: List[Action], noise: float = 0.15):
        self.noise = noise
        self.action_observation_space: Dict[str, List[str]] = {}
        self._build_from_actions(actions)

    def _build_from_actions(self, actions: List[Action]) -> None:
        """
        grounded action 리스트를 받아
        action.name -> 가능한 observation fact 후보들
        형태의 테이블을 만든다.
        """
        for action in actions:
            obs_candidates = getattr(action, "observation", []) or []
            self.action_observation_space[action.name] = obs_candidates.copy()


    def get_observation_candidates(self, action: Action) -> List[str]:
        return self.action_observation_space.get(action.name, [])


    def sample(self, state: State, action: Action) -> Observation:
        """
        현재 state와 action을 보고 observation을 샘플링한다.

        기본 정책:
        1. action이 observation 후보를 가지고 있지 않으면 빈 observation 반환
        2. observation 후보 중 현재 state에 실제로 있는 fact는 높은 확률로 관측
        3. 실제로 없는 fact는 낮은 확률로 false positive 관측
        """
        candidates = self.get_observation_candidates(action)

        if not candidates:
            return Observation([])

        observed: State = State()

        for fact in candidates:
            if state.has_fact(fact):
                if random.random() > self.noise:
                    observed.add_fact(fact)
            else:
                if random.random() < self.noise:
                    observed.add_fact(fact)

        return Observation(observed)


    def likelihood(self, observation: Observation, state: State, action: Action) -> float:
        """
        P(o | s, a)를 대충 계산하는 골격.
        지금은 independent Bernoulli 가정으로 매우 러프하게 계산.
        """
        candidates = self.get_observation_candidates(action)

        if not candidates:
            return 1.0 if observation.is_empty() else 0.0

        obs_set = set(observation.facts)
        prob = 1.0

        for fact in candidates:
            in_state = state.has_fact(fact)
            observed = fact in obs_set

            if in_state:
                prob *= (1.0 - self.noise) if observed else self.noise
            else:
                prob *= self.noise if observed else (1.0 - self.noise)

        return prob