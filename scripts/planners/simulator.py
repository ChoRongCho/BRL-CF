from __future__ import annotations

from dataclasses import dataclass
import random

from models.action import Action
from models.observation import Observation
from models.reward import RewardModel
from models.state import State
from models.transition import TransitionModel


@dataclass(slots=True)
class SimulationStep:
    next_state: State
    observation: Observation
    reward: float


class Simulator:
    """
    Generative model used only inside POMCP planning.

    This keeps transition, observation, and reward sampling behind one API so
    simulate() and rollout() use the same black-box model.  It must not be used
    for real environment execution.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
    ):
        self.transition_model = transition_model
        self.reward_model = reward_model

    def sample(self, state: State, action: Action) -> SimulationStep:
        next_state = self._sample_transition_candidate(state, action)
        observation = Observation(next_state.copy())
        reward = self.reward_model.get_reward(state, action, next_state)
        return SimulationStep(
            next_state=next_state,
            observation=observation,
            reward=reward,
        )

    def _sample_transition_candidate(self, state: State, action: Action) -> State:
        candidates = self.transition_model.get_next_state_distribution(state, action)
        if not candidates:
            return state.copy()

        candidate = random.choices(
            candidates,
            weights=[candidate.probability for candidate in candidates],
            k=1,
        )[0]
        return candidate.next_state
