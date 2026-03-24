# planners/planner.py

from __future__ import annotations

import random
from typing import List, Optional

from models.action import Action
from models.belief import Belief
from models.transition import TransitionModel
from models.observation import ObservationModel
from models.reward import RewardModel
from planners.pomct import POMCP


class Planner:
    def __init__(
        self,
        args,
        actions: List[Action],
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        reward_model: RewardModel,
    ):
        self.actions = actions
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model

        self.gamma = args.gamma
        self.c = args.c
        self.max_depth = args.max_depth
        self.n_simulations = args.n_simulations
        self.n_particles = args.n_particles

        self.pomcp = POMCP(
            actions=self.actions,
            transition_model=self.transition_model,
            observation_model=self.observation_model,
            reward_model=self.reward_model,
            gamma=self.gamma,
            c=self.c,
            max_depth=self.max_depth,
        )
        

    def reset(self) -> None:
        self.pomcp.root = self.pomcp.root.__class__()

    def sample_action(self, belief: Belief) -> Action:
        """
        Belief(frontier, weights)를 받아
        weighted particle list로 변환한 뒤
        POMCP.search()를 호출해서 action을 반환한다.
        """
        belief_particles = self._belief_to_particles(belief)

        if not belief_particles:
            raise ValueError("belief_particles is empty.")

        action = self.pomcp.search(
            belief_particles=belief_particles,
            n_simulations=self.n_simulations
        )
        return action

    def update(self, action: Action, observation) -> None:
        """
        실제 환경에서 action 실행 후 받은 observation으로
        POMCP tree root를 갱신한다.
        """
        self.pomcp.update_root(action, observation)

    def _belief_to_particles(self, belief: Belief):
        """
        belief.frontier와 belief.weights를 이용하여
        weighted sampling 기반 particle list를 만든다.
        """
        if belief is None:
            return []

        if len(belief.frontier) == 0:
            return []

        if len(belief.frontier) != len(belief.weights):
            raise ValueError(
                f"Belief mismatch: len(frontier)={len(belief.frontier)} "
                f"!= len(weights)={len(belief.weights)}"
            )

        total_weight = float(sum(belief.weights))
        if total_weight <= 0.0:
            # fallback: 균등 샘플링
            return random.choices(
                population=belief.frontier,
                k=self.n_particles
            )

        return random.choices(
            population=belief.frontier,
            weights=belief.weights,
            k=self.n_particles
        )