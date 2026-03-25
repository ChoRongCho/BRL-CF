"""
Source from https://github.com/GeorgePik/POMCP.git

"""
from __future__ import annotations

import random
from typing import List, Any
import numpy as np
from numpy.random import binomial, choice
from joblib import Parallel, delayed, parallel_backend
import multiprocessing

from models.state import State
from models.action import Action
from models.belief import Belief
from models.transition import TransitionModel
from models.observation import Observation, ObservationModel
from models.reward import RewardModel
from planners.auxilliary import BuildTree, UCB

# planners/pomct.py


from planners.auxilliary import BuildTree, UCB


class POMCP:
    def __init__(self,
                 actions: List[Action],
                 transition_model: TransitionModel,
                 observation_model: ObservationModel,
                 reward_model: RewardModel,
                 gamma: float = 0.95,
                 c: float = 1.0,
                 threshold: float = 0.005,
                 max_time: int = 10000,
                 no_particles: int = 1200):
        
        self.actions = actions
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model

        self.gamma = gamma
        if gamma >= 1:
            raise ValueError("gamma should be less than 1.")

        self.e = threshold
        self.c = c
        self.max_time = max_time
        self.no_particles = no_particles
        self.tree = BuildTree()


    def initialize(self, belief: Belief, observations: Observation):
        self.belief = belief if belief is not None else None
        self.observations = observations if observations is not None else None


    def _sample_generative_model(self, state, action):
        """
        기존 Generator(state, action)을 대체하는 함수.
        반환: next_state, observation, reward
        """
        # 1. transition 샘플링
        next_state = self.transition_model.sample_next_state(state, action)

        # 2. observation 샘플링
        observation = self.observation_model.sample(next_state, action)
        
        # 3. reward 계산
        reward = self.reward_model.get_reward(state, action, next_state)

        return next_state, observation, reward


    def search_best(self, h, UseUCB=True):
        max_value = None
        result = None
        resulta = None

        if UseUCB:
            if self.tree.nodes[h][4] != -1:
                children = self.tree.nodes[h][1]
                for action, child in children.items():
                    if self.tree.nodes[child][2] == 0:
                        return action, child

                    ucb = UCB(
                        self.tree.nodes[h][2],
                        self.tree.nodes[child][2],
                        self.tree.nodes[child][3],
                        self.c,
                    )

                    if max_value is None or max_value < ucb:
                        max_value = ucb
                        result = child
                        resulta = action
            return resulta, result

        else:
            if self.tree.nodes[h][4] != -1:
                children = self.tree.nodes[h][1]
                for action, child in children.items():
                    node_value = self.tree.nodes[child][3]
                    if max_value is None or max_value < node_value:
                        max_value = node_value
                        result = child
                        resulta = action
            return resulta, result




    def search(self):
        belief_history = self.tree.nodes[-1][4].copy()

        for _ in range(self.max_time):
            if not belief_history:
                if not self.states:
                    raise ValueError(
                        "Root belief is empty and no fallback state set exists. "
                        "Initialize particles or provide states."
                    )
                s = random.choice(self.states)
                
            else:
                s = random.choice(belief_history)

            self.simulate(s, -1, 0)

        action, _ = self.search_best(-1, UseUCB=False)
        return action





    def get_observation_node(self, h, sample_observation): # h = next_node
        if sample_observation not in list(self.tree.nodes[h][1].keys()):
            self.tree.expand_tree_from(h, sample_observation)

        next_node = self.tree.nodes[h][1][sample_observation]
        return next_node


    def rollout(self, s, depth):
        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0.0

        action = random.choice(self.actions)
        sample_state, _, reward = self._sample_generative_model(s, action)

        return reward + self.gamma * self.rollout(sample_state, depth + 1)

    def simulate(self, s, h, depth):
        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0.0

        if self.tree.isLeafNode(h):
            for action in self.actions:
                self.tree.expand_tree_from(h, action, IsAction=True)

            new_value = self.rollout(s, depth)
            self.tree.nodes[h][2] += 1
            self.tree.nodes[h][3] = new_value
            return new_value

        next_action, next_node = self.search_best(h, UseUCB=True)

        sample_state, sample_observation, reward = self._sample_generative_model(
            s, next_action
        )

        obs_node = self.get_observation_node(next_node, sample_observation)

        cum_reward = reward + self.gamma * self.simulate(
            sample_state, obs_node, depth + 1
        )

        self.tree.nodes[h][4].append(s)
        if len(self.tree.nodes[h][4]) > self.no_particles:
            self.tree.nodes[h][4] = self.tree.nodes[h][4][1:]

        self.tree.nodes[h][2] += 1
        self.tree.nodes[next_node][2] += 1
        self.tree.nodes[next_node][3] += (
            cum_reward - self.tree.nodes[next_node][3]
        ) / self.tree.nodes[next_node][2]

        return cum_reward


    def posterior_sample(self, Bh, action, observation):
        if not Bh:
            if not self.belief:
                raise ValueError(
                    "Belief is empty and no fallback state set exists."
                )
            s = random.choice(self.belief)
        else:
            s = random.choice(Bh)

        s_next, o_next, _ = self._sample_generative_model(s, action)

        if o_next == observation:
            return s_next

        return self.posterior_sample(Bh, action, observation)

    def update_belief(self, action, observation):
        prior = self.tree.nodes[-1][4].copy()
        self.tree.nodes[-1][4] = []

        for _ in range(self.no_particles):
            sampled_state = self.posterior_sample(prior, action, observation)
            self.tree.nodes[-1][4].append(sampled_state)