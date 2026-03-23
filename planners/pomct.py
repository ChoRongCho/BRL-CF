# planners/pomct.py

import random
import math

from models.action import Action
from models.transition import TransitionModel
from models.observation import ObservationModel
from models.reward import RewardModel

"""
핵심 구조 요약

이 코드가 하는 일은 다음과 같습니다.

belief → particle로 샘플링

simulation → 하나의 state trajectory

tree → history 기반으로 확장

Q 업데이트 → Monte Carlo 평균
"""

class Node:
    def __init__(self):
        self.N = 0  # visit count
        self.children = {}  # action -> ActionNode


class ActionNode:
    def __init__(self):
        self.N = 0
        self.Q = 0.0
        self.children = {}  # observation -> Node
        
        
class POMCP:
    def __init__(
        self,
        actions: Action,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        reward_model: RewardModel,
        gamma=0.95,
        c=1.0,
        max_depth=10,
    ):
        self.actions = actions
        self.T = transition_model
        self.O = observation_model
        self.R = reward_model

        self.gamma = gamma
        self.c = c
        self.max_depth = max_depth

        self.root = Node()
        
        
    def search(self, belief_particles, n_simulations=1000):
        for _ in range(n_simulations):
            s = random.choice(belief_particles)
            self.simulate(s, self.root, depth=0)

        # best action 선택
        best_action = max(
            self.root.children.items(),
            key=lambda x: x[1].Q
        )[0]

        return best_action

    def simulate(self, state, node, depth):
        if depth >= self.max_depth:
            return 0.0

        # leaf node면 rollout
        if not node.children:
            return self.rollout(state, depth)

        # UCB로 action 선택
        action = self.select_action(node)

        action_node = node.children[action]

        # generative model
        next_state, observation, reward = self.generative_model(state, action)

        # 다음 노드
        if observation not in action_node.children:
            action_node.children[observation] = Node()

        next_node = action_node.children[observation]

        # recursive
        value = reward + self.gamma * self.simulate(next_state, next_node, depth + 1)

        # 업데이트
        action_node.N += 1
        node.N += 1

        action_node.Q += (value - action_node.Q) / action_node.N

        return value

    
    def rollout(self, state, depth):
        if depth >= self.max_depth:
            return 0.0

        action = random.choice(self.actions)
        next_state, observation, reward = self.generative_model(state, action)
        return reward + self.gamma * self.rollout(next_state, depth + 1)
    
    
    def select_action(self, node):
        best_score = -float("inf")
        best_action = None

        for action in self.actions:
            if action not in node.children:
                node.children[action] = ActionNode()

            a_node = node.children[action]

            if a_node.N == 0:
                return action

            ucb = a_node.Q + self.c * math.sqrt(
                math.log(node.N + 1) / a_node.N
            )

            if ucb > best_score:
                best_score = ucb
                best_action = action

        return best_action
    
    
    # TODO
    def generative_model(self, state, action):
        # transition
        outcomes = self.T.get_next_state_distribution(state, action)

        next_states = []
        probs = []

        for o in outcomes:
            next_states.append(o.next_state)
            probs.append(o.probability)

        next_state = random.choices(next_states, weights=probs)[0]

        # observation
        observation = self.O.sample(next_state, action)

        # reward
        reward = self.R.get_reward(state, action, next_state)

        return next_state, observation, reward
    

    def update_root(self, action, observation):
        if action in self.root.children:
            a_node = self.root.children[action]
            if observation in a_node.children:
                self.root = a_node.children[observation]
                return

        # 없으면 새로
        self.root = Node()