from __future__ import annotations

import random
import numpy as np
from numpy.random import binomial, choice

from models.state import State
from models.action import Action
from models.belief import Belief
from models.belief_update import BeliefManager
from models.transition import TransitionModel
from models.observation import ObservationModel, Observation
from models.reward import RewardModel
from planners.tree import POMDPTree
from environments.env import Environment



def UCB(N, n, V, c=1.0):
    if n == 0:
        return float("inf")   # exploration 우선

    return V + c * np.sqrt(np.log(max(1, N)) / n)


class POMCPPlanner:
    
    def __init__(self, args, env: Environment, belief_manager: BeliefManager):
        self.args = args
        
        # set parameters
        self.n_simulations = args.n_simulations
        self.gamma = 0.95
        self.epsilon = 0.005
        self.c = 1.0
        
        # call tree manager
        self.tree = POMDPTree()
        
        # all grounded actions
        self.actions = env.actions
        self.action_map = {action.name: action for action in self.actions}
        self.belief: Belief = None
        
        # model settings
        self.env: Environment = env
        self.belief_manager: BeliefManager = belief_manager
        self.transition_model: TransitionModel = self.belief_manager.transition_model
        self.observation_model: ObservationModel = self.belief_manager.observation_model
        self.reward_model: RewardModel = self.env.reward_model
        
        self.initialize(self.env.state)    
    
    def get_applicable_actions(self, knowledge):
        applicable_actions = [a for a in self.actions if a.is_applicable(knowledge)]    
        if not applicable_actions:
            return []
        else:
            return applicable_actions
    
    def initialize(self, init_state: State):
        self.belief = self.belief_manager.initialize_belief(init_state)
    
    

    def search(self, belief: Belief):
        
        history = self.tree.root_id
        
        for _ in range(self.n_simulations):
            
            state = belief.knowledge
            self.simulate(state=state, history=history, depth=0)

        best_action, _ = self.search_best(history, use_ucb=False)
        
        
        return best_action
                
    
    
    def rollout(self, state, depth):
        if (self.gamma ** depth < self.epsilon or self.gamma == 0) and depth != 0:
            return 0.0

        applicable_actions = self.get_applicable_actions(state)
        if not applicable_actions:
            return 0.0        
        
        action = random.choice(applicable_actions)
        
        sample_state = self.transition_model.sample_next_state(state, action)
        reward = self.reward_model.get_reward(state, action, sample_state)

        return_val = reward + self.gamma * self.rollout(sample_state, depth + 1)
        
        return return_val
    
    
    
    def simulate(self, state, history, depth):
        """
        Docstring for simulate
        
        :param state: 지금 시뮬레이션이 깔고 있는 상태가 무엇인가
        :param history: 현재 search tree의 어느 노드인가
        :param depth: Description
        """
        # 1. stopping condition
        if (self.gamma ** depth < self.epsilon or self.gamma == 0) and depth != 0:
            return 0.0
        
        # 2. leaf expansion
        if self.tree.is_leaf_node(history):
            actions = self.get_applicable_actions(state)
                        
            if not actions:
                self.tree.increment_visit(history)
                self.tree.set_value_if_first(history, 0.0)
                return 0.0
    
            for action in actions:
                self.tree.expand_tree_from(history, action, is_action=True)
                
            value = self.rollout(state, depth)
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, value)
            
            return value
        
        # 3. selection
        action, action_node = self.search_best(history, state)
                
        # 4. generative model
        next_state = self.transition_model.sample_next_state(state, action)
        observation = self.observation_model.sample(next_state, action)
        reward = self.reward_model.get_reward(state, action, next_state)
        
        # 5. observation child
        obs_node = self.tree.get_observation_node(action_node, observation)
        
        # 6. recursive simulation
        total_return = reward + self.gamma * self.simulate(next_state, obs_node, depth + 1)

        # 7. backup
        self.tree.increment_visit(history)
        self.tree.increment_visit(action_node)
        self.tree.update_action_value(action_node, total_return)

        return total_return
    
    
    def search_best(self, h, use_ucb=True) -> Action:
        node = self.tree.get_node(h)
        children = node.children

        if not children:
            return None, None

        best_action = None
        best_action_node = None
        best_score = -float("inf")

        parent_visits = max(1, node.visits)

        for action_key, action_node_id in children.items():
            action_node = self.tree.get_node(action_node_id)

            if use_ucb:
                if action_node.visits == 0:
                    score = float("inf")
                else:
                    score = UCB(
                        N=parent_visits,
                        n=action_node.visits,
                        V=action_node.value,
                        c=self.c
                    )
            else:
                score = action_node.visits

            if score > best_score:
                best_score = score
                best_action_key = action_key
                best_action_node = action_node_id

        # 트리에는 문자열 key가 저장되어 있으므로 다시 Action 객체로 복원        
        best_action = self.action_map[best_action_key]
    
        return best_action, best_action_node
    