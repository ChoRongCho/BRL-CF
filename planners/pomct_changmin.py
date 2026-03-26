from __future__ import annotations

import random
import numpy as np
from numpy.random import binomial, choice
from typing import List

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
    
    
    def _ensure_applicable_children(self, history, state):
        applicable = self.get_applicable_actions(state)
        existing = self.tree.get_action_children(history)
        existing_names = {a.name for a, _ in existing}

        for action in applicable:
            if action.name not in existing_names:
                self.tree.expand_tree_from(history, action, is_action=True)
                
    def search(self, belief: Belief):
        
        self.tree.reset()
        history = self.tree.root_id
        
        # for kb in belief.knowledge.facts:
        #     print(kb)
        
        for k in range(self.n_simulations):
            state = belief.knowledge.copy()
            self.simulate(state=state, history=history, depth=0)
        
        
        candidates = self.tree.get_action_children(history)
        
        for action, node_id in candidates:
            print(
                f"[ROOT] action={action.name} "
                f"visits={self.tree.get_visit(node_id)} "
                f"value={self.tree.get_value(node_id):.4f}"
            )
            
        best_action, _ = self.search_best(history, state, use_ucb=False)        
        print("Selected:", best_action.name if best_action else None)
        
        return best_action
                
    
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
        self._ensure_applicable_children(history, state)

        candidates = self.tree.get_action_children(history)
        if not candidates:
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0
        
        # 3. selection
        action, action_node = self.search_best(history, state)
        if action is None:
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0
                
        # 4. generative model
        # TODO
        next_state = self.transition_model.sample_next_state(state, action)
        observation = self.observation_model.sample(next_state, action)
        reward = self.reward_model.get_reward(state, action, next_state)
        
        # 5. observation child
        obs_node = self.tree.get_observation_node(action_node, observation)        
    
        # 6. recursive simulation
        future = self.simulate(next_state, obs_node, depth + 1)
        total_return = reward + self.gamma * future

        # 7. backup
        self.tree.increment_visit(history)
        self.tree.increment_visit(action_node)
        self.tree.update_action_value(action_node, total_return)


        # print(
        #     f"[SIM] depth={depth} action={action.name} "
        #     f"reward={reward:.4f} future={future:.4f} total={total_return:.4f} "
        #     f"Q={self.tree.get_value(action_node):.4f} N={self.tree.get_visit(action_node)}"
        # )

        return total_return
    
        
    
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
    
    
    def search_best(self, h, state, use_ucb=True) -> Action:
        
        candidates = self.tree.get_action_children(h)
        if not candidates:
            print("There is no candidates")
        
        if state is not None:
            applicable = self.get_applicable_actions(state)
            applicable_names = {a.name for a in applicable}
            
            candidates = [(a, nid) for a, nid in candidates if a.name in applicable_names]
        
        if not candidates:
            print("There is no candidates")
            for a in applicable:
                print(a)
                
            print("Original Candidate")
            candidates = self.tree.get_action_children(h)
            print(candidates)
            return None, None
            
        if use_ucb:
            parent_visits = max(1, self.tree.get_visit(h))
            best_score = float("-inf")
            best = None

            for action, node_id in candidates:
                child_visits = self.tree.get_visit(node_id)
                child_value = self.tree.get_value(node_id)

                if child_visits == 0:
                    score = float("inf")
                else:
                    score = child_value + self.c * ((parent_visits ** 0.5) / (1 + child_visits))

                if score > best_score:
                    best_score = score
                    best = (action, node_id)

            return best

        else:
            best = max(candidates, key=lambda x: self.tree.get_value(x[1]))
            return best
        
    