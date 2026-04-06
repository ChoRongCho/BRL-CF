from __future__ import annotations

import random

from models.state import State
from models.action import Action
from models.belief import Belief
from models.belief_update import BeliefManager
from models.transition import TransitionModel
from models.observation import ObservationModel
from models.reward import RewardModel
from planners.tree import POMDPTree
from environments.env import Environment

DEBUG = False

class POMCPPlanner:
    def __init__(self, args, env: Environment, belief_manager: BeliefManager):
        self.args = args

        self.n_simulations = args.n_simulations
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.c = args.c
        self.max_node_particles = getattr(args, "max_node_particles", 128)

        self.tree = POMDPTree()

        self.actions = env.actions
        self.action_map = {action.name: action for action in self.actions}
        self.belief: Belief = None

        self.env: Environment = env
        self.belief_manager: BeliefManager = belief_manager
        self.transition_model: TransitionModel = self.belief_manager.transition_model
        self.observation_model: ObservationModel = self.belief_manager.observation_model
        self.reward_model: RewardModel = self.env.reward_model

        self.initialize(self.env.state)

    def get_applicable_actions(self, knowledge: State):
        return [action for action in self.actions if action.is_applicable(knowledge)]

    def initialize(self, init_state: State):
        self.belief = self.belief_manager.initialize_belief(init_state)
        self.tree.add_particle(self.tree.root_id, init_state, self.max_node_particles)

    def _filter_existing_children(self, history: int, state: State):
        candidates = self.tree.get_action_children(history)
        if state is None:
            return candidates

        applicable_names = {action.name for action in self.get_applicable_actions(state)}
        return [(action, node_id) for action, node_id in candidates if action.name in applicable_names]

    def _sample_unexpanded_action(self, history: int, state: State):
        applicable = self.get_applicable_actions(state)
        if not applicable:
            return None, None

        existing_names = {action.name for action, _ in self.tree.get_action_children(history)}
        unexpanded = [action for action in applicable if action.name not in existing_names]

        if not unexpanded:
            return None, None

        action = random.choice(unexpanded)
        action_node = self.tree.expand_tree_from(history, action, is_action=True)
        return action, action_node

    def _expand_applicable_actions(self, history: int, state: State):
        applicable = self.get_applicable_actions(state)
        for action in applicable:
            self.tree.expand_tree_from(history, action, is_action=True)
        return applicable

    def search(self, belief: Belief):
        """
        The planner reasons only over the current symbolic knowledge state.
        The full belief frontier is maintained outside the tree for belief
        update and entropy/information calculations.
        """
        history = self.tree.root_id

        knowledge = belief.knowledge.copy()
        root = self.tree.get_node(history)
        root.knowledge = knowledge.copy()
        
        # if not root.frontiers:
        #     root.frontiers = [particle.copy() for particle in belief.particles]

        # Repeat Simulations until timeout
        for _ in range(self.n_simulations):
            sampled_root_state = self.tree.sample_particle(history)
            if sampled_root_state is None:
                sampled_root_state = knowledge.copy()
            self.simulate(state=sampled_root_state, history=history, depth=0)

        # DEBUG
        if DEBUG:
            # history debugging
            self._debugging(history=history)

        best_action = self._select_action(history, knowledge)
        print("Selected:", best_action.name if best_action else None)
        return best_action

    def simulate(self, state: State, history: int, depth: int):
        """
        Docstring for simulate
        """
        
        if DEBUG:
            self.tree.debugging()
        
        # 1. Check significance of update
        if (self.gamma ** depth < self.epsilon or self.gamma == 0) and depth != 0:
            return 0.0

        # 2. Filter actions by applicability in the current sampled state.
        applicable = self.get_applicable_actions(state)
        if not applicable:
            # 3. If nothing is applicable, treat this observation node as terminal.
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0

        # 4. On the first visit, expand all applicable actions and initialize by rollout.
        if self.tree.is_leaf_node(history):
            self._expand_applicable_actions(history, state)
            new_value = self.rollout(state, depth)
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, new_value)
            return new_value

        # 5. Otherwise select the best existing action child with UCB.
        action, action_node = self.search_best(history, state, use_ucb=True)
        if action is None:
            self.tree.increment_visit(history)
            self.tree.set_value_if_first(history, 0.0)
            return 0.0

        # 6. Sample transition, observation, and immediate reward.
        next_state = self.transition_model.sample_next_state(state, action)
        observation = self.observation_model.sample(next_state, action)
        reward = self.reward_model.get_reward(state, action, next_state)

        # 7. Move to the matching observation child and recurse.
        obs_node = self.tree.get_observation_node(action_node, observation)
        total_return = reward + self.gamma*self.simulate(next_state, obs_node, depth+1)

        # 8. Backtrack
        self.tree.add_particle(history, state, self.max_node_particles)
        self.tree.increment_visit(history)
        self.tree.increment_visit(action_node)
        self.tree.update_action_value(action_node, total_return)
        
        return total_return
    

    def rollout(self, state: State, depth: int):
        if (self.gamma ** depth < self.epsilon or self.gamma == 0) and depth != 0:
            return 0.0

        applicable_actions = self.get_applicable_actions(state)
        if not applicable_actions:
            return 0.0

        action = random.choice(applicable_actions)
        sample_state = self.transition_model.sample_next_state(state, action)
        reward = self.reward_model.get_reward(state, action, sample_state)

        return reward + self.gamma * self.rollout(sample_state, depth + 1)

    def search_best(self, history: int, state: State, use_ucb: bool = True):
        candidates = self._filter_existing_children(history, state)
        if not candidates:
            return None, None

        if use_ucb:
            parent_visits = max(1, self.tree.get_visit(history))
            best = None
            best_score = float("-inf")

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

        return max(candidates, key=lambda item: self.tree.get_value(item[1]))

    def prune_search_tree(self, action: Action, observation):
        self.tree.prune_after_action(action, observation)

    
    def _select_action(self, history, knowledge):
        best_action, _ = self.search_best(history, knowledge, use_ucb=False)
        if best_action is not None:
            return best_action

        unexpanded_action, _ = self._sample_unexpanded_action(history, knowledge)
        if unexpanded_action is not None:
            return unexpanded_action

        applicable_actions = self.get_applicable_actions(knowledge)
        if applicable_actions:
            return random.choice(applicable_actions)

        return None
    
    def _debugging(self, history):
        root_action_children = self.tree.get_action_children(history)
        action_node_ids = [node_id for _, node_id in root_action_children]
        obs_node_ids = []
        for _, action_node_id in root_action_children:
            obs_node_ids.extend(child_id for _, child_id in self.tree.get_observation_children(action_node_id))
        print(
            f"[TREE] root_id={history} "
            f"action_nodes={action_node_ids[:10]} "
            f"obs_nodes={obs_node_ids[:10]} "
            f"total_nodes={len(self.tree.nodes)}"
        )
