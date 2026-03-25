
"""
Particle Belief Approximation for POMDP
=======================================

Belief update (exact):
    b_t(s_t) = eta * O(o_t | s_t, a_{t-1}) * sum_{s_{t-1}} T(s_t | s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1})

where
    - b_t(s_t): posterior belief at time t
    - T(s_t | s_{t-1}, a_{t-1}): transition model
    - O(o_t | s_t, a_{t-1}): observation likelihood
    - eta: normalization constant

Particle approximation:
    b_t(s) ≈ sum_{i=1}^N w_t^(i) * delta(s - s_t^(i))

Bootstrap particle filter:
    1) Sample:
         s_t^(i) ~ T(. | s_{t-1}^(i), a_{t-1})
    2) Weight:
         w_t^(i) ∝ O(o_t | s_t^(i), a_{t-1})
    3) Normalize:
         w_t^(i) <- w_t^(i) / sum_j w_t^(j)
    4) Resample if needed:
         draw N particles from categorical(w_1, ..., w_N)

More general importance sampling form:
    if s_t^(i) ~ q(s_t | s_{t-1}^(i), a_{t-1}, o_t),
    then
         w_t^(i) ∝ w_{t-1}^(i) *
                    [ O(o_t | s_t^(i), a_{t-1}) * T(s_t^(i) | s_{t-1}^(i), a_{t-1}) ]
                    / q(s_t^(i) | s_{t-1}^(i), a_{t-1}, o_t)

In this rough implementation below, we use the bootstrap filter:
    q = T
so the weight simplifies to:
    w_t^(i) ∝ O(o_t | s_t^(i), a_{t-1})

ESS (effective sample size):
    ESS = 1 / sum_i (w_t^(i))^2

Resample when ESS < threshold.
"""
from __future__ import annotations
import numpy as np
import copy
from typing import Any, Dict, List, Optional, Iterable, Tuple


from utils.asp import DomainRuleBridge, solve_asp
from models.belief import Belief
from models.state import State
from models.action import Action
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel, TransitionOutcome, NextStateOutcome


class BeliefManager:
    def __init__(self,
                 transition_model: TransitionModel,
                 observation_model: ObservationModel,
                 asp_bridge: DomainRuleBridge):
        """
        Docstring for __init__
        
        """
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.asp_bridge = asp_bridge

        self.belief: Belief = None
        self.initial_belief = 1.0
        self.conf_threshold = 0.7
        
        
    def initialize_belief(self, init_state: State):
        self.belief = Belief(knowledge=init_state,
                             frontier=[State()],
                             frontier_weights=np.array([self.initial_belief], dtype=float))
        return self.belief
           
    def get_belief(self) -> Belief:
        return self.belief
    
    
    def update_belief(self, belief: Belief, obs: Observation, action: Action) -> Belief:
        """
        """
        if belief is None:
            raise ValueError("b_prev is None. 초기 belief를 먼저 설정하세요.")

        if belief.is_empty():
            self.belief = Belief(frontier=[], weights=np.array([], dtype=float))
            return self.belief
        
        # 1. Transition expansion from knowledge base
        outcomes = self._get_transition_outcomes(belief.knowledge, action)
        
        tran_row = []
        obs_row = []
        frontier = []
        
        for outcome in outcomes:   
            next_state, trans_prob = outcome.next_state, outcome.probability
            frontier.append(next_state)
            tran_row.append(trans_prob)

            # 2. Observation            
            obs_likelihood = self._get_observation_likelihood(obs, next_state, action)
            obs_row.append(obs_likelihood)
            
        transition_matrix = np.array(tran_row)
        observation_matrix = np.array(obs_row)
        
        # DEBUG
        # print(transition_matrix, transition_matrix.size)
        # print(observation_matrix, observation_matrix.size)
        
        if transition_matrix.size != observation_matrix.size:
            raise SystemError("The size of Transition and Observation is Different. Modify the code!!!")
        
        # 2. Update belief, posterior ∝ transition * observation
        unnormalized = transition_matrix * observation_matrix
        total = float(np.sum(unnormalized))
        if total <= 0.0:
            weights = np.ones(len(frontier), dtype=float) / len(frontier)
        else:
            weights = unnormalized / total
        
        b_next = Belief(knowledge=belief.knowledge, frontier=frontier, frontier_weights=weights)
        
        # 3. verify frontier
        b_next = self.verify_frontier(belief=b_next)
        
        # 4. calcluate confidence
        confidence = self.calculate_confidence(b_next.frontier_weights)
        
        # 5. append belief into the knowledge
        self.advance_observation(belief=b_next, confidence=confidence)
        
        
        
        
        self.belief = b_next
        
        return b_next, confidence


    def _get_transition_outcomes(self, state: State, action: Action) -> List[NextStateOutcome]:
        return self.transition_model.get_next_state_distribution(state, action)
        

    def _get_observation_likelihood(
        self,
        obs: Observation,
        state: State,
        action: Any,
    ) -> float:

        if self.observation_model is None:
            return 1.0
                
        value = self.observation_model.likelihood(obs, state, action)
        return max(0.0, float(value))

    
    @staticmethod
    def _state_to_key(state: State) -> str:
        if hasattr(state, "facts"):
            try:
                return "||".join(sorted(str(f) for f in state.facts))
            except Exception:
                pass
        return repr(state)
        
    def calculate_confidence(self, weights) -> float:
        """
        1. weights를 max 기준으로 정규화
        2. entropy 계산
        3. 1 - H/theta 반환

        theta가 None이면 max entropy(log N) 사용
        """

        if len(weights) == 0:
            return 0.0
        elif len(weights) == 1:
            return 1.0

        # 1. entropy normalization
        theta = np.log2(len(weights))

        # 확률처럼 쓰기 위해 다시 normalize (sum=1)
        

        # 2. entropy 계산 (log 안정성)
        eps = 1e-12
        h = -np.sum(weights * np.log2(weights + eps))

        # 3. confidence
        confidence = 1.0 - (h / theta)

        # numerical safety
        return confidence
        
        
    def verify_frontier(self, belief: Belief):
        """
        ASP로 belief를 검증하고, 불가능한 belief는 0으로 한 뒤, 강제 normalize
        
        :param self: Description
        :param belief: Description
        """
        self.asp_bridge.clear_runtime()
        
        for i, frontier in enumerate(belief.frontier):
            self.asp_bridge.add_runtime_facts(frontier.facts)
            program = self.asp_bridge.build_certain_worlds()
            worlds = solve_asp(program)
            
            if not len(worlds)==1:
                # print(f"The {i}th worlds is not possible. Avaialble: {len(worlds)}")
                belief.frontier_weights[i] = 0.0 + 1e-12
            else:
                # print(f"The {i}th worlds is possible {belief.frontier_weights[i]}. Avaialble: {len(worlds)}")
                # belief.frontier_weights[i] = 0.0 + 1e-12
                pass
            self.asp_bridge.clear_runtime()
        
        # normalize belief
        total = float(np.sum(belief.frontier_weights))
        belief.frontier_weights = belief.frontier_weights / total
        
        return belief
    
    
    def advance_observation(self, belief: Belief, confidence: float):
        if confidence < self.conf_threshold:
            print(f"[BeliefManager] the confidence {confidence} is low. ")
            
            new_obs = self.dummy_function()
            
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]
            
            
            belief.knowledge.merge_state(max_frontier)
            belief.reset_belief()
            
            return belief
        
        else:
            print(f"[BeliefManager] the confidence {confidence} is sufficiently high. ")
            
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]
            
            belief.knowledge.merge_state(max_frontier)
            belief.reset_belief()
            
            return belief
        
        
        
    def dummy_function(self) -> Observation:
        new_observation = Observation(facts=["at(changmin)"])
        return new_observation