
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
from typing import Any, Dict, List, Optional, Iterable, Tuple

from models.belief import Belief
from models.state import State
from models.action import Action
from models.observation import ObservationModel, Observation
from models.transition import TransitionModel, TransitionOutcome, NextStateOutcome


class BeliefModel:
    def __init__(
        self,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        top_k: int = None,
        prune_threshold: float = 1e-12,
    ):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.prune_threshold = prune_threshold
        self.top_k = top_k

        self.belief = None
        self.initial_belief = 1.0


    def set_initial_belief(self, init_state: State) -> Belief:
        """
        초기 belief state 생성
        
        :param init_state: State 구조의 초기 상태, belief는 1로 유지
        """
        self.belief = Belief(frontier=[init_state], weights=np.array([self.initial_belief], dtype=float))
        return self.belief
    

    def set_belief(self, belief: Belief) -> None:
        self.belief = belief
        

    def get_belief(self) -> Optional[Belief]:
        return self.belief
    

    def update_belief(
        self,
        b_prev: Belief,
        obs: Observation,
        action: Any,
    ) -> Belief:
        """
        reachable state만 유지하는 sparse belief update.

        흐름:
        1. 각 이전 state에서 action에 따른 transition outcome 생성
        2. 같은 next_state는 probability를 합산
        3. observation likelihood로 reweight
        4. pruning + normalize
        
        - self.observation_model.likelihood(obs, state, action) -> float
          또는
        - self.observation_model.probability(obs, state, action) -> float
        """
        if b_prev is None:
            raise ValueError("b_prev is None. 초기 belief를 먼저 설정하세요.")

        if b_prev.is_empty():
            self.belief = Belief(frontier=[], weights=np.array([], dtype=float))
            return self.belief

        merged: Dict[str, float] = {}
        state_map: Dict[str, State] = {}
        
        # 1. Transition expansion
        for prev_state, prev_weight in zip(b_prev.frontier, b_prev.weights):
            outcomes = self._get_transition_outcomes(prev_state, action)
            
            for outcome in outcomes:
                
                next_state, trans_prob = self._parse_transition_outcome(outcome)

                if trans_prob <= 0.0:
                    continue
                
                def _state_to_key(state: State) -> str:
                    if hasattr(state, "facts"):
                        try:
                            return "||".join(sorted(str(f) for f in state.facts))
                        except Exception:
                            pass
                    return repr(state)
                
                key = _state_to_key(next_state)
                merged[key] = merged.get(key, 0.0) + prev_weight * trans_prob
                state_map[key] = next_state
                
                
                

        # transition 결과가 하나도 없을 때: 이전 belief 유지 혹은 empty 처리
        if not merged:
            # 여기서는 보수적으로 이전 belief 유지
            self.belief = Belief(
                frontier=list(b_prev.frontier),
                weights=np.array(b_prev.weights, dtype=float),
            )
            return self.belief

        # 2. Observation correction
        if obs is not None and self.observation_model is not None:
            for key in list(merged.keys()):
                state = state_map[key]
                obs_likelihood = self._get_observation_likelihood(obs, state, action)
                merged[key] *= obs_likelihood

        # 3. Pruning
        merged = {
            s: w
            for s, w in merged.items()
            if w > self.prune_threshold
        }

        if not merged:
            # observation 때문에 다 사라졌다면,
            # fallback으로 transition 후 belief를 observation 없이 정규화
            fallback_states = list(merged.keys())
            # merged가 이미 비어 있으므로 transition 직후 상태를 다시 계산
            merged = {}
            for prev_state, prev_weight in zip(b_prev.frontier, b_prev.weights):
                outcomes = self._get_transition_outcomes(prev_state, action)
                for outcome in outcomes:
                    next_state, trans_prob = self._parse_transition_outcome(outcome)
                    if trans_prob <= 0.0:
                        continue
                    merged[next_state] = merged.get(next_state, 0.0) + prev_weight * trans_prob

        # 4. Optional top-k
        items = [(state_map[k], w) for k, w in merged.items() if w > self.prune_threshold]
        items.sort(key=lambda x: x[1], reverse=True)

        if self.top_k is not None:
            items = items[:self.top_k]

        frontier = [s for s, _ in items]
        weights = np.array([w for _, w in items], dtype=float)

        b_next = Belief(frontier=frontier, weights=weights)
        confidence = self.calculate_confidence(weights)
        self.belief = b_next
        
        return b_next, confidence


    def _get_transition_outcomes(self, state: State, action: Action) -> List[NextStateOutcome]:
        return self.transition_model.get_next_state_distribution(state, action)
        

    def _parse_transition_outcome(self, outcome: Any) -> Tuple[State, float]:
        """
        TransitionOutcome의 구체 필드명을 아직 모르므로 유연하게 파싱.
        지원 예시:
        - outcome.next_state, outcome.prob
        - outcome.state, outcome.probability
        - {"next_state": ..., "prob": ...}
        - (state, prob)
        """
        if isinstance(outcome, tuple) and len(outcome) == 2:
            next_state, prob = outcome
            return next_state, float(prob)

        if isinstance(outcome, dict):
            next_state = (
                outcome.get("next_state")
                or outcome.get("state")
            )
            prob = (
                outcome.get("prob")
                or outcome.get("probability")
                or outcome.get("p")
                or outcome.get("weight")
            )
            if next_state is None or prob is None:
                raise ValueError(f"Invalid transition dict outcome: {outcome}")
            return next_state, float(prob)

        next_state = None
        prob = None

        for state_attr in ("next_state", "state"):
            if hasattr(outcome, state_attr):
                next_state = getattr(outcome, state_attr)
                break

        for prob_attr in ("prob", "probability", "p", "weight"):
            if hasattr(outcome, prob_attr):
                prob = getattr(outcome, prob_attr)
                break

        if next_state is None or prob is None:
            raise ValueError(
                "Could not parse TransitionOutcome. "
                "Expected fields like (next_state/state) and (prob/probability/p/weight)."
            )

        return next_state, float(prob)

    def _get_observation_likelihood(
        self,
        obs: Observation,
        state: State,
        action: Any,
    ) -> float:
        if self.observation_model is None:
            return 1.0

        if hasattr(self.observation_model, "likelihood"):
            value = self.observation_model.likelihood(obs, state, action)
            return max(0.0, float(value))

        if hasattr(self.observation_model, "probability"):
            value = self.observation_model.probability(obs, state, action)
            return max(0.0, float(value))

        if hasattr(self.observation_model, "score"):
            value = self.observation_model.score(obs, state, action)
            return max(0.0, float(value))

        raise AttributeError(
            "observation_model must implement one of "
            "'likelihood(obs, state, action)', "
            "'probability(obs, state, action)', "
            "or 'score(obs, state, action)'."
        )
        
        
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
        