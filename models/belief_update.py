
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
        outcomes = self._get_transition_outcomes(belief.knowledge, action) # 우리는 어떤 액션을 수행하였을 때, 그것의 기대 효과를 알고 있음
        
        tran_row = []
        obs_row = []
        frontiers = []
        
        for outcome in outcomes:   
            next_state, trans_prob = outcome.next_state, outcome.probability
            frontiers.append(next_state)
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
            raise SystemError("The size of Transition and Observation is Different. Modify code!!!")
        
        # 2. Update belief, posterior ∝ transition * observation
        unnormalized = transition_matrix * observation_matrix
        total = float(np.sum(unnormalized))
        if total <= 0.0:
            weights = np.ones(len(frontiers), dtype=float) / len(frontiers)
        else:
            weights = unnormalized / total
        
        b_next = Belief(knowledge=belief.knowledge, frontier=frontiers, frontier_weights=weights)
        
        # 3. verify frontier, CSP
        b_next = self.verify_frontier(belief=b_next)
        
        # 4. calcluate confidence
        confidence = self.calculate_confidence(b_next.frontier_weights)
        
        # 5. append belief into the knowledge / expand knowledge
        b_next = self.advance_observation(belief=b_next, confidence=confidence)
        
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
    
    @staticmethod
    def get_changed_facts_from_knowledge(knowledge, frontier):
        """
        knowledge와 비교해서 frontier들에서 달라지는 fact 후보를 모은다.
        """
        knowledge_set = set(knowledge.facts)
        candidate_facts = set()

        for state in frontier:
            state_set = set(state.facts)

            added = state_set - knowledge_set
            removed = knowledge_set - state_set

            candidate_facts.update(added)
            candidate_facts.update(removed)

        return sorted(candidate_facts)
        
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
                print(f"The {i}th worlds is not possible. Avaialble: {len(worlds)}")
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
    
    def select_best_fact_to_ask(belief):
        """
        frontier의 uncertainty를 가장 많이 줄여줄 fact를 고른다.
        """
        frontier = belief.frontier
        weights = normalize(belief.frontier_weights)

        candidate_facts = get_changed_facts_from_knowledge(belief.knowledge, frontier)
        if not candidate_facts:
            return None

        current_entropy = entropy(weights)

        best_fact = None
        best_gain = -float("inf")

        for fact in candidate_facts:
            exp_h = expected_entropy_after_asking(frontier, weights, fact)
            info_gain = current_entropy - exp_h

            if info_gain > best_gain:
                best_gain = info_gain
                best_fact = fact

        return best_fact

    
    def advance_observation(self, belief: Belief, confidence: float):
        """
        목적:
        frontier 불확실성을 만드는 fact를 찾아 질문하고,
        confidence가 threshold 이상이 될 때까지 반복한다.
        마지막에는 max frontier를 knowledge로 덮어쓴다.
        """

        while confidence < self.conf_threshold:
            # 1. 질문할 가치가 가장 큰 fact 선택
            target_fact = select_best_fact_to_ask(belief)

            # 더 이상 질문할 fact가 없으면 종료
            if target_fact is None:
                break

            # 2. 사람에게 질문
            # dummy_function은 예를 들어 True / False 반환한다고 가정
            answer = self.dummy_function(target_fact)

            # 3. 답변에 맞게 belief 갱신
            belief = apply_fact_answer_to_belief(belief, target_fact, answer)

            # 4. confidence 재계산
            confidence = compute_confidence(belief.frontier_weights)

        # 5. threshold를 넘었거나, 더 이상 질문할 게 없으면
        #    가장 가능성 높은 frontier를 knowledge로 채택
        if len(belief.frontier) > 0:
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]

            belief.knowledge = max_frontier
            belief.reset_belief()

        return belief
    
    
    
    def advance_observation(self, belief: Belief, confidence: float):
        """
        목적: frontier에서 어느 부분이 confidence 감소에 영향을 미쳤는지 추적하는 것
        
        example: belief.frontier에 총 6개의 State가 나옴
        각각 w1, w2, ... , w6 이라고 해보겠음
        
        각각의 belief.fronier_weights 가 [0.3, 0.2, 0.2, 0.1, 0.1, 0.1] 라고 해볼게
        
        그러면 confidence 값은 다음과 같이 계산이 됨
        theta = np.log2(len(weights)) 
        eps = 1e-12
        h = -np.sum(weights * np.log2(weights + eps))
        confidence = 1.0 - (h / theta)
        
        만약 confidence 값이 self.conf_threshold 보다 작으면 질문을 하게 됨.
        
        이때 내가 하고 싶은 건 다음과 같음
        1. belief.knowledge와 비교해서 각각 프론티어에서 변화된 fact을 확인
        2. 엔트로피가 높게 나오는 원인인 fact를 파악, 수정하였을 때, 엔트로피가 가장 낮아질 수 있는 fact를 선택
        3. 그 부분만 정제해서 사람에게 질문 (dumm_function에서 된다고 가정할 거임)
        4. 다시 confidence를 계산하고
        5. 그래서 self.conf_threshold 이 높을 때 까지 계속 질문.
        6. self.conf_threshold 보다 결국 높아지면 max_frontier를 찾아서 knoweldge에 덮어쓰기
        아래 코드 참고
        max_idx = np.argmax(belief.frontier_weights)
        max_frontier = belief.frontier[max_idx]
            
        belief.knowledge = max_frontier
        belief.reset_belief()
        """
        
        # belief.frontier와 같은 크기의 리스트를 생성
        # 형식 belief.frontier = List[State]
        # State.facts = List[str]
        # 어떤 액션 a 로 인한 변화를 추정해야 함
        """
        
        
        """
        
        
        if confidence < self.conf_threshold:            
            dummy = self.dummy_function()
            
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]
            
            
            belief.knowledge = max_frontier
            belief.reset_belief()
            
            return belief
        
        else:            
            max_idx = np.argmax(belief.frontier_weights)
            max_frontier = belief.frontier[max_idx]
            
            belief.knowledge = max_frontier
            belief.reset_belief()
            
            return belief
        
        
        
    def dummy_function(self) -> Observation:
        new_observation = Observation(facts=["at(changmin)"])
        return new_observation