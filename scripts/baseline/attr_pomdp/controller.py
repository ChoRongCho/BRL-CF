"""Adapter from BRL symbolic frontier beliefs to Attr-POMDP questions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planner import AttrPOMDPPlanner, binary_attribute_question


@dataclass(frozen=True)
class AttrPOMDPConfig:
    depth: int = 3
    attribute_cost: float = 0.1
    answer_accuracy: float = 0.99
    correct_commit_reward: float = 1.0
    wrong_commit_reward: float = -1.0
    discount: float = 1.0
    max_questions_per_action: int = 10
    max_candidate_questions: int = 8


class AttrPOMDPController:
    def __init__(self, feedback_manager, config: AttrPOMDPConfig) -> None:
        self.feedback_manager = feedback_manager
        self.config = config
        self.planner = AttrPOMDPPlanner(
            depth=config.depth,
            correct_commit_reward=config.correct_commit_reward,
            wrong_commit_reward=config.wrong_commit_reward,
            discount=config.discount,
        )

    def build_questions(self, belief):
        facts = self.feedback_manager.get_changed_facts(
            belief.knowledge,
            belief.frontier,
        )
        # The paper plans over a small set of attribute concepts. A symbolic
        # frontier can expose dozens of facts, so retain the most immediately
        # discriminative facts before performing the depth-limited tree search.
        facts = sorted(
            facts,
            key=lambda fact: self.feedback_manager.expected_entropy_after_asking(
                belief.frontier,
                belief.frontier_weights,
                fact,
            ),
        )[: self.config.max_candidate_questions]
        return [
            binary_attribute_question(
                fact,
                [state.has_fact(fact) for state in belief.frontier],
                accuracy=self.config.answer_accuracy,
                cost=self.config.attribute_cost,
            )
            for fact in facts
        ]

    def update_brl_belief(self, belief, question, answer):
        belief.frontier_weights = self.planner.update_belief(
            belief.frontier_weights,
            question,
            bool(answer),
        )
        return belief

    @staticmethod
    def commit_map(belief):
        if belief.frontier:
            map_index = int(np.argmax(belief.frontier_weights))
            belief.knowledge = belief.frontier[map_index]
            belief.reset_belief()
        return belief

    def get_new_observation(
        self,
        belief,
        *,
        step,
        action,
        oracle_prior_state,
        observation_facts,
    ):
        manager = self.feedback_manager
        oracle_successor_facts = manager._sample_oracle_successor_facts(
            action=action,
            prior_state=oracle_prior_state,
        )

        for question_index in range(self.config.max_questions_per_action):
            questions = self.build_questions(belief)
            if not questions:
                break
            decision = self.planner.plan(belief.frontier_weights, questions)
            if decision.kind == "commit" or decision.question is None:
                break

            question = decision.question
            confidence_before = manager.compute_confidence(
                belief.frontier_weights
            )
            target_fact = str(question.key)
            answer = manager.call_feedback(
                target_fact,
                action.name,
                observation_facts=observation_facts,
                oracle_state_facts=belief.knowledge.facts,
                oracle_successor_facts=oracle_successor_facts,
            )
            belief = self.update_brl_belief(belief, question, answer)
            confidence_after = manager.compute_confidence(
                belief.frontier_weights
            )
            manager.num_of_query += 1
            manager.query_log.append({
                "step": step,
                "action": action.name,
                "observation": list(observation_facts or []),
                "question": target_fact,
                "answer": bool(answer),
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "baseline": "Attr-POMDP",
                "attr_pomdp_question_index": question_index + 1,
                "attr_pomdp_action_value": decision.value,
            })
            print(
                f"    [Attr-POMDP] AskAttr({target_fact})={bool(answer)}; "
                f"confidence {confidence_before:.4f}->{confidence_after:.4f}; "
                f"value={decision.value:.4f}"
            )

        return self.commit_map(belief)
