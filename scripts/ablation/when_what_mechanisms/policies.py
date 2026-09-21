from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .policy_types import PolicyContext, WhatDecision, WhenDecision


class BeliefThresholdWhen:
    name = "belief_threshold"

    def __init__(self, threshold: float):
        self.threshold = float(threshold)

    def should_start(self, context: PolicyContext) -> WhenDecision:
        confidence = context.feedback_manager.compute_confidence(
            context.belief.frontier_weights
        )
        return WhenDecision(
            start=confidence < self.threshold,
            policy=self.name,
            score=confidence,
            reason="below_threshold" if confidence < self.threshold else "at_or_above_threshold",
            diagnostics={"confidence": confidence, "threshold": self.threshold},
        )


class InformationGainWhat:
    name = "information_gain"

    def select_fact(self, context: PolicyContext) -> WhatDecision | None:
        manager = context.feedback_manager
        candidates = context.ambiguous_facts()
        if not candidates:
            return None
        weights = manager.normalize(context.belief.frontier_weights)
        current_entropy = manager.entropy(weights)
        scores = {}
        for fact in candidates:
            expected = manager.expected_entropy_after_asking(
                context.belief.frontier,
                weights,
                fact,
            )
            scores[fact] = current_entropy - expected
        best = min(candidates, key=lambda fact: (-scores[fact], fact))
        return WhatDecision(
            fact=best,
            policy=self.name,
            score=float(scores[best]),
            diagnostics={
                "current_entropy": current_entropy,
                "candidate_scores": scores,
            },
        )


class ConformalActionAmbiguityWhen:
    name = "cp_action_ambiguity"

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator

    def should_start(self, context: PolicyContext) -> WhenDecision:
        result = self.evaluator.evaluate(context)
        prediction_set = result.prediction_set
        start, reason = cp_trigger(prediction_set, result.fallback_token)
        return WhenDecision(
            start=start,
            policy=self.name,
            score=None,
            reason=reason,
            diagnostics=result.as_dict(),
        )


class QueryValueWhen:
    name = "query_value"

    def __init__(self, evaluator: Any, information_policy: InformationGainWhat):
        self.evaluator = evaluator
        self.information_policy = information_policy

    def should_start(self, context: PolicyContext) -> WhenDecision:
        candidate = self.information_policy.select_fact(context)
        if candidate is None:
            return WhenDecision(False, self.name, reason="no_query_candidate")
        values = self.evaluator.evaluate(
            context,
            query_facts=[candidate.fact],
            root_mode="compare",
        )
        best_physical = max(values.physical_q.values(), default=float("-inf"))
        query_value = values.query_q.get(candidate.fact, float("-inf"))
        start = query_value > best_physical
        diagnostics = values.as_dict()
        diagnostics.update({
            "eig_fact": candidate.fact,
            "eig_score": candidate.score,
            "best_query_q": query_value,
            "best_physical_q": best_physical,
            "margin": query_value - best_physical,
        })
        return WhenDecision(
            start=start,
            policy=self.name,
            score=query_value - best_physical,
            reason="query_value_higher" if start else "physical_value_at_least_as_high",
            diagnostics=diagnostics,
        )


class QueryValueWhat:
    name = "query_value"

    def __init__(self, evaluator: Any):
        self.evaluator = evaluator

    def select_fact(self, context: PolicyContext) -> WhatDecision | None:
        candidates = context.ambiguous_facts()
        if not candidates:
            return None
        values = self.evaluator.evaluate(
            context,
            query_facts=candidates,
            root_mode="query_only",
        )
        if set(values.query_q) != set(candidates):
            missing = sorted(set(candidates) - set(values.query_q))
            raise RuntimeError(f"Value evaluator omitted query candidates: {missing}")
        best = min(candidates, key=lambda fact: (-values.query_q[fact], fact))
        return WhatDecision(
            fact=best,
            policy=self.name,
            score=float(values.query_q[best]),
            diagnostics=values.as_dict(),
        )


def cp_trigger(prediction_set: list[str], fallback_token: str) -> tuple[bool, str]:
    if not prediction_set:
        return True, "empty"
    if fallback_token in prediction_set:
        return True, "contains_noopt"
    if len(prediction_set) != 1:
        return True, "multiple"
    return False, "singleton_action"


@dataclass(frozen=True)
class PolicyBundle:
    condition: str
    when: Any
    what: Any


def build_policy_bundle(condition, *, threshold, cp_evaluator=None, value_evaluator=None):
    information = InformationGainWhat()
    if condition == "ours":
        return PolicyBundle(condition, BeliefThresholdWhen(threshold), information)
    if condition == "cp_when":
        if cp_evaluator is None:
            raise ValueError("cp_when requires a CP evaluator")
        return PolicyBundle(condition, ConformalActionAmbiguityWhen(cp_evaluator), information)
    if condition == "value_when":
        if value_evaluator is None:
            raise ValueError("value_when requires a value evaluator")
        return PolicyBundle(condition, QueryValueWhen(value_evaluator, information), information)
    if condition == "value_what":
        if value_evaluator is None:
            raise ValueError("value_what requires a value evaluator")
        return PolicyBundle(
            condition,
            BeliefThresholdWhen(threshold),
            QueryValueWhat(value_evaluator),
        )
    raise ValueError(f"Unknown mechanism condition: {condition}")

