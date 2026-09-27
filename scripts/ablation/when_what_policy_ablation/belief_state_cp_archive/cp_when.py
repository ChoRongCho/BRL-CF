"""Conformal prediction over the shared belief-state representation.

The hypothesis space is the complete joint truth assignment of symbolic facts
that differ across the current belief frontier. Ours EIG selects the actual
question inside the conformal prediction set. The nonconformity score is
``1 - p_b(true_assignment)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .policy_types import PolicyContext


EPISTEMIC_BOOKKEEPING_PREDICATES = frozenset({"detected", "observed", "scanned"})


def _predicate(fact: str) -> str:
    return fact.split("(", 1)[0]


def ambiguous_facts(belief: Any) -> list[str]:
    """Return every fact whose truth value differs across frontier states."""
    universe = sorted({str(fact) for state in belief.frontier for fact in state.facts})
    return [
        fact for fact in universe
        if len({fact in set(map(str, state.facts)) for state in belief.frontier}) > 1
    ]


def task_state_facts(candidate_facts: list[str]) -> list[str]:
    """Facts with external oracle semantics, used only to ground calibration."""
    return [
        fact for fact in candidate_facts
        if _predicate(fact) not in EPISTEMIC_BOOKKEEPING_PREDICATES
    ]


def state_key(state: Any, candidate_facts: list[str]) -> tuple[str, ...]:
    """Project a state to one query-relevant joint fact assignment."""
    present = set(map(str, state.facts))
    return tuple(fact for fact in candidate_facts if fact in present)


def state_label(key: tuple[str, ...]) -> str:
    return " & ".join(key) if key else "<all ambiguous facts false>"


def aggregate_state_probabilities(
    belief: Any,
    candidate_facts: list[str] | None = None,
) -> dict[tuple[str, ...], float]:
    weights = np.asarray(belief.frontier_weights, dtype=float)
    if len(weights) != len(belief.frontier):
        raise ValueError("Belief frontier and weights have different lengths")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Belief weights must have positive mass")
    candidates = ambiguous_facts(belief) if candidate_facts is None else candidate_facts
    probabilities: dict[tuple[str, ...], float] = {}
    for state, probability in zip(belief.frontier, weights / total):
        key = state_key(state, candidates)
        probabilities[key] = probabilities.get(key, 0.0) + float(probability)
    return probabilities


@dataclass(frozen=True)
class StateCPResult:
    prediction_set: list[str]
    prediction_keys: list[tuple[str, ...]]
    probabilities: dict[str, float]
    qhat: float
    candidate_facts: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_set": list(self.prediction_set),
            "prediction_set_size": len(self.prediction_set),
            "probabilities": dict(self.probabilities),
            "qhat": self.qhat,
            "probability_cutoff": 1.0 - self.qhat,
            "candidate_facts": list(self.candidate_facts),
        }


class StateCPWhenEvaluator:
    """Build a prediction set directly from the current belief distribution."""

    def __init__(self, *, qhat: float):
        if not 0.0 <= qhat <= 1.0:
            raise ValueError("State CP qhat must be in [0, 1]")
        self.qhat = float(qhat)

    def evaluate(self, context: PolicyContext) -> StateCPResult:
        candidates = ambiguous_facts(context.belief)
        aggregated = aggregate_state_probabilities(context.belief, candidates)
        labelled = {state_label(key): probability for key, probability in aggregated.items()}
        cutoff = 1.0 - self.qhat
        prediction_set = sorted(
            label for label, probability in labelled.items() if probability >= cutoff
        )
        prediction_keys = sorted(
            key for key, probability in aggregated.items() if probability >= cutoff
        )
        return StateCPResult(
            prediction_set=prediction_set,
            prediction_keys=prediction_keys,
            probabilities=dict(sorted(labelled.items())),
            qhat=self.qhat,
            candidate_facts=candidates,
        )


def calibration_nonconformity(belief: Any, true_state: Any) -> float:
    """Compute ``1 - p(true assignment)`` for held-out calibration."""
    candidates = ambiguous_facts(belief)
    probability = aggregate_state_probabilities(belief, candidates).get(
        state_key(true_state, candidates), 0.0
    )
    return 1.0 - probability
