"""Resolve oracle truth into the planner's complete belief-state representation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .cp_when import aggregate_state_probabilities, task_state_facts


@dataclass(frozen=True)
class CalibrationTruth:
    facts: set[str]
    resolution: str
    matching_hypotheses: int


def _normalize(fact: object) -> str:
    return str(fact).replace(" ", "")


def build_calibration_truth_facts(
    belief,
    candidate_facts: Iterable[object],
    oracle_answer: Callable[[str], bool],
) -> CalibrationTruth:
    """Return the unique full belief hypothesis matching oracle task truth.

    The domain oracle grounds externally meaningful task facts. Bookkeeping
    facts such as ``detected`` are completed from the belief representation,
    but only when the oracle task assignment identifies one unique full joint
    hypothesis. Ambiguous or unsupported labels fail instead of being guessed.
    """
    candidates = list(map(_normalize, candidate_facts))
    task_candidates = task_state_facts(candidates)
    oracle_task_truth = {
        fact
        for fact in task_candidates
        if bool(oracle_answer(fact))
    }
    probabilities = aggregate_state_probabilities(belief, candidates)
    matches = [
        signature
        for signature in probabilities
        if {
            fact for fact in task_candidates if fact in set(signature)
        } == oracle_task_truth
    ]
    if len(matches) == 1:
        return CalibrationTruth(set(matches[0]), "unique_full_state", 1)
    if not matches:
        # Preserve a genuine support miss. The bookkeeping completion does not
        # affect p_true because no hypothesis matches the oracle task facts.
        raw_truth = {
            fact for fact in candidates if bool(oracle_answer(fact))
        }
        return CalibrationTruth(raw_truth, "out_of_support", 0)
    if len(matches) > 1:
        raise ValueError(
            "Oracle task assignment matches multiple full belief states: "
            f"matches={len(matches)}, task_truth={sorted(oracle_task_truth)}"
        )
    raise AssertionError("unreachable")
