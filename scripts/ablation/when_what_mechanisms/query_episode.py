from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .policy_types import PolicyContext


def commit_map_state(belief):
    if belief.frontier:
        belief.sync_knowledge_to_map()
        belief.reset_belief()
    return belief


def run_query_episode(
    *,
    belief,
    bundle,
    feedback_manager,
    env,
    observation_facts,
    action_history,
    step,
    seed,
    action,
) -> tuple[Any, dict[str, Any]]:
    """Apply one controlled When/What policy after a physical observation."""
    base_context = PolicyContext(
        belief=belief,
        feedback_manager=feedback_manager,
        env=env,
        observation_facts=list(observation_facts or []),
        action_history=list(action_history),
        step=step,
        query_index=0,
        seed=seed,
    )
    when = bundle.when.should_start(base_context)
    trace = {
        "condition": bundle.condition,
        "when": asdict(when),
        "questions": [],
        "stop_reason": "not_triggered" if not when.start else None,
    }
    if not when.start:
        return commit_map_state(belief), trace

    confidence = feedback_manager.compute_confidence(belief.frontier_weights)
    threshold = float(feedback_manager.conf_threshold)
    asked_facts: set[str] = set()
    initial_candidates = base_context.ambiguous_facts()
    max_questions = len(initial_candidates)
    if max_questions == 0:
        trace["stop_reason"] = "no_query_candidate"
        return commit_map_state(belief), trace

    while len(asked_facts) < max_questions and (
        not asked_facts or confidence < threshold
    ):
        context = PolicyContext(
            belief=belief,
            feedback_manager=feedback_manager,
            env=env,
            observation_facts=list(observation_facts or []),
            action_history=list(action_history),
            step=step,
            query_index=len(asked_facts),
            seed=seed,
            excluded_facts=set(asked_facts),
        )
        what = bundle.what.select_fact(context)
        if what is None:
            trace["stop_reason"] = "no_query_candidate"
            break
        if what.fact in asked_facts:
            raise RuntimeError(f"What policy repeated a fact: {what.fact}")

        before_count = len(belief.frontier)
        confidence_before = confidence
        answer = feedback_manager.call_feedback(
            what.fact,
            action.name,
            observation_facts=observation_facts,
            oracle_state_facts=env.true_state.facts,
            oracle_successor_facts=env.true_state.facts,
        )
        feedback_manager.num_of_query += 1
        asked_facts.add(what.fact)
        belief = feedback_manager.apply_fact_answer_to_belief(
            belief,
            what.fact,
            bool(answer),
        )
        confidence = feedback_manager.compute_confidence(belief.frontier_weights)
        record = {
            "step": step,
            "action": action.name,
            "observation": list(observation_facts or []),
            "question": what.fact,
            "answer": bool(answer),
            "confidence_before": confidence_before,
            "confidence_after": confidence,
            "frontier_before": before_count,
            "frontier_after": len(belief.frontier),
            "when_policy": when.policy,
            "what_policy": what.policy,
            "what_score": what.score,
            "what_diagnostics": what.diagnostics,
            "condition": bundle.condition,
        }
        feedback_manager.query_log.append(record)
        trace["questions"].append(record)
        if len(belief.frontier) >= before_count:
            trace["stop_reason"] = "belief_not_reduced"
            break

    if trace["stop_reason"] is None:
        if confidence >= threshold:
            trace["stop_reason"] = "confidence_reached"
        elif len(asked_facts) >= max_questions:
            trace["stop_reason"] = "question_limit"
        else:
            trace["stop_reason"] = "ended"
    trace["final_confidence"] = confidence
    return commit_map_state(belief), trace

