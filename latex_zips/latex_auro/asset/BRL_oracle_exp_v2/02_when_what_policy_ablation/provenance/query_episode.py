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
    *, belief, bundle, feedback_manager, env, observation_facts,
    action_history, step, seed, action,
) -> tuple[Any, dict[str, Any]]:
    """Gate entry once with When; continue questions using a common threshold."""
    asked_facts: set[str] = set()
    trace = {
        "condition": bundle.condition,
        "when": None,
        "when_decisions": [],
        "questions": [],
        "stop_reason": None,
    }

    while True:
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
        if not asked_facts:
            when = bundle.when.should_start(context)
            serialized_when = asdict(when)
            trace["when_decisions"].append(serialized_when)
            trace["when"] = serialized_when
            if not when.start:
                trace["stop_reason"] = "not_triggered"
                break

        what = bundle.what.select_fact(context)
        if what is None:
            trace["stop_reason"] = "no_query_candidate"
            break
        if not asked_facts and when.query_fact is not None and what.fact != when.query_fact:
            raise RuntimeError(
                "When policy required a different question from What: "
                f"trigger={when.query_fact}, selected={what.fact}"
            )
        if what.fact in asked_facts:
            raise RuntimeError(f"What policy repeated a fact: {what.fact}")

        before_count = len(belief.frontier)
        confidence_before = feedback_manager.compute_confidence(
            belief.frontier_weights
        )
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
            belief, what.fact, bool(answer)
        )
        confidence_after = feedback_manager.compute_confidence(
            belief.frontier_weights
        )
        record = {
            "step": step,
            "action": action.name,
            "observation": list(observation_facts or []),
            "question": what.fact,
            "answer": bool(answer),
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "frontier_before": before_count,
            "frontier_after": len(belief.frontier),
            "when_policy": when.policy,
            "when_decision": serialized_when,
            "what_policy": what.policy,
            "what_score": what.score,
            "what_diagnostics": what.diagnostics,
            "condition": bundle.condition,
        }
        feedback_manager.query_log.append(record)
        trace["questions"].append(record)

        if confidence_after >= bundle.threshold:
            trace["stop_reason"] = "threshold_reached"
            break
        if len(belief.frontier) >= before_count:
            trace["stop_reason"] = "belief_not_reduced"
            break

    trace["final_confidence"] = feedback_manager.compute_confidence(
        belief.frontier_weights
    )
    return commit_map_state(belief), trace
