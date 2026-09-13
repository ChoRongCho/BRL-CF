"""Run the joint task/query POMCP baseline in a BRL domain."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from time import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from environments.env import Environment
from main import build_action_schema_summary, print_step_timing
from models.belief_update import BeliefManager
from scripts.baseline.targeted_query_pomdp.planner import QueryAsActionPOMCPPlanner
from scripts.baseline.targeted_query_pomdp.query_actions import QueryAction
from utils.arguments import parse_args
from utils.logger import logger_exp


def parse_baseline_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-cost", type=float, default=1.0)
    parser.add_argument("--answer-accuracy", type=float, default=1.0)
    parser.add_argument("--max-consecutive-queries", type=int, default=30)
    baseline, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = parse_args("tomato")
    finally:
        sys.argv = original
    if args.answer_type != "auto":
        parser.error("This baseline currently uses the existing Boolean auto Oracle")
    if baseline.query_cost < 0.0:
        parser.error("--query-cost must be non-negative")
    if baseline.max_consecutive_queries < 1:
        parser.error("--max-consecutive-queries must be positive")
    args.query_as_action = baseline
    return args


def main() -> None:
    args = parse_baseline_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = Environment(args)
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    planner = QueryAsActionPOMCPPlanner(
        args,
        env,
        belief_manager,
        query_cost=args.query_as_action.query_cost,
        answer_accuracy=args.query_as_action.answer_accuracy,
    )
    env.reset()
    belief = belief_manager.initialize_belief(env.state)
    manager = belief_manager.feedback_manager

    step_logs = []
    decision_log = []
    totals = {key: 0.0 for key in (
        "search_time", "execute_time", "update_time",
        "interaction_time", "pruning_time", "step_total_time",
    )}
    cumulative_reward = 0.0
    plan_success = False
    end_reason = None
    wall_start = time()
    physical_step = 0
    decision_index = 0
    consecutive_queries = 0

    last_task_action = None
    last_observation_facts = []
    oracle_successor_facts = None

    while True:
        decision_index += 1
        decision_start = time()
        phase_start = time()
        action = planner.search(belief)
        search_time = time() - phase_start
        totals["search_time"] += search_time
        if action is None:
            end_reason = "PLAN FAILURE"
            print("[Planner] PLAN FAILURE")
            break

        decision_log.append(action.name)
        print(f"[Decision {decision_index}] Selected action: {action.name}")

        if isinstance(action, QueryAction):
            consecutive_queries += 1
            if consecutive_queries > args.query_as_action.max_consecutive_queries:
                end_reason = "MAX CONSECUTIVE QUERY"
                break

            confidence_before = manager.compute_confidence(belief.frontier_weights)
            phase_start = time()
            oracle_answer = bool(manager.call_feedback(
                action.target_fact,
                # The query is an independent action, not post-processing of
                # the preceding detect/scan action. Passing its own schema
                # keeps the existing Oracle while avoiding sensor-specific
                # visibility rules intended for post-action questions.
                action.name,
                observation_facts=last_observation_facts,
                oracle_state_facts=belief.knowledge.facts,
                oracle_successor_facts=oracle_successor_facts,
            ))
            answer = (
                oracle_answer
                if random.random() <= args.query_as_action.answer_accuracy
                else not oracle_answer
            )
            interaction_time = time() - phase_start
            totals["interaction_time"] += interaction_time

            before_count = len(belief.frontier)
            belief = manager.apply_fact_answer_to_belief(
                belief,
                action.target_fact,
                bool(answer),
            )
            # The posterior remains a distribution. Its MAP state is the
            # symbolic KB used for applicability; the query-confirmed fact (or
            # its Boolean complement encoded by absence) is therefore promoted.
            belief.sync_knowledge_to_map()
            confidence_after = manager.compute_confidence(belief.frontier_weights)
            manager.num_of_query += 1
            manager.query_log.append({
                "step": physical_step,
                "decision": decision_index,
                "action": (
                    last_task_action.name if last_task_action is not None else None
                ),
                "query_action": action.name,
                "question": action.target_fact,
                "answer": bool(answer),
                "oracle_answer": oracle_answer,
                "answer_type": "boolean",
                "observation": list(last_observation_facts),
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "baseline": "Query-as-Action (QaA)",
            })
            cumulative_reward -= action.cost
            print(
                f"    [Boolean Query] {action.name} -> {bool(answer)}; "
                f"belief {before_count}->{len(belief.frontier)}, "
                f"confidence {confidence_before:.4f}->{confidence_after:.4f}"
            )

            done = env.check_done(belief)
            if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
                end_reason = done
                plan_success = done == "GOAL DONE"
                break
            continue

        consecutive_queries = 0
        physical_step += 1
        step_log = {
            "step": physical_step,
            "decision": decision_index,
            "action": action.name,
            "search_time": search_time,
            "execute_time": 0.0,
            "update_time": 0.0,
            "interaction_time": 0.0,
            "pruning_time": 0.0,
            "step_total_time": 0.0,
        }

        oracle_prior_state = belief.knowledge.copy()
        phase_start = time()
        observation, reward, _, _ = env.step(action)
        execute_time = time() - phase_start
        totals["execute_time"] += execute_time
        cumulative_reward += reward
        step_log["execute_time"] = execute_time
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulative_reward
        print("[Planner] Observation:", observation.state.facts)

        phase_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        # Keep uncertainty for future query actions while maintaining the same
        # MAP knowledge convention used by the original BRL action planner.
        belief.sync_knowledge_to_map()
        update_time = time() - phase_start
        totals["update_time"] += update_time
        step_log["update_time"] = update_time

        last_task_action = action
        last_observation_facts = list(observation.state.facts)
        oracle_successor_facts = manager._sample_oracle_successor_facts(
            action=action,
            prior_state=oracle_prior_state,
        )

        done = env.check_done(belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            end_reason = done
            plan_success = done == "GOAL DONE"

        step_time = time() - decision_start
        totals["step_total_time"] += step_time
        step_log["step_total_time"] = step_time
        step_logs.append(step_log)
        print_step_timing(step_log)
        if end_reason is not None:
            break

    if end_reason is None:
        end_reason = "PLAN FAILURE"
    elapsed = time() - wall_start
    count = max(len(step_logs), 1)
    timing = {
        key: {"total": value, "avg": value / count}
        for key, value in totals.items()
    }
    timing["total_time"] = elapsed
    result = {
        "meta": {
            "experiment": "query_as_action",
            "policy": "Query-as-Action (QaA)",
            "query_interface": "enumerated_grounded_boolean_actions",
            "domain": args.domain,
            "initial_state": args.initial_state,
            "seed": args.seed,
            "max_step": args.max_step,
            "n_simulations": args.n_simulations,
            "max_depth": args.max_depth,
            "query_cost": args.query_as_action.query_cost,
            "answer_accuracy": args.query_as_action.answer_accuracy,
            "max_consecutive_queries": args.query_as_action.max_consecutive_queries,
            "log_dir": args.log_dir,
        },
        "success": plan_success,
        "end_reason": end_reason,
        "steps": len(step_logs),
        "reward": {"cumulated": cumulative_reward},
        "actions": step_logs,
        "decisions": decision_log,
        "timing": timing,
        "questions": manager.query_log,
        "total_questions": manager.num_of_query,
        "action_schema_summary": build_action_schema_summary(
            step_logs,
            manager.query_log,
        ),
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(result, log_dir=args.log_dir)
    print("=============Query-as-Action (QaA)=============")
    print(f"Success: {plan_success} ({end_reason})")
    print(f"Physical steps: {len(step_logs)}, Total Query: {manager.num_of_query}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
