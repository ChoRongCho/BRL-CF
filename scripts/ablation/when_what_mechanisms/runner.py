"""Run the isolated controlled When/What mechanism draft.

This module imports the existing environment, belief update, physical planner,
KnowNo scoring helpers, and QaA planner. It does not modify their source code.
"""

from __future__ import annotations

import argparse
import json
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
from planners.pomcp import POMCPPlanner
from utils.arguments import parse_args
from utils.logger import logger_exp

from scripts.ablation.when_what_mechanisms.cp_when import KnowNoCPWhenEvaluator
from scripts.ablation.when_what_mechanisms.policies import build_policy_bundle
from scripts.ablation.when_what_mechanisms.query_episode import run_query_episode
from scripts.ablation.when_what_mechanisms.value_evaluator import QueryValueEvaluator


CONDITIONS = ("ours", "cp_when", "value_when", "value_what")


def parse_mechanism_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mechanism-condition", required=True, choices=CONDITIONS)
    parser.add_argument("--query-cost", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--answer-accuracy", type=float, default=1.0)
    parser.add_argument("--score-temperature", type=float, default=5.0)
    parser.add_argument("--tomato-qhat", type=float, default=0.8404)
    parser.add_argument("--waste-qhat", type=float, default=0.8704)
    parser.add_argument(
        "--llm-settings",
        default=str(PROJECT_ROOT / "llm_setting.json"),
    )
    parser.add_argument("--api-key", default="")
    mechanism, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = parse_args("tomato")
    finally:
        sys.argv = original

    if args.domain not in {"tomato", "wastesorting"}:
        parser.error("domain must be tomato or wastesorting")
    if args.answer_type != "auto":
        parser.error("mechanism draft currently requires --answer_type auto")
    if mechanism.query_cost < 0 or mechanism.failure_penalty < 0:
        parser.error("query cost and failure penalty must be non-negative")
    if not 0.5 <= mechanism.answer_accuracy <= 1.0:
        parser.error("answer accuracy must be in [0.5, 1.0]")
    if mechanism.score_temperature <= 0:
        parser.error("score temperature must be positive")
    for qhat in (mechanism.tomato_qhat, mechanism.waste_qhat):
        if not 0 <= qhat <= 1:
            parser.error("qhat must be in [0, 1]")
    args.mechanism = mechanism
    return args


def build_bundle(args, env):
    cp_evaluator = None
    value_evaluator = None
    if args.mechanism_condition == "cp_when":
        qhat = (
            args.mechanism.tomato_qhat
            if args.domain == "tomato"
            else args.mechanism.waste_qhat
        )
        cp_evaluator = KnowNoCPWhenEvaluator(
            domain=args.domain,
            qhat=qhat,
            score_temperature=args.mechanism.score_temperature,
            settings_path=args.mechanism.llm_settings,
            api_key=args.mechanism.api_key,
        )
    if args.mechanism_condition in {"value_when", "value_what"}:
        value_evaluator = QueryValueEvaluator(
            args=args,
            env=env,
            query_cost=args.mechanism.query_cost,
            answer_accuracy=args.mechanism.answer_accuracy,
            failure_penalty=args.mechanism.failure_penalty,
            base_simulations=args.n_simulations,
        )
    return build_policy_bundle(
        args.mechanism_condition,
        threshold=args.threshold,
        cp_evaluator=cp_evaluator,
        value_evaluator=value_evaluator,
    )


def main() -> int:
    args = parse_mechanism_args()
    args.mechanism_condition = args.mechanism.mechanism_condition
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = Environment(args)
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    physical_planner = POMCPPlanner(
        args=args,
        env=env,
        belief_manager=belief_manager,
    )
    env.reset()
    belief = belief_manager.initialize_belief(env.state)
    bundle = build_bundle(args, env)
    manager = belief_manager.feedback_manager

    step_logs = []
    action_history: list[str] = []
    policy_traces = []
    totals = {key: 0.0 for key in (
        "search_time", "execute_time", "update_time",
        "interaction_time", "pruning_time", "step_total_time",
    )}
    cumulative_reward = 0.0
    plan_success = False
    end_reason = None
    wall_start = time()

    while True:
        step = len(step_logs) + 1
        step_start = time()
        search_start = time()
        action = physical_planner.search(belief)
        search_time = time() - search_start
        totals["search_time"] += search_time
        if action is None:
            end_reason = "PLAN FAILURE"
            break

        print(f"[Step {step}] Selected physical action: {action.name}")
        action_history.append(action.name)
        step_log = {
            "step": step,
            "action": action.name,
            "search_time": search_time,
            "execute_time": 0.0,
            "update_time": 0.0,
            "interaction_time": 0.0,
            "pruning_time": 0.0,
            "step_total_time": 0.0,
        }

        phase = time()
        observation, reward, _, _ = env.step(action)
        step_log["execute_time"] = time() - phase
        totals["execute_time"] += step_log["execute_time"]
        cumulative_reward += reward
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulative_reward

        phase = time()
        belief = belief_manager.update_belief(belief, observation, action)
        # The posterior frontier represents the post-action state, while
        # update_belief deliberately retains the prior symbolic knowledge.
        # CP prompts and root physical-action applicability must both describe
        # the current posterior, so promote its MAP state without collapsing
        # the distribution used by the When/What policies.
        belief.sync_knowledge_to_map()
        step_log["update_time"] = time() - phase
        totals["update_time"] += step_log["update_time"]

        phase = time()
        belief, policy_trace = run_query_episode(
            belief=belief,
            bundle=bundle,
            feedback_manager=manager,
            env=env,
            observation_facts=observation.state.facts,
            action_history=action_history,
            step=step,
            seed=args.seed,
            action=action,
        )
        step_log["interaction_time"] = time() - phase
        totals["interaction_time"] += step_log["interaction_time"]
        step_log["query_count"] = len(policy_trace["questions"])
        step_log["when_decision"] = policy_trace["when"]
        step_log["query_stop_reason"] = policy_trace["stop_reason"]
        policy_traces.append(policy_trace)

        done = env.check_done(belief=belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            end_reason = done
            plan_success = done == "GOAL DONE"
        else:
            phase = time()
            physical_planner.prune_search_tree(action=action, obs=belief.knowledge)
            step_log["pruning_time"] = time() - phase
            totals["pruning_time"] += step_log["pruning_time"]

        step_log["step_total_time"] = time() - step_start
        totals["step_total_time"] += step_log["step_total_time"]
        step_logs.append(step_log)
        print_step_timing(step_log)
        if end_reason is not None:
            break

    if end_reason is None:
        end_reason = "PLAN FAILURE"
    wall_time = time() - wall_start
    executed = max(1, len(step_logs))
    timing = {
        key: {"total": value, "avg": value / executed}
        for key, value in totals.items()
    }
    timing["total_time"] = wall_time
    log_data = {
        "meta": {
            "experiment": "when_what_mechanisms_draft",
            "condition": args.mechanism_condition,
            "domain": args.domain,
            "initial_state": str(args.initial_state),
            "env_setting": str(args.env_setting),
            "seed": args.seed,
            "threshold": args.threshold,
            "max_step": args.max_step,
            "n_simulations": args.n_simulations,
            "gamma": args.gamma,
            "max_depth": args.max_depth,
            "c": args.c,
            "epsilon": args.epsilon,
            "query_cost": args.mechanism.query_cost,
            "failure_penalty": args.mechanism.failure_penalty,
            "answer_accuracy": args.mechanism.answer_accuracy,
            "score_temperature": args.mechanism.score_temperature,
            "tomato_qhat": args.mechanism.tomato_qhat,
            "waste_qhat": args.mechanism.waste_qhat,
        },
        "success": plan_success,
        "end_reason": end_reason,
        "steps": len(step_logs),
        "reward": {"cumulated": cumulative_reward},
        "actions": step_logs,
        "questions": manager.query_log,
        "total_questions": manager.num_of_query,
        "policy_traces": policy_traces,
        "timing": timing,
        "action_schema_summary": build_action_schema_summary(
            step_logs,
            manager.query_log,
        ),
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    trace_path = Path(args.log_dir) / "mechanism_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "meta": log_data["meta"],
                "success": plan_success,
                "end_reason": end_reason,
                "total_questions": manager.num_of_query,
                "policy_traces": policy_traces,
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        ) + "\n",
        encoding="utf-8",
    )
    path = logger_exp(log_data, log_dir=args.log_dir)
    print("=============When/What Mechanism Draft=============")
    print(f"Condition: {args.mechanism_condition}")
    print(f"Success: {plan_success} ({end_reason})")
    print(f"Physical steps: {len(step_logs)}, Total Query: {manager.num_of_query}")
    print(f"Log: {path}")
    print(f"Mechanism trace: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
