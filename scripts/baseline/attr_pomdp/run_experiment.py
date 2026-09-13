"""Run the paper-based Attr-POMDP baseline in a BRL domain."""

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
from planners.pomcp import POMCPPlanner
from scripts.baseline.attr_pomdp.controller import (
    AttrPOMDPConfig,
    AttrPOMDPController,
)
from utils.arguments import parse_args
from utils.logger import logger_exp


def parse_attr_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attr-depth", type=int, default=3)
    parser.add_argument("--attribute-cost", type=float, default=0.1)
    parser.add_argument("--answer-accuracy", type=float, default=0.99)
    parser.add_argument("--commit-correct-reward", type=float, default=1.0)
    parser.add_argument("--commit-wrong-reward", type=float, default=-1.0)
    parser.add_argument("--attr-discount", type=float, default=1.0)
    parser.add_argument("--max-questions-per-action", type=int, default=10)
    parser.add_argument("--max-candidate-questions", type=int, default=8)
    attr_args, remaining = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = parse_args("tomato")
    finally:
        sys.argv = original_argv
    if args.answer_type != "auto":
        parser.error("Attr-POMDP BRL experiments use the existing auto Oracle")
    if attr_args.max_questions_per_action < 1:
        parser.error("--max-questions-per-action must be positive")
    if attr_args.max_candidate_questions < 1:
        parser.error("--max-candidate-questions must be positive")
    args.attr = attr_args
    return args


def main() -> None:
    args = parse_attr_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = Environment(args)
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    task_planner = POMCPPlanner(args=args, env=env, belief_manager=belief_manager)
    attr_config = AttrPOMDPConfig(
        depth=args.attr.attr_depth,
        attribute_cost=args.attr.attribute_cost,
        answer_accuracy=args.attr.answer_accuracy,
        correct_commit_reward=args.attr.commit_correct_reward,
        wrong_commit_reward=args.attr.commit_wrong_reward,
        discount=args.attr.attr_discount,
        max_questions_per_action=args.attr.max_questions_per_action,
        max_candidate_questions=args.attr.max_candidate_questions,
    )
    attr_controller = AttrPOMDPController(
        belief_manager.feedback_manager,
        attr_config,
    )

    env.reset()
    belief = belief_manager.initialize_belief(env.state)
    step_logs = []
    action_log = []
    totals = {
        "search_time": 0.0,
        "execute_time": 0.0,
        "update_time": 0.0,
        "interaction_time": 0.0,
        "pruning_time": 0.0,
        "step_total_time": 0.0,
    }
    plan_success = False
    end_reason = None
    cumulative_reward = 0.0
    wall_start = time()
    step = 0

    while True:
        step += 1
        step_start = time()
        print(f"Step: {step}")

        phase_start = time()
        action = task_planner.search(belief)
        search_time = time() - phase_start
        totals["search_time"] += search_time
        if action is None:
            end_reason = "PLAN FAILURE"
            print("[Planner] PLAN FAILURE")
            break

        action_log.append(action.name)
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
        print(f"[Planner] Selected action: {action.name}")

        phase_start = time()
        oracle_prior_state = belief.knowledge.copy()
        observation, reward, _, _ = env.step(action)
        execute_time = time() - phase_start
        totals["execute_time"] += execute_time
        cumulative_reward += reward
        step_log["execute_time"] = execute_time
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulative_reward
        print("[Planner] Observation: ", observation.state.facts)

        phase_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        update_time = time() - phase_start
        totals["update_time"] += update_time
        step_log["update_time"] = update_time

        phase_start = time()
        belief = attr_controller.get_new_observation(
            belief,
            step=step,
            action=action,
            oracle_prior_state=oracle_prior_state,
            observation_facts=observation.state.facts,
        )
        interaction_time = time() - phase_start
        totals["interaction_time"] += interaction_time
        step_log["interaction_time"] = interaction_time

        done = env.check_done(belief=belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            end_reason = done
            plan_success = done == "GOAL DONE"
        else:
            phase_start = time()
            task_planner.prune_search_tree(action=action, obs=belief.knowledge)
            pruning_time = time() - phase_start
            totals["pruning_time"] += pruning_time
            step_log["pruning_time"] = pruning_time

        step_time = time() - step_start
        totals["step_total_time"] += step_time
        step_log["step_total_time"] = step_time
        step_logs.append(step_log)
        print_step_timing(step_log)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            break

    if end_reason is None:
        end_reason = "PLAN FAILURE"

    elapsed = time() - wall_start
    count = max(len(step_logs), 1)
    manager = belief_manager.feedback_manager
    timing = {
        key: {"total": value, "avg": value / count}
        for key, value in totals.items()
    }
    timing["total_time"] = elapsed
    summary = build_action_schema_summary(step_logs, manager.query_log)
    result = {
        "meta": {
            "experiment": "attr_pomdp",
            "policy": "Attr-POMDP",
            "implementation": "paper_based_reimplementation",
            "domain": args.domain,
            "initial_state": args.initial_state,
            "seed": args.seed,
            "max_step": args.max_step,
            "n_simulations": args.n_simulations,
            "max_belief_particles": args.max_belief_particles,
            "attr_depth": attr_config.depth,
            "attribute_cost": attr_config.attribute_cost,
            "answer_accuracy": attr_config.answer_accuracy,
            "commit_correct_reward": attr_config.correct_commit_reward,
            "commit_wrong_reward": attr_config.wrong_commit_reward,
            "attr_discount": attr_config.discount,
            "max_questions_per_action": attr_config.max_questions_per_action,
            "max_candidate_questions": attr_config.max_candidate_questions,
            "log_dir": args.log_dir,
        },
        "success": plan_success,
        "end_reason": end_reason,
        "steps": len(step_logs),
        "reward": {"cumulated": cumulative_reward},
        "actions": step_logs,
        "timing": timing,
        "questions": manager.query_log,
        "total_questions": manager.num_of_query,
        "action_schema_summary": summary,
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(result, log_dir=args.log_dir)
    print("=============Attr-POMDP=============")
    print(f"Success: {plan_success} ({end_reason})")
    print(f"Steps: {len(step_logs)}, Total Query: {manager.num_of_query}")
    print("=============Action Log=============")
    for index, action_name in enumerate(action_log, start=1):
        print(f"[STEP {index}]: {action_name}")
    print("=============Log File=============")
    print(log_path)


if __name__ == "__main__":
    main()
