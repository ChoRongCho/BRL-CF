"""Independent runner for the 2x2 When/What query-policy ablation.

This entry point intentionally does not reuse ``when_main.py``.  It keeps the
standard planner/environment loop and the existing FeedbackManger Oracle, while
varying only (1) when a query episode starts and (2) which fact is queried.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from time import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from environments.env import Environment
from main import build_action_schema_summary, print_step_timing
from models.belief_update import BeliefManager
from planners.pomcp import POMCPPlanner
from utils.arguments import parse_args
from utils.logger import logger_exp


@dataclass(frozen=True)
class AblationPolicy:
    label: str
    when_policy: str
    what_policy: str


ABLATION_POLICIES = {
    "random": AblationPolicy("Random", "random", "random"),
    "ours-when-only": AblationPolicy(
        "Ours-when-only", "proposed", "random"
    ),
    "ours-what-only": AblationPolicy(
        "Ours-what-only", "random", "proposed"
    ),
    "ours": AblationPolicy("Ours", "proposed", "proposed"),
}


def should_start_query(policy, confidence, threshold, random_query_prob, rng):
    """Decide whether this action starts a query episode."""
    if policy.when_policy == "proposed":
        return confidence < threshold
    return rng.random() < random_query_prob


def select_question(policy, feedback_manager, belief, rng):
    """Select a discriminative fact according to the What policy."""
    if policy.what_policy == "proposed":
        return feedback_manager.select_best_fact_to_ask(belief)

    candidates = feedback_manager.get_changed_facts(
        belief.knowledge,
        belief.frontier,
    )
    return rng.choice(candidates) if candidates else None


def commit_most_likely_state(belief):
    """Commit the MAP frontier state, matching the standard feedback flow."""
    if not belief.frontier:
        return belief

    max_idx = int(np.argmax(belief.frontier_weights))
    max_frontier = belief.frontier[max_idx]
    previous_facts = set(belief.knowledge.facts)
    final_facts = set(max_frontier.facts)
    added_facts = [fact for fact in max_frontier.facts if fact not in previous_facts]
    deleted_facts = [fact for fact in belief.knowledge.facts if fact not in final_facts]
    belief.knowledge = max_frontier
    belief.reset_belief()
    print("    [Belief Diff]")
    print(f"      + add ({len(added_facts)}): {', '.join(added_facts) if added_facts else '-'}")
    print(f"      - del ({len(deleted_facts)}): {', '.join(deleted_facts) if deleted_facts else '-'}")
    return belief


def apply_ablation_feedback(
    belief,
    feedback_manager,
    policy,
    threshold,
    random_query_prob,
    when_rng,
    what_rng,
    *,
    step,
    action,
    oracle_prior_state,
    oracle_successor_facts=None,
    observation_facts,
):
    """Run one action's query episode using the existing domain Oracle."""
    confidence = feedback_manager.compute_confidence(belief.frontier_weights)
    query_started = should_start_query(
        policy,
        confidence,
        threshold,
        random_query_prob,
        when_rng,
    )

    # Use the outcome actually executed by the environment. Retain a fallback
    # for standalone callers that do not provide it.
    if not feedback_manager.is_human_answer and oracle_successor_facts is None:
        oracle_successor_facts = feedback_manager._sample_oracle_successor_facts(
            action=action,
            prior_state=oracle_prior_state,
        )

    asked = 0
    max_questions = max(1, len(belief.frontier))
    while query_started and (asked == 0 or confidence < threshold):
        target_fact = select_question(policy, feedback_manager, belief, what_rng)
        if target_fact is None:
            break

        frontier_size_before = len(belief.frontier)
        original_belief = copy.deepcopy(belief)
        print(f"    [Planner] Current confidence: {confidence}")
        answer = feedback_manager.call_feedback(
            target_fact,
            action.name,
            observation_facts=observation_facts,
            oracle_state_facts=oracle_successor_facts,
            oracle_successor_facts=oracle_successor_facts,
        )
        feedback_manager.num_of_query += 1
        asked += 1
        print(f"    [Query] A: {answer}")

        belief = feedback_manager.apply_fact_answer_to_belief(
            belief,
            target_fact,
            answer,
        )
        updated_confidence = feedback_manager.compute_confidence(
            belief.frontier_weights
        )
        feedback_manager.query_log.append({
            "step": step,
            "action": action.name,
            "observation": list(observation_facts or []),
            "question": target_fact,
            "answer": answer,
            "confidence_before": confidence,
            "confidence_after": updated_confidence,
            "when_policy": policy.when_policy,
            "what_policy": policy.what_policy,
        })
        confidence = updated_confidence
        print(
            f"    [Planner] Reduced {len(original_belief.frontier)} "
            f"--> {len(belief.frontier)}"
        )

        # An Oracle answer outside the represented frontier leaves the belief
        # unchanged. Do not repeatedly ask the same unresolvable question.
        if len(belief.frontier) >= frontier_size_before or asked >= max_questions:
            break

    print(f"    [Planner] Updated confidence: {confidence}")
    return commit_most_likely_state(belief)


def parse_ablation_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--ablation-condition",
        required=True,
        choices=tuple(ABLATION_POLICIES),
    )
    ablation_args, remaining = pre_parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = parse_args("tomato")
    finally:
        sys.argv = original_argv

    if not 0.0 <= args.random_query_prob <= 1.0:
        pre_parser.error("--random_query_prob must be between 0 and 1")
    if args.answer_type != "auto":
        pre_parser.error("When/What ablation requires the existing auto Oracle")
    args.ablation_condition = ablation_args.ablation_condition
    return args


def main():
    args = parse_ablation_args()
    policy = ABLATION_POLICIES[args.ablation_condition]
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Independent streams keep Random-What draws from shifting Random-When.
    when_rng = random.Random(args.seed ^ 0x5748454E)
    what_rng = random.Random(args.seed ^ 0x57484154)

    env = Environment(args)
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    planner = POMCPPlanner(args=args, env=env, belief_manager=belief_manager)
    env.reset()
    belief = belief_manager.initialize_belief(env.state)

    done = False
    step = 0
    plan_success = False
    plan_end_reason = None
    cumulated_reward = 0.0
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
    wall_start = time()

    while not done:
        step += 1
        step_wall_start = time()
        print(f"Step: {step}")

        search_start = time()
        action = planner.search(belief)
        search_elapsed = time() - search_start
        totals["search_time"] += search_elapsed
        if not action:
            print("[Planner] PLAN FAILURE")
            plan_end_reason = "PLAN FAILURE"
            break

        action_log.append(action.name)
        step_log = {
            "step": step,
            "action": action.name,
            "search_time": search_elapsed,
            "execute_time": 0.0,
            "update_time": 0.0,
            "interaction_time": 0.0,
            "pruning_time": 0.0,
            "step_total_time": 0.0,
        }
        print(f"[Planner] Selected action: {action.name}")

        execute_start = time()
        oracle_prior_state = belief.knowledge.copy()
        observation, reward, _, _ = env.step(action)
        execute_elapsed = time() - execute_start
        totals["execute_time"] += execute_elapsed
        cumulated_reward += reward
        step_log["execute_time"] = execute_elapsed
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulated_reward
        print("[Planner] Observation: ", observation.state.facts)

        update_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        update_elapsed = time() - update_start
        totals["update_time"] += update_elapsed
        step_log["update_time"] = update_elapsed

        interaction_start = time()
        belief = apply_ablation_feedback(
            belief,
            belief_manager.feedback_manager,
            policy,
            args.threshold,
            args.random_query_prob,
            when_rng,
            what_rng,
            step=step,
            action=action,
            oracle_prior_state=oracle_prior_state,
            oracle_successor_facts=env.true_state.facts,
            observation_facts=observation.state.facts,
        )
        interaction_elapsed = time() - interaction_start
        totals["interaction_time"] += interaction_elapsed
        step_log["interaction_time"] = interaction_elapsed

        done = env.check_done(belief=belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            plan_success = done == "GOAL DONE"
            plan_end_reason = done
        else:
            prune_start = time()
            planner.prune_search_tree(action=action, obs=belief.knowledge)
            prune_elapsed = time() - prune_start
            totals["pruning_time"] += prune_elapsed
            step_log["pruning_time"] = prune_elapsed

        step_elapsed = time() - step_wall_start
        totals["step_total_time"] += step_elapsed
        step_log["step_total_time"] = step_elapsed
        print_step_timing(step_log)
        step_logs.append(step_log)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            break
        print("==================\n")

    total_wall_time = time() - wall_start
    if plan_end_reason is None:
        plan_end_reason = "PLAN FAILURE"
    executed_steps = max(len(step_logs), 1)
    feedback_manager = belief_manager.feedback_manager
    action_schema_summary = build_action_schema_summary(
        step_logs,
        feedback_manager.query_log,
    )
    timing = {
        name: {"total": value, "avg": value / executed_steps}
        for name, value in totals.items()
    }
    timing["total_time"] = total_wall_time
    log_data = {
        "meta": {
            "experiment": "when_what_ablation",
            "ablation_condition": policy.label,
            "when_policy": policy.when_policy,
            "what_policy": policy.what_policy,
            "random_query_prob": args.random_query_prob,
            "domain": args.domain,
            "initial_state": args.initial_state,
            "threshold": args.threshold,
            "log_dir": args.log_dir,
            "seed": args.seed,
            "max_step": args.max_step,
            "max_particles": args.max_particles,
            "max_belief_particles": args.max_belief_particles,
            "n_simulations": args.n_simulations,
            "gamma": args.gamma,
            "c": args.c,
            "max_depth": args.max_depth,
            "epsilon": args.epsilon,
        },
        "success": plan_success,
        "end_reason": plan_end_reason,
        "steps": len(step_logs),
        "reward": {"cumulated": cumulated_reward},
        "actions": step_logs,
        "timing": timing,
        "questions": feedback_manager.query_log,
        "total_questions": feedback_manager.num_of_query,
        "action_schema_summary": action_schema_summary,
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(log_data, log_dir=args.log_dir)

    print("=============When/What Ablation=============")
    print(
        f"{policy.label}: when={policy.when_policy}, "
        f"what={policy.what_policy}"
    )
    print(f"Success: {plan_success} ({plan_end_reason})")
    print(f"Steps: {len(step_logs)}, Total Query: {feedback_manager.num_of_query}")
    print("=============Action Log=============")
    for index, action_name in enumerate(action_log, start=1):
        print(f"[STEP {index}]: {action_name}")
    print("=============Log File=============")
    print(log_path)


if __name__ == "__main__":
    main()
