from __future__ import annotations

from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.arguments import parse_args
from environments.env import Environment
from models.belief_update import BeliefManager
from planners.pomcp import POMCPPlanner
from utils.logger import logger_exp


def action_schema_name(action_name: str) -> str:
    return action_name.replace(" ", "").split("(", 1)[0]


def build_action_schema_summary(step_logs: list[dict[str, Any]], query_logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    queried_action_runs = set()

    for step_log in step_logs:
        schema = action_schema_name(step_log["action"])
        step_log["action_schema"] = schema
        summary.setdefault(schema, {
            "action_count": 0,
            "question_count": 0,
            "actions_with_question": 0,
            "question_probability": 0.0,
            "questions_per_action": 0.0,
        })
        summary[schema]["action_count"] += 1

    for query_log in query_logs:
        action_name = query_log.get("action")
        if not action_name:
            continue
        schema = action_schema_name(action_name)
        summary.setdefault(schema, {
            "action_count": 0,
            "question_count": 0,
            "actions_with_question": 0,
            "question_probability": 0.0,
            "questions_per_action": 0.0,
        })
        summary[schema]["question_count"] += 1
        queried_action_runs.add((query_log.get("step"), schema))

    for _step, schema in queried_action_runs:
        if schema in summary:
            summary[schema]["actions_with_question"] += 1

    for schema_stats in summary.values():
        action_count = schema_stats["action_count"]
        if action_count > 0:
            schema_stats["question_probability"] = schema_stats["actions_with_question"] / action_count
            schema_stats["questions_per_action"] = schema_stats["question_count"] / action_count

    return dict(sorted(summary.items()))


def tree_depths(planner: POMCPPlanner) -> list[int]:
    depths: list[int] = []
    for node_id in planner.tree.nodes:
        depth = 0
        current = node_id
        while planner.tree.nodes[current].parent_id is not None:
            current = planner.tree.nodes[current].parent_id
            depth += 1
        depths.append(depth)
    return depths


def root_action_metrics(planner: POMCPPlanner, selected_action_name: str | None) -> dict[str, Any]:
    root_id = planner.tree.root_id
    children = planner.tree.get_action_children(root_id)
    visits = [planner.tree.get_visit(node_id) for _action, node_id in children]
    total_visits = sum(visits)
    selected_visits = 0
    selected_value = 0.0
    if selected_action_name is not None:
        for action, node_id in children:
            if action.name == selected_action_name:
                selected_visits = planner.tree.get_visit(node_id)
                selected_value = planner.tree.get_value(node_id)
                break

    entropy = 0.0
    if total_visits > 0:
        probs = np.array([visit / total_visits for visit in visits if visit > 0], dtype=float)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    return {
        "root_visit_count": planner.tree.get_visit(root_id),
        "root_action_count": len(children),
        "root_action_total_visits": total_visits,
        "root_action_visit_entropy": entropy,
        "selected_action_visits": selected_visits,
        "selected_action_value": selected_value,
    }


def search_metrics(planner: POMCPPlanner, nodes_before: int, selected_action_name: str | None) -> dict[str, Any]:
    depths = tree_depths(planner)
    node_count = len(planner.tree.nodes)
    action_nodes = sum(1 for node in planner.tree.nodes.values() if node.is_action_node)
    observation_nodes = sum(1 for node in planner.tree.nodes.values() if node.is_observation_node)
    metrics = {
        "tree_node_count": node_count,
        "tree_nodes_expanded_this_step": max(0, node_count - nodes_before),
        "action_node_count": action_nodes,
        "observation_node_count": observation_nodes,
        "max_tree_depth": max(depths, default=0),
        "avg_tree_depth": float(np.mean(depths)) if depths else 0.0,
        "n_simulations": planner.n_simulations,
    }
    metrics.update(root_action_metrics(planner, selected_action_name))
    return metrics


def belief_metrics(belief) -> dict[str, Any]:
    weights = getattr(belief, "frontier_weights", np.array([], dtype=float))
    particle_count = len(getattr(belief, "frontier", []) or [])
    nonzero = int(np.sum(np.array(weights) > 1e-12)) if len(weights) else 0
    entropy = 0.0
    if len(weights) > 1:
        entropy = float(-np.sum(weights * np.log2(weights + 1e-12)))
    return {
        "belief_frontier_size": particle_count,
        "belief_nonzero_frontier_size": nonzero,
        "belief_entropy": entropy,
        "belief_max_weight": float(np.max(weights)) if len(weights) else 0.0,
    }


def summarize_numeric(step_logs: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in step_logs if key in item]
    if not values:
        return {"total": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "total": float(np.sum(values)),
        "avg": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def print_step_timing(step_log: dict[str, Any]) -> None:
    print(
        "[SCALE] "
        f"step={step_log['step']} "
        f"search={step_log.get('search_time', 0.0):.4f}s "
        f"nodes={step_log.get('tree_node_count', 0)} "
        f"expanded={step_log.get('tree_nodes_expanded_this_step', 0)} "
        f"root_actions={step_log.get('root_action_count', 0)} "
        f"belief={step_log.get('post_update_belief_frontier_size', 0)}"
    )


def main() -> None:
    args = parse_args("tomato")
    args.f_strategy = 3
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    total_search_time = 0.0
    total_execute_time = 0.0
    total_update_time = 0.0
    total_interaction_time = 0.0
    total_prune_time = 0.0
    total_step_wall_time = 0.0
    action_log: list[str] = []
    step_logs: list[dict[str, Any]] = []
    plan_success = False
    plan_end_reason = None
    wall_start = time()
    cumulated_reward = 0.0

    step = 0
    while True:
        step += 1
        step_wall_start = time()
        print(f"Step: {step}")

        nodes_before = len(planner.tree.nodes)
        search_start = time()
        action = planner.search(belief)
        search_elapsed = time() - search_start
        total_search_time += search_elapsed

        if action is None:
            print("[Planner] PLAN FAILURE")
            plan_end_reason = "PLAN FAILURE"
            break

        action_log.append(action.name)
        step_log: dict[str, Any] = {
            "step": step,
            "action": action.name,
            "search_time": search_elapsed,
            "execute_time": 0.0,
            "update_time": 0.0,
            "interaction_time": 0.0,
            "pruning_time": 0.0,
            "step_total_time": 0.0,
        }
        step_log.update(search_metrics(planner, nodes_before, action.name))
        step_log.update({f"pre_{key}": value for key, value in belief_metrics(belief).items()})
        print(f"[Planner] Selected action: {action.name}")

        execute_start = time()
        observation, reward, _, _info = env.step(action)
        cumulated_reward += reward
        execute_elapsed = time() - execute_start
        total_execute_time += execute_elapsed
        step_log["execute_time"] = execute_elapsed
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulated_reward

        update_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        update_elapsed = time() - update_start
        total_update_time += update_elapsed
        step_log["update_time"] = update_elapsed
        step_log.update({f"post_update_{key}": value for key, value in belief_metrics(belief).items()})

        interaction_start = time()
        belief = belief_manager.feedback_manager.get_new_observation(
            belief=belief,
            step=step,
            action_name=action.name,
        )
        interaction_elapsed = time() - interaction_start
        total_interaction_time += interaction_elapsed
        step_log["interaction_time"] = interaction_elapsed
        step_log.update({f"post_query_{key}": value for key, value in belief_metrics(belief).items()})

        done = env.check_done(belief=belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            plan_success = done == "GOAL DONE"
            plan_end_reason = done
            step_total_elapsed = time() - step_wall_start
            total_step_wall_time += step_total_elapsed
            step_log["step_total_time"] = step_total_elapsed
            print_step_timing(step_log)
            step_logs.append(step_log)
            break

        prune_start = time()
        planner.prune_search_tree(action=action, obs=belief.knowledge)
        prune_elapsed = time() - prune_start
        total_prune_time += prune_elapsed
        step_log["pruning_time"] = prune_elapsed
        step_total_elapsed = time() - step_wall_start
        total_step_wall_time += step_total_elapsed
        step_log["step_total_time"] = step_total_elapsed
        print_step_timing(step_log)
        print("==================\n")
        step_logs.append(step_log)

    total_wall_time = time() - wall_start
    executed_steps = max(len(step_logs), 1)
    if plan_end_reason is None:
        plan_end_reason = "PLAN FAILURE"

    action_schema_summary = build_action_schema_summary(
        step_logs,
        belief_manager.feedback_manager.query_log,
    )
    scale_metrics = {
        "tree_node_count": summarize_numeric(step_logs, "tree_node_count"),
        "tree_nodes_expanded_this_step": summarize_numeric(step_logs, "tree_nodes_expanded_this_step"),
        "root_action_count": summarize_numeric(step_logs, "root_action_count"),
        "max_tree_depth": summarize_numeric(step_logs, "max_tree_depth"),
        "belief_frontier_size": summarize_numeric(step_logs, "post_update_belief_frontier_size"),
        "belief_entropy": summarize_numeric(step_logs, "post_update_belief_entropy"),
    }

    log_data = {
        "meta": {
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
            "experiment": "scalability",
            "query_policy": "ours",
        },
        "success": plan_success,
        "end_reason": plan_end_reason,
        "steps": len(step_logs),
        "reward": {"cumulated": cumulated_reward},
        "actions": step_logs,
        "timing": {
            "search_time": {"total": total_search_time, "avg": total_search_time / executed_steps},
            "execute_time": {"total": total_execute_time, "avg": total_execute_time / executed_steps},
            "update_time": {"total": total_update_time, "avg": total_update_time / executed_steps},
            "interaction_time": {"total": total_interaction_time, "avg": total_interaction_time / executed_steps},
            "pruning_time": {"total": total_prune_time, "avg": total_prune_time / executed_steps},
            "step_total_time": {"total": total_step_wall_time, "avg": total_step_wall_time / executed_steps},
            "total_time": total_wall_time,
        },
        "scale_metrics": scale_metrics,
        "questions": belief_manager.feedback_manager.query_log,
        "total_questions": belief_manager.feedback_manager.num_of_query,
        "action_schema_summary": action_schema_summary,
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(log_data, log_dir=args.log_dir)

    print("[SCALE SUMMARY]")
    print(f"success: {plan_success}")
    print(f"end_reason: {plan_end_reason}")
    print(f"steps: {len(step_logs)}")
    print(f"total_questions: {belief_manager.feedback_manager.num_of_query}")
    print(f"avg search: {total_search_time / executed_steps:.4f}s")
    for key, value in scale_metrics.items():
        print(f"{key}: avg={value['avg']:.4f}, max={value['max']:.4f}")
    print("=============Action Log=============")
    for idx, action_name in enumerate(action_log, start=1):
        print(f"[STEP {idx}]: {action_name}")
    print("=============Log File=============")
    print(log_path)


if __name__ == "__main__":
    main()
