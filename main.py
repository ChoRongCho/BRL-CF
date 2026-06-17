"""
Docstring for main
"""

from pathlib import Path
import random
import sys
from time import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.arguments import parse_args
from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefManager
from models.action import Action
from planners.pomcp import POMCPPlanner
from utils.logger import logger_exp


# available domain
DOMAIN = ["tomato", "wastesorting"]


def action_schema_name(action_name):
    return action_name.replace(" ", "").split("(", 1)[0]


def build_action_schema_summary(step_logs, query_logs):
    summary = {}
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

    for step, schema in queried_action_runs:
        if schema in summary:
            summary[schema]["actions_with_question"] += 1

    for schema_stats in summary.values():
        action_count = schema_stats["action_count"]
        if action_count > 0:
            schema_stats["question_probability"] = schema_stats["actions_with_question"] / action_count
            schema_stats["questions_per_action"] = schema_stats["question_count"] / action_count

    return dict(sorted(summary.items()))


def main():
    args = parse_args("tomato")
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    env = Environment(args)
    
    belief_manager = BeliefManager(args,
                                   env.transition_model, 
                                   env.observation_model, 
                                   env.asp_bridge)
    
    planner = POMCPPlanner(args=args, 
                           env=env, 
                           belief_manager=belief_manager)
    
    observation = env.reset()
    
    belief = belief_manager.initialize_belief(env.state)
    
    done = False
    i = 0
    total_search_time = 0.0
    total_execute_time = 0.0
    total_update_time = 0.0
    total_interaction_time = 0.0
    total_prune_time = 0.0
    action_log = []
    step_logs = []
    plan_success = False
    plan_end_reason = None
    wall_start = time()
    cumulated_reward = 0.0
    
    while not done:     
        i += 1
        print(f"Step: {i}")
        search_start = time()
        
        # =========== 1. Select action ===========
        action = planner.search(belief)
        if action:
            action_log.append(action.name)
            search_elapsed = time() - search_start
            total_search_time += search_elapsed
            step_log = {
                "step": i,
                "action": action.name,
                "search_time": search_elapsed,
                "execute_time": 0.0,
                "update_time": 0.0,
                "interaction_time": 0.0,
                "pruning_time": 0.0,
            }
            print(f"[Planner] Selected action: {action.name}")
        else:
            print("[Planner] PLAN FAILURE")
            plan_end_reason = "PLAN FAILURE"
            break
        
        # =========== 2. excute action and get observation ===========
        execute_start = time()
        observation, r, _, info = env.step(action)
        cumulated_reward += r
        print("[Planner] Observation: ", observation.state.facts)
        execute_elapsed = time() - execute_start
        total_execute_time += execute_elapsed
        step_log["execute_time"] = execute_elapsed
        step_log["step_reward"] = r
        step_log["cumulated_reward"] = cumulated_reward
        
        
        # =========== 3. Update belief and Query ===========
        update_start = time()
        belief= belief_manager.update_belief(belief, observation, action)
        update_elapsed = time() - update_start
        total_update_time += update_elapsed
        step_log["update_time"] = update_elapsed
        

        # =========== 4. compute confidence and human ask ===========
        interaction_start = time()
        belief = belief_manager.feedback_manager.get_new_observation(
            belief=belief,
            step=i,
            action_name=action.name,
        )
        interaction_elapsed = time() - interaction_start
        total_interaction_time += interaction_elapsed
        step_log["interaction_time"] = interaction_elapsed
        
        # Check Done
        done = env.check_done(belief=belief)
        if done == "GOAL DONE":
            plan_success = True
            plan_end_reason = done
            step_logs.append(step_log)
            break
        
        elif done == "MAX STEP":
            plan_success = False
            plan_end_reason = done
            step_logs.append(step_log)
            break
        
        elif done == "PLAN FAILURE":
            plan_success = False
            plan_end_reason = done
            step_logs.append(step_log)
            break
        
        
        print("==================\n")
        # print("Total Query", belief_manager.feedback_manager.num_of_query)

        # =========== 5. Pruning belief ===========
        prune_start = time()
        planner.prune_search_tree(action=action, obs=belief.knowledge)
        prune_elapsed = time() - prune_start
        total_prune_time += prune_elapsed
        step_log["pruning_time"] = prune_elapsed
        step_logs.append(step_log)
        
        

    total_wall_time = time() - wall_start
    executed_steps = max(len(step_logs), 1)
    if plan_end_reason is None:
        plan_end_reason = "PLAN FAILURE"
    action_schema_summary = build_action_schema_summary(
        step_logs,
        belief_manager.feedback_manager.query_log,
    )
    log_data = {
        "meta": {
            "domain": args.domain,
            "initial_state": args.initial_state,
            "threshold": args.threshold,
            "log_dir": args.log_dir,
            "seed": args.seed,
            "max_step": args.max_step,
            "max_particles": args.max_particles,
            "n_simulations": args.n_simulations,
            "gamma": args.gamma,
            "c": args.c,
            "max_depth": args.max_depth,
            "epsilon": args.epsilon,
        },
        "success": plan_success,
        "end_reason": plan_end_reason,
        "steps": len(step_logs),
        "reward": {
            "cumulated": cumulated_reward,
        },
        "actions": step_logs,
        "timing": {
            "search_time": {
                "total": total_search_time,
                "avg": total_search_time / executed_steps,
            },
            "execute_time": {
                "total": total_execute_time,
                "avg": total_execute_time / executed_steps,
            },
            "update_time": {
                "total": total_update_time,
                "avg": total_update_time / executed_steps,
            },
            "interaction_time": {
                "total": total_interaction_time,
                "avg": total_interaction_time / executed_steps,
            },
            "pruning_time": {
                "total": total_prune_time,
                "avg": total_prune_time / executed_steps,
            },
            "total_time": total_wall_time,
        },
        "questions": belief_manager.feedback_manager.query_log,
        "total_questions": belief_manager.feedback_manager.num_of_query,
        "action_schema_summary": action_schema_summary,
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = None
    if plan_end_reason != "MAX STEP":
        log_path = logger_exp(log_data, log_dir=args.log_dir)

    print("[TIME SUMMARY]")
    print(f"avg search: {total_search_time / executed_steps:.4f}s")
    print(f"avg execute: {total_execute_time / executed_steps:.4f}s")
    print(f"avg update: {total_update_time / executed_steps:.4f}s")
    print(f"avg interaction: {total_interaction_time / executed_steps:.4f}s")
    print(f"avg pruning: {total_prune_time / executed_steps:.4f}s")
    print(f"total wall clock: {total_wall_time:.4f}s")
    
    print("=============Action Log=============")
    for j, a in enumerate(action_log):
        print(f"[STEP {j+1}]: {a}")
    
    print("=============Final Knowledge=============")
    for s in sorted(belief.knowledge.facts):
        print(s)
    for key, value in belief.knowledge.fluents.items():
        print(key, value)
        
    print("=============Query=============")
    print("Total Query", belief_manager.feedback_manager.num_of_query)
    print("=============Action Schema Summary=============")
    for schema, stats in action_schema_summary.items():
        print(
            f"{schema}: actions={stats['action_count']}, "
            f"questions={stats['question_count']}, "
            f"expected_questions_per_action={stats['questions_per_action']:.4f}"
        )
    print("=============Log File=============")
    print(log_path if log_path is not None else "Skipped because plan ended by MAX STEP")


if __name__ == "__main__":
    main()
