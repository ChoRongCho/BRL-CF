"""
Docstring for main
"""

from pathlib import Path
import random
import sys
from time import time

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils.arguments import parse_args
from environments.env import Environment
from models.belief_update import BeliefManager
from planners.pomcp import POMCPPlanner
from utils.logger import logger_exp
import numpy as np


# available domain
DOMAIN = ["tomato", "wastesorting"]
WHEN_THRESHOLD = 0.8
STRATEGY_NAMES = {
    1: "no",
    2: "all",
    3: "ours",
    4: "random",
}


def _should_trigger_query(
    strategy: int,
    confidence: float,
    threshold: float,
    random_query_prob: float,
    random_query_rng: random.Random,
) -> bool:
    if strategy == 1:
        return False
    if strategy == 2:
        return True
    if strategy == 3:
        return confidence < threshold
    if strategy == 4:
        return random_query_rng.random() < random_query_prob
    raise ValueError(f"Unknown f_strategy={strategy}; expected 1(no), 2(all), 3(ours), or 4(random)")


def _ask_one_question(belief_manager: BeliefManager, belief, step: int, action_name: str, confidence: float):
    target_fact = belief_manager.feedback_manager.select_best_fact_to_ask(belief)
    if target_fact is None:
        return belief, confidence, False

    print(f"    [Query] Q: {target_fact} is True?")
    answer = belief_manager.feedback_manager.query_human(target_fact, action_name)
    belief_manager.feedback_manager.num_of_query += 1
    print(f"    [Query] A: {answer}")

    belief = belief_manager.feedback_manager.apply_fact_answer_to_belief(belief, target_fact, answer)
    updated_confidence = belief_manager.feedback_manager.compute_confidence(belief.frontier_weights)
    belief_manager.feedback_manager.query_log.append({
        "step": step,
        "action": action_name,
        "question": target_fact,
        "answer": answer,
        "confidence_before": confidence,
        "confidence_after": updated_confidence,
    })
    print(f"    [Planner] Updated confidence: {updated_confidence}")
    return belief, updated_confidence, True


def _run_feedback_policy(
    args,
    belief_manager: BeliefManager,
    belief,
    step: int,
    action_name: str,
    random_query_rng: random.Random,
):
    """
    Run the selected query timing policy after belief update.

    Random query timing uses its own RNG so POMCP rollout randomness does not
    change whether a feedback question is triggered. If random triggers, the
    caller asks at least one question regardless of the current confidence.
    """
    threshold = belief_manager.feedback_manager.conf_threshold
    strategy = args.f_strategy
    strategy_name = STRATEGY_NAMES.get(strategy, str(strategy))
    confidence = belief_manager.feedback_manager.compute_confidence(belief.frontier_weights)
    
    trigger_query = _should_trigger_query(
        strategy=strategy,
        confidence=confidence,
        threshold=threshold,
        random_query_prob=args.random_query_prob,
        random_query_rng=random_query_rng,
    )

    print(
        f"    [When] strategy={strategy_name}, confidence={confidence:.4f}, "
        f"threshold={threshold:.4f}, trigger={trigger_query}"
    )

    if trigger_query:
        queries_this_step = 0
        while queries_this_step < 1 or confidence < threshold:
            print(f"    [Planner] Current confidence: {confidence}")
            belief, confidence, asked = _ask_one_question(
                belief_manager=belief_manager,
                belief=belief,
                step=step,
                action_name=action_name,
                confidence=confidence,
            )
            if not asked:
                break
            queries_this_step += 1

    print(f"    [Planner] Final confidence: {confidence}")

    if len(belief.frontier) > 0:
        max_idx = np.argmax(belief.frontier_weights)
        max_frontier = belief.frontier[max_idx]
        prev_facts = set(belief.knowledge.facts)
        final_facts = set(max_frontier.facts)
        added_facts = [fact for fact in max_frontier.facts if fact not in prev_facts]
        deleted_facts = [fact for fact in belief.knowledge.facts if fact not in final_facts]
        belief.knowledge = max_frontier
        belief.reset_belief()
        print("    [Belief Diff]")
        print(f"      + add ({len(added_facts)}): {', '.join(added_facts) if added_facts else '-'}")
        print(f"      - del ({len(deleted_facts)}): {', '.join(deleted_facts) if deleted_facts else '-'}")

    return belief

def main():
    args = parse_args("tomato")
    args.threshold = WHEN_THRESHOLD
    random.seed(args.seed)
    np.random.seed(args.seed)
    random_query_rng = random.Random(args.seed)
    
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
    plan_end_reason = "NOT_STARTED"
    wall_start = time()
    cumulated_reward = 0.0
    belief_manager.feedback_manager.conf_threshold = WHEN_THRESHOLD
    
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
            plan_success = False
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
        belief = _run_feedback_policy(
            args=args,
            belief_manager=belief_manager,
            belief=belief,
            step=i,
            action_name=action.name,
            random_query_rng=random_query_rng,
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
    if plan_end_reason == "NOT_STARTED":
        plan_end_reason = "NO_STEPS_EXECUTED"
    log_path = logger_exp({
        "meta": {
            "domain": args.domain,
            "initial_state": args.initial_state,
            "threshold": args.threshold,
            "when_strategy": STRATEGY_NAMES.get(args.f_strategy, args.f_strategy),
            "random_query_prob": args.random_query_prob,
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
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }, log_dir=args.log_dir)

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
    print("=============Log File=============")
    print("Plan Success: ", done)
    print(log_path)


if __name__ == "__main__":
    main()
