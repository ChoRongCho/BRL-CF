"""
Docstring for main
"""

from time import time
from utils.arguments import parse_args
from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefManager
from models.action import Action
from planners.pomcp import POMCPPlanner
from utils.logger import logger_exp


# available domain
DOMAIN = ["tomato", "wastesorting"]


def main():
    args = parse_args("wastesorting")
    
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
            print("[Planner] Dead-End")
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
            step_logs.append(step_log)
            break
        
        elif done == "MAX STEP":
            plan_success = False
            step_logs.append(step_log)
            break
        
        elif done == "PLAN FAILURE":
            plan_success = False
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
    log_path = logger_exp({
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
    print(log_path)


if __name__ == "__main__":
    main()
