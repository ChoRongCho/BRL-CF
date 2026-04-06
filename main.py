"""
Docstring for main
"""

from time import time
from utils.arguments import parse_args
from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefManager
from models.action import Action
from planners.pomct import POMCPPlanner


# available domain
DOMAIN = ["tomato", "blocksworld", "wastesorting"]


def print_fluents(tag, fluents, objects=None):
    print(f"[FLUENTS] {tag}")

    if not fluents and not objects:
        print("  <empty>")
        return

    target_objects = objects or sorted(fluents.keys())
    fluent_keys = ["pose_x", "pose_y", "pose_z"]

    for obj in target_objects:
        values = fluents.get(obj, {})
        formatted = ", ".join(
            f"{key}={values.get(key, '<missing>')}"
            for key in fluent_keys
        )
        print(f"  {obj}: {formatted}")



def main():

    args = parse_args("tomato")
    env = Environment(args)
    
    # for key, values in env.transition_model.transition_table.items():
    #     print()
    #     print(key)
    #     for val in values:
    #         print(val.add_facts, val.probability)
        
    
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
    total_step_time = 0.0
    total_update_time = 0.0
    total_prune_time = 0.0
    action_log = []
    tomato_objects = env.obj_type.get("tomato(T)", [])
    
    
    # print_fluents("initial env.state", env.state.fluents, tomato_objects)
    # print_fluents("initial belief.knowledge", belief.knowledge.fluents, tomato_objects)

    # for fact in env.state.facts:
    #     print(fact)
    # asdf
    while not done:     
        i += 1
        print(f"Step: {i}")
        search_start = time()
        action = planner.search(belief)
        action_log.append(action.name)
        search_elapsed = time() - search_start
        total_search_time += search_elapsed
        print(f"[ACTION] {action.name}")
        
        # asdf

        step_start = time()
        observation, reward, done, info = env.step(action)
        step_elapsed = time() - step_start
        total_step_time += step_elapsed
        # print_fluents("env.state after step", env.state.fluents, tomato_objects)
        # print_fluents("observation", observation.state.fluents, tomato_objects)
        
        update_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        update_elapsed = time() - update_start
        total_update_time += update_elapsed
        # print_fluents("belief.knowledge after update", belief.knowledge.fluents, tomato_objects)

        prune_start = time()
        planner.prune_search_tree(action=action, observation=observation)
        prune_elapsed = time() - prune_start
        total_prune_time += prune_elapsed
        
        print("==================\n")

    if i > 0:
        print("[TIME SUMMARY]")
        print(f"avg search: {total_search_time / i:.4f}s")
        print(f"avg env.step: {total_step_time / i:.4f}s")
        print(f"avg update_belief: {total_update_time / i:.4f}s")
        print(f"avg prune_search_tree: {total_prune_time / i:.4f}s")
        
        print("=============Action Log=============")
        for j, a in enumerate(action_log):
            print(f"[STEP {j+1}]: {a}")
        
        print("=============Final Knowledge=============")
        for s in sorted(belief.knowledge.facts):
            print(s)
        for key, value in belief.knowledge.fluents.items():
            print(key, value)


if __name__ == "__main__":
    main()
