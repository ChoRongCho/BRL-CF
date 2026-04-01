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



def main():

    args = parse_args("tomato")
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
    
    while not done:     
        i += 1
        print(f"Step: {i}")
        action = planner.search(belief)    

        observation, reward, done, info = env.step(action)
        
        belief = belief_manager.update_belief(belief, observation, action)

        planner.prune_search_tree(action=action, observation=observation)
        
        print("==================\n")


if __name__ == "__main__":
    main()