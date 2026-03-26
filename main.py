"""
Docstring for main
"""

from time import time
from utils.arguments import parse_args
from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefManager
from models.action import Action
from planners.pomct_changmin import POMCPPlanner


# available domain
DOMAIN = ["tomato", "blocksworld", "wastesorting"]



def main():

    args = parse_args("tomato")
    env = Environment(args)
    belief_manager = BeliefManager(env.transition_model, env.observation_model, env.asp_bridge)
    planner = POMCPPlanner(args=args, env=env, belief_manager=belief_manager)
    
    observation = env.reset()
    
    belief = belief_manager.initialize_belief(env.state)

    done = False

    
    while not done:     
        
        action = planner.search(belief)    

        observation, reward, done, info = env.step(action)
                
        belief, confidence = belief_manager.update_belief(belief, observation, action)

        print("Confidence: ", confidence)
        print("==================\n")
        # print(belief)
        # asdf



if __name__ == "__main__":
    main()