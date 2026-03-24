"""
Docstring for main
"""

import random
from time import time

from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefModel
from planners.planner import Planner
from utils.arguments import parse_args


# available domain
DOMAIN = ["tomato", "blocksworld", "wastesorting"]



def main():

    args = parse_args("tomato")
    env = Environment(args)
    belief = BeliefModel(env.transition_model, env.observation_model)
    planner = Planner(
        args=args,
        actions=env.actions,
        transition_model=env.transition_model,
        observation_model=env.observation_model,
        reward_model=env.reward_model,
    )
    
    observation = env.reset()
    b = belief.set_initial_belief(env.state)
        
    done = False
    
    while not done:     
        
        action = planner.sample_action(b)
        # ====================== Random action ======================
        applicable_actions = [
            a for a in env.actions if a.is_applicable(env.state)
        ]
        # for app in applicable_actions:
        #     print("Applicable action: ", app.name)
        #     pass
        if not applicable_actions:
            print("No applicable actions. Terminating.")
            break
        action = random.choice(applicable_actions)
        # action = planner.sample_action(belief=b)
        # ===========================================================
        
        observation, reward, done, info = env.step(action)
        
        n_b, confidence = belief.update_belief(b, observation, action)
        
        # enhancing observation
        b = n_b
        
        
        print(f"\nStep: {info['step_count']}")
        print(f"Action: {action.name}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Confidence: {confidence}")



if __name__ == "__main__":
    main()