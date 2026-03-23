"""
Docstring for main
"""

from environments.env import Environment
from models.belief import Belief
from planners.planner import Planner
from utils.arguments import parse_args
import random
from utils.asp import solve_asp
from time import time

DOMAIN = ["tomato", "blocksworld", "wastesorting"]

def main():

    args = parse_args("tomato")
    env = Environment(args)
    # belief = Belief(args)
    planner = Planner(actions=env.actions)
    
    observation = env.reset()
    # b = belief.get_belief(env.state)
        
    done = False
    
    # for a in env.actions:
        
    
    #     print(a.name, a.observation)
    
    return 0

    while not done:     
        """
        Docstring for planning loop
        
        1. action = policy.get_action(b)
        
        2. observation, reward, done, info = env.step(action)
        
        3. n_b = belief.update_belief(b, observation, action | TransModel, ObsModel)
        
        4. n_b = enhancing_obs(belief, action, observation)
        
        5. policy.train(n_b, action, reward)
        
        6. b = n_b
        """
        
        # action = policy(belief, observation)        
        # 임시 정책 (random policy)

        # env.render()

        action = planner.sample_action(belief=b)
        # applicable_actions = [
        #     a for a in env.actions if a.is_applicable(env.state)
        # ]
        # for app in applicable_actions:
        #     print("  Applicable action: ", app.name)
        # if not applicable_actions:
        #     print("No applicable actions. Terminating.")
        #     break
        # action = random.choice(applicable_actions)
        
        
        observation, reward, done, info = env.step(action)

        print(f"Step: {info['step_count']}")
        print(f"Action: {action.name}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}\n")

        # done = True



if __name__ == "__main__":
    main()