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
    # planner = Planner(actions=env.actions)
    
    observation = env.reset()
    b = belief.set_initial_belief(env.state)
        
    done = False
    
    print(b)
        
    # return 0

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

        # action = planner.sample_action(belief=b)
        applicable_actions = [
            a for a in env.actions if a.is_applicable(env.state)
        ]
        for app in applicable_actions:
            print("Applicable action: ", app.name)
        if not applicable_actions:
            print("No applicable actions. Terminating.")
            break
        action = random.choice(applicable_actions)
        
        
        observation, reward, done, info = env.step(action)
        
        env.observation_model.pretty_print_distribution(env.state, action=action)
        

        print(f"Step: {info['step_count']}")
        print(f"Action: {action.name}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}\n")
        
        n_b = belief.update_belief(b, observation, action)
        for front, weight in zip(n_b.frontier, n_b.weights):
            print(front.facts, weight)
        # done = True



if __name__ == "__main__":
    main()