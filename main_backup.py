"""
Docstring for main
"""

from time import time

from utils.asp import solve_asp
from environments.env import Environment
from models.belief_update import BeliefManager
from models.action import Action

from planners.pomct_changmin import POMCPPlanner

from utils.arguments import parse_args




# available domain
DOMAIN = ["tomato", "blocksworld", "wastesorting"]



def main():

    args = parse_args("tomato")
    env = Environment(args)
    belief = BeliefManager(env.transition_model, env.observation_model, env.asp_bridge)
    planner = POMCPPlanner(args=args, env=env, belief_manager=belief)

    """
    time = 0
    while time <= 10:
        time += 1
        
        1. action = planner.Search()
        
        print(planner.tree.nodes[-1][:4])
        print(action)
        2. observation = choice(O)
        
        3. planner.tree.prune_after_action(action,observation)
        
        4. planner.UpdateBelief(action, observation)
    """
    
    observation = env.reset()
    
    b = belief.initialize_belief(env.state)
        
    done = False
    
    # asdf
    
    while not done:     
        
        # env.render()
        
        # action = planner.sample_action(b)
        action = planner.search()    
        
        asdf
        
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
        
        for act in applicable_actions:
            if 'detect' in act.name:
                action = act
                break
            
        # action = random.choice(applicable_actions)
        print(action.name)

        # action = planner.sample_action(belief=b)
        # ===========================================================
        # action = planner.sample_action(belief=b)
        
        observation, reward, done, info = env.step(action)
        
        next_b, confidence = belief.update_belief(b, observation, action)
        
        
        
        
        print(next_b)
        print(confidence)
        # enhancing observation
        
        
        asdf
        print(f"\nStep: {info['step_count']}")
        print(f"Action: {action.name}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Confidence: {confidence}")



if __name__ == "__main__":
    main()