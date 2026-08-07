from __future__ import annotations

from pathlib import Path
import random
import sys
from pprint import pprint

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.environments.env import Environment
from scripts.models.belief_update import BeliefManager
from scripts.planners.pomcp import POMCPPlanner
from scripts.utils.arguments import parse_args


DOMAIN = ["tomato", "wastesorting", "watering", "blocksworld", "rover", "kitchen", ]

def main():
    args = parse_args("wastesorting")
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Arguments:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key}: {value}")
    print("\n")
    
    env = Environment(args)
    
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    planner = POMCPPlanner(
        args=args,
        env=env,
        belief_manager=belief_manager,
    )

    env.reset()
    belief = belief_manager.initialize_belief(env.state)

    plan = []
    reward = 0.0
    end_reason = None

    while True:
        
        # 1. Plan next action
        print(f"\nStep {len(plan) + 1}:")
        action = planner.search(belief)
        if action is None:
            end_reason = "PLAN FAILURE"
            break
        plan.append(action.name)

        # print(action.name)
        # adsf
        
        # 2. Execute action and get observation
        observation, step_reward, _, _ = env.step(action)

        # 3. Update belief with observation
        belief = belief_manager.update_belief(belief, observation, action)
        
        # 4. Get new observation from feedback manager
        belief = belief_manager.feedback_manager.get_new_observation(
            belief=belief,
            step=len(plan),
            action_name=action.name,
        )

        reward += step_reward
        done = env.check_done(belief=belief)

        if done:
            end_reason = done
            break
        
        # 5. Prune search tree based on the new observation            
        planner.prune_search_tree(action=action, obs=belief.knowledge)


    final_result = {
        "success": end_reason == "GOAL DONE",
        "end_reason": end_reason,
        "steps": len(plan),
        "reward": reward,
        "actions": plan,
        "total_questions": belief_manager.feedback_manager.num_of_query,
        "final_facts": sorted(belief.knowledge.facts),
        "final_fluents": belief.knowledge.fluents,
    }
    print("\n")
    pprint(final_result, sort_dicts=False)



if __name__ == "__main__":
    main()
