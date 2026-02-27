from scripts import GT_MODEL_CONFIDENCE, ACTIONS
from scripts.planner import Planner, Action



def main():
    planner = Planner()
    plan = planner.generate_random_plan(horizon=100)
    planner.dump_plan(plan)
    
    # fm = 







if __name__ == "__main__":
    main()
