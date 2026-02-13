from scripts.knowledge_base import KnowledgeBase, generate_kb
from scripts.planner import Planner, generate_action
from scripts.feedback_manager import FeedbackManager
from scripts.utils.plot_confidence import plot_history


def main():
    kb = generate_kb()
    kb.pretty_print()
    
    # prior
    actions = generate_action()
    planner = Planner(actions=actions)
    plan = planner.generate_random_plan(horizon=1000)
    
    # executing plan and query
    fm = FeedbackManager(lam=2.0, n_sample=20)
    history = fm.run_and_record(plan, kb)
    plot_history(history, save_dir='data/lam2_nsample20_seed43_hori1000')  
    
    kb.pretty_print()


if __name__ == "__main__":
    main()
