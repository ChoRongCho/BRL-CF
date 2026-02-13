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
    fm = FeedbackManager(lam=1, n_sample=10)
    history, query_call = fm.run_and_record(plan, kb)
    plot_history(history, save_dir='data/lam1_nsample10_seed43_hori20')  
    
    kb.pretty_print()
    print(f"The rate of total query: {query_call}%")


if __name__ == "__main__":
    main()
