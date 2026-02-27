
from scripts import GT_MODEL_CONFIDENCE, ACTIONS
from scripts.knowledge_base import KnowledgeBase, generate_kb
from scripts.planner import Planner
from scripts.feedback_manager import FeedbackManager
from scripts.utils.util import plot_history, save_as_json


def main():
    kb = generate_kb()
    kb.pretty_print()
    
    # settings
    lam = 1
    nsample = 10
    hori = 100
    save_dir = f'data/lam{lam}_nsample{nsample}_seed43_hori{hori}'
    
    # prior
    planner = Planner()
    plan = planner.generate_random_plan(horizon=hori)
    planner.dump_plan(plan)
    
    # executing plan and query
    fm = FeedbackManager(lam=lam, n_sample=nsample)
    history, query_call, cls_stats, logs = fm.run(plan, kb)
    
    filename = f"data/log_lam{lam}_nsample{nsample}_seed43_hori{hori}.json"
    save_as_json(query_rate=query_call, cls_stats=cls_stats, logs=logs, filename=filename)
    # plot_history(history, save_dir=save_dir)  
    
    kb.pretty_print()
    print(f"The rate of total query: {query_call}%")
    print(f"The rate of stats: {cls_stats}%")


if __name__ == "__main__":
    main()
