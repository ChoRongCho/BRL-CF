
from scripts import GT_MODEL_CONFIDENCE, ACTIONS
from scripts.knowledge_base import KnowledgeBase, generate_kb
from scripts.planner import Planner
from scripts.feedback_manager import FeedbackManager
from scripts.utils.util import plot_history, save_as_json


def main():
    kb = generate_kb()
    # kb.pretty_print()
    
    # settings
    lam = 5  # the confidence of the generated value => high variance
    nsample = 20  # the number of the sample => expected value
    hori = 100  # horizon of the plan
    gamma = 0.95  # the tracking speed of the GT
    
    # folders
    save_dir = f'data/v4/lam{lam}_nsample{nsample}_gamma{gamma}_hori{hori}'
    filename = f"data/v4/log_lam{lam}_nsample{nsample}_gamma{gamma}_hori{hori}.json"
    
    # prior
    planner = Planner()
    plan = planner.generate_random_plan(horizon=hori)
    planner.dump_plan(plan)
    # planner.dump_actions()
    
    target = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    target = [0.5]
    
    # # executing plan and query
    for thres in target:
        fm = FeedbackManager(lam=lam, n_sample=nsample, gamma=gamma, no_oracle=True,
                            use_threshold=True, threshold=thres, 
                            use_timevariant=True)
        history, gt_history, query_call, cls_stats, logs = fm.run(plan, kb)
        
        plot_history(history, gt_history, save_dir=save_dir)  
        save_as_json(query_rate=query_call, cls_stats=cls_stats, logs=logs, filename=filename)
        
        # kb.pretty_print()
        print(f"The target threshold: {thres}")
        print(f"The rate of total query: {query_call}%")
        print(f"The stats: {cls_stats}\n")
        # rates = cls_stats["rates"]


if __name__ == "__main__":
    main()
