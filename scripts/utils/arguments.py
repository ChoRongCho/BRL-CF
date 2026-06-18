import argparse
import random
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]


def parse_args(domain: str):
    """
    CLI arguments를 파싱하여 반환한다.
    domain 인자는 기본 domain 이름으로 사용된다.
    """

    parser = argparse.ArgumentParser(description="POMDP Runner")

    default_domain_rule = SCRIPTS_ROOT / "domain" / domain / "domain_rule.yaml"
    default_initial_state = SCRIPTS_ROOT / "domain" / domain / "scene_01.yaml"
    default_robot_skill = SCRIPTS_ROOT / "domain" / domain / "robot_skill.yaml"

    parser.add_argument("--domain", type=str, default=domain, help="Domain name")
    parser.add_argument("--domain_rule", type=str, default=str(default_domain_rule), help="Path to domain rule yaml")
    parser.add_argument("--initial_state", type=str, default=str(default_initial_state), help="Path to initial state yaml")
    parser.add_argument("--robot_skill", type=str, default=str(default_robot_skill), help="Path to robot-skill yaml")

    # POMCP settings
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for future rewards (0 < gamma ≤ 1)")
    parser.add_argument("--c", type=float, default=1.0, help="Exploration constant for UCB in tree search.")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum simulation depth for each rollout in POMCP.")
    parser.add_argument("--n_simulations", type=int, default=100, help="Number of Monte Carlo simulations per planning step.")
    parser.add_argument("--max_node_particles", type=int, default=None, help="Maximum number of particles cached at each POMCP tree node. Defaults to max_belief_particles.")
    parser.add_argument("--epsilon", type=float, default=0.005, help="")
    
    # Experiments settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, a random seed is generated.")
    parser.add_argument("--max_step", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--max_particles", type=int, default=250, help="Legacy alias for the maximum number of belief particles to keep after update")
    parser.add_argument("--max_belief_particles", type=int, default=250, help="Maximum number of transition-outcome particles sampled for each belief update")
    parser.add_argument("--threshold", type=float, default=0.8, help="")
    parser.add_argument("--log_dir", type=str, default="experiments_logs/system_log", help="Directory where planning logs are saved")
    
    # Numeric fluents settings
    parser.add_argument("--fluent_sample_sigma", type=float, default=0.05, help="Gaussian support width for observed fluent particles")
    parser.add_argument("--pick_fluent_sigma", type=float, default=0.05, help="Execution tolerance for comparing commanded and particle fluent values")
    parser.add_argument("--pick_success_rate", type=float, default=0.88, help="Nominal pick success probability at the ML fluent command")
    parser.add_argument("--pick_success_floor", type=float, default=0.05, help="Minimum pick success probability for distant fluent particles")
    
    # Ablation study
    parser.add_argument("--f_strategy", type=int, default=1, help="1: no, 2: all, 3: ours, 4:random")
    parser.add_argument("--q_strategy", type=int, default=1, help="1: ours 2: LLM")
    parser.add_argument("--answer_type", type=str, default="auto", help="auto: auto answer, human: you answer")
    parser.add_argument("--random_query_prob", type=float, default=0.3, help="Query trigger probability for f_strategy=4 in when_main.py")
    
    
    args = parser.parse_args()

    args.domain_rule = Path(args.domain_rule)
    args.initial_state = Path(args.initial_state)
    args.robot_skill = Path(args.robot_skill)
    if args.domain != domain:
        domain_root = SCRIPTS_ROOT / "domain" / args.domain
        if args.domain_rule == default_domain_rule:
            args.domain_rule = domain_root / "domain_rule.yaml"
        if args.initial_state == default_initial_state:
            args.initial_state = domain_root / "scene_01.yaml"
        if args.robot_skill == default_robot_skill:
            args.robot_skill = domain_root / "robot_skill.yaml"
    args.log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randrange(0, 2**32 - 1)
    
    return args
