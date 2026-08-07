import argparse
import random
from pathlib import Path

import yaml


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
ARGUMENT_DEFAULTS = Path(__file__).resolve().with_name("argument_defaults.yaml")


def _load_argument_defaults(domain: str) -> dict:
    if not ARGUMENT_DEFAULTS.exists():
        return {}

    with ARGUMENT_DEFAULTS.open("r", encoding="utf-8") as f:
        raw_defaults = yaml.safe_load(f) or {}

    defaults = dict(raw_defaults.get("default", {}) or {})
    domain_defaults = (raw_defaults.get("domains", {}) or {}).get(domain, {}) or {}
    defaults.update(domain_defaults)
    return defaults


def parse_args(domain: str):
    """
    CLI arguments를 파싱하여 반환한다.
    domain 인자는 기본 domain 이름으로 사용된다.
    """

    domain_parser = argparse.ArgumentParser(add_help=False)
    domain_parser.add_argument("--domain", type=str, default=domain)
    domain_args, _ = domain_parser.parse_known_args()
    selected_domain = domain_args.domain
    defaults = _load_argument_defaults(selected_domain)

    parser = argparse.ArgumentParser(description="POMDP Runner")

    default_domain_rule = SCRIPTS_ROOT / "domain" / selected_domain / "domain_rule.yaml"
    default_initial_state = SCRIPTS_ROOT / "domain" / selected_domain / "scene_01.yaml"
    default_robot_skill = SCRIPTS_ROOT / "domain" / selected_domain / "robot_skill.yaml"

    parser.add_argument("--domain", type=str, default=selected_domain, help="Domain name")
    parser.add_argument("--domain_rule", type=str, default=str(default_domain_rule), help="Path to domain rule yaml")
    parser.add_argument("--initial_state", type=str, default=str(default_initial_state), help="Path to initial state yaml")
    parser.add_argument("--robot_skill", type=str, default=str(default_robot_skill), help="Path to robot-skill yaml")

    # POMCP settings
    parser.add_argument("--gamma", type=float, default=defaults.get("gamma", 0.95), help="Discount factor for future rewards (0 < gamma ≤ 1)")
    parser.add_argument("--c", type=float, default=defaults.get("c", 1.0), help="Exploration constant for UCB in tree search.")
    parser.add_argument("--max_depth", type=int, default=defaults.get("max_depth", 25), help="Maximum simulation depth for each rollout in POMCP.")
    parser.add_argument("--n_simulations", type=int, default=defaults.get("n_simulations", 100), help="Number of Monte Carlo simulations per planning step.")
    parser.add_argument("--max_node_particles", type=int, default=defaults.get("max_node_particles"), help="Maximum number of particles cached at each POMCP tree node. Defaults to max_belief_particles.")
    parser.add_argument("--epsilon", type=float, default=defaults.get("epsilon", 0.005), help="")
    parser.add_argument("--profile_search", action="store_true", default=defaults.get("profile_search", False), help="Print detailed timing breakdown for POMCP search.")
    
    # Experiments settings
    parser.add_argument("--seed", type=int, default=defaults.get("seed"), help="Random seed. If omitted, a random seed is generated.")
    parser.add_argument("--max_step", type=int, default=defaults.get("max_step", 25), help="Maximum steps per episode")
    parser.add_argument("--max_particles", type=int, default=defaults.get("max_particles", 250), help="Legacy alias for the maximum number of belief particles to keep after update")
    parser.add_argument("--max_belief_particles", type=int, default=defaults.get("max_belief_particles", 8000), help="Maximum number of transition-outcome particles sampled for each belief update")
    parser.add_argument("--threshold", type=float, default=defaults.get("threshold", 0.8), help="")
    parser.add_argument("--log_dir", type=str, default=defaults.get("log_dir", "experiments_logs/system_log"), help="Directory where planning logs are saved")
    
    # Numeric fluents settings
    parser.add_argument("--fluent_sample_sigma", type=float, default=defaults.get("fluent_sample_sigma", 0.05), help="Gaussian support width for observed fluent particles")
    parser.add_argument("--pick_fluent_sigma", type=float, default=defaults.get("pick_fluent_sigma", 0.05), help="Execution tolerance for comparing commanded and particle fluent values")
    parser.add_argument("--pick_success_rate", type=float, default=defaults.get("pick_success_rate", 0.88), help="Nominal pick success probability at the ML fluent command")
    parser.add_argument("--pick_success_floor", type=float, default=defaults.get("pick_success_floor", 0.05), help="Minimum pick success probability for distant fluent particles")
    
    # Ablation study
    parser.add_argument("--f_strategy", type=int, default=defaults.get("f_strategy", 1), help="1: no, 2: all, 3: ours, 4:random")
    parser.add_argument("--q_strategy", type=int, default=defaults.get("q_strategy", 1), help="1: ours 2: LLM")
    parser.add_argument(
        "--answer_type",
        type=str,
        default=defaults.get("answer_type", "oracle"),
        choices=["oracle", "noisy-oracle", "human-proxy", "random", "human", "auto"],
        help="Feedback answer mode. oracle: domain oracle answer, noisy-oracle: noisy domain oracle answer, human-proxy: domain proxy, random: random answer, human: terminal input, auto: alias for oracle",
    )
    parser.add_argument(
        "--noisy_oracle_error_rate",
        type=float,
        default=defaults.get("noisy_oracle_error_rate", 0.1),
        help="Flip probability for answer_type=noisy-oracle.",
    )
    parser.add_argument("--random_query_prob", type=float, default=defaults.get("random_query_prob", 0.3), help="Query trigger probability for FEEDBACK_METHOD=ours-random-when in main.py")
    
    
    args = parser.parse_args()

    args.domain_rule = Path(args.domain_rule)
    args.initial_state = Path(args.initial_state)
    args.robot_skill = Path(args.robot_skill)
    args.log_dir = Path(args.log_dir)
    if args.seed is None:
        args.seed = random.randrange(0, 2**32 - 1)
    if args.noisy_oracle_error_rate > 1.0:
        args.noisy_oracle_error_rate /= 100.0
    if not 0.0 <= args.noisy_oracle_error_rate <= 1.0:
        parser.error("--noisy_oracle_error_rate must be in [0, 1] or given as a percentage in [0, 100].")
    
    return args
