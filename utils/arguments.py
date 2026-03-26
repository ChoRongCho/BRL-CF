import argparse
from pathlib import Path


def parse_args(domain: str):
    """
    CLI arguments를 파싱하여 반환한다.
    domain 인자는 기본 domain 이름으로 사용된다.
    """

    parser = argparse.ArgumentParser(description="POMDP Runner")

    parser.add_argument(
        "--domain",
        type=str,
        default=domain,
        help="Domain name"
    )

    parser.add_argument(
        "--domain_rule",
        type=str,
        default=f"domain/{domain}/domain_rule.yaml",
        help="Path to domain rule yaml"
    )

    parser.add_argument(
        "--initial_state",
        type=str,
        default=f"domain/{domain}/initial_state.yaml",
        # default=f"domain/{domain}/init_test.yaml",
        help="Path to initial state yaml"
    )
    
    parser.add_argument(
        "--robot_skill",
        type=str,
        default=f"domain/{domain}/robot_skill.yaml",
        help="Path to robot-skill yaml"
    )
    
    parser.add_argument(
        "--stochastic_action",
        type=bool,
        default=False,
        help="Is action stochastic"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed"
    )

    parser.add_argument(
        "--max_step",
        type=int,
        default=30,
        help="Maximum steps per episode"
    )

    # POMCP settings
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for future rewards (0 < gamma ≤ 1). Higher values prioritize long-term rewards."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Exploration constant for UCB in tree search. Larger values encourage exploration over exploitation."
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum simulation depth (planning horizon) for each rollout in POMCP."
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations (tree traversals) per planning step."
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=100,
        help="Number of particles sampled from belief to approximate the state distribution."
    )
    
    args = parser.parse_args()

    args.domain_rule = Path(args.domain_rule)
    args.initial_state = Path(args.initial_state)
    args.robot_skill = Path(args.robot_skill)
    
    return args