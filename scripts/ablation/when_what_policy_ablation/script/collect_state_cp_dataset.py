#!/usr/bin/env python3
"""Collect held-out post-observation beliefs for State-CP calibration."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from environments.env import Environment
from models.belief_update import BeliefManager
from planners.pomcp import POMCPPlanner

from .calibration_truth import build_calibration_truth_facts
from .cp_when import ambiguous_facts
from .state_cp_dataset import SCHEMA_VERSION, make_belief_record


def episode_seed(base_seed: int, domain: str, scene: int, episode: int) -> int:
    payload = f"{base_seed}|{domain}|{scene}|{episode}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_args(config: argparse.Namespace, domain: str, scene: int, seed: int):
    root = PROJECT_ROOT / "scripts" / "domain" / domain
    return SimpleNamespace(
        domain=domain,
        domain_rule=root / "domain_rule.yaml",
        initial_state=root / f"scene_{scene:02}.yaml",
        robot_skill=root / "robot_skill.yaml",
        env_setting=root / "env_setting.yaml",
        gamma=config.gamma,
        c=config.c,
        max_depth=config.max_depth,
        n_simulations=config.n_simulations,
        max_node_particles=config.max_node_particles,
        epsilon=config.epsilon,
        seed=seed,
        max_step=config.max_step,
        max_particles=config.max_particles,
        max_belief_particles=config.max_belief_particles,
        threshold=0.8,
        log_dir=Path("/tmp/state_cp_collection"),
        fluent_sample_sigma=0.05,
        pick_fluent_sigma=0.05,
        pick_success_rate=0.88,
        pick_success_floor=0.05,
        f_strategy=1,
        q_strategy=1,
        answer_type="auto",
        use_interface=False,
        interface_host="0.0.0.0",
        interface_port=9765,
        interface_timeout=300.0,
        random_query_prob=0.0,
    )


def collect_episode(
    config: argparse.Namespace,
    domain: str,
    scene: int,
    episode: int,
    seed: int,
) -> tuple[dict[str, Any] | None, str]:
    random.seed(seed)
    np.random.seed(seed)
    args = build_args(config, domain, scene, seed)
    env = Environment(args)
    manager = BeliefManager(
        args, env.transition_model, env.observation_model, env.asp_bridge
    )
    planner = POMCPPlanner(args=args, env=env, belief_manager=manager)
    env.reset()
    belief = manager.initialize_belief(env.state)
    candidates = []
    end_reason = "MAX STEP"
    output = None if config.verbose else io.StringIO()

    for step in range(1, config.max_step + 1):
        with redirect_stdout(output) if output is not None else _null_context():
            action = planner.search(belief)
        if action is None:
            end_reason = "PLAN FAILURE"
            break
        observation, _, _, _ = env.step(action)
        belief = manager.update_belief(belief, observation, action)
        try:
            calibration_truth = build_calibration_truth_facts(
                belief=belief,
                candidate_facts=ambiguous_facts(belief),
                oracle_answer=lambda fact: manager.feedback_manager.query_oracle(
                    fact,
                    action.name,
                    observation_facts=observation.state.facts,
                    oracle_state_facts=env.true_state.facts,
                    oracle_successor_facts=env.true_state.facts,
                ),
            )
        except ValueError as error:
            raise ValueError(
                f"Calibration truth failed: domain={domain}, scene={scene}, "
                f"episode={episode}, seed={seed}, step={step}, action={action.name}: "
                f"{error}"
            ) from error
        record = make_belief_record(
            belief,
            calibration_truth.facts,
            domain=domain,
            scene=scene,
            episode=episode,
            seed=seed,
            step=step,
            action=action.name,
            observation_facts=observation.state.facts,
        )
        if record is not None:
            record["truth_resolution"] = calibration_truth.resolution
            record["matching_truth_hypotheses"] = calibration_truth.matching_hypotheses
            candidates.append(record)

        belief.sync_knowledge_to_map()
        belief.reset_belief()
        done = env.check_done(belief=belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            end_reason = done
            break
        planner.prune_search_tree(action=action, obs=belief.knowledge)

    if not candidates:
        return None, end_reason
    chooser = random.Random(seed ^ 0x43504441)
    selected = candidates[chooser.randrange(len(candidates))]
    selected["available_contexts_in_episode"] = len(candidates)
    selected["episode_end_reason"] = end_reason
    return selected, end_reason


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def assign_splits(
    records: list[dict[str, Any]], calibration_fraction: float, base_seed: int
) -> None:
    for domain in sorted({record["domain"] for record in records}):
        subset = [record for record in records if record["domain"] == domain]
        random.Random(base_seed ^ sum(map(ord, domain))).shuffle(subset)
        calibration_n = int(len(subset) * calibration_fraction)
        for index, record in enumerate(subset):
            record["split"] = "calibration" if index < calibration_n else "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["tomato", "wastesorting"])
    parser.add_argument("--scenes", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes-per-scene", type=int, default=25)
    parser.add_argument("--calibration-fraction", type=float, default=0.8)
    parser.add_argument("--base-seed", type=int, default=2026092101)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--max-step", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--max-particles", type=int, default=250)
    parser.add_argument("--max-belief-particles", type=int, default=8000)
    parser.add_argument("--max-node-particles", type=int, default=8000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if set(args.domains) - {"tomato", "wastesorting"}:
        parser.error("domains must be tomato and/or wastesorting")
    if args.episodes_per_scene < 1 or not args.scenes or min(args.scenes) < 1:
        parser.error("episodes and scenes must be positive")
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("calibration fraction must be between 0 and 1")
    return args


def main() -> int:
    config = parse_args()
    records = []
    missing = []
    total = len(config.domains) * len(config.scenes) * config.episodes_per_scene
    index = 0
    for domain in config.domains:
        for scene in config.scenes:
            for episode in range(1, config.episodes_per_scene + 1):
                index += 1
                seed = episode_seed(config.base_seed, domain, scene, episode)
                record, reason = collect_episode(
                    config, domain, scene, episode, seed
                )
                if record is None:
                    missing.append({
                        "domain": domain, "scene": scene, "episode": episode,
                        "seed": seed, "end_reason": reason,
                    })
                else:
                    records.append(record)
                print(
                    f"\rProgress: {index}/{total}, records={len(records)}, "
                    f"missing={len(missing)}",
                    end="",
                    flush=True,
                )
    print()
    assign_splits(records, config.calibration_fraction, config.base_seed)
    output = Path(config.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "collector": "held-out physical POMCP episodes without queries",
            "one_context_per_episode": True,
            "domains": config.domains,
            "scenes": config.scenes,
            "episodes_per_scene": config.episodes_per_scene,
            "calibration_fraction": config.calibration_fraction,
            "base_seed": config.base_seed,
            "gamma": config.gamma,
            "n_simulations": config.n_simulations,
            "max_step": config.max_step,
            "evaluation_seed_overlap_allowed": False,
        },
        "records": records,
        "episodes_without_ambiguous_context": missing,
    }
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(records)} independent episode records: {output}")
    if missing:
        print(f"Episodes without an ambiguous context: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
