#!/usr/bin/env python3
"""Attr-POMDP algorithm only: no logging, saving, timing, or console output."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline.attr_pomdp.scripts.belief import ParticleBeliefModel
from baseline.attr_pomdp.scripts.config import Defaults, parse_configuration
from baseline.attr_pomdp.scripts.initial_belief import initialize_attribute_belief
from baseline.attr_pomdp.scripts.planner import AttrPOMDPPlanner
from baseline.attr_pomdp.scripts.providers import make_provider
from baseline.attr_pomdp.scripts.query import QueryAction, QueryFactory
from environments.env import Environment


# Edit these values to run without spelling out CLI options.
# Explicit CLI options override the corresponding values below.
DOMAIN = "tomato"
SCENE = "01"
FEEDBACK_SOURCE = "oracle"
QUERY_COST = 1.0
ACTION_LIMIT = 20
QUERY_LIMIT = 20
MAX_DEPTH = 2
MAX_STEP = 25
SEED = None


def main():
    defaults = Defaults(
        domain=DOMAIN,
        scene=SCENE,
        feedback_source=FEEDBACK_SOURCE,
        query_cost=QUERY_COST,
        action_limit=ACTION_LIMIT,
        query_limit=QUERY_LIMIT,
        max_depth=MAX_DEPTH,
        max_step=MAX_STEP,
        seed=SEED,
    )
    args, attr_args = parse_configuration(sys.argv[1:], defaults)

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = Environment(args)
    provider = make_provider(attr_args.feedback_source, args.domain)
    belief_model = ParticleBeliefModel(
        env.transition_model,
        env.observation_model,
        getattr(args, "max_belief_particles", 8000),
    )
    query_factory = QueryFactory(
        args.domain,
        provider=attr_args.feedback_source,
        query_cost=attr_args.query_cost,
    )
    planner = AttrPOMDPPlanner(
        args,
        env,
        belief_model,
        query_factory,
        provider,
        action_limit=attr_args.action_limit,
        query_limit=attr_args.query_limit,
    )

    env.reset()
    belief = initialize_attribute_belief(args.domain, env.state, env.obj_type)
    asked_facts: set[str] = set()
    context_action_name: str | None = None

    for _ in range(args.max_step):
        
        print(f"Step {_ + 1}: Belief has {len(belief.particles)} particles.")
        
        plan = planner.search(
            belief,
            context_action_name=context_action_name,
            asked_facts=frozenset(asked_facts),
        )
        
        if plan.action is None:
            break

        if isinstance(plan.action, QueryAction):
            answer = provider.answer(plan.action, env.gt_init_state, env.state)
            belief = belief_model.query_update(belief, plan.action, answer, provider)
            if answer:
                env.state.add_fact(plan.action.target_fact)
            asked_facts.add(plan.action.target_fact)
        else:
            observation, _, _, _ = env.step(plan.action)
            belief = belief_model.update(belief, plan.action, observation)
            context_action_name = plan.action.name

        if env.check_done(belief=belief):
            break

    return belief


if __name__ == "__main__":
    main()
