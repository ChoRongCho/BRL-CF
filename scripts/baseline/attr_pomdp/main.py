#!/usr/bin/env python3
"""Run the grounded-fact adaptation of Attr-POMDP."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from time import perf_counter

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
from baseline.attr_pomdp.scripts.rewards import negative_entropy
from environments.env import Environment
from utils.logger import logger_exp


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


def _defaults() -> Defaults:
    return Defaults(
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


def main() -> Path:
    args, attr_args = parse_configuration(sys.argv[1:], _defaults())

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = Environment(args)
    provider = make_provider(attr_args.feedback_source, args.domain)
    # Fail before the experiment starts instead of silently substituting Oracle.
    if attr_args.feedback_source != "oracle":
        provider.answer(None, env.gt_init_state, env.state)  # type: ignore[arg-type]

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
    action_logs = []
    question_logs = []
    cumulative_reward = 0.0
    start = perf_counter()
    end_reason = "PLAN FAILURE"

    while len(action_logs) < args.max_step:
        search_start = perf_counter()
        plan = planner.search(
            belief,
            context_action_name=context_action_name,
            asked_facts=frozenset(asked_facts),
        )
        search_time = perf_counter() - search_start
        if plan.action is None:
            break

        entropy_before = negative_entropy(belief.particle_weights)
        execute_start = perf_counter()
        answer = None
        if isinstance(plan.action, QueryAction):
            answer = provider.answer(plan.action, env.gt_init_state, env.state)
            belief = belief_model.query_update(belief, plan.action, answer, provider)
            if answer:
                env.state.add_fact(plan.action.target_fact)
            asked_facts.add(plan.action.target_fact)
            reward = -float(plan.action.cost)
            question_logs.append(
                {
                    "step": len(action_logs) + 1,
                    "target_fact": plan.action.target_fact,
                    "context_action_name": plan.action.context_action_name,
                    "provider": plan.action.provider,
                    "answer": answer,
                    "cost": float(plan.action.cost),
                }
            )
            action_type = "query"
        else:
            observation, reward, _, _ = env.step(plan.action)
            belief = belief_model.update(belief, plan.action, observation)
            context_action_name = plan.action.name
            action_type = "task"

        execute_time = perf_counter() - execute_start
        cumulative_reward += float(reward)
        entropy_after = negative_entropy(belief.particle_weights)
        action_logs.append(
            {
                "step": len(action_logs) + 1,
                "action": plan.action.name,
                "action_type": action_type,
                "target_fact": getattr(plan.action, "target_fact", None),
                "context_action_name": context_action_name,
                "provider": attr_args.feedback_source if action_type == "query" else None,
                "answer": answer,
                "q_value": plan.value,
                "action_values": plan.action_values,
                "step_reward": float(reward),
                "cumulated_reward": cumulative_reward,
                "belief_particles": len(belief.particles),
                "belief_entropy_before": entropy_before,
                "belief_entropy_after": entropy_after,
                "search_time": search_time,
                "execute_time": execute_time,
            }
        )
        print(
            f"[Attr-POMDP] step={len(action_logs)} type={action_type} "
            f"action={plan.action.name} Q={plan.value:.4f} "
            f"H={entropy_before:.4f}->{entropy_after:.4f}"
        )

        done = env.check_done(belief=belief)
        if done:
            end_reason = done
            break
    else:
        end_reason = "MAX STEP"

    result = {
        "meta": {
            "method": "active-search",
            "feedback_source": attr_args.feedback_source,
            "implementation": "adapted_attr_pomdp",
            "reference": "Yang et al., Interactive Robotic Grasping with Attribute-Guided Disambiguation, ICRA 2022",
            "adaptation": "attribute questions replaced by grounded Boolean fact queries",
            "domain": args.domain,
            "initial_state": str(args.initial_state),
            "seed": args.seed,
            "max_step": args.max_step,
            "horizon": args.max_depth,
            "gamma": args.gamma,
            "query_cost": attr_args.query_cost,
            "solver": "online finite-horizon particle-belief expectimax",
        },
        "success": end_reason == "GOAL DONE",
        "end_reason": end_reason,
        "steps": len(action_logs),
        "reward": {"cumulated": cumulative_reward},
        "actions": action_logs,
        "questions": question_logs,
        "total_questions": len(question_logs),
        "timing": {"total_time": perf_counter() - start},
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(result, args.log_dir)
    attr_log = log_path.with_suffix(".active_search.json")
    attr_log.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Attr-POMDP] success={result['success']} reason={end_reason} "
        f"steps={len(action_logs)} questions={len(question_logs)} "
        f"reward={cumulative_reward:.4f}"
    )
    print(f"[Attr-POMDP] log={attr_log}")
    return attr_log


if __name__ == "__main__":
    main()
