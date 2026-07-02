from __future__ import annotations

from typing import Any

from models.belief import Belief


def _parse_fact(raw_fact: str):
    fact = raw_fact.replace(" ", "")
    if not fact.endswith(")"):
        return None, ()

    predicate, sep, args = fact[:-1].partition("(")
    if not sep:
        return None, ()

    return predicate, tuple(args.split(","))


def check_done(env: Any, belief: Belief):
    """Goal 달성 또는 max_step 초과 시 episode 종료"""
    if env.domain_name == "tomato":
        true_at_tomatoes = set()
        true_moved_tomatoes = set()
        unripe_tomatoes = set()
        ripe_tomatoes = set()
        rotten_tomatoes = set()
        held_tomatoes = set()
        discarded_tomatoes = set()
        loaded_tomatoes = set()

        for raw_fact in env.true_state.facts:
            predicate, args = _parse_fact(raw_fact)
            if not args:
                continue

            if predicate == "at":
                true_at_tomatoes.add(args[0])
            elif predicate in {"holding", "holded", "loaded", "discarded"}:
                tomato = args[1] if predicate == "holding" and len(args) >= 2 else args[0]
                true_moved_tomatoes.add(tomato)

        for raw_fact in belief.knowledge.facts:
            predicate, args = _parse_fact(raw_fact)
            if not args:
                continue

            tomato = args[0]
            if predicate == "unripe":
                unripe_tomatoes.add(tomato)
            elif predicate == "ripe":
                ripe_tomatoes.add(tomato)
            elif predicate == "rotten":
                rotten_tomatoes.add(tomato)
            elif predicate == "holding" and len(args) >= 2:
                held_tomatoes.add(args[1])
            elif predicate == "discarded":
                discarded_tomatoes.add(tomato)
            elif predicate == "loaded":
                loaded_tomatoes.add(tomato)

        if (
            held_tomatoes & unripe_tomatoes
            or discarded_tomatoes & ripe_tomatoes
            or loaded_tomatoes & rotten_tomatoes
        ):
            return "PLAN FAILURE"

        if true_at_tomatoes & true_moved_tomatoes:
            return "PLAN FAILURE"

    elif env.domain_name == "wastesorting":
        goal_bin_by_waste = {}

        if env.goal:
            for goal_fact in env.goal.facts:
                predicate, args = _parse_fact(goal_fact)
                if predicate == "in_bin" and len(args) >= 2:
                    goal_bin_by_waste[args[0]] = args[1]

        for raw_fact in belief.knowledge.facts:
            predicate, args = _parse_fact(raw_fact)
            if not args:
                continue

            if predicate != "in_bin" or len(args) < 2:
                continue

            waste, bin_name = args[:2]
            goal_bin = goal_bin_by_waste.get(waste)
            if goal_bin and bin_name != goal_bin:
                return "PLAN FAILURE"

    if env.goal and all(belief.knowledge.has_fact(f) for f in env.goal.facts):
        return "GOAL DONE"

    if env.step_count >= env.max_step:
        return "MAX STEP"

    return False
