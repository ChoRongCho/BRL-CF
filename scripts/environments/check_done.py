from __future__ import annotations

from typing import Any

from models.belief import Belief


def _normalize_fact(raw_fact: str) -> str:
    return str(raw_fact).replace(" ", "")


def _parse_fact(raw_fact: str):
    fact = _normalize_fact(raw_fact)
    if "(" not in fact and ")" not in fact:
        return fact, ()

    if not fact.endswith(")"):
        return None, ()

    predicate, sep, args = fact[:-1].partition("(")
    if not sep:
        return None, ()

    if args == "":
        return predicate, ()
    return predicate, tuple(args.split(","))


def _knowledge_facts(env: Any, belief: Belief | None) -> set[str]:
    if belief is not None and getattr(belief, "knowledge", None) is not None:
        return {_normalize_fact(fact) for fact in belief.knowledge.facts}
    return {_normalize_fact(fact) for fact in env.state.facts}


def _goal_facts(env: Any) -> set[str]:
    if not getattr(env, "goal", None):
        return set()
    return {_normalize_fact(fact) for fact in env.goal.facts}


def _goal_subset_done(env: Any, belief: Belief | None) -> bool:
    goals = _goal_facts(env)
    return bool(goals) and goals.issubset(_knowledge_facts(env, belief))


def _tomato_goal_done(env: Any, belief: Belief | None) -> bool:
    # loaded/discarded goal facts are already listed explicitly in each scene.
    return _goal_subset_done(env, belief)


def _wastesorting_goal_done(env: Any, belief: Belief | None) -> bool:
    # Each waste must be observed in its target bin.
    return _goal_subset_done(env, belief)


def _blocksworld_goal_done(env: Any, belief: Belief | None) -> bool:
    # Target stack relations include zero-arity facts such as handempty.
    return _goal_subset_done(env, belief)


def _kitchen_goal_done(env: Any, belief: Belief | None) -> bool:
    # Dish-serving predicates such as served_tomato_soup(dish1) define success.
    return _goal_subset_done(env, belief)


def _rover_goal_done(env: Any, belief: Belief | None) -> bool:
    # Success requires communicated science data facts.
    return _goal_subset_done(env, belief)


def _watering_goal_done(env: Any, belief: Belief | None) -> bool:
    # All target plants must be watered.
    return _goal_subset_done(env, belief)


GOAL_CHECKERS = {
    "tomato": _tomato_goal_done,
    "wastesorting": _wastesorting_goal_done,
    "blocksworld": _blocksworld_goal_done,
    "kitchen": _kitchen_goal_done,
    "rover": _rover_goal_done,
    "watering": _watering_goal_done,
}


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

        for raw_fact in _knowledge_facts(env, belief):
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

        for raw_fact in _knowledge_facts(env, belief):
            predicate, args = _parse_fact(raw_fact)
            if not args:
                continue

            if predicate != "in_bin" or len(args) < 2:
                continue

            waste, bin_name = args[:2]
            goal_bin = goal_bin_by_waste.get(waste)
            if goal_bin and bin_name != goal_bin:
                return "PLAN FAILURE"

    goal_checker = GOAL_CHECKERS.get(env.domain_name, _goal_subset_done)
    if goal_checker(env, belief):
        return "GOAL DONE"

    if env.step_count >= env.max_step:
        return "MAX STEP"

    return False
