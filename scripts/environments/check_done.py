from __future__ import annotations

from typing import Any

from models.belief import Belief
from utils.utils import _parse_fact


GOAL_DONE = "GOAL DONE"
MAX_STEP = "MAX STEP"
PLAN_FAILURE = "PLAN FAILURE"


def _normalize_fact(raw_fact: str) -> str:
    return str(raw_fact).replace(" ", "")


def _get_knowledge_facts(env: Any, belief: Belief | None) -> set[str]:
    facts = belief.knowledge.facts
    return {_normalize_fact(fact) for fact in facts}


def _get_goal_facts(env: Any) -> set[str]:
    goal = env.goal
    if goal is None:
        raise ValueError("Goal is not defined")
    return {_normalize_fact(fact) for fact in goal.facts}


def _check_tomato_deadend(env: Any, belief: Belief | None) -> bool:
    """
    토마토 도메인의 dead-end는 되돌릴 수 없는 잘못된 토마토 처리 상태를 의미한다.
    현재 state와 true state를 각각 명시적으로 만든 뒤, true state 기준으로 불가능한
    처리 조건을 나열하고 현재 state가 그 조건에 걸리는지 확인한다.

    dead-end 조건:
    1. unripe 토마토를 holding한 경우
    2. ripe 또는 unripe 토마토를 discarded한 경우
    3. rotten 또는 unripe 토마토를 loaded한 경우
    """

    current_state = {"unripe": set(), "ripe": set(), "rotten": set(), 
                     "holding": set(), "discarded": set(), "loaded": set()}
    true_state = {"at": set(), "unripe": set(), "ripe": set(), "rotten": set()}

    # 1. Build true state
    for raw_fact in env.true_state.facts:
        predicate, args = _parse_fact(raw_fact)
        if not args:
            continue

        tomato = args[0]
        if predicate == "at":
            true_state["at"].add(tomato)
        elif predicate == "unripe":
            true_state["unripe"].add(tomato)
        elif predicate == "ripe":
            true_state["ripe"].add(tomato)
        elif predicate == "rotten":
            true_state["rotten"].add(tomato)

    # 2. Build current state
    for raw_fact in _get_knowledge_facts(env, belief):
        predicate, args = _parse_fact(raw_fact)
        if not args:
            continue

        tomato = args[0]
        if predicate == "unripe":
            current_state["unripe"].add(tomato)
        elif predicate == "ripe":
            current_state["ripe"].add(tomato)
        elif predicate == "rotten":
            current_state["rotten"].add(tomato)
        elif predicate == "holding" and len(args) >= 2:
            held_tomato = args[1]
            current_state["holding"].add(held_tomato)
        elif predicate == "discarded":
            current_state["discarded"].add(tomato)
        elif predicate == "loaded":
            current_state["loaded"].add(tomato)

    # 3. Generate dead-end conditions
    deadend_conditions = (
        # 3-1. Is any held tomato unripe?
        current_state["holding"] & true_state["unripe"],
        # 3-2. Is any discarded tomato ripe?
        current_state["discarded"] & true_state["ripe"],
        # 3-3. Is any loaded tomato rotten?
        current_state["loaded"] & true_state["rotten"],
    )

    return any(deadend_conditions)


def _check_wastesorting_deadend(env: Any, belief: Belief | None) -> bool:
    """
    Wastesorting dead-end means waste has been placed in a bin different from
    the goal bin.
    """

    goal_state = {"in_bin": set(), "waste": set()}
    current_state = {"in_bin": set()}

    # 1. Build goal state
    for raw_fact in _get_goal_facts(env):
        predicate, args = _parse_fact(raw_fact)
        if predicate == "in_bin" and len(args) >= 2:
            waste, goal_bin = args[:2]
            goal_state["in_bin"].add((waste, goal_bin))
            goal_state["waste"].add(waste)

    # 2. Build current state
    for raw_fact in _get_knowledge_facts(env, belief):
        predicate, args = _parse_fact(raw_fact)
        if predicate == "in_bin" and len(args) >= 2:
            waste, bin_name = args[:2]
            current_state["in_bin"].add((waste, bin_name))

    # 3. Generate dead-end conditions
    deadend_conditions = (
        # 3-1. Is any waste placed in a bin different from its goal bin?
        current_state["in_bin"] - goal_state["in_bin"],
    )
    return any(deadend_conditions)


def _check_blocksworld_deadend(env: Any, belief: Belief | None) -> bool:
    "There is no dead-end condition in blocksworld domain."
    return False


def _check_kitchen_deadend(env: Any, belief: Belief | None) -> bool:
    "There is no dead-end condition in kitchen domain."
    return False


def _check_rover_deadend(env: Any, belief: Belief | None) -> bool:
    "There is no dead-end condition in rover domain."
    return False


def _check_watering_deadend(env: Any, belief: Belief | None) -> bool:
    """
    Watering dead-end means a plant has been watered while the current believed
    plant location differs from its true location.
    """

    true_state = {"at": set()}
    current_state = {"at": set(), "watered": set()}

    # 1. Build true state
    for raw_fact in env.true_state.facts:
        predicate, args = _parse_fact(raw_fact)
        if predicate == "at" and len(args) >= 2:
            item, room = args[:2]
            true_state["at"].add((item, room))

    # 2. Build current state
    for raw_fact in _get_knowledge_facts(env, belief):
        predicate, args = _parse_fact(raw_fact)
        if predicate == "at" and len(args) >= 2:
            item, room = args[:2]
            current_state["at"].add((item, room))
        elif predicate == "watered" and args:
            plant = args[0]
            current_state["watered"].add(plant)

    # 3. Generate dead-end conditions
    watered_plant_locations = {
        (plant, room)
        for plant, room in current_state["at"]
        if plant in current_state["watered"]
    }
    deadend_conditions = (
        # 3-1. Is any watered plant believed to be in a wrong room?
        watered_plant_locations - true_state["at"],
    )

    return any(deadend_conditions)



def evaluate_done(env: Any, belief: Belief | None):
    """
    Return the episode termination reason, or False if the episode continues.
    """
    # 1. Get domain name
    domain_name = env.domain_name
    done_reason: str | bool = None

    # 2. Check domain-specific dead-end conditions
    if domain_name == "tomato":
        if _check_tomato_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason
    elif domain_name == "wastesorting":
        if _check_wastesorting_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason
    elif domain_name == "blocksworld":
        if _check_blocksworld_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason
    elif domain_name == "kitchen":
        if _check_kitchen_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason
    elif domain_name == "rover":
        if _check_rover_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason
    elif domain_name == "watering":
        if _check_watering_deadend(env, belief):
            done_reason = PLAN_FAILURE
            return done_reason

    # 3. Check goal completion
    goals = _get_goal_facts(env)
    knowledge = _get_knowledge_facts(env, belief)
    if goals.issubset(knowledge):
        done_reason = GOAL_DONE
        return done_reason

    # 4. Check max step
    if env.step_count >= env.max_step:
        done_reason = MAX_STEP
        return done_reason
    
    return done_reason
