from __future__ import annotations

import random
from typing import Any

from tomato_utils import parse_tomato_action
from wastesorting_utils import parse_waste_action
from scripts.knowno_action_validation import (
    match_remaining_object,
    validate_tomato_action,
    validate_waste_action,
)

# Auto-answer policy used for scripted KnowNo help queries.
#
# Decision rules:
# 1. Build a feasible set first. Feasible follows the same terminal-failure
#    checks used by knowno_multistep_tomato.py and
#    knowno_multistep_wastesorting.py: an action is infeasible if selecting it
#    would immediately stop the run as invalid or PLAN FAILURE.
# 2. If no feasible option exists, select the fallback option E
#    ("an option not listed here").
# 3. If feasible options exist, select by deterministic goal-progress rules:
#    Tomato:
#    - If holding a tomato and its scan result is still unknown, scan it again
#      when scan is available.
#    - If holding a tomato and no scan option is available, place it when it is
#      true ripe and discard it when it is true rotten.
#    - If not holding anything, pick an observed actionable tomato at the current
#      stem before navigating away.
#    - If no pick is available and the current stem still has unknown tomatoes,
#      choose detect at the current stem.
#    - If the current stem is done, navigate to the first stem that still has a
#      remaining goal-relevant tomato.
#    Waste:
#    - If holding an object, place it in its true bin.
#    - If not holding anything, pick the first visible remaining object.
#    - If no pick is available, detect visible objects if detect is offered.
# 4. Tie-breaking is by natural object/location order: tomato1 before tomato2,
#    waste1 before waste2, stem_01 before stem_02, etc.

FALLBACK_OPTION_TEXT = "an option not listed here"


def select_tomato_answer(
    options,
    tokens,
    add_mc_prefix,
    robot_location,
    active_tomatoes,
    hidden_properties,
    hidden_locations,
    observed_properties,
    observed_locations,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
    scanned_properties=None,
) -> dict[str, Any]:
    scanned_properties = scanned_properties or {}
    active = set(active_tomatoes)
    handled = set(loaded_tomatoes) | set(discarded_tomatoes)
    records = [
        _tomato_record(
            token,
            option,
            add_mc_prefix,
            robot_location,
            active,
            handled,
            hidden_properties,
            hidden_locations,
            observed_properties,
            observed_locations,
            held_tomato,
        )
        for token, option in zip(tokens, options)
    ]
    feasible = [record for record in records if record["feasible"]]

    if feasible:
        selected, rule = _select_tomato_by_rule(
            feasible,
            robot_location,
            active,
            handled,
            hidden_properties,
            hidden_locations,
            observed_properties,
            held_tomato,
            scanned_properties,
        )
    else:
        selected = add_mc_prefix
        rule = "no feasible option; select fallback E"

    return {"selected_token": selected, "rule": rule, "options": records}


def _tomato_record(
    token,
    option,
    add_mc_prefix,
    robot_location,
    active,
    handled,
    hidden_properties,
    hidden_locations,
    observed_properties,
    observed_locations,
    held_tomato,
):
    action_type, action_arg = parse_tomato_action(option)
    feasible, reason = validate_tomato_action(
        action_type,
        action_arg,
        robot_location,
        active,
        hidden_properties,
        observed_locations,
        held_tomato,
        loaded_tomatoes=[],
        discarded_tomatoes=handled,
    )
    record = {
        "token": token,
        "option": option,
        "action_type": action_type,
        "target": action_arg,
        "feasible": feasible,
        "reason": reason,
    }

    if token == add_mc_prefix or option == FALLBACK_OPTION_TEXT:
        record["feasible"] = False
        record["reason"] = "fallback option"
        return record
    return record


def _select_tomato_by_rule(
    feasible,
    robot_location,
    active,
    handled,
    hidden_properties,
    hidden_locations,
    observed_properties,
    held_tomato,
    scanned_properties,
):
    if held_tomato is not None:
        scanned_property = scanned_properties.get(held_tomato, "unknown")
        if scanned_property == "unknown":
            match = _first_matching(feasible, "scan", None)
            if match:
                return match["token"], f"held {held_tomato} has unknown scan result; scan it again"
        true_prop = hidden_properties.get(held_tomato)
        if true_prop == "ripe":
            match = _first_matching(feasible, "place", held_tomato)
            if match:
                return match["token"], f"held {held_tomato} is true ripe; place it"
        if true_prop == "rotten":
            match = _first_matching(feasible, "discard", held_tomato)
            if match:
                return match["token"], f"held {held_tomato} is true rotten; discard it"
        match = _first_matching(feasible, "scan", None)
        if match:
            return match["token"], f"held {held_tomato} cannot be directly resolved; scan it"
        selected = random.choice(feasible)
        return selected["token"], f"no preferred held-tomato rule matched; randomly choose feasible option {selected['token']}"

    pick_options = [record for record in feasible if record["action_type"] == "pick"]
    if pick_options:
        selected = min(pick_options, key=lambda record: _natural_index(record["target"]))
        return selected["token"], f"pick first actionable tomato: {selected['target']}"

    current_unknown = _unknown_tomatoes_at(
        robot_location,
        active,
        handled,
        hidden_locations,
        observed_properties,
    )
    if current_unknown:
        detect_options = [record for record in feasible if record["action_type"] == "detect"]
        if detect_options:
            selected = min(detect_options, key=lambda record: record["token"])
            return selected["token"], f"detect current stem; unknown tomatoes remain: {', '.join(current_unknown)}"

    current_has_work = _stem_has_goal_work(
        robot_location,
        active,
        handled,
        hidden_properties,
        hidden_locations,
        observed_properties,
    )
    if current_has_work:
        detect_options = [record for record in feasible if record["action_type"] == "detect"]
        if detect_options:
            selected = min(detect_options, key=lambda record: record["token"])
            return selected["token"], "detect current stem; goal-relevant tomato remains unresolved"

    if not current_has_work:
        navigate_options = [
            record
            for record in feasible
            if record["action_type"] == "navigate"
            and _stem_has_goal_work(
                record["target"],
                active,
                handled,
                hidden_properties,
                hidden_locations,
                observed_properties,
            )
        ]
        if navigate_options:
            selected = min(navigate_options, key=lambda record: _natural_index(record["target"]))
            return selected["token"], f"current stem is done; navigate to {selected['target']}"

    selected = random.choice(feasible)
    return selected["token"], f"no preferred tomato rule matched; randomly choose feasible option {selected['token']}"


def select_waste_answer(options, tokens, add_mc_prefix, remaining_objects, hidden_attributes, held_object, occlusions):
    records = [
        _waste_record(token, option, add_mc_prefix, remaining_objects, hidden_attributes, held_object, occlusions)
        for token, option in zip(tokens, options)
    ]
    feasible = [record for record in records if record["feasible"]]

    if feasible:
        selected, rule = _select_waste_by_rule(feasible, remaining_objects, hidden_attributes, held_object, occlusions)
    else:
        selected = add_mc_prefix
        rule = "no feasible option; select fallback E"

    return {"selected_token": selected, "rule": rule, "options": records}


def _waste_record(token, option, add_mc_prefix, remaining_objects, hidden_attributes, held_object, occlusions):
    action_type, action_arg = parse_waste_action(option)
    feasible, reason = validate_waste_action(
        action_type,
        action_arg,
        remaining_objects,
        hidden_attributes,
        held_object,
        occlusions,
    )
    record = {
        "token": token,
        "option": option,
        "action_type": action_type,
        "target": action_arg,
        "feasible": feasible,
        "reason": reason,
    }

    if token == add_mc_prefix or option == FALLBACK_OPTION_TEXT:
        record["feasible"] = False
        record["reason"] = "fallback option"
        return record

    if action_type == "pick":
        matched_object = match_remaining_object(action_arg, remaining_objects)
        if matched_object is None:
            return record
        record["target"] = matched_object

    return record


def _select_waste_by_rule(feasible, remaining_objects, hidden_attributes, held_object, occlusions):
    if held_object is not None:
        place_options = [record for record in feasible if record["action_type"] == "place"]
        if place_options:
            selected = min(place_options, key=lambda record: _natural_index(held_object))
            return selected["token"], f"place held {held_object} into its true bin"
        selected = random.choice(feasible)
        return selected["token"], f"no preferred held-object rule matched; randomly choose feasible option {selected['token']}"

    pick_options = [record for record in feasible if record["action_type"] == "pick"]
    if pick_options:
        selected = min(pick_options, key=lambda record: _natural_index(record["target"]))
        return selected["token"], f"pick first visible object: {selected['target']}"

    detect_options = [record for record in feasible if record["action_type"] == "detect"]
    if detect_options:
        selected = min(detect_options, key=lambda record: record["token"])
        return selected["token"], "detect visible objects"

    selected = random.choice(feasible)
    return selected["token"], f"no preferred waste rule matched; randomly choose feasible option {selected['token']}"


def _first_matching(records, action_type, target):
    for record in sorted(records, key=lambda item: item["token"]):
        if record["action_type"] != action_type:
            continue
        if target is not None and record["target"] != target:
            continue
        return record
    return None


def _unknown_tomatoes_at(robot_location, active, handled, hidden_locations, observed_properties):
    return sorted(
        [
            tomato
            for tomato in active
            if tomato not in handled
            and hidden_locations.get(tomato) == robot_location
            and observed_properties.get(tomato) is None
        ],
        key=_natural_index,
    )


def _stem_has_goal_work(stem, active, handled, hidden_properties, hidden_locations, observed_properties):
    for tomato in active:
        if tomato in handled or hidden_locations.get(tomato) != stem:
            continue
        if observed_properties.get(tomato) is None:
            return True
        if hidden_properties.get(tomato) in {"ripe", "rotten"}:
            return True
    return False


def _natural_index(name: str | None) -> int:
    if not name:
        return 999
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999
