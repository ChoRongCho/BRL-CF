from __future__ import annotations

from typing import Any

from tomato_utils import parse_tomato_action
from wastesorting_utils import parse_waste_action
from scripts.knowno_action_validation import (
    match_remaining_object,
    validate_tomato_action,
    validate_waste_action,
)

FALLBACK_OPTION_TEXT = "an option not listed here"


def _natural_index(name):
    if not name:
        return 999
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999


def select_tomato_answer(
    options,
    tokens,
    add_mc_prefix,
    robot_location,
    active_tomatoes,
    hidden_ripeness,
    hidden_freshness,
    hidden_locations,
    observed_properties,
    observed_locations,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
    scanned_properties=None,
    allowed_tokens=None,
) -> dict[str, Any]:
    """Choose the true-state-correct option when KnowNo requests help."""
    scanned_properties = scanned_properties or {}
    handled = set(loaded_tomatoes) | set(discarded_tomatoes)
    records = []
    for token, option in zip(tokens, options):
        action_type, action_arg = parse_tomato_action(option)
        feasible, reason = validate_tomato_action(
            action_type,
            action_arg,
            robot_location,
            active_tomatoes,
            hidden_ripeness,
            hidden_freshness,
            observed_properties,
            observed_locations,
            scanned_properties,
            held_tomato,
            loaded_tomatoes,
            discarded_tomatoes,
        )
        if token == add_mc_prefix or option == FALLBACK_OPTION_TEXT:
            feasible, reason = False, "fallback option"
        records.append({
            "token": token,
            "option": option,
            "action_type": action_type,
            "target": action_arg,
            "feasible": feasible,
            "reason": reason,
        })

    allowed = set(tokens if allowed_tokens is None else allowed_tokens)
    feasible = [record for record in records if record["feasible"] and record["token"] in allowed]
    if not feasible:
        return {"selected_token": add_mc_prefix, "rule": "no correct feasible option", "options": records}

    if held_tomato is not None:
        if scanned_properties.get(held_tomato, "unknown") == "unknown":
            scans = [r for r in feasible if r["action_type"] == "scan"]
            if scans:
                selected = min(scans, key=lambda r: r["token"])
                return _result(selected, f"scan held {held_tomato}", records)
        desired = "place" if hidden_freshness.get(held_tomato) == "fresh" else "discard"
        matches = [r for r in feasible if r["action_type"] == desired and r["target"] == held_tomato]
        if matches:
            selected = min(matches, key=lambda r: r["token"])
            return _result(selected, f"{desired} true {hidden_freshness.get(held_tomato)} {held_tomato}", records)

    picks = [r for r in feasible if r["action_type"] == "pick"]
    if picks:
        selected = min(picks, key=lambda r: (_natural_index(r["target"]), r["token"]))
        return _result(selected, f"pick actionable {selected['target']}", records)

    unknown_here = [
        tomato for tomato in active_tomatoes
        if tomato not in handled
        and hidden_locations.get(tomato) == robot_location
        and observed_properties.get(tomato) is None
    ]
    detects = [r for r in feasible if r["action_type"] == "detect"]
    if unknown_here and detects:
        selected = min(detects, key=lambda r: r["token"])
        return _result(selected, "detect unresolved tomatoes at current stem", records)

    def stem_has_work(stem):
        return any(
            tomato not in handled
            and hidden_locations.get(tomato) == stem
            and (observed_properties.get(tomato) is None or hidden_ripeness.get(tomato) == "ripe")
            for tomato in active_tomatoes
        )

    if not stem_has_work(robot_location):
        navs = [r for r in feasible if r["action_type"] == "navigate" and stem_has_work(r["target"])]
        if navs:
            selected = min(navs, key=lambda r: (_natural_index(r["target"]), r["token"]))
            return _result(selected, f"navigate to remaining work at {selected['target']}", records)
    if detects:
        selected = min(detects, key=lambda r: r["token"])
        return _result(selected, "detect unresolved current state", records)

    selected = min(feasible, key=lambda r: r["token"])
    return _result(selected, "first correct feasible option", records)


def select_waste_answer(
    options,
    tokens,
    add_mc_prefix,
    remaining_objects,
    hidden_attributes,
    observed_attributes,
    held_object,
    occlusions,
    allowed_tokens=None,
):
    """Choose the true-state-correct option when KnowNo requests help."""
    records = []
    for token, option in zip(tokens, options):
        action_type, action_arg = parse_waste_action(option)
        feasible, reason = validate_waste_action(
            action_type,
            action_arg,
            remaining_objects,
            hidden_attributes,
            observed_attributes,
            held_object,
            occlusions,
        )
        target = action_arg
        if token == add_mc_prefix or option == FALLBACK_OPTION_TEXT:
            feasible, reason = False, "fallback option"
        elif action_type == "pick":
            target = match_remaining_object(action_arg, remaining_objects)
        records.append({
            "token": token,
            "option": option,
            "action_type": action_type,
            "target": target,
            "feasible": feasible,
            "reason": reason,
        })

    allowed = set(tokens if allowed_tokens is None else allowed_tokens)
    feasible = [record for record in records if record["feasible"] and record["token"] in allowed]
    if not feasible:
        return {"selected_token": add_mc_prefix, "rule": "no correct feasible option", "options": records}
    if held_object is not None:
        places = [r for r in feasible if r["action_type"] == "place"]
        if places:
            selected = min(places, key=lambda r: r["token"])
            return _result(selected, f"place {held_object} into its true bin", records)
    picks = [r for r in feasible if r["action_type"] == "pick"]
    if picks:
        selected = min(picks, key=lambda r: (_natural_index(r["target"]), r["token"]))
        return _result(selected, f"pick visible {selected['target']}", records)
    detects = [r for r in feasible if r["action_type"] == "detect"]
    if detects:
        selected = min(detects, key=lambda r: r["token"])
        return _result(selected, "detect visible objects", records)
    selected = min(feasible, key=lambda r: r["token"])
    return _result(selected, "first correct feasible option", records)


def _result(selected, rule, records):
    return {"selected_token": selected["token"], "rule": rule, "options": records}
