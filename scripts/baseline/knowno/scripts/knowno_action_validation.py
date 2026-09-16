from __future__ import annotations

from tomato_utils import STEMS


def validate_tomato_action(
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
):
    """Return whether an action is feasible and correct in the true tomato state."""
    if action_type == "done":
        return False, "planner selected terminal action"
    if action_type is None:
        return False, "non-executable action"

    active = set(active_tomatoes)
    handled = set(loaded_tomatoes) | set(discarded_tomatoes)
    if action_type == "navigate":
        if held_tomato is not None:
            return False, "invalid navigate while holding tomato"
        return True, f"can navigate to {action_arg}"
    if action_type == "detect":
        if robot_location not in STEMS:
            return False, "invalid detect outside stem"
        return True, f"can detect at {robot_location}"
    if action_type == "pick":
        tomato = action_arg
        if tomato not in active or tomato in handled:
            return False, f"invalid pick unavailable tomato: {tomato}"
        if held_tomato is not None:
            return False, "invalid pick while holding tomato"
        if observed_locations.get(tomato) != robot_location:
            return False, f"invalid pick tomato not observed here: {tomato}"
        if observed_properties.get(tomato) != "ripe":
            return False, f"invalid pick tomato not observed ripe: {tomato}"
        if hidden_ripeness.get(tomato) != "ripe":
            return False, f"incorrect pick true unripe tomato: {tomato}"
        return True, f"can pick true {hidden_ripeness.get(tomato)} {tomato}"
    if action_type == "scan":
        if held_tomato is None:
            return False, "invalid scan without held tomato"
        if action_arg is not None and action_arg != held_tomato:
            return False, "invalid scan target mismatch"
        return True, f"can scan held {held_tomato}"
    if action_type == "place":
        if held_tomato != action_arg:
            return False, "invalid place target mismatch"
        if scanned_properties.get(action_arg) != "fresh":
            return False, f"invalid place tomato not scanned fresh: {action_arg}"
        if hidden_freshness.get(action_arg) != "fresh":
            return False, f"invalid place non-fresh tomato: {action_arg}"
        return True, f"can place true fresh {action_arg}"
    if action_type == "discard":
        if held_tomato != action_arg:
            return False, "invalid discard target mismatch"
        if hidden_freshness.get(action_arg) != "rotten":
            return False, f"invalid discard non-rotten tomato: {action_arg}"
        return True, f"can discard true rotten {action_arg}"
    return False, "unsupported executable action"


def validate_waste_action(
    action_type,
    action_arg,
    remaining_objects,
    hidden_attributes,
    observed_attributes,
    held_object,
    occlusions,
):
    """Return whether an action is feasible and correct in the true waste state."""
    if action_type == "done":
        return False, "planner selected terminal action"
    if action_type is None:
        return False, "non-executable action"
    visible = set(visible_waste_objects(remaining_objects, occlusions))
    if action_type == "detect":
        return True, "can detect"
    if action_type == "pick":
        matched = match_remaining_object(action_arg, remaining_objects)
        if matched is None:
            return False, f"invalid pick unavailable object: {action_arg}"
        if matched not in visible:
            return False, f"invalid pick occluded object: {matched}"
        if matched not in observed_attributes:
            return False, f"invalid pick undetected object: {matched}"
        if held_object is not None:
            return False, "invalid pick while holding object"
        return True, f"can pick visible {matched}"
    if action_type == "place":
        if held_object is None:
            return False, "invalid place without held object"
        place_object, target_bin = action_arg
        if not (place_object in held_object or held_object in place_object):
            return False, "invalid place target mismatch"
        true_bin = f"{hidden_attributes.get(held_object)} bin"
        if target_bin != true_bin:
            return False, f"invalid place: true bin is {true_bin}"
        return True, f"can place held {held_object} into true bin {true_bin}"
    return False, "unsupported executable action"


def match_remaining_object(target, remaining_objects):
    for obj in remaining_objects:
        if target in obj or obj in target:
            return obj
    return None


def visible_waste_objects(remaining_objects, occlusions):
    remaining = set(remaining_objects)
    return [obj for obj in remaining_objects if occlusions.get(obj) not in remaining]
