from __future__ import annotations

from tomato_utils import STEMS


def validate_tomato_action(
    action_type,
    action_arg,
    robot_location,
    active_tomatoes,
    hidden_properties,
    observed_locations,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
):
    """Return whether a tomato action avoids the runner's terminal-failure cases."""
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
        if tomato not in active:
            return False, f"invalid pick inactive tomato: {tomato}"
        if tomato in handled:
            return False, f"invalid pick already handled tomato: {tomato}"
        if held_tomato is not None:
            return False, "invalid pick while holding tomato"
        if observed_locations.get(tomato) != robot_location:
            return False, f"invalid pick tomato not observed here: {tomato}"
        if hidden_properties.get(tomato) == "unripe":
            return False, f"invalid pick non-ripe or unknown tomato: {tomato}"
        return True, f"can pick true {hidden_properties.get(tomato)} {tomato}"

    if action_type == "scan":
        if held_tomato is None:
            return False, "invalid scan without held tomato"
        if action_arg is not None and action_arg != held_tomato:
            return False, "invalid scan target mismatch"
        return True, f"can scan held {held_tomato}"

    if action_type == "place":
        tomato = action_arg
        if held_tomato != tomato:
            return False, "invalid place target mismatch"
        if hidden_properties.get(tomato) != "ripe":
            return False, f"invalid place non-ripe tomato: {tomato}"
        return True, f"can place true ripe {tomato}"

    if action_type == "discard":
        tomato = action_arg
        if held_tomato != tomato:
            return False, "invalid discard target mismatch"
        if hidden_properties.get(tomato) == "ripe":
            return False, f"PLAN FAILURE: true ripe tomato {tomato} was discarded"
        return True, f"can discard true {hidden_properties.get(tomato)} {tomato}"

    return False, "unsupported executable action"


def validate_waste_action(
    action_type,
    action_arg,
    remaining_objects,
    hidden_attributes,
    held_object,
    occlusions,
):
    """Return whether a waste action avoids the runner's terminal-failure cases."""
    if action_type == "done":
        return False, "planner selected terminal action"
    if action_type is None:
        return False, "non-executable action"

    visible = set(visible_waste_objects(remaining_objects, occlusions))

    if action_type == "detect":
        return True, "can detect"

    if action_type == "pick":
        matched_object = match_remaining_object(action_arg, remaining_objects)
        if matched_object is None:
            return False, f"invalid pick unavailable object: {action_arg}"
        if matched_object not in visible:
            return False, f"invalid pick occluded object: {matched_object}"
        if held_object is not None:
            return False, "invalid pick while holding object"
        return True, f"can pick visible {matched_object}"

    if action_type == "place":
        if held_object is None:
            return False, "invalid place without held object"
        place_object, target_bin = action_arg
        if not (place_object in held_object or held_object in place_object):
            return False, "invalid place target mismatch"
        true_bin = f"{hidden_attributes.get(held_object)} bin"
        if target_bin != true_bin:
            return False, (
                f"PLAN FAILURE: true {hidden_attributes.get(held_object)} waste "
                f"{held_object} was placed into {target_bin}"
            )
        return True, f"can place held {held_object} into true bin {true_bin}"

    return False, "unsupported executable action"


def match_remaining_object(target, remaining_objects):
    for obj in remaining_objects:
        if target in obj or obj in target:
            return obj
    return None


def visible_waste_objects(remaining_objects: list[str], occlusions: dict[str, str]) -> list[str]:
    remaining = set(remaining_objects)
    return [obj for obj in remaining_objects if occlusions.get(obj) not in remaining]
