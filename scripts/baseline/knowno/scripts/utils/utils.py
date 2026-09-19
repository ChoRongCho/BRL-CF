"""KnowNo multi-step 실행에서 공유하는 상태 판정과 작은 보조 함수."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

from tomato_utils import STEMS, TOMATO_SCAN_RESULTS
from wastesorting_utils import WASTE_ATTRIBUTES
from scripts.utils.env_setting import draw_probability


GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass(frozen=True)
class PlanningResult:
    options_text: str
    options: list[str]
    fallback_token: str
    scored_tokens: list[str]
    logprobs: list[float]
    scores: object
    prediction_set: list[str]


@dataclass(frozen=True)
class ActionSelection:
    token: Optional[str]
    action: Optional[str]
    help_needed: bool
    oracle_provided_action: bool = False
    error: Optional[str] = None
    oracle_answer: Optional[dict[str, Any]] = None


def tomato_observation_mismatches(
    observed_properties,
    observed_locations,
    scanned_properties,
    hidden_ripeness,
    hidden_locations,
    hidden_freshness,
):
    """Return observed tomato facts that disagree with the hidden world."""
    mismatches = []
    for tomato, observed in sorted(observed_properties.items()):
        expected = hidden_ripeness.get(tomato)
        if expected is not None and observed != expected:
            mismatches.append({
                "fact": "ripeness",
                "object": tomato,
                "observed": observed,
                "true": expected,
            })
    for tomato, observed in sorted(observed_locations.items()):
        expected = hidden_locations.get(tomato)
        if expected is not None and observed != expected:
            mismatches.append({
                "fact": "location",
                "object": tomato,
                "observed": observed,
                "true": expected,
            })
    for tomato, observed in sorted(scanned_properties.items()):
        expected = hidden_freshness.get(tomato)
        if expected is not None and observed != expected:
            mismatches.append({
                "fact": "freshness",
                "object": tomato,
                "observed": observed,
                "true": expected,
            })
    return mismatches


def waste_observation_mismatches(observed_attributes, hidden_attributes):
    """Return observed waste labels that disagree with the hidden world."""
    return [
        {
            "fact": "waste_type",
            "object": waste,
            "observed": observed,
            "true": hidden_attributes[waste],
        }
        for waste, observed in sorted(observed_attributes.items())
        if waste in hidden_attributes and observed != hidden_attributes[waste]
    ]


def analyze_asked_prediction_set(selection, prediction_set, observation_mismatches):
    """Classify a queried decision using the exact-oracle option audit.

    A wrong observation makes the request justified for this diagnostic.  The
    ``what`` failure is whether the prediction set omitted every action that
    the hidden-state oracle considered correct and feasible.  NoOpt recovery
    may still let the episode continue, so callers record this as a failure
    mode event independently of final task success.
    """
    if not selection.help_needed or selection.oracle_answer is None:
        return None

    allowed = set(prediction_set)
    option_records = selection.oracle_answer.get("options", [])
    correct_tokens = sorted(
        record["token"]
        for record in option_records
        if record.get("feasible") and record.get("token") in allowed
    )
    correct_option_available = bool(correct_tokens)
    observation_error_present = bool(observation_mismatches)

    category = None
    if not correct_option_available:
        category = (
            "what_missing_correct_option_after_observation_error"
            if observation_error_present
            else "when_what_missing_correct_option_without_observation_error"
        )

    return {
        "category": category,
        "help_requested": True,
        "observation_error_present": observation_error_present,
        "observation_mismatches": list(observation_mismatches),
        "prediction_set": list(prediction_set),
        "correct_option_available": correct_option_available,
        "correct_option_tokens": correct_tokens,
        "selected_token": selection.token,
        "noopt_recovery": selection.oracle_provided_action,
    }


def usage_total(usage) -> int:
    if not usage:
        return 0
    return (
        usage.get("total_tokens")
        or usage.get("total_token_count")
        or sum(value for key, value in usage.items() if key.endswith("tokens") and isinstance(value, int))
    )


def tomato_dead_end_reason(hidden_ripeness, hidden_freshness, held_tomato, loaded_tomatoes, discarded_tomatoes):
    for tomato in discarded_tomatoes:
        if hidden_freshness.get(tomato) == "fresh":
            return f"PLAN FAILURE: true fresh tomato {tomato} was discarded."
    for tomato in loaded_tomatoes:
        if hidden_freshness.get(tomato) == "rotten":
            return f"PLAN FAILURE: true rotten tomato {tomato} was loaded."
    if held_tomato is not None and hidden_ripeness.get(held_tomato) == "unripe":
        return f"PLAN FAILURE: true unripe tomato {held_tomato} is held."
    return None


def tomato_success(hidden_ripeness, hidden_freshness, held_tomato, loaded_tomatoes, discarded_tomatoes):
    if held_tomato is not None:
        return False
    for tomato, ripeness in hidden_ripeness.items():
        if ripeness != "ripe":
            continue
        if hidden_freshness[tomato] == "fresh" and tomato not in loaded_tomatoes:
            return False
        if hidden_freshness[tomato] == "rotten" and tomato not in discarded_tomatoes:
            return False
    return True


def required_tomato_next_action(
    robot_location,
    active_tomatoes,
    observed_properties,
    observed_locations,
    scanned_properties,
    held_tomato,
    detected_stems,
):
    if held_tomato is not None:
        scanned_property = scanned_properties.get(held_tomato, "unknown")
        if scanned_property == "fresh":
            return f"place {held_tomato}"
        if scanned_property == "rotten":
            return f"discard {held_tomato}"
        return f"scan {held_tomato}"

    if robot_location not in STEMS:
        return "navigate to stem_01 or navigate to stem_02"

    for tomato in active_tomatoes:
        if observed_locations.get(tomato) == robot_location and observed_properties.get(tomato) == "ripe":
            return f"pick {tomato}"

    if robot_location in detected_stems:
        other_stems = [stem for stem in STEMS if stem != robot_location]
        if other_stems and any(tomato not in observed_locations for tomato in active_tomatoes):
            return f"navigate to {other_stems[0]}"

    return f"detect {robot_location}"


def waste_dead_end_reason(hidden_attributes, placed_objects):
    for obj, target_bin in placed_objects.items():
        true_bin = f"{hidden_attributes[obj]} bin"
        if target_bin != true_bin:
            return f"PLAN FAILURE: true {hidden_attributes[obj]} waste {obj} was placed into {target_bin}."
    return None


def waste_success(hidden_attributes, remaining_objects, held_object, placed_objects):
    if held_object is not None or remaining_objects:
        return False
    return all(placed_objects.get(obj) == f"{label} bin" for obj, label in hidden_attributes.items())


def parse_occlusions(text: str) -> dict[str, str]:
    occlusions = {}
    if not text:
        return occlusions
    for item in text.split(","):
        if ":" not in item:
            raise ValueError('Occlusions must use "hidden:blocker" format, e.g. "waste4:waste3".')
        hidden, blocker = [part.strip().lower() for part in item.split(":", 1)]
        occlusions[hidden] = blocker
    return occlusions


def visible_objects(remaining_objects: list[str], occlusions: dict[str, str]) -> list[str]:
    remaining = set(remaining_objects)
    return [obj for obj in remaining_objects if occlusions.get(obj) not in remaining]


def occlusion_text(occlusions: dict[str, str], remaining_objects: list[str]) -> str:
    remaining = set(remaining_objects)
    active = [
        f"{hidden} is under {blocker} and cannot be detected until {blocker} is placed"
        for hidden, blocker in sorted(occlusions.items())
        if hidden in remaining and blocker in remaining
    ]
    return "; ".join(active) if active else "None"


def execute_tomato_action(
    action_type,
    action_arg,
    *,
    args,
    step,
    robot_location,
    active_tomatoes,
    hidden_ripeness,
    hidden_freshness,
    hidden_locations,
    observed_properties,
    observed_locations,
    scanned_properties,
    held_tomato,
    loaded_tomatoes,
    discarded_tomatoes,
    detected_stems,
    action_history,
    console,
    log_json,
):
    """토마토 action 하나를 실행하고 변경된 scalar state와 결과를 반환한다."""
    action_failure_count = 0

    if action_type == "navigate":
        if held_tomato is not None:
            console("Robot is holding a tomato; place or discard it before navigating.")
            return robot_location, held_tomato, None, 0, "invalid navigate while holding tomato"
        failure_roll = random.random()
        failed = failure_roll <= args.navigate_failure_prob
        log_json(f"Step {step} navigate roll:", {
            "target_location": action_arg,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"navigate to {action_arg} (failed)")
            result_text = f"Navigate failed: stayed at {robot_location}"
        else:
            robot_location = action_arg
            action_history.append(f"navigate to {robot_location}")
            result_text = f"Executed: navigate to {robot_location}"

    elif action_type == "detect":
        if robot_location not in STEMS:
            console("Detect requires the robot to be at a stem.")
            return robot_location, held_tomato, None, 0, "invalid detect outside stem"
        new_observations = []
        detect_rolls = []
        for tomato in active_tomatoes:
            at_target = hidden_locations[tomato] == robot_location
            detect_probability = (
                draw_probability(args.detect_success_prob)
                if at_target
                else args.detect_false_positive_prob
            )
            previous_property = observed_properties.get(tomato)
            detect_roll = random.random()
            roll_info = {
                "tomato": tomato,
                "true_ripeness": hidden_ripeness[tomato],
                "true_location": hidden_locations[tomato],
                "previous_property": previous_property,
                "at_target": at_target,
                "detect_probability": detect_probability,
                "detect_roll": detect_roll,
                "detected": detect_roll <= detect_probability,
            }
            if detect_roll <= detect_probability:
                detected_property = hidden_ripeness[tomato]
                label_error_roll = random.random()
                roll_info["label_error_roll"] = label_error_roll
                roll_info["label_error"] = label_error_roll <= args.detect_label_error_prob
                if label_error_roll <= args.detect_label_error_prob:
                    candidates = [label for label in ["ripe", "unripe"] if label != detected_property]
                    observed_properties[tomato] = random.choice(candidates)
                else:
                    observed_properties[tomato] = detected_property
                observed_locations[tomato] = robot_location
                roll_info["observed_property"] = observed_properties[tomato]
                new_observations.append(f"{tomato}: {observed_properties[tomato]} at {robot_location}")
            detect_rolls.append(roll_info)
        log_json(f"Step {step} detect rolls:", detect_rolls)
        detected_stems.add(robot_location)
        action_history.append(f"detect {robot_location}")
        result_text = "Detect result: " + (", ".join(new_observations) if new_observations else "no new tomato observed")

    elif action_type == "pick":
        tomato = action_arg
        if tomato not in active_tomatoes:
            console("Selected tomato is not active.")
            return robot_location, held_tomato, None, 0, f"invalid pick inactive tomato: {tomato}"
        if held_tomato is not None:
            console("Robot is already holding a tomato.")
            return robot_location, held_tomato, None, 0, "invalid pick while holding tomato"
        if observed_locations.get(tomato) != robot_location:
            console("Tomato is not observed at the current robot location.")
            return robot_location, held_tomato, None, 0, f"invalid pick tomato not observed here: {tomato}"
        if observed_properties.get(tomato) != "ripe":
            console("Pick requires an observed ripe tomato.")
            return robot_location, held_tomato, None, 0, f"invalid pick non-ripe or unknown tomato: {tomato}"
        failure_roll = random.random()
        failed = failure_roll <= args.pick_failure_prob
        log_json(f"Step {step} pick roll:", {
            "tomato": tomato,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"pick {tomato} (failed)")
            result_text = f"Pick failed: {tomato} was not picked"
        else:
            held_tomato = tomato
            action_history.append(f"pick {tomato}")
            result_text = f"Executed: pick {tomato}"

    elif action_type == "scan":
        if held_tomato is None:
            console("Scan requires a held tomato.")
            return robot_location, held_tomato, None, 0, "invalid scan without held tomato"
        if action_arg is not None and action_arg != held_tomato:
            console("Scan action does not match the held tomato.")
            return robot_location, held_tomato, None, 0, "invalid scan target mismatch"
        scan_roll = random.random()
        scan_probability = draw_probability(args.scan_success_prob)
        scan_info = {
            "tomato": held_tomato,
            "true_freshness": hidden_freshness[held_tomato],
            "scan_roll": scan_roll,
            "scan_probability": scan_probability,
            "scanned": scan_roll <= scan_probability,
        }
        if scan_roll <= scan_probability:
            true_scan_result = hidden_freshness[held_tomato]
            label_error_roll = random.random()
            scan_info["label_error_roll"] = label_error_roll
            scan_info["label_error"] = label_error_roll <= args.scan_label_error_prob
            if label_error_roll <= args.scan_label_error_prob:
                candidates = [label for label in TOMATO_SCAN_RESULTS if label != true_scan_result]
                scanned_properties[held_tomato] = random.choice(candidates)
            else:
                scanned_properties[held_tomato] = true_scan_result
            scan_info["scanned_property"] = scanned_properties[held_tomato]
            result_text = f"Scan result: {held_tomato}: {scanned_properties[held_tomato]}"
        else:
            result_text = "Scan result: no property observed"
        log_json(f"Step {step} scan roll:", scan_info)
        action_history.append(f"scan {held_tomato}")

    elif action_type == "place":
        tomato = action_arg
        if held_tomato != tomato:
            console("Place action does not match the held tomato.")
            return robot_location, held_tomato, None, 0, "invalid place target mismatch"
        if scanned_properties.get(tomato) != "fresh":
            console("Place requires a held tomato scanned as fresh.")
            return robot_location, held_tomato, None, 0, f"invalid place non-fresh tomato: {tomato}"
        failure_roll = random.random()
        failed = failure_roll <= args.place_failure_prob
        log_json(f"Step {step} place roll:", {
            "tomato": tomato,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"place {tomato} (failed)")
            result_text = f"Place failed: still holding {tomato}"
        else:
            held_tomato = None
            loaded_tomatoes.append(tomato)
            action_history.append(f"place {tomato}")
            result_text = f"Executed: place {tomato}"

    elif action_type == "discard":
        tomato = action_arg
        if held_tomato != tomato:
            console("Discard action does not match the held tomato.")
            return robot_location, held_tomato, None, 0, "invalid discard target mismatch"
        failure_roll = random.random()
        failed = failure_roll <= args.discard_failure_prob
        log_json(f"Step {step} discard roll:", {
            "tomato": tomato,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"discard {tomato} (failed)")
            result_text = f"Discard failed: still holding {tomato}"
        else:
            held_tomato = None
            discarded_tomatoes.append(tomato)
            action_history.append(f"discard {tomato}")
            result_text = f"Executed: discard {tomato}"
    else:
        return robot_location, held_tomato, None, 0, f"unsupported action type: {action_type}"

    return robot_location, held_tomato, result_text, action_failure_count, None


def execute_waste_action(
    action_type,
    action_arg,
    *,
    args,
    step,
    remaining_objects,
    hidden_attributes,
    observed_attributes,
    held_object,
    placed_objects,
    occlusions,
    action_history,
    console,
    log_json,
):
    """폐기물 action 하나를 실행하고 변경된 held state와 결과를 반환한다."""
    action_failure_count = 0
    if action_type == "detect":
        new_observations = []
        detect_rolls = []
        for obj in visible_objects(remaining_objects, occlusions):
            if obj in observed_attributes:
                continue
            detect_roll = random.random()
            detect_probability = draw_probability(args.detect_success_prob)
            roll_info = {
                "object": obj,
                "true_label": hidden_attributes[obj],
                "detect_roll": detect_roll,
                "detect_probability": detect_probability,
                "detected": detect_roll <= detect_probability,
            }
            if detect_roll <= detect_probability:
                true_label = hidden_attributes[obj]
                label_error_roll = random.random()
                roll_info["label_error_roll"] = label_error_roll
                roll_info["label_error"] = label_error_roll <= args.detect_label_error_prob
                if label_error_roll <= args.detect_label_error_prob:
                    candidates = [label for label in WASTE_ATTRIBUTES if label != true_label]
                    observed_attributes[obj] = random.choice(candidates)
                else:
                    observed_attributes[obj] = true_label
                roll_info["observed_label"] = observed_attributes[obj]
                new_observations.append(f"{obj}: {observed_attributes[obj]}")
            detect_rolls.append(roll_info)
        log_json(f"Step {step} detect rolls:", detect_rolls)
        action_history.append("detect")
        result_text = "Detect result: " + (", ".join(new_observations) if new_observations else "no new attribute observed")

    elif action_type == "pick":
        matched_object = next((obj for obj in remaining_objects if action_arg in obj or obj in action_arg), None)
        if matched_object is None:
            console("Selected object is not in the current state. Stopping to avoid compounding error.")
            return held_object, None, 0, f"invalid pick unavailable object: {action_arg}"
        if matched_object not in visible_objects(remaining_objects, occlusions):
            blocker = occlusions.get(matched_object)
            console(f"Selected object is occluded by {blocker}. Place the blocker before picking it.")
            return held_object, None, 0, f"invalid pick occluded object: {matched_object}"
        if held_object is not None:
            console("Robot is already holding an object. Place it before picking another one.")
            return held_object, None, 0, "invalid pick while holding object"
        failure_roll = random.random()
        failed = failure_roll <= args.pick_failure_prob
        log_json(f"Step {step} pick roll:", {
            "object": matched_object,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"pick {matched_object} (failed)")
            result_text = f"Pick failed: {matched_object} was not picked"
        else:
            held_object = matched_object
            action_history.append(f"pick {matched_object}")
            result_text = f"Executed: pick {matched_object}"

    elif action_type == "place":
        if held_object is None:
            console("Robot is not holding anything. Pick an object before placing.")
            return held_object, None, 0, "invalid place without held object"
        place_object, target_bin = action_arg
        if not (place_object in held_object or held_object in place_object):
            console("Place action does not match the held object. Stopping to avoid compounding error.")
            return held_object, None, 0, "invalid place target mismatch"
        placed_object = held_object
        failure_roll = random.random()
        failed = failure_roll <= args.place_failure_prob
        log_json(f"Step {step} place roll:", {
            "object": placed_object,
            "target_bin": target_bin,
            "failure_roll": failure_roll,
            "failed": failed,
        })
        if failed:
            action_failure_count = 1
            action_history.append(f"place {placed_object} into {target_bin} (failed)")
            result_text = f"Place failed: still holding {placed_object}"
        else:
            held_object = None
            remaining_objects.remove(placed_object)
            observed_attributes.pop(placed_object, None)
            placed_objects[placed_object] = target_bin
            action_history.append(f"place {placed_object} into {target_bin}")
            result_text = f"Executed: place {placed_object} into {target_bin}"
    else:
        return held_object, None, 0, f"unsupported action type: {action_type}"

    return held_object, result_text, action_failure_count, None
