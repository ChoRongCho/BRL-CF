from __future__ import annotations

import math
import random
from typing import Any

from domains.wastesorting.semantics import parse_waste_action


FALLBACK_OPTION_TEXT = "an option not listed here"
ANSWER_MODES = {"oracle", "noisy-oracle", "human-proxy", "human"}
NOISY_ORACLE_ERROR_RATE = 0.1
HUMAN_PROXY_M_MAX = 4.0

HUMAN_PROXY_DIFFICULTY_BY_SITUATION = {
    "held_observed_place": 2,
    "held_unobserved_place": 5,
    "held_unresolved": 5,
    "pick_visible_observed": 2,
    "pick_visible_unobserved": 5,
    "detect_unknown_visible": 5,
    "detect_only_occluded_remaining": 5,
    "detect_no_remaining_work": 2,
    "fallback": 5,
    "default": 5,
}


def select_waste_answer(
    options,
    tokens,
    add_mc_prefix,
    remaining_objects,
    hidden_attributes,
    held_object,
    occlusions,
    observed_attributes=None,
    mode: str = "oracle",
    noisy_oracle_error_rate: float = NOISY_ORACLE_ERROR_RATE,
) -> dict[str, Any]:
    if mode not in ANSWER_MODES:
        raise ValueError(f"Unsupported KnowNo wastesorting answer mode: {mode}")
    if mode == "human":
        raise NotImplementedError("human mode must be handled by manual option input.")

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

    human_proxy_context = _human_proxy_context(
        remaining_objects,
        hidden_attributes,
        held_object,
        occlusions,
        observed_attributes or {},
    )
    selected, rule, noise_applied = _apply_answer_mode(
        selected,
        rule,
        records,
        mode,
        noisy_oracle_error_rate,
        human_proxy_context,
    )
    return {
        "mode": mode,
        "selected_token": selected,
        "rule": rule,
        "noise_applied": noise_applied,
        "options": records,
    }


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
        if matched_object is not None:
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


def match_remaining_object(target, remaining_objects):
    for obj in remaining_objects:
        if target in obj or obj in target:
            return obj
    return None


def visible_waste_objects(remaining_objects: list[str], occlusions: dict[str, str]) -> list[str]:
    remaining = set(remaining_objects)
    return [obj for obj in remaining_objects if occlusions.get(obj) not in remaining]


def _natural_index(name: str | None) -> int:
    if not name:
        return 999
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999


def _human_proxy_context(remaining_objects, hidden_attributes, held_object, occlusions, observed_attributes):
    visible = set(visible_waste_objects(remaining_objects, occlusions))
    observed = set(observed_attributes)
    return {
        "held_object": held_object,
        "held_label": hidden_attributes.get(held_object) if held_object is not None else None,
        "held_observed": held_object in observed if held_object is not None else False,
        "visible_objects": visible,
        "observed_objects": observed,
        "unknown_visible_objects": visible - observed,
        "occluded_objects": set(remaining_objects) - visible,
        "remaining_objects": set(remaining_objects),
    }


def _apply_answer_mode(selected_token, rule, records, mode, noisy_oracle_error_rate, human_proxy_context=None):
    if mode == "oracle":
        return selected_token, rule, False
    if mode == "human-proxy":
        return _apply_human_proxy(selected_token, rule, records, human_proxy_context or {})
    if mode != "noisy-oracle":
        return selected_token, rule, False

    error_rate = _normalize_error_rate(noisy_oracle_error_rate)
    if random.random() >= error_rate:
        return selected_token, rule, False

    alternatives = [record for record in records if record["token"] != selected_token]
    if not alternatives:
        return selected_token, f"noisy-oracle could not flip; no alternative option. {rule}", False

    noisy_record = random.choice(alternatives)
    noisy_rule = (
        f"noisy-oracle flipped oracle option {selected_token} to "
        f"{noisy_record['token']} with error_rate={error_rate}. oracle rule: {rule}"
    )
    return noisy_record["token"], noisy_rule, True


def _apply_human_proxy(selected_token, rule, records, context):
    selected_record = _record_by_token(records, selected_token)
    difficulty, situation = _human_proxy_difficulty(selected_record, context)
    if _sample_human_proxy_correct(difficulty):
        return selected_token, f"human-proxy kept oracle option {selected_token} ({situation}): {rule}", False

    alternatives = [record for record in records if record["token"] != selected_token]
    if not alternatives:
        return selected_token, f"human-proxy could not flip; no alternative option. {rule}", False

    confused_record = random.choice(alternatives)
    confused_rule = (
        f"human-proxy confused oracle option {selected_token} to "
        f"{confused_record['token']} with situation={situation}, difficulty={difficulty}. oracle rule: {rule}"
    )
    return confused_record["token"], confused_rule, True


def _record_by_token(records, token):
    for record in records:
        if record["token"] == token:
            return record
    return {"token": token, "action_type": "fallback", "feasible": False}


def _human_proxy_difficulty(record, context) -> tuple[float, str]:
    if not record.get("feasible", False):
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["fallback"], "fallback"

    action_type = record.get("action_type", "")
    held_object = context.get("held_object")
    if held_object is not None:
        if action_type == "place":
            if context.get("held_observed"):
                return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_observed_place"], "held_observed_place"
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_unobserved_place"], "held_unobserved_place"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_unresolved"], "held_unresolved"

    if action_type == "pick":
        target = record.get("target")
        if target in context.get("observed_objects", set()):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["pick_visible_observed"], "pick_visible_observed"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["pick_visible_unobserved"], "pick_visible_unobserved"

    if action_type == "detect":
        if context.get("unknown_visible_objects"):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_unknown_visible"], "detect_unknown_visible"
        if context.get("occluded_objects"):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_only_occluded_remaining"], "detect_only_occluded_remaining"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_no_remaining_work"], "detect_no_remaining_work"

    return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["default"], "default"


def _sample_human_proxy_correct(difficulty: float) -> bool:
    difficulty = min(max(float(difficulty), 0.0), 10.0)
    margin = HUMAN_PROXY_M_MAX * (1.0 - difficulty / 10.0)
    p_correct = 1.0 / (1.0 + math.exp(-margin))
    return random.random() < p_correct


def _normalize_error_rate(error_rate):
    value = float(error_rate)
    if value > 1.0:
        value /= 100.0
    return min(max(value, 0.0), 1.0)
