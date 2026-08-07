from __future__ import annotations

import math
import random
from typing import Any

from domains.tomato.utils import STEMS, parse_tomato_action


FALLBACK_OPTION_TEXT = "an option not listed here"
ANSWER_MODES = {"oracle", "noisy-oracle", "human-proxy", "human"}
NOISY_ORACLE_ERROR_RATE = 0.1
HUMAN_PROXY_M_MAX = 4.0

HUMAN_PROXY_DIFFICULTY_BY_SITUATION = {
    "held_unknown_scan": 4,
    "held_ripe_place": 2,
    "held_rotten_discard": 2,
    "held_unresolved": 5,
    "pick_actionable": 1,
    "detect_unknown_current_stem": 5,
    "detect_goal_work_current_stem": 5,
    "detect_done_current_stem": 3,
    "navigate_to_work_stem": 3,
    "navigate_no_known_work": 5,
    "fallback": 5,
    "default": 5,
}


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
    mode: str = "oracle",
    noisy_oracle_error_rate: float = NOISY_ORACLE_ERROR_RATE,
) -> dict[str, Any]:
    if mode not in ANSWER_MODES:
        raise ValueError(f"Unsupported KnowNo tomato answer mode: {mode}")
    if mode == "human":
        raise NotImplementedError("human mode must be handled by manual option input.")

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

    human_proxy_context = _human_proxy_context(
        robot_location,
        active,
        handled,
        hidden_properties,
        hidden_locations,
        observed_properties,
        held_tomato,
        scanned_properties,
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


def _tomato_record(
    token,
    option,
    add_mc_prefix,
    robot_location,
    active,
    handled,
    hidden_properties,
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


def _human_proxy_context(
    robot_location,
    active,
    handled,
    hidden_properties,
    hidden_locations,
    observed_properties,
    held_tomato,
    scanned_properties,
):
    work_stems = {
        stem
        for stem in STEMS
        if _stem_has_goal_work(stem, active, handled, hidden_properties, hidden_locations, observed_properties)
    }
    return {
        "robot_location": robot_location,
        "held_tomato": held_tomato,
        "held_property": hidden_properties.get(held_tomato) if held_tomato is not None else None,
        "held_scanned_property": scanned_properties.get(held_tomato, "unknown") if held_tomato is not None else None,
        "current_unknown": _unknown_tomatoes_at(
            robot_location,
            active,
            handled,
            hidden_locations,
            observed_properties,
        ),
        "current_has_work": robot_location in work_stems,
        "work_stems": work_stems,
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
    held_tomato = context.get("held_tomato")
    if held_tomato is not None:
        if action_type == "scan" and context.get("held_scanned_property") == "unknown":
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_unknown_scan"], "held_unknown_scan"
        if action_type == "place" and context.get("held_property") == "ripe":
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_ripe_place"], "held_ripe_place"
        if action_type == "discard" and context.get("held_property") == "rotten":
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_rotten_discard"], "held_rotten_discard"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["held_unresolved"], "held_unresolved"

    if action_type == "pick":
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["pick_actionable"], "pick_actionable"

    if action_type == "detect":
        if context.get("current_unknown"):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_unknown_current_stem"], "detect_unknown_current_stem"
        if context.get("current_has_work"):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_goal_work_current_stem"], "detect_goal_work_current_stem"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["detect_done_current_stem"], "detect_done_current_stem"

    if action_type == "navigate":
        target = record.get("target")
        if target in context.get("work_stems", set()):
            return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["navigate_to_work_stem"], "navigate_to_work_stem"
        return HUMAN_PROXY_DIFFICULTY_BY_SITUATION["navigate_no_known_work"], "navigate_no_known_work"

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
