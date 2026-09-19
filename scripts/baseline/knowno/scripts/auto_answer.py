"""KnowNo 자동 실험에서 사람의 답변을 대신하는 정답 oracle.

KnowNo가 도움을 요청하면 prediction set에 포함된 각 행동을 실제 hidden
state와 비교한다. 실행 조건을 만족하지 않거나 실제 상태에서 잘못된 결과를
만드는 행동은 제외하고, 남은 후보 중 아래 우선순위에 따라 하나를 답한다.

토마토 작업의 답변 규칙:
1. 로봇이 토마토를 들고 있고 아직 scan 결과가 없으면 ``scan``을 선택한다.
2. 들고 있는 토마토가 실제로 fresh이면 ``place``를, rotten이면 ``discard``를
   선택한다. 단, 후보가 들고 있는 토마토를 대상으로 해야 하며, ``place``는
   관측된 scan 결과도 fresh여야 한다.
3. 로봇이 아무것도 들고 있지 않으면 현재 위치에서 관측된 ripe 토마토 중
   실제 상태도 ripe인 토마토를 대상으로 하는 ``pick``을 선택한다.
4. 현재 stem에 아직 관측되지 않은 토마토가 있고 ``detect`` 후보가 있으면
   ``detect``를 선택한다.
5. 현재 위치에 처리할 토마토가 없고, 다른 stem에 아직 관측되지 않았거나
   실제로 ripe인 미처리 토마토가 있으면 해당 stem으로 가는 ``navigate``를
   선택한다.
6. 위 조건에 해당하지 않지만 실행 가능한 ``detect``가 있으면 이를 선택하고,
   그래도 없으면 남아 있는 첫 번째 실행 가능 후보를 선택한다.

폐기물 작업의 답변 규칙:
1. 로봇이 물체를 들고 있으면 그 물체를 실제 분류와 일치하는 bin에 넣는
   ``place``를 선택한다.
2. 로봇이 아무것도 들고 있지 않으면 현재 보이고 이미 종류가 관측된 물체를
   대상으로 하는 ``pick``을 선택한다.
3. 위 후보가 없고 ``detect``가 실행 가능하면 ``detect``를 선택한다.
4. 그래도 없으면 남아 있는 첫 번째 실행 가능 후보를 선택한다.

먼저 위 판단을 prediction set 안의 후보에만 적용한다. set 안에 올바른 실행
행동이 있으면 그 행동을 답하며, 없고 NoOpt가 set에 포함되어 있으면 NoOpt를
선택한다. NoOpt가 set에 함께 있어도 올바른 일반 행동을 선택했다면 기존처럼
그 행동을 그대로 실행한다.

Oracle이 prediction set 안에서 NoOpt를 실제로 선택한 경우에는 hidden
state와 현재 실행 상태를 보고 유한한 domain action 중 다음에 실행할
행동을 바로 결정한다. LLM이 생성했지만 prediction set에서 제외된
후보는 다시 검색하지 않는다. 제공된 행동은 실행 전에 다시 검증한다.
NoOpt가 단순히 prediction set에 포함되었다는 이유만으로 이 복구 절차를
수행하지는 않는다.
"""

from __future__ import annotations

from typing import Any

from tomato_utils import parse_tomato_action
from wastesorting_utils import parse_waste_action
from scripts.knowno_action_validation import (
    match_remaining_object,
    validate_tomato_action,
    validate_waste_action,
    visible_waste_objects,
)

FALLBACK_OPTION_TEXT = "an option not listed here"


def _natural_index(name):
    if not name:
        return 999
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 999


def _revealed_waste_count(target, remaining_objects, occlusions):
    """Count objects transitively revealed after removing ``target``."""
    remaining = set(remaining_objects)
    revealed = set()
    frontier = [target]
    while frontier:
        occluder = frontier.pop()
        for obj in remaining:
            if obj not in revealed and occlusions.get(obj) == occluder:
                revealed.add(obj)
                frontier.append(obj)
    return len(revealed)


def _tomato_oracle_action(
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
):
    """Create a canonical action when NoOpt is selected.

    Logic
    토마토를 들고 있을 때:
    1. 아직 scan하지 않았다면 scan
    2. 실제로 rotten이면 discard
    3. 실제로 fresh인데 scan 결과가 fresh가 아니면 다시 scan
    4. 실제로 fresh이고 scan 결과도 fresh이면 place

    토마토를 들고 있지 않을 때:
    1. 현재 위치에서 실제로 ripe + ripe로 관측됨 + 위치도 현재 위치로 관측됨인 토마토를 pick
    2. 현재 위치에 실제 ripe 토마토가 있지만 아직 올바르게 관측되지 않았다면 detect
    3. 다른 stem에 실제 ripe 토마토가 남아 있으면 그 stem으로 navigate
    4. 처리할 ripe 토마토가 없으면 행동을 반환하지 않음
    navigate는 남은 ripe 토마토가 가장 많은 stem을 선택하고,
    완전히 동률인 대상은 번호순으로 선택합니다.
    """
    handled = set(loaded_tomatoes) | set(discarded_tomatoes)
    if held_tomato is not None:
        scan_result = scanned_properties.get(held_tomato, "unknown")
        if scan_result == "unknown":
            return f"scan {held_tomato}", "scan held tomato before deciding its destination"
        if hidden_freshness.get(held_tomato) == "rotten":
            return f"discard {held_tomato}", "discard true rotten held tomato"
        if scan_result != "fresh":
            return f"scan {held_tomato}", "rescan true fresh tomato after an incorrect observation"
        return f"place {held_tomato}", "place true fresh held tomato"

    actionable = [
        tomato for tomato in active_tomatoes
        if tomato not in handled
        and hidden_ripeness.get(tomato) == "ripe"
        and observed_properties.get(tomato) == "ripe"
        and observed_locations.get(tomato) == robot_location
    ]
    if actionable:
        target = min(actionable, key=_natural_index)
        return f"pick {target}", f"pick actionable true ripe {target}"

    unresolved_here = [
        tomato for tomato in active_tomatoes
        if tomato not in handled
        and hidden_ripeness.get(tomato) == "ripe"
        and hidden_locations.get(tomato) == robot_location
    ]
    if unresolved_here:
        return f"detect {robot_location}", "detect unresolved ripe tomatoes at current stem"

    ripe_by_stem = {}
    for tomato in active_tomatoes:
        if tomato not in handled and hidden_ripeness.get(tomato) == "ripe":
            stem = hidden_locations[tomato]
            ripe_by_stem[stem] = ripe_by_stem.get(stem, 0) + 1
    if ripe_by_stem:
        # Visit the stem with the most remaining work to avoid extra navigation.
        target = min(ripe_by_stem, key=lambda stem: (-ripe_by_stem[stem], _natural_index(stem)))
        if target != robot_location:
            return (
                f"navigate to {target}",
                f"navigate to {target} with {ripe_by_stem[target]} remaining ripe tomatoes",
            )
        return f"detect {target}", f"detect remaining ripe tomatoes at {target}"
    return None, "no task-progressing tomato action exists"


def _waste_oracle_action(
    remaining_objects,
    hidden_attributes,
    observed_attributes,
    held_object,
    occlusions,
):
    """Create a canonical action when NoOpt is selected.

    Logic

    물체를 들고 있을 때:
    1. hidden state의 실제 종류를 확인
    2. 실제 종류에 맞는 bin으로 place

    물체를 들고 있지 않을 때:
    1. 가려지지 않았고 종류가 이미 관측된 물체가 있으면 pick
    2. 보이는 물체는 있지만 종류가 관측되지 않았다면 detect
    3. 보이는 물체가 없으면 행동을 반환하지 않음
    여러 물체가 가능하면 집었을 때 드러나는 물체가 가장 많은 대상을
    선택하고, 완전히 동률이면 번호순으로 선택합니다.
    """
    if held_object is not None:
        target_bin = f"{hidden_attributes[held_object]} bin"
        return f"place {held_object} into {target_bin}", f"place held object into true {target_bin}"

    visible = set(visible_waste_objects(remaining_objects, occlusions))
    actionable = [obj for obj in remaining_objects if obj in visible and obj in observed_attributes]
    if actionable:
        target = min(
            actionable,
            key=lambda obj: (-_revealed_waste_count(obj, remaining_objects, occlusions), _natural_index(obj)),
        )
        revealed = _revealed_waste_count(target, remaining_objects, occlusions)
        return f"pick {target}", f"pick observed visible {target}, revealing {revealed} blocked objects"
    if visible:
        return "detect", "detect unresolved visible waste"
    return None, "no task-progressing waste action exists"


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
        result = {"selected_token": add_mc_prefix, "rule": "no correct feasible option", "options": records}
        if add_mc_prefix in allowed:
            recovery = recover_tomato_noopt_action(
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
                scanned_properties,
            )
            result["provided_action"] = recovery["provided_action"]
            result["noopt_recovery"] = recovery
        return result

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
        result = {"selected_token": add_mc_prefix, "rule": "no correct feasible option", "options": records}
        if add_mc_prefix in allowed:
            recovery = recover_waste_noopt_action(
                remaining_objects,
                hidden_attributes,
                observed_attributes,
                held_object,
                occlusions,
            )
            result["provided_action"] = recovery["provided_action"]
            result["noopt_recovery"] = recovery
        return result
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


def recover_tomato_noopt_action(
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
):
    """Choose a finite domain action directly from the true simulator state."""
    scanned_properties = scanned_properties or {}
    action, rule = _tomato_oracle_action(
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
    )
    return {
        "provided_action": action,
        "source": "hidden-state oracle",
        "rule": rule,
    }


def recover_waste_noopt_action(
    remaining_objects,
    hidden_attributes,
    observed_attributes,
    held_object,
    occlusions,
):
    """Choose a finite domain action directly from the true simulator state."""
    action, rule = _waste_oracle_action(
        remaining_objects,
        hidden_attributes,
        observed_attributes,
        held_object,
        occlusions,
    )
    return {
        "provided_action": action,
        "source": "hidden-state oracle",
        "rule": rule,
    }


def _result(selected, rule, records):
    return {"selected_token": selected["token"], "rule": rule, "options": records}
