"""토마토 작업에서 KnowNo를 여러 단계에 걸쳐 실행하는 실험 loop.

각 단계에서 (1) LLM으로 다음 행동 후보를 생성하고, (2) 후보별 점수로
conformal prediction set을 만들며, (3) 필요한 경우 사람 또는 자동 oracle에
질의한다. 이어서 (4) 선택되거나 oracle이 제공한 action을 실행하고,
(5) 작업의 성공·실패·진행 여부를 판단한다.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from scripts.calibration import (
    run_knowno_calibration,
    write_template,
)
from scripts.llm import call_llm, configure_openai
from scripts.env import TOMATO_MC_PROMPT_FILE
from scripts.prompt import process_mc_raw, temperature_scaling, top_choice_logprobs
from scripts.auto_answer import select_tomato_answer
from scripts.knowno_action_validation import validate_tomato_action
from scripts.utils.logging import build_run_summary, finish_run, show_tomato_decision, start_tomato_run
from scripts.utils.env_setting import add_env_setting_argument, apply_env_setting
from scripts.utils.utils import (
    ActionSelection,
    PlanningResult,
    execute_tomato_action,
    required_tomato_next_action,
    analyze_asked_prediction_set,
    tomato_dead_end_reason,
    tomato_observation_mismatches,
    tomato_success,
    usage_total,
)
from tomato_utils import (
    LOCATIONS,
    TOMATO_BACKGROUND,
    TOMATO_CALIBRATION_TEMPLATE,
    build_tomato_calibration_prompt,
    build_tomato_generation_prompt,
    build_tomato_score_prompt,
    format_tomato_state,
    initialize_tomato_world,
    parse_tomato_action,
)
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-step KnowNo planning for tomato harvesting.")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[4] / "llm_setting.json"),
    )
    parser.add_argument("--instruction", default="Harvest all ripe tomatoes and discard rotten tomatoes.")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--tomatoes", default="tomato1, tomato2, tomato3, tomato4")
    parser.add_argument("--qhat", type=float, default=0.8404)
    parser.add_argument("--score-temperature", "--temperature", dest="score_temperature", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--detect-success-prob", type=float, default=None)
    parser.add_argument("--detect-label-error-prob", type=float, default=None)
    parser.add_argument("--scan-success-prob", type=float, default=None)
    parser.add_argument("--scan-label-error-prob", type=float, default=None)
    parser.add_argument("--navigate-failure-prob", type=float, default=None)
    parser.add_argument("--pick-failure-prob", type=float, default=None)
    parser.add_argument("--place-failure-prob", type=float, default=None)
    parser.add_argument("--discard-failure-prob", type=float, default=None)
    add_env_setting_argument(parser, "tomato")
    parser.add_argument("--ripeness", default="", help='True ripeness, e.g. "tomato1:ripe,tomato2:unripe".')
    parser.add_argument("--freshness", default="", help='True marketability, e.g. "tomato1:fresh,tomato2:rotten".')
    parser.add_argument("--labels", default="", help=argparse.SUPPRESS)
    parser.add_argument("--locations", default="", help='Optional tomato locations, e.g. "tomato1:stem_01,tomato2:stem_02".')
    parser.add_argument("--log-file", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Also print detailed log records to the terminal.")
    parser.add_argument("--auto-answer", action="store_true", help="Use the exact task-state oracle when KnowNo asks for help.")
    parser.add_argument("--calibration-file", default=str(TOMATO_MC_PROMPT_FILE))
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--write-calibration-template", default="")
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--num-calibration", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=0)
    return parser.parse_args()


def plan_tomato_step(
    *,
    args,
    step,
    qhat,
    tokens,
    tomatoes,
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
    total_usage,
    logger,
    call_llm,
):
    """후보 생성부터 conformal prediction set 생성까지 한 planning 단계를 수행한다."""
    history_text = "\n".join(f"{i + 1}. {action}" for i, action in enumerate(action_history)) or "None"
    tomato_state_text = format_tomato_state(
        tomatoes,
        observed_properties,
        scanned_properties,
        held_tomato,
        loaded_tomatoes,
        discarded_tomatoes,
    )
    required_next_action_text = required_tomato_next_action(
        robot_location,
        active_tomatoes,
        observed_properties,
        observed_locations,
        scanned_properties,
        held_tomato,
        detected_stems,
    )
    generation_prompt = build_tomato_generation_prompt(
        args,
        robot_location,
        active_tomatoes,
        tomato_state_text,
        held_tomato,
        loaded_tomatoes,
        discarded_tomatoes,
        history_text,
        required_next_action_text,
    )
    logger.json(f"Step {step} start:", {
        "robot_location": robot_location,
        "active_tomatoes": active_tomatoes,
        "observed_properties": observed_properties,
        "observed_locations": observed_locations,
        "scanned_properties": scanned_properties,
        "held_tomato": held_tomato,
        "loaded_tomatoes": loaded_tomatoes,
        "discarded_tomatoes": discarded_tomatoes,
        "action_history": action_history,
        "detected_stems": sorted(detected_stems),
        "required_next_action": required_next_action_text,
    })
    if args.verbose:
        logger.file_only(f"\n====== Step {step} generation prompt ======")
        logger.file_only(generation_prompt)

    generation_start = time.perf_counter()
    generation_response, generation_text = call_llm(generation_prompt, stop_seq=["We:"], logit_bias={})
    generation_usage = generation_response.get("usage")
    total_usage["generation"] += usage_total(generation_usage)
    total_usage["overall"] += usage_total(generation_usage)
    logger.json(f"Step {step} generation:", {
        "elapsed_sec": time.perf_counter() - generation_start,
        "usage": generation_usage,
        "raw_text": generation_text,
    })
    options_text, options, fallback_token = process_mc_raw(generation_text.strip())

    score_prompt = build_tomato_score_prompt(
        args,
        robot_location,
        active_tomatoes,
        tomato_state_text,
        held_tomato,
        loaded_tomatoes,
        discarded_tomatoes,
        history_text,
        options_text,
        required_next_action_text,
    )
    if args.verbose:
        logger.file_only(f"\n====== Step {step} scoring prompt ======")
        logger.file_only(score_prompt)

    scoring_start = time.perf_counter()
    response, score_text = call_llm(score_prompt, max_tokens=1, logprobs=5, logit_bias={})
    score_usage = response.get("usage")
    total_usage["scoring"] += usage_total(score_usage)
    total_usage["overall"] += usage_total(score_usage)
    logger.json(f"Step {step} scoring:", {
        "elapsed_sec": time.perf_counter() - scoring_start,
        "usage": score_usage,
        "text": score_text,
    })

    _, _, raw_logprobs = top_choice_logprobs(response)
    option_logprobs = {}
    for raw_token, logprob in raw_logprobs.items():
        token = raw_token.strip().strip("'\"").upper()
        if token in tokens:
            option_logprobs[token] = max(logprob, option_logprobs.get(token, -np.inf))
    if not option_logprobs:
        raise ValueError(f"LLM did not return any A/B/C/D/E logprobs: {raw_logprobs}")

    scored_tokens = list(option_logprobs)
    logprobs = list(option_logprobs.values())
    scores = temperature_scaling(logprobs, temperature=args.score_temperature)
    prediction_set = [token for token, score in zip(scored_tokens, scores) if score >= 1 - qhat]
    logger.json(f"Step {step} decision data:", {
        "options": options,
        "add_mc_prefix": fallback_token,
        "option_logprobs": option_logprobs,
        "scores": scores.tolist(),
        "threshold": 1 - qhat,
        "prediction_set": prediction_set,
    })
    show_tomato_decision(
        logger,
        step,
        robot_location,
        active_tomatoes,
        hidden_ripeness,
        hidden_freshness,
        hidden_locations,
        tomato_state_text,
        held_tomato,
        options_text,
        prediction_set,
        scored_tokens,
        logprobs,
        scores,
    )
    return PlanningResult(options_text, options, fallback_token, scored_tokens, logprobs, scores, prediction_set)


def select_tomato_action(
    planning,
    *,
    args,
    step,
    tokens,
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
    logger,
):
    """Prediction set에서 실행할 행동을 고르고 필요하면 외부에 질의한다."""
    prediction_set = planning.prediction_set
    help_needed = len(prediction_set) != 1 or planning.fallback_token in prediction_set
    provided_action = None

    if help_needed and args.auto_answer:
        answer = select_tomato_answer(
            planning.options,
            tokens,
            planning.fallback_token,
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
            prediction_set,
        )
        selected_token = answer["selected_token"]
        provided_action = answer.get("provided_action")
        logger.console(f"Oracle selected option {selected_token}.")
        if provided_action is not None:
            logger.console(f"Oracle provided action for NoOpt: {provided_action}")
        logger.json(f"Step {step} oracle answer:", answer)
    elif help_needed:
        if not prediction_set:
            logger.console("Prediction set is empty. PLAN FAILURE reached.")
            return ActionSelection(None, None, True, error="empty prediction set")
        while True:
            selected_token = input("Help needed. Choose an option (A/B/C/D/E): ").strip().upper()
            if selected_token in prediction_set:
                break
            logger.console(f"Invalid option. Please choose from the prediction set: {prediction_set}")
    else:
        selected_token = prediction_set[0]

    if selected_token not in tokens:
        raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")
    selected_action = provided_action or planning.options[tokens.index(selected_token)]
    return ActionSelection(
        selected_token,
        selected_action,
        help_needed,
        oracle_provided_action=provided_action is not None,
        oracle_answer=answer if help_needed and args.auto_answer else None,
    )


def main(call_llm=call_llm, baseline_name="KnowNo") -> None:
    args = parse_args()
    apply_env_setting(args, "tomato")
    run_start = time.perf_counter()
    if args.write_calibration_template:
        write_template(args.write_calibration_template, TOMATO_CALIBRATION_TEMPLATE)
        return
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    settings = configure_openai(args.api_key, args.settings)
    qhat = settings.get("qhat", args.qhat)

    tomatoes = [obj.strip().lower() for obj in args.tomatoes.split(",") if obj.strip()]
    hidden_ripeness, hidden_freshness, hidden_locations = initialize_tomato_world(
        tomatoes,
        args.ripeness,
        args.freshness,
        args.locations,
        legacy_labels=args.labels,
    )

    robot_location = "dock_station"
    observed_properties = {}
    observed_locations = {}
    scanned_properties = {}
    held_tomato = None
    loaded_tomatoes = []
    discarded_tomatoes = []
    action_history = []
    detected_stems = set()
    tokens = ["A", "B", "C", "D", "E"]

    logger = start_tomato_run(
        __file__,
        args,
        baseline_name,
        settings,
        tomatoes,
        LOCATIONS,
        hidden_ripeness,
        hidden_freshness,
        hidden_locations,
    )
    console = logger.console
    log_json = logger.json
    total_usage = {"generation": 0, "scoring": 0, "overall": 0}
    qhat = args.qhat
    if args.run_calibration:
        qhat = run_knowno_calibration(
            args.calibration_file,
            args.num_calibration,
            args.num_test,
            args.target_success,
            domain_name="tomato",
            background=TOMATO_BACKGROUND,
            generation_prompt_builder=build_tomato_calibration_prompt,
        )
    console("qhat:", qhat)
    console("Detailed log file:", logger.path)

    completed_iterations = 0
    help_count = 0
    autonomous_count = 0
    fallback_in_prediction_count = 0
    candidate_counts = []
    help_candidate_counts = []
    prediction_set_sizes = []
    help_prediction_set_sizes = []
    action_failure_count = 0
    noopt_recovery_count = 0
    failure_mode_events = []
    terminal_failure_category = None
    stop_reason = "unknown"

    for step in range(1, args.max_steps + 1):
        step_start = time.perf_counter()

        # 1. 현재 상태에서 작업이 이미 종료되었는지 확인한다.
        dead_end_reason = tomato_dead_end_reason(
            hidden_ripeness,
            hidden_freshness,
            held_tomato,
            loaded_tomatoes,
            discarded_tomatoes,
        )
        if dead_end_reason is not None:
            console(dead_end_reason)
            stop_reason = dead_end_reason
            log_json(f"Step {step} plan failure:", {
                "reason": dead_end_reason,
                "held_tomato": held_tomato,
                "loaded_tomatoes": loaded_tomatoes,
                "discarded_tomatoes": discarded_tomatoes,
            })
            break

        if tomato_success(hidden_ripeness, hidden_freshness, held_tomato, loaded_tomatoes, discarded_tomatoes):
            console("\nAll target tomatoes have been handled successfully.")
            stop_reason = "success"
            log_json(f"Step {step} success:", {
                "held_tomato": held_tomato,
                "loaded_tomatoes": loaded_tomatoes,
                "discarded_tomatoes": discarded_tomatoes,
            })
            break

        active_tomatoes = [t for t in tomatoes if t not in loaded_tomatoes and t not in discarded_tomatoes]
        if not active_tomatoes and held_tomato is None:
            console("\nAll target tomatoes have been handled.")
            stop_reason = "all tomatoes handled"
            break

        # 2. 행동 후보와 conformal prediction set을 만든다.
        planning = plan_tomato_step(
            args=args,
            step=step,
            qhat=qhat,
            tokens=tokens,
            tomatoes=tomatoes,
            robot_location=robot_location,
            active_tomatoes=active_tomatoes,
            hidden_ripeness=hidden_ripeness,
            hidden_freshness=hidden_freshness,
            hidden_locations=hidden_locations,
            observed_properties=observed_properties,
            observed_locations=observed_locations,
            scanned_properties=scanned_properties,
            held_tomato=held_tomato,
            loaded_tomatoes=loaded_tomatoes,
            discarded_tomatoes=discarded_tomatoes,
            detected_stems=detected_stems,
            action_history=action_history,
            total_usage=total_usage,
            logger=logger,
            call_llm=call_llm,
        )
        completed_iterations = step
        candidate_counts.append(len(planning.options))
        prediction_set_sizes.append(len(planning.prediction_set))

        # 3. 자율 실행하거나 외부 답변을 받아 실행할 행동을 정한다.
        if planning.fallback_token in planning.prediction_set:
            fallback_in_prediction_count += 1
        selection = select_tomato_action(
            planning,
            args=args,
            step=step,
            tokens=tokens,
            robot_location=robot_location,
            active_tomatoes=active_tomatoes,
            hidden_ripeness=hidden_ripeness,
            hidden_freshness=hidden_freshness,
            hidden_locations=hidden_locations,
            observed_properties=observed_properties,
            observed_locations=observed_locations,
            scanned_properties=scanned_properties,
            held_tomato=held_tomato,
            loaded_tomatoes=loaded_tomatoes,
            discarded_tomatoes=discarded_tomatoes,
            logger=logger,
        )
        if selection.help_needed:
            help_count += 1
            help_candidate_counts.append(len(planning.options))
            help_prediction_set_sizes.append(len(planning.prediction_set))
        else:
            autonomous_count += 1

        observation_mismatches = tomato_observation_mismatches(
            observed_properties,
            observed_locations,
            scanned_properties,
            hidden_ripeness,
            hidden_locations,
            hidden_freshness,
        )
        asked_diagnostic = analyze_asked_prediction_set(
            selection,
            planning.prediction_set,
            observation_mismatches,
        )
        if asked_diagnostic is not None:
            asked_diagnostic = {"step": step, **asked_diagnostic}
            log_json(f"Step {step} query diagnostic:", asked_diagnostic)
            if asked_diagnostic["category"] is not None:
                failure_mode_events.append(asked_diagnostic)
        if selection.error is not None:
            if asked_diagnostic is not None:
                terminal_failure_category = asked_diagnostic["category"]
            stop_reason = selection.error
            break

        selected_token = selection.token
        selected_action = selection.action
        oracle_provided_action = selected_action if selection.oracle_provided_action else None
        if selection.oracle_provided_action:
            noopt_recovery_count += 1
        # 4. 선택된 행동을 검증하고 환경 상태에 적용한다.
        action_type, action_arg = parse_tomato_action(selected_action)
        # 5. 실행 결과를 기록하고 성공·실패·진행 여부를 판단한다.
        if oracle_provided_action is not None:
            recovery_feasible, recovery_reason = validate_tomato_action(
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
            log_json(f"Step {step} oracle NoOpt validation:", {
                "action": selected_action,
                "feasible": recovery_feasible,
                "reason": recovery_reason,
            })
            if not recovery_feasible:
                stop_reason = f"invalid oracle NoOpt action: {recovery_reason}"
                console(stop_reason)
                break
        if action_type == "done":
            console("Planner selected a terminal action. Stopping.")
            stop_reason = "planner selected terminal action"
            break
        if action_type is None:
            console(f"Planner: {action_type}")
            console("Planner selected a terminal or non-executable action. Stopping.")
            stop_reason = f"non-executable action selected: {selected_action}"
            break

        robot_location, held_tomato, result_text, failure_delta, execution_error = execute_tomato_action(
            action_type,
            action_arg,
            args=args,
            step=step,
            robot_location=robot_location,
            active_tomatoes=active_tomatoes,
            hidden_ripeness=hidden_ripeness,
            hidden_freshness=hidden_freshness,
            hidden_locations=hidden_locations,
            observed_properties=observed_properties,
            observed_locations=observed_locations,
            scanned_properties=scanned_properties,
            held_tomato=held_tomato,
            loaded_tomatoes=loaded_tomatoes,
            discarded_tomatoes=discarded_tomatoes,
            detected_stems=detected_stems,
            action_history=action_history,
            console=console,
            log_json=log_json,
        )
        action_failure_count += failure_delta
        if execution_error is not None:
            stop_reason = execution_error
            break

        if oracle_provided_action is not None:
            source = "oracle NoOpt recovery"
        else:
            source = "oracle" if selection.help_needed and args.auto_answer else ("user" if selection.help_needed else "prediction set")
        console(f"Selected/Executed ({source}, option {selected_token}): {selected_action} -> {result_text}")
        dead_end_reason = tomato_dead_end_reason(
            hidden_ripeness,
            hidden_freshness,
            held_tomato,
            loaded_tomatoes,
            discarded_tomatoes,
        )
        log_json(f"Step {step} end:", {
            "robot_location": robot_location,
            "observed_properties": observed_properties,
            "observed_locations": observed_locations,
            "scanned_properties": scanned_properties,
            "held_tomato": held_tomato,
            "loaded_tomatoes": loaded_tomatoes,
            "discarded_tomatoes": discarded_tomatoes,
            "detected_stems": sorted(detected_stems),
            "action_history": action_history,
            "dead_end_reason": dead_end_reason,
            "step_elapsed_sec": time.perf_counter() - step_start,
        })
        if dead_end_reason is not None:
            if not selection.help_needed:
                terminal_failure_category = "when_missed_query_dead_end"
                failure_event = {
                    "step": step,
                    "category": terminal_failure_category,
                    "help_requested": False,
                    "selected_token": selected_token,
                    "selected_action": selected_action,
                    "observation_error_present": bool(observation_mismatches),
                    "observation_mismatches": observation_mismatches,
                    "dead_end_reason": dead_end_reason,
                }
                failure_mode_events.append(failure_event)
                log_json(f"Step {step} failure diagnostic:", failure_event)
            elif asked_diagnostic is not None and asked_diagnostic["category"] is not None:
                terminal_failure_category = asked_diagnostic["category"]
            else:
                terminal_failure_category = "other_failure_after_query"
            console(dead_end_reason)
            stop_reason = dead_end_reason
            break
        if tomato_success(hidden_ripeness, hidden_freshness, held_tomato, loaded_tomatoes, discarded_tomatoes):
            console("\nAll target tomatoes have been handled successfully.")
            stop_reason = "success"
            break

    else:
        console("\nReached max steps before all tomatoes were handled.")
        stop_reason = "max steps reached"

    remaining = [t for t in tomatoes if t not in loaded_tomatoes and t not in discarded_tomatoes]
    total_elapsed = time.perf_counter() - run_start
    final_success = tomato_success(hidden_ripeness, hidden_freshness, held_tomato, loaded_tomatoes, discarded_tomatoes)
    failure_mode_counts = {
        category: sum(event["category"] == category for event in failure_mode_events)
        for category in (
            "when_missed_query_dead_end",
            "what_missing_correct_option_after_observation_error",
            "when_what_missing_correct_option_without_observation_error",
        )
    }
    summary = build_run_summary(
        success=final_success,
        stop_reason=stop_reason,
        action_history=action_history,
        planning_iterations=completed_iterations,
        question_count=help_count,
        autonomous_action_count=autonomous_count,
        fallback_in_prediction_count=fallback_in_prediction_count,
        noopt_recovery_count=noopt_recovery_count,
        candidate_counts=candidate_counts,
        help_candidate_counts=help_candidate_counts,
        prediction_set_sizes=prediction_set_sizes,
        help_prediction_set_sizes=help_prediction_set_sizes,
        action_failure_count=action_failure_count,
        terminal_failure_category=terminal_failure_category,
        failure_mode_counts=failure_mode_counts,
        failure_mode_events=failure_mode_events,
        token_usage=total_usage,
        total_elapsed_seconds=total_elapsed,
        loaded_tomatoes=loaded_tomatoes,
        discarded_tomatoes=discarded_tomatoes,
        held_tomato=held_tomato,
        unhandled_tomatoes=remaining,
    )
    finish_run(
        logger,
        action_history,
        total_usage,
        total_elapsed,
        summary,
        held_label="Held tomato:",
        held_value=held_tomato,
        remaining_label="Unhandled tomatoes:",
        remaining_values=remaining,
    )


if __name__ == "__main__":
    main()
