from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

from scripts.calibration import (
    run_knowno_calibration,
    write_template,
)
from log_tomato import RunLogger
from scripts.llm import call_llm, configure_openai
from scripts.env import TOMATO_MC_PROMPT_FILE
from scripts.prompt import process_mc_raw, temperature_scaling, top_choice_logprobs
from utils import GREEN, RESET, YELLOW, usage_total
from tomato_utils import (
    LOCATIONS,
    STEMS,
    TOMATO_BACKGROUND,
    TOMATO_CALIBRATION_TEMPLATE,
    TOMATO_PROPERTIES,
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
    parser.add_argument("--settings", default=str(Path(__file__).with_name("llm_setting.json")))
    parser.add_argument("--instruction", default="Harvest all ripe tomatoes and discard rotten tomatoes.")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--tomatoes", default="tomato1, tomato2, tomato3, tomato4")
    parser.add_argument("--qhat", type=float, default=0.928)
    parser.add_argument("--score-temperature", "--temperature", dest="score_temperature", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--detect-success-prob", type=float, default=0.8)
    parser.add_argument("--detect-label-error-prob", type=float, default=0.0)
    parser.add_argument("--scan-success-prob", type=float, default=0.9)
    parser.add_argument("--scan-label-error-prob", type=float, default=0.0)
    parser.add_argument("--navigate-failure-prob", type=float, default=0.0)
    parser.add_argument("--pick-failure-prob", type=float, default=0.0)
    parser.add_argument("--place-failure-prob", type=float, default=0.0)
    parser.add_argument("--discard-failure-prob", type=float, default=0.0)
    parser.add_argument("--labels", default="", help='Optional tomato properties, e.g. "tomato1:ripe,tomato2:rotten".')
    parser.add_argument("--locations", default="", help='Optional tomato locations, e.g. "tomato1:stem_01,tomato2:stem_02".')
    parser.add_argument("--log-file", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Also print detailed log records to the terminal.")
    parser.add_argument("--calibration-file", default=str(TOMATO_MC_PROMPT_FILE))
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--write-calibration-template", default="")
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--num-calibration", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=0)
    return parser.parse_args()


def tomato_dead_end_reason(hidden_properties, held_tomato, loaded_tomatoes, discarded_tomatoes):
    for tomato in discarded_tomatoes:
        if hidden_properties.get(tomato) == "ripe":
            return f"Dead-end: true ripe tomato {tomato} was discarded."
    for tomato in loaded_tomatoes:
        if hidden_properties.get(tomato) == "rotten":
            return f"Dead-end: true rotten tomato {tomato} was loaded."
    if held_tomato is not None and hidden_properties.get(held_tomato) == "unripe":
        return f"Dead-end: true unripe tomato {held_tomato} is held."
    return None


def tomato_success(hidden_properties, held_tomato, loaded_tomatoes, discarded_tomatoes):
    if held_tomato is not None:
        return False
    for tomato, prop in hidden_properties.items():
        if prop == "ripe" and tomato not in loaded_tomatoes:
            return False
        if prop == "rotten" and tomato not in discarded_tomatoes:
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
        if scanned_property == "ripe":
            return f"place {held_tomato}"
        if scanned_property in {"rotten", "unripe"}:
            return f"discard {held_tomato}"
        return "scan"

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


def main() -> None:
    args = parse_args()
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
    hidden_properties, hidden_locations = initialize_tomato_world(tomatoes, args.labels, args.locations)

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

    logger = RunLogger(__file__, args.log_file, args.verbose, prefix="knowno_multistep_tomato")
    console = logger.console
    console_colored = logger.colored
    log = logger.file_only
    log_json = logger.json
    total_usage = {"generation": 0, "scoring": 0, "overall": 0}
    console("====== Multi-step Tomato KnowNo ======")
    log_json("Run metadata:", {
        "argv": sys.argv,
        "model": settings.get("model") or settings.get("model_name"),
        "prompt_version": args.prompt_version,
        "seed": args.seed,
    })
    console("Instruction:", args.instruction)
    console("Prompt version:", args.prompt_version)
    console("Tomatoes:", ", ".join(tomatoes))
    console("Locations:", ", ".join(LOCATIONS))
    console("True properties:", ", ".join(f"{obj}: {hidden_properties[obj]}" for obj in sorted(hidden_properties)))
    console("True locations:", ", ".join(f"{obj}: {hidden_locations[obj]}" for obj in sorted(hidden_locations)))
    console("Detect success/error:", args.detect_success_prob, "/", args.detect_label_error_prob)
    console("Scan success/error:", args.scan_success_prob, "/", args.scan_label_error_prob)
    console("Action failure probabilities:",
            f"navigate={args.navigate_failure_prob}, pick={args.pick_failure_prob}, "
            f"place={args.place_failure_prob}, discard={args.discard_failure_prob}")
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
    stop_reason = "unknown"

    for step in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        dead_end_reason = tomato_dead_end_reason(
            hidden_properties,
            held_tomato,
            loaded_tomatoes,
            discarded_tomatoes,
        )
        if dead_end_reason is not None:
            console(dead_end_reason)
            stop_reason = dead_end_reason
            log_json(f"Step {step} dead-end:", {
                "reason": dead_end_reason,
                "held_tomato": held_tomato,
                "loaded_tomatoes": loaded_tomatoes,
                "discarded_tomatoes": discarded_tomatoes,
            })
            break

        if tomato_success(hidden_properties, held_tomato, loaded_tomatoes, discarded_tomatoes):
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

        history_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(action_history)) or "None"
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

        # mc_gen_prompt
        mc_gen_prompt = build_tomato_generation_prompt(
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

        log_json(f"Step {step} start:", {
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
            log(f"\n====== Step {step} generation prompt ======")
            log(mc_gen_prompt)

        gen_start = time.perf_counter()
        gen_response, mc_gen_raw = call_llm(mc_gen_prompt, stop_seq=["We:"], logit_bias={})
        gen_elapsed = time.perf_counter() - gen_start
        gen_usage = gen_response.get("usage")
        total_usage["generation"] += usage_total(gen_usage)
        total_usage["overall"] += usage_total(gen_usage)
        log_json(f"Step {step} generation:", {
            "elapsed_sec": gen_elapsed,
            "usage": gen_usage,
            "raw_text": mc_gen_raw,
        })
        mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_raw.strip())
        completed_iterations = step
        candidate_counts.append(len(mc_gen_all))

        score_prompt = build_tomato_score_prompt(
            args,
            robot_location,
            active_tomatoes,
            tomato_state_text,
            held_tomato,
            loaded_tomatoes,
            discarded_tomatoes,
            history_text,
            mc_gen_full,
            required_next_action_text,
        )

        if args.verbose:
            log(f"\n====== Step {step} scoring prompt ======")
            log(score_prompt)

        score_start = time.perf_counter()
        response, score_text = call_llm(score_prompt, max_tokens=1, logprobs=5)
        score_elapsed = time.perf_counter() - score_start
        score_usage = response.get("usage")
        total_usage["scoring"] += usage_total(score_usage)
        total_usage["overall"] += usage_total(score_usage)
        log_json(f"Step {step} scoring:", {
            "elapsed_sec": score_elapsed,
            "usage": score_usage,
            "text": score_text,
        })
        _, _, top_logprobs_full = top_choice_logprobs(response)
        option_logprobs = {}
        for raw_token, logprob in top_logprobs_full.items():
            token = raw_token.strip().strip("'\"").upper()
            if token in tokens:
                option_logprobs[token] = max(logprob, option_logprobs.get(token, -np.inf))
        if not option_logprobs:
            raise ValueError(f"LLM did not return any A/B/C/D/E logprobs: {top_logprobs_full}")
        top_tokens = list(option_logprobs.keys())
        top_logprobs = list(option_logprobs.values())
        scores = temperature_scaling(top_logprobs, temperature=args.score_temperature)
        preds = [token for token, score in zip(top_tokens, scores) if score >= 1 - qhat]
        prediction_set_sizes.append(len(preds))
        log_json(f"Step {step} decision data:", {
            "options": mc_gen_all,
            "add_mc_prefix": add_mc_prefix,
            "option_logprobs": option_logprobs,
            "scores": scores.tolist(),
            "threshold": 1 - qhat,
            "prediction_set": preds,
        })

        console(f"\n====== Step {step} ======")
        console("Robot location:", robot_location)
        console("Active tomatoes:", ", ".join(active_tomatoes) if active_tomatoes else "None")
        true_prop_text = ", ".join(f"{t}: {hidden_properties[t]}" for t in sorted(hidden_properties))
        true_loc_text = ", ".join(f"{t}: {hidden_locations[t]}" for t in sorted(hidden_locations))
        console_colored(f"{YELLOW}True tomato properties: {true_prop_text}{RESET}", f"True tomato properties: {true_prop_text}")
        console_colored(f"{YELLOW}True tomato locations: {true_loc_text}{RESET}", f"True tomato locations: {true_loc_text}")
        console("Tomato states:")
        console(tomato_state_text)
        console("Held tomato:", held_tomato if held_tomato else "None")
        console("\nGenerated options:")
        highlighted_options = []
        for option_line in mc_gen_full.splitlines():
            option_token = option_line[:1].upper()
            if option_token in preds:
                highlighted_options.append(f"{GREEN}{option_line}{RESET}")
            else:
                highlighted_options.append(option_line)
        console_colored("\n".join(highlighted_options), mc_gen_full)
        console("\nOption scores:")
        for token, logprob, score in zip(top_tokens, top_logprobs, scores):
            console("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
        console("Prediction set:", preds)

        help_needed = len(preds) != 1 or add_mc_prefix in preds
        if add_mc_prefix in preds:
            fallback_in_prediction_count += 1
        if help_needed:
            help_count += 1
            help_candidate_counts.append(len(mc_gen_all))
            help_prediction_set_sizes.append(len(preds))
            while True:
                selected_token = input("Help needed. Choose an option (A/B/C/D/E): ").strip().upper()
                if selected_token in tokens:
                    break
                console("Invalid option. Please enter one of A, B, C, D, or E.")
        else:
            autonomous_count += 1
            selected_token = preds[0]
        if selected_token not in tokens:
            raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")

        selected_action = mc_gen_all[tokens.index(selected_token)]
        action_type, action_arg = parse_tomato_action(selected_action)
        if action_type == "done":
            console("Planner selected a terminal action. Stopping.")
            stop_reason = "planner selected terminal action"
            break
        if action_type is None:
            console(f"Planner: {action_type}")
            console("Planner selected a terminal or non-executable action. Stopping.")
            stop_reason = f"non-executable action selected: {selected_action}"
            break

        if action_type == "navigate":
            if held_tomato is not None:
                console("Robot is holding a tomato; place or discard it before navigating.")
                stop_reason = "invalid navigate while holding tomato"
                break
            failure_roll = random.random()
            failed = failure_roll <= args.navigate_failure_prob
            log_json(f"Step {step} navigate roll:", {
                "target_location": action_arg,
                "failure_roll": failure_roll,
                "failed": failed,
            })
            if failed:
                action_failure_count += 1
                action_history.append(f"navigate to {action_arg} (failed)")
                result_text = f"Navigate failed: stayed at {robot_location}"
            else:
                robot_location = action_arg
                action_history.append(f"navigate to {robot_location}")
                result_text = f"Executed: navigate to {robot_location}"

        elif action_type == "detect":
            if robot_location not in STEMS:
                console("Detect requires the robot to be at a stem.")
                stop_reason = "invalid detect outside stem"
                break
            new_observations = []
            detect_rolls = []
            for tomato in active_tomatoes:
                if hidden_locations[tomato] != robot_location:
                    continue
                previous_property = observed_properties.get(tomato)
                detect_roll = random.random()
                roll_info = {
                    "tomato": tomato,
                    "true_property": hidden_properties[tomato],
                    "true_location": hidden_locations[tomato],
                    "previous_property": previous_property,
                    "detect_roll": detect_roll,
                    "detected": detect_roll <= args.detect_success_prob,
                }
                if detect_roll <= args.detect_success_prob:
                    true_property = hidden_properties[tomato]
                    detected_property = "ripe" if true_property == "rotten" else true_property
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
                stop_reason = f"invalid pick inactive tomato: {tomato}"
                break
            if held_tomato is not None:
                console("Robot is already holding a tomato.")
                stop_reason = "invalid pick while holding tomato"
                break
            if observed_locations.get(tomato) != robot_location:
                console("Tomato is not observed at the current robot location.")
                stop_reason = f"invalid pick tomato not observed here: {tomato}"
                break
            if observed_properties.get(tomato) != "ripe":
                console("Pick requires an observed ripe tomato.")
                stop_reason = f"invalid pick non-ripe or unknown tomato: {tomato}"
                break
            failure_roll = random.random()
            failed = failure_roll <= args.pick_failure_prob
            log_json(f"Step {step} pick roll:", {
                "tomato": tomato,
                "failure_roll": failure_roll,
                "failed": failed,
            })
            if failed:
                action_failure_count += 1
                action_history.append(f"pick {tomato} (failed)")
                result_text = f"Pick failed: {tomato} was not picked"
            else:
                held_tomato = tomato
                action_history.append(f"pick {tomato}")
                result_text = f"Executed: pick {tomato}"

        elif action_type == "scan":
            if held_tomato is None:
                console("Scan requires a held tomato.")
                stop_reason = "invalid scan without held tomato"
                break
            if action_arg is not None and action_arg != held_tomato:
                console("Scan action does not match the held tomato.")
                stop_reason = "invalid scan target mismatch"
                break
            scan_roll = random.random()
            scan_info = {
                "tomato": held_tomato,
                "true_property": hidden_properties[held_tomato],
                "scan_roll": scan_roll,
                "scanned": scan_roll <= args.scan_success_prob,
            }
            if scan_roll <= args.scan_success_prob:
                true_property = hidden_properties[held_tomato]
                label_error_roll = random.random()
                scan_info["label_error_roll"] = label_error_roll
                scan_info["label_error"] = label_error_roll <= args.scan_label_error_prob
                if label_error_roll <= args.scan_label_error_prob:
                    candidates = [label for label in TOMATO_PROPERTIES if label != true_property]
                    scanned_properties[held_tomato] = random.choice(candidates)
                else:
                    scanned_properties[held_tomato] = true_property
                scan_info["scanned_property"] = scanned_properties[held_tomato]
                result_text = f"Scan result: {held_tomato}: {scanned_properties[held_tomato]}"
            else:
                result_text = "Scan result: no property observed"
            log_json(f"Step {step} scan roll:", scan_info)
            action_history.append("scan")

        elif action_type == "place":
            tomato = action_arg
            if held_tomato != tomato:
                console("Place action does not match the held tomato.")
                stop_reason = "invalid place target mismatch"
                break
            if scanned_properties.get(tomato, observed_properties.get(tomato)) != "ripe":
                console("Place is intended for ripe tomatoes; discard non-ripe/rotten tomatoes.")
                stop_reason = f"invalid place non-ripe tomato: {tomato}"
                break
            failure_roll = random.random()
            failed = failure_roll <= args.place_failure_prob
            log_json(f"Step {step} place roll:", {
                "tomato": tomato,
                "failure_roll": failure_roll,
                "failed": failed,
            })
            if failed:
                action_failure_count += 1
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
                stop_reason = "invalid discard target mismatch"
                break
            failure_roll = random.random()
            failed = failure_roll <= args.discard_failure_prob
            log_json(f"Step {step} discard roll:", {
                "tomato": tomato,
                "failure_roll": failure_roll,
                "failed": failed,
            })
            if failed:
                action_failure_count += 1
                action_history.append(f"discard {tomato} (failed)")
                result_text = f"Discard failed: still holding {tomato}"
            else:
                held_tomato = None
                discarded_tomatoes.append(tomato)
                action_history.append(f"discard {tomato}")
                result_text = f"Executed: discard {tomato}"

        source = "user" if help_needed else "prediction set"
        console(f"Selected/Executed ({source}, option {selected_token}): {selected_action} -> {result_text}")
        dead_end_reason = tomato_dead_end_reason(
            hidden_properties,
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
            console(dead_end_reason)
            stop_reason = dead_end_reason
            break
        if tomato_success(hidden_properties, held_tomato, loaded_tomatoes, discarded_tomatoes):
            console("\nAll target tomatoes have been handled successfully.")
            stop_reason = "success"
            break

    else:
        console("\nReached max steps before all tomatoes were handled.")
        stop_reason = "max steps reached"

    console("\n====== Final Plan ======")
    if action_history:
        for i, action in enumerate(action_history, start=1):
            console(f"{i}. {action}")
    else:
        console("No action executed.")
    if held_tomato is not None:
        console("Held tomato:", held_tomato)
    remaining = [t for t in tomatoes if t not in loaded_tomatoes and t not in discarded_tomatoes]
    if remaining:
        console("Unhandled tomatoes:", ", ".join(remaining))
    log_json("Token usage totals:", total_usage)
    total_elapsed = time.perf_counter() - run_start
    console("Total elapsed seconds:", total_elapsed)
    final_success = tomato_success(hidden_properties, held_tomato, loaded_tomatoes, discarded_tomatoes)
    summary = {
        "success": final_success,
        "stop_reason": stop_reason,
        "planning_length": len(action_history),
        "planning_iterations": completed_iterations,
        "question_count": help_count,
        "autonomous_action_count": autonomous_count,
        "fallback_in_prediction_count": fallback_in_prediction_count,
        "average_candidate_count": (sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
        "average_candidate_count_when_asked": (
            sum(help_candidate_counts) / len(help_candidate_counts)
            if help_candidate_counts else 0.0
        ),
        "average_prediction_set_size": (
            sum(prediction_set_sizes) / len(prediction_set_sizes)
            if prediction_set_sizes else 0.0
        ),
        "average_prediction_set_size_when_asked": (
            sum(help_prediction_set_sizes) / len(help_prediction_set_sizes)
            if help_prediction_set_sizes else 0.0
        ),
        "action_failure_count": action_failure_count,
        "loaded_tomatoes": loaded_tomatoes,
        "discarded_tomatoes": discarded_tomatoes,
        "held_tomato": held_tomato,
        "unhandled_tomatoes": remaining,
        "token_usage": total_usage,
        "total_elapsed_seconds": total_elapsed,
    }
    log_json("Summary:", summary)
    console("\n====== Summary ======")
    console("Success:", final_success)
    console("Stop reason:", stop_reason)
    console("Planning length:", len(action_history))
    console("Planning iterations:", completed_iterations)
    console("Question count:", help_count)
    console("Average candidate count when asked:", summary["average_candidate_count_when_asked"])
    console("Average prediction set size when asked:", summary["average_prediction_set_size_when_asked"])
    console("Autonomous action count:", autonomous_count)
    console("Fallback in prediction count:", fallback_in_prediction_count)
    console("Action failure count:", action_failure_count)
    logger.close()


if __name__ == "__main__":
    main()
