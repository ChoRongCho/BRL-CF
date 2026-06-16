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
from log_waste import RunLogger
from scripts.llm import call_llm, configure_openai
from scripts.env import WASTE_MC_PROMPT_FILE
from scripts.prompt import process_mc_raw, temperature_scaling, top_choice_logprobs
from utils import GREEN, RESET, YELLOW, usage_total
from wastesorting_utils import (
    AVAILABLE_BINS,
    WASTE_ATTRIBUTES,
    WASTE_BACKGROUND,
    WASTE_CALIBRATION_TEMPLATE,
    build_waste_calibration_prompt,
    build_waste_generation_prompt,
    build_waste_score_prompt,
    initialize_hidden_attributes,
    parse_waste_action,
)
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-step KnowNo planning for waste sorting.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--settings", default=str(Path(__file__).with_name("llm_setting.json")))
    parser.add_argument("--instruction", default="Discard all waste.")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--scene-objects", default="waste1, waste2, waste3, waste4")
    parser.add_argument("--qhat", type=float, default=0.928)
    parser.add_argument("--score-temperature", "--temperature", dest="score_temperature", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--detect-success-prob", type=float, default=0.8)
    parser.add_argument("--detect-label-error-prob", type=float, default=0.0)
    parser.add_argument("--labels", default="", help='Optional true labels, e.g. "waste1:can,waste2:paper".')
    parser.add_argument("--log-file", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Also print detailed log records to the terminal.")
    parser.add_argument("--calibration-file", default=str(WASTE_MC_PROMPT_FILE))
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--write-calibration-template", default="")
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--num-calibration", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_start = time.perf_counter()
    if args.write_calibration_template:
        write_template(args.write_calibration_template, WASTE_CALIBRATION_TEMPLATE)
        return
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    settings = configure_openai(args.api_key, args.settings)
    qhat = settings.get("qhat", args.qhat)

    # This file intentionally keeps the pipeline explicit:
    # observe current state -> generate next-action candidates -> score candidates
    # -> build prediction set -> execute or ask for help -> update state -> repeat.
    remaining_objects = [obj.strip().lower() for obj in args.scene_objects.split(",") if obj.strip()]
    hidden_attributes = initialize_hidden_attributes(remaining_objects, args.labels)
    observed_attributes = {}
    held_object = None
    action_history = []
    tokens = ["A", "B", "C", "D", "E"]

    logger = RunLogger(__file__, args.log_file, args.verbose, prefix="knowno_multistep_waste")
    console = logger.console
    console_colored = logger.colored
    log = logger.file_only
    log_json = logger.json
    total_usage = {"generation": 0, "scoring": 0, "overall": 0}
    console("====== Multi-step Waste Sorting KnowNo ======")
    log_json("Run metadata:", {
        "argv": sys.argv,
        "model": settings.get("model") or settings.get("model_name"),
        "prompt_version": args.prompt_version,
        "seed": args.seed,
    })
    console("Instruction:", args.instruction)
    console("Prompt version:", args.prompt_version)
    console("Initial objects:", ", ".join(remaining_objects))
    console("Available bins:", ", ".join(AVAILABLE_BINS))
    console("True labels:", ", ".join(f"{obj}: {label}" for obj, label in sorted(hidden_attributes.items())))
    console("Detect success probability:", args.detect_success_prob)
    console("Detect label error probability:", args.detect_label_error_prob)
    qhat = args.qhat
    
    # 1. Run Calibration
    if args.run_calibration:
        qhat = run_knowno_calibration(
            args.calibration_file,
            args.num_calibration,
            args.num_test,
            args.target_success,
            domain_name="waste",
            background=WASTE_BACKGROUND,
            generation_prompt_builder=build_waste_calibration_prompt,
        )
        
    console("qhat:", qhat)
    console("Detailed log file:", logger.path)

    for step in range(1, args.max_steps + 1):
        step_start = time.perf_counter()
        if not remaining_objects and held_object is None:
            console("\nAll objects have been sorted.")
            break

        history_text = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(action_history))
        if not history_text:
            history_text = "None"
        held_text = held_object if held_object is not None else "None"
        observed_text = (
            ", ".join(f"{obj}: {attr}" for obj, attr in sorted(observed_attributes.items()))
            if observed_attributes
            else "None"
        )

        mc_gen_prompt = build_waste_generation_prompt(
            args.instruction,
            remaining_objects,
            observed_text,
            held_text,
            history_text,
            args.prompt_version,
        )

        log_json(f"Step {step} start:", {
            "remaining_objects": remaining_objects,
            "observed_attributes": observed_attributes,
            "held_object": held_object,
            "action_history": action_history,
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

        score_prompt = build_waste_score_prompt(
            args.instruction,
            remaining_objects,
            observed_text,
            held_text,
            history_text,
            mc_gen_full,
            args.prompt_version,
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
        log_json(f"Step {step} decision data:", {
            "options": mc_gen_all,
            "add_mc_prefix": add_mc_prefix,
            "option_logprobs": option_logprobs,
            "scores": scores.tolist(),
            "threshold": 1 - qhat,
            "prediction_set": preds,
        })

        console(f"\n====== Step {step} ======")
        console("Remaining objects:", ", ".join(remaining_objects) if remaining_objects else "None")
        true_text = ", ".join(f"{obj}: {label}" for obj, label in sorted(hidden_attributes.items()))
        console_colored(f"{YELLOW}True waste attributes: {true_text}{RESET}", f"True waste attributes: {true_text}")
        console("Observed waste attributes:", observed_text)
        console("Held object:", held_text)
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

        # KnowNo decision rule: act only when the prediction set is a singleton
        # and not the fallback "option not listed here"; otherwise ask for help.
        if preds == [add_mc_prefix]:
            console("Prediction set only includes 'an option not listed here'. Dead-end reached.")
            break
        help_needed = len(preds) != 1 or add_mc_prefix in preds
        if help_needed:
            while True:
                selected_token = input("Help needed. Choose an option (A/B/C/D/E): ").strip().upper()
                if selected_token in tokens:
                    break
                console("Invalid option. Please enter one of A, B, C, D, or E.")
        else:
            selected_token = preds[0]
        if selected_token not in tokens:
            raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")

        selected_action = mc_gen_all[tokens.index(selected_token)]
        if selected_action == add_mc_prefix:
            console("Selected 'an option not listed here'. Dead-end reached.")
            break

        action_type, action_arg = parse_waste_action(selected_action)
        if action_type == "done":
            console("Planner selected a terminal action. Stopping.")
            break
        if action_type is None:
            console("Planner selected a terminal or non-executable action. Stopping.")
            break

        # Lightweight state update for the high-level planner view. This is not
        # a physics simulation; it tracks observed attributes, held object, and
        # removes an object only after a separate place action.
        if action_type == "detect":
            new_observations = []
            detect_rolls = []
            for obj in remaining_objects:
                if obj in observed_attributes:
                    continue
                detect_roll = random.random()
                roll_info = {
                    "object": obj,
                    "true_label": hidden_attributes[obj],
                    "detect_roll": detect_roll,
                    "detected": detect_roll <= args.detect_success_prob,
                }
                if detect_roll <= args.detect_success_prob:
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
            if new_observations:
                result_text = "Detect result: " + ", ".join(new_observations)
            else:
                result_text = "Detect result: no new attribute observed"

        elif action_type == "pick":
            matched_object = None
            for obj in remaining_objects:
                if action_arg in obj or obj in action_arg:
                    matched_object = obj
                    break
            if matched_object is None:
                console("Selected object is not in the current state. Stopping to avoid compounding error.")
                break

            if held_object is not None:
                console("Robot is already holding an object. Place it before picking another one.")
                break
            held_object = matched_object
            action_history.append(f"pick {matched_object}")
            result_text = f"Executed: pick {matched_object}"

        elif action_type == "place":
            if held_object is None:
                console("Robot is not holding anything. Pick an object before placing.")
                break
            place_object, target_bin = action_arg
            if not (place_object in held_object or held_object in place_object):
                console("Place action does not match the held object. Stopping to avoid compounding error.")
                break
            placed_object = held_object
            held_object = None
            remaining_objects.remove(placed_object)
            observed_attributes.pop(placed_object, None)
            action_history.append(f"place {placed_object} into {target_bin}")
            result_text = f"Executed: place {placed_object} into {target_bin}"

        source = "user" if help_needed else "prediction set"
        console(f"Selected/Executed ({source}, option {selected_token}): {selected_action} -> {result_text}")
        log_json(f"Step {step} end:", {
            "remaining_objects": remaining_objects,
            "observed_attributes": observed_attributes,
            "held_object": held_object,
            "action_history": action_history,
            "step_elapsed_sec": time.perf_counter() - step_start,
        })

    else:
        console("\nReached max steps before all objects were sorted.")

    console("\n====== Final Plan ======")
    if action_history:
        for i, action in enumerate(action_history, start=1):
            console(f"{i}. {action}")
    else:
        console("No action executed.")
    if held_object is not None:
        console("Held object:", held_object)
    if remaining_objects:
        console("Unsorted objects:", ", ".join(remaining_objects))
    log_json("Token usage totals:", total_usage)
    console("Total elapsed seconds:", time.perf_counter() - run_start)
    logger.close()


if __name__ == "__main__":
    main()
