"""KnowNo 실험에서 공통으로 사용하는 로그 생성과 출력."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from scripts.utils.utils import GREEN, RESET, YELLOW


class RunLogger:
    def __init__(self, script_path: str, log_file: str = "", verbose: bool = False, prefix: str = "knowno_multistep"):
        log_dir = Path(script_path).resolve().parent / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_file) if log_file else log_dir / f"{prefix}_{timestamp}.txt"
        self.file = self.path.open("w", encoding="utf-8")
        self.verbose = verbose

    def file_only(self, *values):
        text = " ".join(str(value) for value in values)
        self.file.write(text + "\n")
        self.file.flush()
        if self.verbose:
            print(text)

    def console(self, *values):
        text = " ".join(str(value) for value in values)
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def colored(self, text: str, plain_text: str):
        print(text)
        self.file.write(plain_text + "\n")
        self.file.flush()

    def json(self, title: str, data):
        self.file_only(title)
        self.file_only(json.dumps(data, indent=2, sort_keys=True, default=str))

    def close(self):
        self.file.close()


def _start_run(script_path, args, baseline_name, settings, prefix, title):
    logger = RunLogger(script_path, args.log_file, args.verbose, prefix=prefix)
    logger.console(f"====== {title} {baseline_name} ======")
    logger.json("Run metadata:", {
        "argv": sys.argv,
        "baseline": baseline_name,
        "model": settings.get("model") or settings.get("model_name"),
        "prompt_version": args.prompt_version,
        "seed": args.seed,
        "expert": "exact_oracle" if args.auto_answer else "human_input",
        "env_setting": args.env_setting,
    })
    return logger


def start_tomato_run(
    script_path,
    args,
    baseline_name,
    settings,
    tomatoes,
    locations,
    hidden_ripeness,
    hidden_freshness,
    hidden_locations,
):
    logger = _start_run(
        script_path,
        args,
        baseline_name,
        settings,
        prefix="knowno_multistep_tomato",
        title="Multi-step Tomato",
    )
    console = logger.console
    console("Instruction:", args.instruction)
    console("Prompt version:", args.prompt_version)
    console("Tomatoes:", ", ".join(tomatoes))
    console("Locations:", ", ".join(locations))
    console("True ripeness:", ", ".join(f"{obj}: {hidden_ripeness[obj]}" for obj in sorted(hidden_ripeness)))
    console("True freshness:", ", ".join(f"{obj}: {hidden_freshness[obj]}" for obj in sorted(hidden_freshness)))
    console("True locations:", ", ".join(f"{obj}: {hidden_locations[obj]}" for obj in sorted(hidden_locations)))
    console("Detect success/error:", args.detect_success_prob, "/", args.detect_label_error_prob)
    console("Scan success/error:", args.scan_success_prob, "/", args.scan_label_error_prob)
    console(
        "Action failure probabilities:",
        f"navigate={args.navigate_failure_prob}, pick={args.pick_failure_prob}, "
        f"place={args.place_failure_prob}, discard={args.discard_failure_prob}",
    )
    return logger


def start_waste_run(script_path, args, baseline_name, settings, remaining_objects, available_bins, hidden_attributes):
    logger = _start_run(
        script_path,
        args,
        baseline_name,
        settings,
        prefix="knowno_multistep_waste",
        title="Multi-step Waste Sorting",
    )
    console = logger.console
    console("Instruction:", args.instruction)
    console("Prompt version:", args.prompt_version)
    console("Initial objects:", ", ".join(remaining_objects))
    console("Available bins:", ", ".join(available_bins))
    console("True labels:", ", ".join(f"{obj}: {label}" for obj, label in sorted(hidden_attributes.items())))
    console("Detect success probability:", args.detect_success_prob)
    console("Detect label error probability:", args.detect_label_error_prob)
    console(
        "Action failure probabilities:",
        f"pick={args.pick_failure_prob}, place={args.place_failure_prob}",
    )
    return logger


def _show_options(logger, options_text, prediction_set, tokens, logprobs, scores):
    highlighted_options = []
    for option_line in options_text.splitlines():
        option_token = option_line[:1].upper()
        highlighted_options.append(
            f"{GREEN}{option_line}{RESET}" if option_token in prediction_set else option_line
        )
    logger.console("\nGenerated options:")
    logger.colored("\n".join(highlighted_options), options_text)
    logger.console("\nOption scores:")
    for token, logprob, score in zip(tokens, logprobs, scores):
        logger.console("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
    logger.console("Prediction set:", prediction_set)


def show_tomato_decision(
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
    tokens,
    logprobs,
    scores,
):
    console = logger.console
    console(f"\n====== Step {step} ======")
    console("Robot location:", robot_location)
    console("Active tomatoes:", ", ".join(active_tomatoes) if active_tomatoes else "None")
    true_ripeness = ", ".join(f"{tomato}: {hidden_ripeness[tomato]}" for tomato in sorted(hidden_ripeness))
    true_freshness = ", ".join(f"{tomato}: {hidden_freshness[tomato]}" for tomato in sorted(hidden_freshness))
    true_locations = ", ".join(f"{tomato}: {hidden_locations[tomato]}" for tomato in sorted(hidden_locations))
    logger.colored(f"{YELLOW}True tomato ripeness: {true_ripeness}{RESET}", f"True tomato ripeness: {true_ripeness}")
    logger.colored(f"{YELLOW}True tomato freshness: {true_freshness}{RESET}", f"True tomato freshness: {true_freshness}")
    logger.colored(f"{YELLOW}True tomato locations: {true_locations}{RESET}", f"True tomato locations: {true_locations}")
    console("Tomato states:")
    console(tomato_state_text)
    console("Held tomato:", held_tomato if held_tomato else "None")
    _show_options(logger, options_text, prediction_set, tokens, logprobs, scores)


def show_waste_decision(
    logger,
    step,
    remaining_objects,
    hidden_attributes,
    observed_text,
    occlusion_description,
    held_text,
    options_text,
    prediction_set,
    tokens,
    logprobs,
    scores,
):
    console = logger.console
    console(f"\n====== Step {step} ======")
    console("Remaining objects:", ", ".join(remaining_objects) if remaining_objects else "None")
    true_attributes = ", ".join(f"{obj}: {label}" for obj, label in sorted(hidden_attributes.items()))
    logger.colored(
        f"{YELLOW}True waste attributes: {true_attributes}{RESET}",
        f"True waste attributes: {true_attributes}",
    )
    console("Observed waste attributes:", observed_text)
    console("Occluded waste objects:", occlusion_description)
    console("Held object:", held_text)
    _show_options(logger, options_text, prediction_set, tokens, logprobs, scores)


def build_run_summary(
    *,
    success,
    stop_reason,
    action_history,
    planning_iterations,
    question_count,
    autonomous_action_count,
    fallback_in_prediction_count,
    noopt_recovery_count,
    candidate_counts,
    help_candidate_counts,
    prediction_set_sizes,
    help_prediction_set_sizes,
    action_failure_count,
    token_usage,
    total_elapsed_seconds,
    **domain_state,
):
    def average(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "success": success,
        "stop_reason": stop_reason,
        "planning_length": len(action_history),
        "planning_iterations": planning_iterations,
        "question_count": question_count,
        "autonomous_action_count": autonomous_action_count,
        "fallback_in_prediction_count": fallback_in_prediction_count,
        "noopt_recovery_count": noopt_recovery_count,
        "average_candidate_count": average(candidate_counts),
        "average_candidate_count_when_asked": average(help_candidate_counts),
        "average_prediction_set_size": average(prediction_set_sizes),
        "average_prediction_set_size_when_asked": average(help_prediction_set_sizes),
        "action_failure_count": action_failure_count,
        **domain_state,
        "token_usage": token_usage,
        "total_elapsed_seconds": total_elapsed_seconds,
    }


def finish_run(
    logger,
    action_history,
    total_usage,
    total_elapsed,
    summary,
    held_label,
    held_value,
    remaining_label,
    remaining_values,
):
    console = logger.console
    console("\n====== Final Plan ======")
    if action_history:
        for index, action in enumerate(action_history, start=1):
            console(f"{index}. {action}")
    else:
        console("No action executed.")
    if held_value is not None:
        console(held_label, held_value)
    if remaining_values:
        console(remaining_label, ", ".join(remaining_values))

    logger.json("Token usage totals:", total_usage)
    console("Total elapsed seconds:", total_elapsed)
    logger.json("Summary:", summary)
    console("\n====== Summary ======")
    console("Success:", summary["success"])
    console("Stop reason:", summary["stop_reason"])
    console("Planning length:", summary["planning_length"])
    console("Planning iterations:", summary["planning_iterations"])
    console("Question count:", summary["question_count"])
    console("Average candidate count when asked:", summary["average_candidate_count_when_asked"])
    console("Average prediction set size when asked:", summary["average_prediction_set_size_when_asked"])
    console("Autonomous action count:", summary["autonomous_action_count"])
    console("Fallback in prediction count:", summary["fallback_in_prediction_count"])
    console("NoOpt recovery count:", summary["noopt_recovery_count"])
    console("Action failure count:", summary["action_failure_count"])
    console("Terminal failure category:", summary.get("terminal_failure_category"))
    console("Failure mode counts:", summary.get("failure_mode_counts", {}))
    logger.close()
