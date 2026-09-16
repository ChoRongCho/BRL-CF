from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from scripts.llm import call_llm, configure_openai
from scripts.prompt import prediction_set, process_mc_raw, top_choice_logprobs


WASTE_BACKGROUND = (
    "You are a robot operating in a waste sorting station. You are in front "
    "of a counter. There are four bins: a general bin, a plastic bin, a "
    "paper bin, and a can bin."
)
WASTE_ATTRIBUTES = ["general", "plastic", "paper", "can"]
AVAILABLE_BINS = [f"{attribute} bin" for attribute in WASTE_ATTRIBUTES]


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-step KnowNo planning for waste sorting.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--settings", default=str(Path(__file__).with_name("llm_setting.json")))
    parser.add_argument("--instruction", default="Discard all waste.")
    parser.add_argument("--scene-objects", default="waste1, waste2, waste3, waste4")
    parser.add_argument("--qhat", type=float, default=0.928)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--detect-success-prob", type=float, default=0.8)
    return parser.parse_args()


def generate_choices(prompt: str):
    """Ask the LLM for four action options and retry if the format is unusable."""
    last_raw = ""
    format_hint = (
        "\nAnswer with exactly four options, one per line, formatted as:\n"
        "A) ...\nB) ...\nC) ...\nD) ..."
    )
    for attempt in range(3):
        cur_prompt = prompt if attempt == 0 else prompt + format_hint
        _, text = call_llm(cur_prompt, stop_seq=["We:"], logit_bias={})
        last_raw = text.strip()
        try:
            return last_raw, *process_mc_raw(last_raw)
        except ValueError:
            pass
    raise ValueError(f"Cannot extract four options from LLM output:\n{last_raw}")


def parse_waste_action(action: str):
    """Parse one atomic action: detect, pick <object>, or place <object> into <bin>."""
    action = action.lower().strip()
    if "done" in action or "no more" in action or "nothing" in action:
        return "done", None

    if re.fullmatch(r"detect", action):
        return "detect", None

    pick_match = re.fullmatch(r"pick (.+)", action)
    if pick_match:
        return "pick", pick_match.group(1).strip()

    place_match = re.fullmatch(r"place (.+) into (general|plastic|paper|can) bin", action)
    if place_match:
        return "place", (place_match.group(1).strip(), f"{place_match.group(2)} bin")

    return None, None


def main() -> None:
    args = parse_args()
    settings = configure_openai(args.api_key, args.settings)
    qhat = settings.get("qhat", args.qhat)

    # This file intentionally keeps the pipeline explicit:
    # observe current state -> generate next-action candidates -> score candidates
    # -> build prediction set -> execute or ask for help -> update state -> repeat.
    remaining_objects = [obj.strip().lower() for obj in args.scene_objects.split(",") if obj.strip()]
    hidden_attributes = {obj: random.choice(WASTE_ATTRIBUTES) for obj in remaining_objects}
    observed_attributes = {}
    held_object = None
    action_history = []
    tokens = ["A", "B", "C", "D", "E"]

    print("====== Multi-step Waste Sorting KnowNo ======")
    print("Instruction:", args.instruction)
    print("Initial objects:", ", ".join(remaining_objects))
    print("Available bins:", ", ".join(AVAILABLE_BINS))
    print("qhat:", qhat)

    for step in range(1, args.max_steps + 1):
        if not remaining_objects and held_object is None:
            print("\nAll objects have been sorted.")
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

        # Generation prompt: unlike the single-step notebook, this prompt carries
        # the overall instruction, remaining objects, and previous actions.
        # Candidate actions must be atomic: exactly one detect, pick, or place.
        mc_gen_prompt = f"""
We: {WASTE_BACKGROUND}

We: Overall instruction: {args.instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(AVAILABLE_BINS)}
We: Observed waste attributes: {observed_text}
We: Object currently held by the robot: {held_text}
We: Actions already completed:
{history_text}
We: Choose exactly one next atomic action. Valid action forms are only: detect, pick <object>, place <object> into <bin>. Do not combine pick and place in one option.
You:
A) detect
B) pick waste1
C) place waste1 into paper bin
D) place waste2 into can bin

We: Overall instruction: {args.instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(AVAILABLE_BINS)}
We: Observed waste attributes: {observed_text}
We: Object currently held by the robot: {held_text}
We: Actions already completed:
{history_text}
We: What should the robot do next? Answer with options that are only detect, pick <object>, or place <object> into <bin>.
You:
""".strip()

        _, mc_gen_full, mc_gen_all, add_mc_prefix = generate_choices(mc_gen_prompt)

        # Scoring prompt: keep the exact same multi-step context, then ask for a
        # single option token so KnowNo can read next-token log probabilities.
        score_prompt = f"""
{WASTE_BACKGROUND}

Overall instruction: {args.instruction}
Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
Available bins: {", ".join(AVAILABLE_BINS)}
Observed waste attributes: {observed_text}
Object currently held by the robot: {held_text}
Actions already completed:
{history_text}

These are candidate next actions:
{mc_gen_full}
We: Which next action is correct? Answer with a single capital letter.
You:
""".strip()

        response, _ = call_llm(score_prompt, max_tokens=1, logprobs=5)
        top_tokens, top_logprobs, _ = top_choice_logprobs(response)
        preds, scores = prediction_set(top_tokens, top_logprobs, qhat=qhat)

        print(f"\n====== Step {step} ======")
        print("Remaining objects:", ", ".join(remaining_objects) if remaining_objects else "None")
        print("Available bins:", ", ".join(AVAILABLE_BINS))
        print("Observed waste attributes:", observed_text)
        print("Held object:", held_text)
        print("\nGenerated options:")
        print(mc_gen_full)
        print("\nOption scores:")
        for token, logprob, score in zip(top_tokens, top_logprobs, scores):
            print("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
        print("Prediction set:", preds)

        # KnowNo decision rule: act only when the prediction set is a singleton
        # and not the fallback "option not listed here"; otherwise ask for help.
        help_needed = len(preds) != 1 or add_mc_prefix in preds
        if help_needed:
            while True:
                selected_token = input("Help needed. Choose an option (A/B/C/D/E): ").strip().upper()
                if selected_token in tokens:
                    break
                print("Invalid option. Please enter one of A, B, C, D, or E.")
        else:
            selected_token = preds[0]
        if selected_token not in tokens:
            raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")

        selected_action = mc_gen_all[tokens.index(selected_token)]
        if help_needed:
            print("Using user option:", selected_token)
        else:
            print("No help needed. Using prediction set option:", selected_token)
        print("Selected action:", selected_action)

        action_type, action_arg = parse_waste_action(selected_action)
        if action_type == "done":
            print("Planner selected a terminal action. Stopping.")
            break
        if action_type is None:
            print("Planner selected a terminal or non-executable action. Stopping.")
            break

        # Lightweight state update for the high-level planner view. This is not
        # a physics simulation; it tracks observed attributes, held object, and
        # removes an object only after a separate place action.
        if action_type == "detect":
            new_observations = []
            for obj in remaining_objects:
                if obj in observed_attributes:
                    continue
                if random.random() <= args.detect_success_prob:
                    observed_attributes[obj] = hidden_attributes[obj]
                    new_observations.append(f"{obj}: {observed_attributes[obj]}")
            action_history.append("detect")
            if new_observations:
                print("Detect result:", ", ".join(new_observations))
            else:
                print("Detect result: no new attribute observed")

        elif action_type == "pick":
            matched_object = None
            for obj in remaining_objects:
                if action_arg in obj or obj in action_arg:
                    matched_object = obj
                    break
            if matched_object is None:
                print("Selected object is not in the current state. Stopping to avoid compounding error.")
                break

            if held_object is not None:
                print("Robot is already holding an object. Place it before picking another one.")
                break
            held_object = matched_object
            action_history.append(f"pick {matched_object}")

        elif action_type == "place":
            if held_object is None:
                print("Robot is not holding anything. Pick an object before placing.")
                break
            place_object, target_bin = action_arg
            if not (place_object in held_object or held_object in place_object):
                print("Place action does not match the held object. Stopping to avoid compounding error.")
                break
            placed_object = held_object
            held_object = None
            remaining_objects.remove(placed_object)
            observed_attributes.pop(placed_object, None)
            action_history.append(f"place {placed_object} into {target_bin}")

        print("Executed:", action_history[-1])

    else:
        print("\nReached max steps before all objects were sorted.")

    print("\n====== Final Plan ======")
    if action_history:
        for i, action in enumerate(action_history, start=1):
            print(f"{i}. {action}")
    else:
        print("No action executed.")
    if held_object is not None:
        print("Held object:", held_object)
    if remaining_objects:
        print("Unsorted objects:", ", ".join(remaining_objects))


if __name__ == "__main__":
    main()
