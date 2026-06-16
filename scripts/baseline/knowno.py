from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from scripts.env import load_mobile_dataset
from scripts.llm import call_llm, configure_openai
from scripts.prompt import (
    DEMO_MC_SCORE_BACKGROUND,
    MOBILE_MC_SCORE_BACKGROUND,
    WASTESORTING_MC_GEN_PROMPT,
    build_demo_mc_prompt,
    build_mobile_mc_prompt,
    build_score_prompt,
    build_tabletop_mc_prompt,
    build_tabletop_score_prompt,
    label_mobile_true_options,
    parse_tabletop_action_option,
    prediction_set,
    process_mc_raw,
    temperature_scaling,
    top_choice_logprobs,
)
from scripts.sim import run_tabletop_demo


def _generate_mc_choices(mc_gen_prompt: str):
    last_raw = ""
    format_instruction = (
        "\nAnswer with exactly four options, one per line, formatted as:\n"
        "A) ...\nB) ...\nC) ...\nD) ..."
    )
    for attempt in range(3):
        prompt = mc_gen_prompt if attempt == 0 else mc_gen_prompt + format_instruction
        _, text = call_llm(prompt, stop_seq=["We:"], logit_bias={})
        last_raw = text.strip()
        try:
            return last_raw, *process_mc_raw(last_raw)
        except ValueError:
            continue
    raise ValueError(f"Cannot extract four options from LLM output:\n{last_raw}")


def _prepare_mobile_choices(dataset: List[dict], generate: bool) -> None:
    for data in dataset:
        if generate or "mc_gen_raw" not in data:
            mc_gen_raw, mc_gen_full, mc_gen_all, add_mc_prefix = _generate_mc_choices(data["mc_gen_prompt"])
            data["mc_gen_raw"] = mc_gen_raw
        else:
            mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(data["mc_gen_raw"].strip())
        score_prompt = build_score_prompt(
            data["mc_gen_prompt"],
            mc_gen_full,
            MOBILE_MC_SCORE_BACKGROUND,
            capital=True,
        )
        data["mc_score_prompt"] = score_prompt
        data["mc_gen_full"] = mc_gen_full
        data["mc_gen_all"] = mc_gen_all
        data["add_mc_prefix"] = add_mc_prefix


def _score_mobile_choices(dataset: List[dict]) -> None:
    for data in dataset:
        response, _ = call_llm(data["mc_score_prompt"], max_tokens=1, logprobs=5)
        top_tokens, top_logprobs, top_logprobs_full = top_choice_logprobs(response)
        data["top_logprobs_full"] = top_logprobs_full
        data["top_tokens"] = top_tokens
        data["top_logprobs"] = top_logprobs
        data["true_options"] = label_mobile_true_options(data)


def calibrate_qhat(calibration_set: List[dict], target_success: float) -> float:
    epsilon = 1 - target_success
    scores = []
    for data in calibration_set:
        mc_smx_all = temperature_scaling(data["top_logprobs"], temperature=5)
        true_label_smx = [
            mc_smx_all[token_ind]
            for token_ind, token in enumerate(data["top_tokens"])
            if token in data["true_options"]
        ]
        scores.append(1 - np.max(true_label_smx))
    q_level = np.ceil((len(calibration_set) + 1) * (1 - epsilon)) / len(calibration_set)
    return float(np.quantile(scores, q_level, method="higher"))


def summarize_mobile(test_set: List[dict], qhat: float, target_success: float) -> None:
    num_coverage = 0
    num_help = 0
    set_size_all = []
    for data in test_set:
        preds, _ = prediction_set(data["top_tokens"], data["top_logprobs"], qhat=qhat)
        set_size_all.append(len(preds))
        flag_coverage = not set(preds).isdisjoint(data["true_options"])
        num_coverage += flag_coverage
        num_help += len(preds) != 1 or data["add_mc_prefix"] in preds

    num_test = len(test_set)
    print("============== Summary ==============")
    print("Number of test data:", num_test)
    print("Quantile value qhat:", qhat)
    print("Average prediction set size:", np.mean(set_size_all))
    print("Marginal coverage guarantee:", target_success)
    print("Empirical coverage:", num_coverage / num_test)
    print("Help rate:", num_help / num_test)
    print("Success rate:", num_coverage / num_test)



def run_demo(args) -> None:
    """
    Run the office-kitchen KnowNo demo.

    The demo asks the LLM to generate multiple-choice robot actions,
    scores those options, and reports whether the calibrated prediction
    set is confident enough to act without help.
    """
    # 1. Build the multiple-choice generation prompt.
    qhat = args.qhat
    mc_gen_prompt = build_demo_mc_prompt(args.instruction, args.scene_objects)

    # 2. Generate and clean candidate options.
    _, mc_gen_full, _, _ = _generate_mc_choices(mc_gen_prompt)

    # 3. Build the scoring prompt and score the option letters.
    score_prompt = build_score_prompt(mc_gen_prompt, mc_gen_full, DEMO_MC_SCORE_BACKGROUND)

    response, _ = call_llm(score_prompt, max_tokens=1, logprobs=5)
    top_tokens, top_logprobs, _ = top_choice_logprobs(response)
    preds, scores = prediction_set(top_tokens, top_logprobs, qhat=qhat)

    # 4. Print prompts, scores, and the final help-needed decision.
    print("====== Prompt for generating possible options ======")
    print(mc_gen_prompt)
    print("====== Generated options ======")
    print(mc_gen_full)
    print("====== Prompt for scoring options ======")
    print(score_prompt)
    print("\n====== Raw log probabilities for each option ======")
    for token, logprob in zip(top_tokens, top_logprobs):
        print("Option:", token, "\t", "log prob:", logprob)
    print("Softmax scores:", scores)
    print("Prediction set:", preds)
    print("Help needed!" if len(preds) != 1 else "No help needed!")
    
    
def run_mobile(args) -> None:
    """
    Run the mobile-manipulation KnowNo flow from the notebook.

    This builds the few-shot prompt for multiple-choice generation, asks
    the LLM to generate candidate robot actions, builds the scoring prompt,
    and reports the calibrated prediction set for the current scenario.
    """
    # 1. Optionally run the notebook calibration pipeline.
    qhat = args.qhat
    if args.run_calibration:
        calibration_set, test_set = load_mobile_dataset(args.num_calibration, args.num_test)
        print("Running MC generation...")
        _prepare_mobile_choices(calibration_set, generate=True)
        _prepare_mobile_choices(test_set, generate=True)
        print("Running MC scoring...")
        _score_mobile_choices(calibration_set)
        _score_mobile_choices(test_set)
        qhat = calibrate_qhat(calibration_set, args.target_success)
        summarize_mobile(test_set, qhat, args.target_success)

    # 2. Build the mobile multiple-choice generation prompt.
    mc_gen_prompt = build_mobile_mc_prompt(args.instruction, args.scene_objects)

    # 3. Generate and normalize candidate robot actions.
    _, mc_gen_full, _, add_mc_prefix = _generate_mc_choices(mc_gen_prompt)

    # 4. Build the notebook-style mobile scoring prompt.
    score_prompt = build_score_prompt(mc_gen_prompt, mc_gen_full, MOBILE_MC_SCORE_BACKGROUND, capital=True)

    # 5. Score option letters and construct the prediction set.
    response, _ = call_llm(score_prompt, max_tokens=1, logprobs=5)
    top_tokens, top_logprobs, _ = top_choice_logprobs(response)
    preds, scores = prediction_set(top_tokens, top_logprobs, qhat=qhat)

    # 6. Print the prompts, scores, and help decision.
    print("====== Mobile prompt for generating possible options ======")
    print(mc_gen_prompt)
    print("====== Mobile generated options ======")
    print(mc_gen_full)
    print("====== Mobile prompt for scoring options ======")
    print(score_prompt)
    print("\n====== Raw log probabilities for each option ======")
    for token, logprob, score in zip(top_tokens, top_logprobs, scores):
        print("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
    print("Prediction set:", preds)
    print("Help needed!" if len(preds) != 1 or add_mc_prefix in preds else "No help needed!")


def run_tabletop(args) -> None:
    qhat = args.qhat
    instruction = args.instruction

    mc_gen_prompt = build_tabletop_mc_prompt(instruction)
    print("====== Tabletop prompt for generating possible options ======")
    print(mc_gen_prompt)

    _, mc_gen_full, mc_gen_all, _ = _generate_mc_choices(mc_gen_prompt)
    print("\n====== Tabletop generated options ======")
    print(mc_gen_full)

    score_prompt = build_tabletop_score_prompt(instruction, mc_gen_full)
    print("\n====== Tabletop prompt for scoring options ======")
    print(score_prompt)

    score_response, _ = call_llm(score_prompt, max_tokens=1, logprobs=5)
    top_tokens, top_logprobs, _ = top_choice_logprobs(score_response)
    preds, scores = prediction_set(top_tokens, top_logprobs, qhat=qhat)

    print("\n====== Tabletop option scores ======")
    for token, logprob, score in zip(top_tokens, top_logprobs, scores):
        print("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
    print("Prediction set:", preds)

    tokens = ["A", "B", "C", "D", "E"]
    selected_token = preds[0] if len(preds) == 1 else args.user_option
    if selected_token not in tokens:
        raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")
    action_option = mc_gen_all[tokens.index(selected_token)]
    if len(preds) != 1:
        print("Help needed. Using user/default option:", selected_token)
    else:
        print("No help needed. Using prediction set option:", selected_token)
    print("Selected action option:", action_option)

    pick_obj, target_obj, relation = parse_tabletop_action_option(action_option)
    print("Parsed pick object:", pick_obj)
    print("Parsed target object:", target_obj)
    print("Parsed spatial relation:", relation)

    result = run_tabletop_demo(pick_obj=pick_obj, target_obj=target_obj, relation=relation)
    print()
    print("[KnowNo] Tabletop PyBullet action:", result["action"])
    if "render_paths" in result:
        print("Rendered before image:", result["render_paths"]["before"])
        print("Rendered after image:", result["render_paths"]["after"])
        if "video" in result["render_paths"]:
            print("Rendered rollout video:", result["render_paths"]["video"])



def run_wastesorting(args):
    qhat = args.qhat
    mc_gen_prompt = (
        WASTESORTING_MC_GEN_PROMPT
        .replace("{scene_objects}", args.scene_objects)
        .replace("{task}", args.instruction)
        .strip()
    )

    _, mc_gen_full, mc_gen_all, add_mc_prefix = _generate_mc_choices(mc_gen_prompt)

    score_background = (
        "You are a robot operating in a waste sorting station. You are in front "
        "of a counter. There are four bins: a general bin, a plastic bin, a "
        "paper bin, and a can bin."
    )
    score_prompt = build_score_prompt(mc_gen_prompt, mc_gen_full, score_background, capital=True)

    response, _ = call_llm(score_prompt, max_tokens=1, logprobs=5)
    top_tokens, top_logprobs, _ = top_choice_logprobs(response)
    preds, scores = prediction_set(top_tokens, top_logprobs, qhat=qhat)

    print("====== Waste sorting prompt for generating possible options ======")
    print(mc_gen_prompt)
    print("====== Waste sorting generated options ======")
    print(mc_gen_full)
    print("====== Waste sorting prompt for scoring options ======")
    print(score_prompt)
    print("\n====== Raw log probabilities for each option ======")
    for token, logprob, score in zip(top_tokens, top_logprobs, scores):
        print("Option:", token, "\tlog prob:", logprob, "\tsoftmax:", score)
    print("Prediction set:", preds)
    print("Help needed!" if len(preds) != 1 or add_mc_prefix in preds else "No help needed!")

    tokens = ["A", "B", "C", "D", "E"]
    selected_token = preds[0] if len(preds) == 1 else args.user_option
    if selected_token not in tokens:
        raise ValueError(f"Selected option must be one of {tokens}, got {selected_token!r}")
    print("Selected waste sorting option:", mc_gen_all[tokens.index(selected_token)])


def run_bimani(args):
    print("Code is not done yet")
    return 


def parse_args():
    parser = argparse.ArgumentParser(description="Run KnowNo baselines without opening notebooks.")
    parser.add_argument("--mode", choices=["demo", "mobile", "tabletop", "wastesorting", "bimani"], default="mobile")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--settings", default=str(Path(__file__).with_name("llm_setting.json")))
    parser.add_argument("--instruction", default="Put the bottled water in the bin.")
    parser.add_argument("--scene-objects", default="energy bar, bottled water, rice chips")
    parser.add_argument("--qhat", type=float, default=0.928)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--num-calibration", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=100)
    parser.add_argument("--user-option", default="D")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Instruction:", args.instruction)
    print("Scene objects:", args.scene_objects)
    
    settings = configure_openai(args.api_key, args.settings)
    args.qhat = settings.get("qhat", args.qhat)
    args.target_success = settings.get("target_success", args.target_success)
    if args.mode == "tabletop" and args.instruction == "Put the bottled water in the bin.":
        args.instruction = settings.get("tabletop_instruction", "put the yellow block next to the green bowl.")
    if args.mode == "demo":
        run_demo(args)
        
    elif args.mode == "mobile":
        run_mobile(args)
        
    elif args.mode == "tabletop":
        run_tabletop(args)
        
    elif args.mode == "bimani":
        run_bimani(args)
    
    elif args.mode == "wastesorting":
        run_wastesorting(args)
    
    else:
        raise ValueError(f"Mode {args.mode} is worng. Select: demo, mobile, tabletop, bimani")
        


if __name__ == "__main__":
    main()
