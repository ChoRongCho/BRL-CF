from __future__ import annotations

from domains.wastesorting.calibration_prompts import WASTE_ACTION_ROLES, WASTE_BACKGROUND

WASTE_ACTION_OUTPUT_RULES = 'Output rules:\n- Each option must be exactly one allowed action from the action roles.\n- Use only these action formats: detect, pick <object>, place <object> into <bin>.\n- Do not generate descriptive phrases, retries, requests for assistance, system checks, or any action outside these formats.'

WASTE_GENERATION_FEW_SHOT_V1 = 'We: Example state:\nObjects still on the counter: waste1, waste2, waste3, waste4\nObserved waste attributes: None\nObject currently held by the robot: None\nYou:\nA) detect\nB) pick waste1\nC) place waste1 into can bin\nD) pick waste2\n\nWe: Example state:\nObjects still on the counter: waste1, waste2, waste3, waste4\nObserved waste attributes: waste1: paper, waste2: general\nObject currently held by the robot: None\nYou:\nA) pick waste1\nB) pick waste2\nC) detect\nD) place waste1 into paper bin\n\nWe: Example state:\nObjects still on the counter: waste2, waste3, waste4\nObserved waste attributes: waste2: general, waste3: plastic, waste4: can\nObject currently held by the robot: waste2\nYou:\nA) place waste2 into general bin\nB) place waste2 into can bin\nC) pick waste3\nD) detect'
WASTE_GENERATION_FEW_SHOT_V2 = 'We: The task is: Discard all waste.\nWe: The waste objects still on the counter are waste1, waste2, waste3, waste4.\nWe: The available bins are general bin, plastic bin, paper bin, can bin.\nWe: The robot has not observed the waste types yet. The robot is not holding any waste object.\nWe: The robot has not completed any actions yet.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) detect\nB) pick waste1\nC) place waste1 into can bin\nD) pick waste2\n\nWe: The task is: Discard all waste.\nWe: The waste objects still on the counter are waste1, waste2, waste3, waste4.\nWe: The available bins are general bin, plastic bin, paper bin, can bin.\nWe: The robot has observed these waste types: waste1: paper, waste2: general. The robot is not holding any waste object.\nWe: So far, the robot has completed these actions: 1. detect.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) pick waste1\nB) pick waste2\nC) detect\nD) place waste1 into paper bin\n\nWe: The task is: Discard all waste.\nWe: The waste objects still on the counter are waste2, waste3, waste4.\nWe: The available bins are general bin, plastic bin, paper bin, can bin.\nWe: The robot has observed these waste types: waste2: general, waste3: plastic, waste4: can. The robot is holding waste2.\nWe: So far, the robot has completed these actions: 1. detect; 2. pick waste2.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) place waste2 into general bin\nB) place waste2 into can bin\nC) pick waste3\nD) detect'

def _build_waste_generation_prompt_text_v1(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    available_bins: list[str],
    occlusion_text: str = "None",
) -> str:
    return f"""
We: {WASTE_BACKGROUND}

{WASTE_ACTION_ROLES}

{WASTE_ACTION_OUTPUT_RULES}

{WASTE_GENERATION_FEW_SHOT_V1}

We: Overall instruction: {instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(available_bins)}
We: Occluded waste objects: {occlusion_text}
We: Observed waste attributes: {observed_text}
We: Object currently held by the robot: {held_text}
We: Actions already completed:
{history_text}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def _build_waste_score_prompt_text_v1(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    mc_gen_full: str,
    available_bins: list[str],
    occlusion_text: str = "None",
) -> str:
    return f"""
{WASTE_BACKGROUND}

We: Overall instruction: {instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(available_bins)}
We: Occluded waste objects: {occlusion_text}
We: Observed waste attributes: {observed_text}
We: Object currently held by the robot: {held_text}
We: Actions already completed:
{history_text}
We: What should the robot do next?
You:
{mc_gen_full}
We: Which option is correct? Answer with a single capital letter.
You:
""".strip()


def _items_text(items: list[str], none_text: str = "none") -> str:
    return ", ".join(items) if items else none_text


def _history_sentence(history_text: str) -> str:
    if history_text.strip() == "None":
        return "The robot has not completed any actions yet."
    compact = "; ".join(line.strip() for line in history_text.splitlines() if line.strip())
    return f"So far, the robot has completed these actions: {compact}."


def _waste_observation_sentence(observed_text: str) -> str:
    if observed_text.strip() == "None":
        return "The robot has not observed the waste types yet."
    return f"The robot has observed these waste types: {observed_text}."


def _waste_scene_text(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    available_bins: list[str],
    occlusion_text: str = "None",
) -> str:
    held_sentence = (
        "The robot is not holding any waste object."
        if held_text.strip() == "None"
        else f"The robot is holding {held_text}."
    )
    return f"""
We: The task is: {instruction}
We: The waste objects still on the counter are {_items_text(remaining_objects)}.
We: The available bins are {_items_text(available_bins)}.
We: The occluded waste objects are {occlusion_text}.
We: {_waste_observation_sentence(observed_text)} {held_sentence}
We: {_history_sentence(history_text)}
""".strip()


def _build_waste_generation_prompt_text_v2(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    available_bins: list[str],
    occlusion_text: str = "None",
) -> str:
    return f"""
We: {WASTE_BACKGROUND}

{WASTE_ACTION_ROLES}

{WASTE_ACTION_OUTPUT_RULES}

{WASTE_GENERATION_FEW_SHOT_V2}

{_waste_scene_text(instruction, remaining_objects, observed_text, held_text, history_text, available_bins, occlusion_text)}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def _build_waste_score_prompt_text_v2(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    mc_gen_full: str,
    available_bins: list[str],
    occlusion_text: str = "None",
) -> str:
    return f"""
{WASTE_BACKGROUND}

{_waste_scene_text(instruction, remaining_objects, observed_text, held_text, history_text, available_bins, occlusion_text)}
We: What should the robot do next?
You:
{mc_gen_full}
We: Which option is correct? Answer with a single capital letter.
You:
""".strip()



def _is_v2(prompt_version: str) -> bool:
    return (prompt_version or "v1").lower() in {"v2", "natural", "natural_language"}


def build_waste_generation_prompt_text(*args, prompt_version: str = "v1", **kwargs) -> str:
    builder = _build_waste_generation_prompt_text_v2 if _is_v2(prompt_version) else _build_waste_generation_prompt_text_v1
    return builder(*args, **kwargs)


def build_waste_score_prompt_text(*args, prompt_version: str = "v1", **kwargs) -> str:
    builder = _build_waste_score_prompt_text_v2 if _is_v2(prompt_version) else _build_waste_score_prompt_text_v1
    return builder(*args, **kwargs)
