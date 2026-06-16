from __future__ import annotations

from scripts.structured_prompts import (
    TOMATO_ACTION_ROLES,
    TOMATO_BACKGROUND,
    WASTE_ACTION_ROLES,
    WASTE_BACKGROUND,
)


def _items_text(items: list[str], none_text: str = "none") -> str:
    return ", ".join(items) if items else none_text


def _history_sentence(history_text: str) -> str:
    if history_text.strip() == "None":
        return "The robot has not completed any actions yet."
    compact = "; ".join(line.strip() for line in history_text.splitlines() if line.strip())
    return f"So far, the robot has completed these actions: {compact}."


def _tomato_state_sentences(tomato_state_text: str) -> str:
    sentences = []
    for raw_line in tomato_state_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        tomato, state_text = [part.strip() for part in line.split(":", 1)]
        parts = [part.strip() for part in state_text.split(",")]
        status = parts[0] if parts else "unknown"
        observed = "unknown"
        scanned = "unknown"
        for part in parts[1:]:
            if part.startswith("observed "):
                observed = part.removeprefix("observed ").strip()
            elif part.startswith("scanned "):
                scanned = part.removeprefix("scanned ").strip()
        sentences.append(
            f"{tomato} is currently {status}; its observed ripeness is {observed}, "
            f"and its scan result is {scanned}."
        )
    return " ".join(sentences) if sentences else "No tomato state details are available."


def _tomato_scene_text(
    instruction: str,
    robot_location: str,
    active_tomatoes: list[str],
    tomato_state_text: str,
    held_tomato: str | None,
    loaded_tomatoes: list[str],
    discarded_tomatoes: list[str],
    history_text: str,
) -> str:
    held_text = f"The robot is holding {held_tomato}." if held_tomato else "The robot is not holding a tomato."
    return f"""
We: The task is: {instruction}
We: The robot is at {robot_location}. {held_text}
We: The tomatoes that still need attention are {_items_text(active_tomatoes)}.
We: {_tomato_state_sentences(tomato_state_text)}
We: The loaded tomatoes are {_items_text(loaded_tomatoes)}. The discarded tomatoes are {_items_text(discarded_tomatoes)}.
We: {_history_sentence(history_text)}
""".strip()


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
We: {_waste_observation_sentence(observed_text)} {held_sentence}
We: {_history_sentence(history_text)}
""".strip()


TOMATO_GENERATION_FEW_SHOT = """
We: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.
We: The robot is at dock_station. The robot is not holding a tomato.
We: The tomatoes that still need attention are tomato1, tomato2.
We: tomato1 is currently unknown; its observed ripeness is unknown, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.
We: The loaded tomatoes are none. The discarded tomatoes are none.
We: The robot has not completed any actions yet.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) navigate to stem_01
B) navigate to stem_02
C) detect stem_01
D) pick tomato1

We: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.
We: The robot is at stem_01. The robot is not holding a tomato.
We: The tomatoes that still need attention are tomato1, tomato2.
We: tomato1 is currently detected; its observed ripeness is ripe, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.
We: The loaded tomatoes are none. The discarded tomatoes are none.
We: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) pick tomato1
B) detect stem_01
C) navigate to stem_02
D) scan

We: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.
We: The robot is at stem_01. The robot is holding tomato1.
We: The tomatoes that still need attention are tomato1, tomato2.
We: tomato1 is currently held; its observed ripeness is ripe, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.
We: The loaded tomatoes are none. The discarded tomatoes are none.
We: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01; 3. pick tomato1.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) scan
B) place tomato1
C) discard tomato1
D) pick tomato2
""".strip()


WASTE_GENERATION_FEW_SHOT = """
We: The task is: Discard all waste.
We: The waste objects still on the counter are waste1, waste2, waste3, waste4.
We: The available bins are general bin, plastic bin, paper bin, can bin.
We: The robot has not observed the waste types yet. The robot is not holding any waste object.
We: The robot has not completed any actions yet.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) detect
B) pick waste1
C) place waste1 into can bin
D) pick waste2

We: The task is: Discard all waste.
We: The waste objects still on the counter are waste1, waste2, waste3, waste4.
We: The available bins are general bin, plastic bin, paper bin, can bin.
We: The robot has observed these waste types: waste1: paper, waste2: general. The robot is not holding any waste object.
We: So far, the robot has completed these actions: 1. detect.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) pick waste1
B) pick waste2
C) detect
D) place waste1 into paper bin

We: The task is: Discard all waste.
We: The waste objects still on the counter are waste2, waste3, waste4.
We: The available bins are general bin, plastic bin, paper bin, can bin.
We: The robot has observed these waste types: waste2: general, waste3: plastic, waste4: can. The robot is holding waste2.
We: So far, the robot has completed these actions: 1. detect; 2. pick waste2.
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
A) place waste2 into general bin
B) place waste2 into can bin
C) pick waste3
D) detect
""".strip()


def build_tomato_calibration_prompt_text(context: str) -> str:
    return f"""
We: {TOMATO_BACKGROUND}

{TOMATO_ACTION_ROLES}

{context}
You:
""".strip()


def build_tomato_generation_prompt_text(
    instruction: str,
    robot_location: str,
    active_tomatoes: list[str],
    tomato_state_text: str,
    held_tomato: str | None,
    loaded_tomatoes: list[str],
    discarded_tomatoes: list[str],
    history_text: str,
    required_next_action_text: str,
) -> str:
    return f"""
We: {TOMATO_BACKGROUND}

{TOMATO_GENERATION_FEW_SHOT}

{_tomato_scene_text(instruction, robot_location, active_tomatoes, tomato_state_text, held_tomato, loaded_tomatoes, discarded_tomatoes, history_text)}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def build_tomato_score_prompt_text(
    instruction: str,
    robot_location: str,
    active_tomatoes: list[str],
    tomato_state_text: str,
    held_tomato: str | None,
    loaded_tomatoes: list[str],
    discarded_tomatoes: list[str],
    history_text: str,
    mc_gen_full: str,
    required_next_action_text: str,
) -> str:
    return f"""
{TOMATO_BACKGROUND}

{_tomato_scene_text(instruction, robot_location, active_tomatoes, tomato_state_text, held_tomato, loaded_tomatoes, discarded_tomatoes, history_text)}
We: What should the robot do next?
You:
{mc_gen_full}
We: Which option is correct? Answer with a single capital letter.
You:
""".strip()


def build_waste_calibration_prompt_text(context: str) -> str:
    return f"""
We: {WASTE_BACKGROUND}

{WASTE_ACTION_ROLES}

{context}
You:
""".strip()


def build_waste_generation_prompt_text(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    available_bins: list[str],
) -> str:
    return f"""
We: {WASTE_BACKGROUND}

{WASTE_GENERATION_FEW_SHOT}

{_waste_scene_text(instruction, remaining_objects, observed_text, held_text, history_text, available_bins)}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def build_waste_score_prompt_text(
    instruction: str,
    remaining_objects: list[str],
    observed_text: str,
    held_text: str,
    history_text: str,
    mc_gen_full: str,
    available_bins: list[str],
) -> str:
    return f"""
{WASTE_BACKGROUND}

{_waste_scene_text(instruction, remaining_objects, observed_text, held_text, history_text, available_bins)}
We: What should the robot do next?
You:
{mc_gen_full}
We: Which option is correct? Answer with a single capital letter.
You:
""".strip()
