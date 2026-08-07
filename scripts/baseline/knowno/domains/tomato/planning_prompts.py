from __future__ import annotations

from domains.tomato.calibration_prompts import TOMATO_ACTION_ROLES, TOMATO_BACKGROUND

TOMATO_ACTION_OUTPUT_RULES = 'Output rules:\n- Each option must be exactly one allowed action from the action roles.\n- Use only these action formats: navigate to <location>, detect <location>, pick <tomato>, scan, scan <tomato>, place <tomato>, discard <tomato>.\n- Do not generate descriptive phrases, retries, requests for assistance, system checks, or any action outside these formats.'

TOMATO_GENERATION_FEW_SHOT_V1 = 'We: Example state:\nRobot location: dock_station\nTomato states:\ntomato1: unknown, observed unknown, scanned unknown\ntomato2: unknown, observed unknown, scanned unknown\nHeld tomato: None\nYou:\nA) navigate to stem_01\nB) navigate to stem_02\nC) detect stem_01\nD) pick tomato1\n\nWe: Example state:\nRobot location: stem_01\nTomato states:\ntomato1: detected, observed ripe, scanned unknown\ntomato2: unknown, observed unknown, scanned unknown\nHeld tomato: None\nYou:\nA) pick tomato1\nB) detect stem_01\nC) navigate to stem_02\nD) scan\n\nWe: Example state:\nRobot location: stem_01\nTomato states:\ntomato1: held, observed ripe, scanned unknown\ntomato2: unknown, observed unknown, scanned unknown\nHeld tomato: tomato1\nYou:\nA) scan\nB) scan\nC) scan\nD) scan\n\nWe: Example state:\nRobot location: stem_01\nTomato states:\ntomato1: held, observed ripe, scanned ripe\ntomato2: unknown, observed unknown, scanned unknown\nHeld tomato: tomato1\nYou:\nA) place tomato1\nB) place tomato1\nC) place tomato1\nD) place tomato1\n\nWe: Example state:\nRobot location: stem_01\nTomato states:\ntomato1: loaded, observed ripe, scanned ripe\ntomato2: discarded, observed ripe, scanned rotten\ntomato3: unknown, observed unknown, scanned unknown\ntomato4: unknown, observed unknown, scanned unknown\nHeld tomato: None\nYou:\nA) navigate to stem_02\nB) detect stem_01\nC) pick tomato3\nD) scan'
TOMATO_GENERATION_FEW_SHOT_V2 = 'We: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at dock_station. The robot is not holding a tomato.\nWe: The tomatoes that still need attention are tomato1, tomato2.\nWe: tomato1 is currently unknown; its observed ripeness is unknown, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.\nWe: The loaded tomatoes are none. The discarded tomatoes are none.\nWe: The robot has not completed any actions yet.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) navigate to stem_01\nB) navigate to stem_02\nC) detect stem_01\nD) pick tomato1\n\nWe: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at stem_01. The robot is not holding a tomato.\nWe: The tomatoes that still need attention are tomato1, tomato2.\nWe: tomato1 is currently detected; its observed ripeness is ripe, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.\nWe: The loaded tomatoes are none. The discarded tomatoes are none.\nWe: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) pick tomato1\nB) detect stem_01\nC) navigate to stem_02\nD) scan\n\nWe: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at stem_01. The robot is holding tomato1.\nWe: The tomatoes that still need attention are tomato1, tomato2.\nWe: tomato1 is currently held; its observed ripeness is ripe, and its scan result is unknown. tomato2 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.\nWe: The loaded tomatoes are none. The discarded tomatoes are none.\nWe: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01; 3. pick tomato1.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) scan\nB) place tomato1\nC) discard tomato1\nD) pick tomato2\n\nWe: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at stem_01. The robot is not holding a tomato.\nWe: The tomatoes that still need attention are tomato3, tomato4.\nWe: tomato1 is currently loaded; its observed ripeness is ripe, and its scan result is ripe. tomato2 is currently discarded; its observed ripeness is ripe, and its scan result is rotten. tomato3 is currently unknown; its observed ripeness is unknown, and its scan result is unknown. tomato4 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.\nWe: The loaded tomatoes are tomato1. The discarded tomatoes are tomato2.\nWe: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01; 3. pick tomato1; 4. scan; 5. place tomato1; 6. pick tomato2; 7. scan; 8. discard tomato2.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) navigate to stem_02\nB) detect stem_01\nC) pick tomato3\nD) scan\n\nWe: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at stem_01. The robot is holding tomato2.\nWe: The tomatoes that still need attention are tomato2, tomato3, tomato4.\nWe: tomato1 is currently loaded; its observed ripeness is ripe, and its scan result is ripe. tomato2 is currently held; its observed ripeness is ripe, and its scan result is rotten. tomato3 is currently unknown; its observed ripeness is unknown, and its scan result is unknown. tomato4 is currently unknown; its observed ripeness is unknown, and its scan result is unknown.\nWe: The loaded tomatoes are tomato1. The discarded tomatoes are none.\nWe: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01; 3. pick tomato1; 4. scan; 5. place tomato1; 6. pick tomato2; 7. scan.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) discard tomato2\nB) place tomato2\nC) navigate to stem_02\nD) pick tomato3\n\nWe: The task is: Harvest all ripe tomatoes and discard rotten tomatoes.\nWe: The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.\nWe: The robot is at stem_02. The robot is not holding a tomato.\nWe: The tomatoes that still need attention are tomato3, tomato4.\nWe: tomato1 is currently loaded; its observed ripeness is ripe, and its scan result is ripe. tomato2 is currently discarded; its observed ripeness is ripe, and its scan result is rotten. tomato3 is currently detected; its observed ripeness is ripe, and its scan result is unknown. tomato4 is currently detected; its observed ripeness is unripe, and its scan result is unknown.\nWe: The loaded tomatoes are tomato1. The discarded tomatoes are tomato2.\nWe: So far, the robot has completed these actions: 1. navigate to stem_01; 2. detect stem_01; 3. pick tomato1; 4. scan; 5. place tomato1; 6. pick tomato2; 7. scan; 8. discard tomato2; 9. navigate to stem_02; 10. detect stem_02.\nWe: What should the robot do next? Answer with four options labeled A), B), C), and D).\nYou:\nA) pick tomato3\nB) pick tomato4\nC) detect stem_02\nD) navigate to stem_01'
TOMATO_ENVIRONMENT_SENTENCE = 'The environment contains four tomatoes: tomato1, tomato2, tomato3, and tomato4. The locations are dock_station, stem_01, and stem_02.'

def _build_tomato_generation_prompt_text_v1(
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

{TOMATO_ACTION_ROLES}

{TOMATO_ACTION_OUTPUT_RULES}

{TOMATO_GENERATION_FEW_SHOT_V1}

We: Overall instruction: {instruction}
We: Robot location: {robot_location}
We: Active tomatoes: {", ".join(active_tomatoes) if active_tomatoes else "None"}
We: Tomato states:
{tomato_state_text}
We: Held tomato: {held_tomato if held_tomato else "None"}
We: Loaded tomatoes: {", ".join(loaded_tomatoes) if loaded_tomatoes else "None"}
We: Discarded tomatoes: {", ".join(discarded_tomatoes) if discarded_tomatoes else "None"}
We: Actions already completed:
{history_text}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def _build_tomato_score_prompt_text_v1(
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

We: Overall instruction: {instruction}
We: Robot location: {robot_location}
We: Active tomatoes: {", ".join(active_tomatoes) if active_tomatoes else "None"}
We: Tomato states:
{tomato_state_text}
We: Held tomato: {held_tomato if held_tomato else "None"}
We: Loaded tomatoes: {", ".join(loaded_tomatoes) if loaded_tomatoes else "None"}
We: Discarded tomatoes: {", ".join(discarded_tomatoes) if discarded_tomatoes else "None"}
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
We: {TOMATO_ENVIRONMENT_SENTENCE}
We: The robot is at {robot_location}. {held_text}
We: The tomatoes that still need attention are {_items_text(active_tomatoes)}.
We: {_tomato_state_sentences(tomato_state_text)}
We: The loaded tomatoes are {_items_text(loaded_tomatoes)}. The discarded tomatoes are {_items_text(discarded_tomatoes)}.
We: {_history_sentence(history_text)}
""".strip()


def _build_tomato_generation_prompt_text_v2(
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

{TOMATO_ACTION_ROLES}

{TOMATO_ACTION_OUTPUT_RULES}

{TOMATO_GENERATION_FEW_SHOT_V2}

{_tomato_scene_text(instruction, robot_location, active_tomatoes, tomato_state_text, held_tomato, loaded_tomatoes, discarded_tomatoes, history_text)}
We: What should the robot do next? Answer with four options labeled A), B), C), and D).
You:
""".strip()


def _build_tomato_score_prompt_text_v2(
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

{TOMATO_ACTION_ROLES}

{_tomato_scene_text(instruction, robot_location, active_tomatoes, tomato_state_text, held_tomato, loaded_tomatoes, discarded_tomatoes, history_text)}
We: What should the robot do next?
You:
{mc_gen_full}
We: Which option is correct? Answer with a single capital letter.
You:
""".strip()



def _is_v2(prompt_version: str) -> bool:
    return (prompt_version or "v1").lower() in {"v2", "natural", "natural_language"}


def build_tomato_generation_prompt_text(*args, prompt_version: str = "v1", **kwargs) -> str:
    builder = _build_tomato_generation_prompt_text_v2 if _is_v2(prompt_version) else _build_tomato_generation_prompt_text_v1
    return builder(*args, **kwargs)


def build_tomato_score_prompt_text(*args, prompt_version: str = "v1", **kwargs) -> str:
    builder = _build_tomato_score_prompt_text_v2 if _is_v2(prompt_version) else _build_tomato_score_prompt_text_v1
    return builder(*args, **kwargs)
