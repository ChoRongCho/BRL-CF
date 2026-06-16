from __future__ import annotations


"""
Tomato
"""

TOMATO_BACKGROUND = (
    "You are a tomato harvesting robot. The robot moves between a dock station "
    "and tomato stems, observes tomatoes, picks ripe tomatoes, scans held "
    "tomatoes for quality, loads good tomatoes, and discards rotten tomatoes."
)

TOMATO_ACTION_ROLES = """
Action roles:
- navigate to <location>: move the robot to dock_station, stem_01, or stem_02.
- detect <location>: observe tomatoes at the current robot stem.
- pick <tomato>: pick one detected ripe tomato at the current robot stem with an empty hand.
- scan <tomato>: inspect the currently held tomato.
- place <tomato>: load the held ripe tomato.
- discard <tomato>: discard the held rotten or bad tomato.
""".strip()

TOMATO_GENERATION_FEW_SHOT = """
We: Example state:
Robot location: dock_station
Tomato states:
tomato1: unknown, observed unknown, scanned unknown
tomato2: unknown, observed unknown, scanned unknown
Held tomato: None
You:
A) navigate to stem_01
B) navigate to stem_02
C) detect stem_01
D) pick tomato1

We: Example state:
Robot location: stem_01
Tomato states:
tomato1: detected, observed ripe, scanned unknown
tomato2: unknown, observed unknown, scanned unknown
Held tomato: None
You:
A) pick tomato1
B) detect stem_01
C) navigate to stem_02
D) scan

We: Example state:
Robot location: stem_01
Tomato states:
tomato1: held, observed ripe, scanned unknown
tomato2: unknown, observed unknown, scanned unknown
Held tomato: tomato1
You:
A) scan
B) scan
C) scan
D) scan

We: Example state:
Robot location: stem_01
Tomato states:
tomato1: held, observed ripe, scanned ripe
tomato2: unknown, observed unknown, scanned unknown
Held tomato: tomato1
You:
A) place tomato1
B) place tomato1
C) place tomato1
D) place tomato1

We: Example state:
Robot location: stem_01
Tomato states:
tomato1: loaded, observed ripe, scanned ripe
tomato2: discarded, observed ripe, scanned rotten
tomato3: unknown, observed unknown, scanned unknown
tomato4: unknown, observed unknown, scanned unknown
Held tomato: None
You:
A) navigate to stem_02
B) detect stem_01
C) pick tomato3
D) scan
""".strip()

WASTE_BACKGROUND = (
    "You are a robot operating in a waste sorting station. You are in front "
    "of a counter. There are four bins: a general bin, a plastic bin, a "
    "paper bin, and a can bin."
)

WASTE_ACTION_ROLES = """
Action roles:
- detect: observe waste attributes for objects on the counter.
- pick <object>: pick one observed object from the counter with an empty hand.
- place <object> into <bin>: place the held object into the bin matching the observed attribute.
""".strip()

WASTE_GENERATION_FEW_SHOT = """
We: Example state:
Objects still on the counter: waste1, waste2, waste3, waste4
Observed waste attributes: None
Object currently held by the robot: None
You:
A) detect
B) pick waste1
C) place waste1 into can bin
D) pick waste2

We: Example state:
Objects still on the counter: waste1, waste2, waste3, waste4
Observed waste attributes: waste1: paper, waste2: general
Object currently held by the robot: None
You:
A) pick waste1
B) pick waste2
C) detect
D) place waste1 into paper bin

We: Example state:
Objects still on the counter: waste2, waste3, waste4
Observed waste attributes: waste2: general, waste3: plastic, waste4: can
Object currently held by the robot: waste2
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

We: Overall instruction: {instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(available_bins)}
We: Observed waste attributes: {observed_text}
We: Object currently held by the robot: {held_text}
We: Actions already completed:
{history_text}
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

We: Overall instruction: {instruction}
We: Objects still on the counter: {", ".join(remaining_objects) if remaining_objects else "None"}
We: Available bins: {", ".join(available_bins)}
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
