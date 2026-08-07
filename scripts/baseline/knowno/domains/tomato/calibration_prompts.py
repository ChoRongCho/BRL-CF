from __future__ import annotations

TOMATO_BACKGROUND = 'You are a tomato harvesting robot. The robot moves between a dock station and tomato stems, observes tomatoes, picks ripe tomatoes, scans held tomatoes for quality, loads good tomatoes, and discards rotten tomatoes.'

TOMATO_ACTION_ROLES = 'Action roles:\n- navigate to <location>: move the robot to dock_station, stem_01, or stem_02.\n- detect <location>: observe tomatoes at the current robot stem.\n- pick <tomato>: pick one detected ripe tomato at the current robot stem with an empty hand.\n- scan <tomato>: inspect the currently held tomato.\n- place <tomato>: load the held ripe tomato.\n- discard <tomato>: discard the held rotten or bad tomato.'

TOMATO_CALIBRATION_TEMPLATE = '# Tomato calibration dataset.\n# Separate examples with --0000--.\n# Options are optional. If omitted, the LLM generates options from Context.\n\n--0000--\nContext:\nOverall instruction: Harvest all ripe tomatoes and discard rotten tomatoes.\nRobot location: stem_01\nActive tomatoes: tomato1, tomato2\nTomato states:\ntomato1: detected, ripe\ntomato2: unknown, unknown\nHeld tomato: None\nLoaded tomatoes: None\nDiscarded tomatoes: None\nActions already completed:\n1. navigate to stem_01\n2. detect\n\nTrue actions:\npick tomato1\n\nOptions:\nA) pick tomato1\nB) detect\nC) navigate to stem_02\nD) scan\nE) an option not listed here\n\nCorrect options:\nA\n'


def build_tomato_calibration_prompt_text(context: str, prompt_version: str = "v1") -> str:
    return f"""
We: {TOMATO_BACKGROUND}

{TOMATO_ACTION_ROLES}

{context}
You:
""".strip()
