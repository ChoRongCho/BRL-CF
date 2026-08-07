from __future__ import annotations

WASTE_BACKGROUND = 'You are a robot operating in a waste sorting station. You are in front of a counter. There are four bins: a general bin, a plastic bin, a paper bin, and a can bin.'

WASTE_ACTION_ROLES = 'Action roles:\n- detect: observe waste attributes for objects on the counter.\n- pick <object>: pick one observed object from the counter with an empty hand.\n- place <object> into <bin>: place the held object into the bin matching the observed attribute.'

WASTE_CALIBRATION_TEMPLATE = '# Waste-sorting calibration dataset.\n# Separate examples with --0000--.\n# Options are optional. If omitted, the LLM generates options from Context.\n\n--0000--\nContext:\nOverall instruction: Discard all waste.\nObjects still on the counter: waste1, waste2\nAvailable bins: general bin, plastic bin, paper bin, can bin\nObserved waste attributes: waste1: can\nObject currently held by the robot: None\nActions already completed:\nNone\n\nTrue actions:\npick waste1\n\nOptions:\nA) detect\nB) pick waste1\nC) place waste1 into can bin\nD) pick waste2\nE) an option not listed here\n\nCorrect options:\nB\n'


def build_waste_calibration_prompt_text(context: str, prompt_version: str = "v1") -> str:
    return f"""
We: {WASTE_BACKGROUND}

{WASTE_ACTION_ROLES}

{context}
You:
""".strip()
