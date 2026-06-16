from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEMO_MC_GEN_PROMPT = """
We: You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.

We: On the counter, there is an orange soda, a Pepsi, and an apple.
We: Put that drink in the top drawer.
You:
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it

We: On the counter, there is an energy bar, a banana, and a microwave.
We: Put the snack next to the microwave.
You:
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana

We: On the counter, there is a Coke, a Sprite, and a sponge.
We: Can you dispose of the can? It should have expired.
You:
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin

We: On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips.
We: I would like a bag of chips.
You:
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips

We: On the counter, there is {scene_objects}
We: {task}
You:
"""


DEMO_MC_SCORE_BACKGROUND = """
You are a robot operating in an office kitchen. You are in front of a counter with two closed drawers, a top one and a bottom one. There is also a landfill bin, a recycling bin, and a compost bin.
""".strip()

MOBILE_MC_GEN_PROMPT = DEMO_MC_GEN_PROMPT

MOBILE_MC_SCORE_BACKGROUND = (
    "You are a robot operating in an office kitchen. "
    "You are in front of a counter with two closed drawers, a top one and a middle "
    "one. There is also a landfill bin, a recycling bin, and a compost bin."
)

TABLETOP_MC_GEN_PROMPT = """
We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: blue block, yellow bowl, yellow block, green bowl, green block, blue bowl.
We: Now, put the grass-colored bowl at the right side of the blue round object
You: These are some options:
A) put blue bowl at the right side of blue block
B) put green bowl at the right side of blue bowl
C) put green block at the right side of blue bowl
D) put yellow bowl at the right side of blue bowl

We: On the table there are these objects: yellow bowl, green bowl, green block, yellow block, blue block, blue bowl.
We: Now, put the yellow square object near the green box
You: These are some options:
A) put yellow block in front of green block
B) put yellow block behind green block
C) put yellow block to the left of green block
D) put yellow block to the right of green block

We: On the table there are these objects: blue bowl, yellow block, green bowl, blue block, green block, yellow bowl.
We: Now, put the yellow bowl along the horizontal axis of the grass-colored block
You: These are some options:
A) put yellow bowl at the front of the green block
B) put yellow bowl at the left side of the green block
C) put yellow bowl at the left side of the blue block
D) put yellow bowl at the right side of the green block

We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow bowl, green block, blue block.
We: Now, {instruction}
You: These are some options:
"""

TABLETOP_MC_SCORE_PROMPT = """
We: You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous.

We: On the table there are these objects: green bowl, yellow block, blue bowl, yellow bowl, green block, blue block.
We: Now, {instruction}
You: These are some options:
{mc}
We: Which option is correct? Answer with a single letter.
You:
"""


WASTESORTING_MC_GEN_PROMPT = """
We: You are a robot operating in a waste sorting station. You are in front of a counter. There are four bins: a general bin, a plastic bin, a paper bin, and a can bin.

We: On the counter, there is a plastic bottle, a banana peel, and a soda can.
We: Dispose of the plastic bottle properly.
You:
A) pick up the plastic bottle and put it in the plastic bin
B) pick up the banana peel and put it in the general bin
C) pick up the soda can and put it in the can bin
D) pick up the plastic bottle and put it in the general bin

We: On the counter, there is an apple core, a paper cup, and a glass bottle.
We: Throw away the food waste.
You:
A) pick up the apple core and put it in the general bin
B) pick up the paper cup and put it in the paper bin
C) pick up the glass bottle and put it in the general bin
D) pick up the apple core and put it in the paper bin

We: On the counter, there is a soda can, a plastic wrapper, and a dirty tissue.
We: Dispose of the recyclable item.
You:
A) pick up the soda can and put it in the can bin
B) pick up the plastic wrapper and put it in the plastic bin
C) pick up the dirty tissue and put it in the paper bin
D) pick up the soda can and put it in the general bin

We: On the counter, there is a cardboard box, a plastic bottle, and a soda can.
We: Throw away the paper item.
You:
A) pick up the cardboard box and put it in the paper bin
B) pick up the plastic bottle and put it in the plastic bin
C) pick up the soda can and put it in the can bin
D) pick up the cardboard box and put it in the general bin

We: On the counter, there is {scene_objects}
We: {task}
You:
"""


def process_mc_raw(mc_raw: str, add_mc: str = "an option not listed here") -> Tuple[str, List[str], str]:
    mc_processed_all = []
    for mc in mc_raw.split("\n"):
        mc = mc.strip()
        if (
            len(mc) < 5
            or mc[0] not in ["a", "b", "c", "d", "A", "B", "C", "D", "1", "2", "3", "4"]
            or mc[1] not in [")", "."]
        ):
            continue
        mc = mc[2:].strip().lower().split(".")[0]
        mc_processed_all.append(mc)
    if len(mc_processed_all) < 4:
        raise ValueError("Cannot extract four options from the raw output.")

    mc_processed_all = list(dict.fromkeys(mc_processed_all))[:4]
    while len(mc_processed_all) < 4:
        mc_processed_all.append("do nothing")

    prefix_all = ["A) ", "B) ", "C) ", "D) "]
    if add_mc is not None:
        mc_processed_all.append(add_mc)
        prefix_all.append("E) ")
    random.shuffle(mc_processed_all)

    mc_prompt = "\n".join(prefix + mc for prefix, mc in zip(prefix_all, mc_processed_all))
    add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
    return mc_prompt, mc_processed_all, add_mc_prefix


def temperature_scaling(logits: Sequence[float], temperature: float) -> np.ndarray:
    logits = np.array(logits)
    logits /= temperature
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    return np.exp(logits)


def build_demo_mc_prompt(instruction: str, scene_objects: str) -> str:
    return DEMO_MC_GEN_PROMPT.replace("{task}", instruction).replace("{scene_objects}", scene_objects)


def build_mobile_mc_prompt(instruction: str, scene_objects: str) -> str:
    return MOBILE_MC_GEN_PROMPT.replace("{task}", instruction).replace("{scene_objects}", scene_objects)


def build_tabletop_mc_prompt(instruction: str) -> str:
    return TABLETOP_MC_GEN_PROMPT.replace("{instruction}", instruction).strip()


def build_tabletop_score_prompt(instruction: str, mc_gen_full: str) -> str:
    return TABLETOP_MC_SCORE_PROMPT.replace("{instruction}", instruction).replace("{mc}", mc_gen_full).strip()


def build_score_prompt(
    mc_gen_prompt: str,
    mc_gen_full: str,
    background_prompt: str = DEMO_MC_SCORE_BACKGROUND,
    capital: bool = False,
) -> str:
    cur_scenario_prompt = mc_gen_prompt.split("\n\n")[-1].strip()
    letter_text = "single capital letter" if capital else "single letter"
    return (
        background_prompt
        + "\n\n"
        + cur_scenario_prompt
        + "\n"
        + mc_gen_full
        + f"\nWe: Which option is correct? Answer with a {letter_text}."
        + "\nYou:"
    )


def top_choice_logprobs(response) -> Tuple[List[str], List[float], Dict[str, float]]:
    top_logprobs_full = response["choices"][0]["logprobs"]["top_logprobs"][0]
    top_tokens = [token.strip() for token in top_logprobs_full.keys()]
    top_logprobs = [value for value in top_logprobs_full.values()]
    return top_tokens, top_logprobs, top_logprobs_full


def prediction_set(top_tokens: Sequence[str], top_logprobs: Sequence[float], qhat: float, temperature: float = 5) -> Tuple[List[str], np.ndarray]:
    mc_smx_all = temperature_scaling(top_logprobs, temperature=temperature)
    preds = [token for token_ind, token in enumerate(top_tokens) if mc_smx_all[token_ind] >= 1 - qhat]
    return preds, mc_smx_all


def parse_tabletop_action_option(action_option: str) -> Tuple[str, str, str]:
    if "not listed here" in action_option or "do nothing" in action_option or "block " not in action_option:
        raise ValueError(f"Invalid tabletop option: {action_option}")

    option_split = action_option.split()
    pick_obj_attr = option_split[option_split.index("block") - 1]
    pick_obj = pick_obj_attr + " block"

    target_obj_attr = option_split[option_split.index("bowl") - 1]
    target_obj = target_obj_attr + " bowl"

    if "left" in action_option:
        relation = "left"
    elif "right" in action_option:
        relation = "right"
    elif "front" in action_option:
        relation = "front"
    elif "back" in action_option or "behind" in action_option:
        relation = "back"
    else:
        relation = "in"
    return pick_obj, target_obj, relation


def label_mobile_true_options(data: Dict) -> List[str]:
    true_options = []
    info = data["info"].split("\n", 1)[1]
    true_obj = info.split("User intent (object): ")[1].split("\n")[0].lower()
    true_obj = [obj.strip() for obj in true_obj.split(",")]
    true_target_loc = info.split("User intent (location): ")[1].split("\n")[0].lower()
    scene_obj = info.split("Scene objects:")[1].split("\n")[0].split(", ")
    scene_obj = [obj.strip().lower() for obj in scene_obj]

    token_all = ["A", "B", "C", "D", "E"]
    for i in range(len(true_obj)):
        if not ("clean sponge" in scene_obj and "dirty sponge with food residue" in scene_obj) and "sponge" in true_obj[i]:
            true_obj[i] = "sponge"

    for mc_ind, mc in enumerate(data["mc_gen_all"]):
        if "not listed here" in mc or "do nothing" in mc:
            continue
        if "clean" in mc and "dirty" in mc:
            continue
        if true_target_loc == "pick-up":
            num_obj_in_mc = sum(1 for obj in scene_obj if obj in mc.lower())
            if num_obj_in_mc > 1:
                continue
            for obj in true_obj:
                if obj in mc and "drawer" not in mc and "bin" not in mc and "microwave" not in mc and "cooktop" not in mc:
                    true_options.append(token_all[mc_ind])
        elif "drawer" in true_target_loc:
            for obj in true_obj:
                if obj in mc and true_target_loc in mc:
                    true_options.append(token_all[mc_ind])
        elif any(loc in true_target_loc for loc in ["recycling", "landfill", "compost", "microwave", "cooktop"]):
            for obj in true_obj:
                if obj in mc and true_target_loc in mc:
                    true_options.append(token_all[mc_ind])
        else:
            parts = mc.split("and")
            if len(parts) < 2:
                continue
            mc_obj_pick_up_phrase = parts[0]
            mc_obj_place_phrase = parts[1]
            for obj in true_obj:
                if obj in mc_obj_pick_up_phrase and true_target_loc in mc_obj_place_phrase:
                    true_options.append(token_all[mc_ind])

    return true_options or [data["add_mc_prefix"]]
