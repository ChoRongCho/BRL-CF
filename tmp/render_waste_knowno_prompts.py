"""Render the actual 02 and 04 Waste KnowNo prompts without calling an LLM."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE02 = ROOT / "02_BRL_POMDP_CODE"
CODE04 = ROOT / "04_BRL_WASTE"
OUTPUT = CODE02 / "tmp" / "waste_knowno_prompt_comparison"


def load_04_planner_class():
    path = (
        CODE04
        / "src"
        / "pomdp_planner"
        / "src"
        / "pomdp_planner"
        / "knowno_planner.py"
    )
    spec = importlib.util.spec_from_file_location("waste_knowno_04", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.KnowNoPlanner


def main():
    sys.path.insert(0, str(CODE02 / "scripts" / "baseline" / "knowno"))
    from wastesorting_utils import (
        build_waste_generation_prompt,
        build_waste_score_prompt,
    )

    history02 = "\n".join(
        [
            "1. detect",
            "2. pick waste1",
            "3. place waste1 into can bin",
            "4. pick waste2",
            "5. place waste2 into paper bin",
            "6. pick waste3",
            "7. place waste3 into general bin",
        ]
    )
    options02 = "\n".join(
        [
            "A) pick waste4",
            "B) place waste4 into general bin",
            "C) detect",
            "D) place waste4 into plastic bin",
            "E) an option not listed here",
        ]
    )
    prompt02_generation = build_waste_generation_prompt(
        "Discard all waste.",
        ["waste4"],
        "None",
        "None",
        history02,
        "v2",
        "waste4 is under waste3 and cannot be detected until waste3 is placed",
    )
    prompt02_score = build_waste_score_prompt(
        "Discard all waste.",
        ["waste4"],
        "None",
        "None",
        history02,
        options02,
        "v2",
        "waste4 is under waste3 and cannot be detected until waste3 is placed",
    )

    Planner04 = load_04_planner_class()
    planner04 = Planner04(client=object())
    planner04.history = [
        "detect",
        "pick w1",
        "place w1 into can bin",
        "pick w2",
        "place w2 into paper bin",
        "pick w3",
        "place w3 into general bin",
    ]
    facts04 = [
        "waste(w1)",
        "waste(w2)",
        "waste(w3)",
        "waste(w4)",
        "in_bin(w1,b_can)",
        "in_bin(w2,b_paper)",
        "in_bin(w3,b_general)",
        "handempty(brl_robot)",
    ]
    state04 = planner04.state_from_facts(facts04)
    options04 = [
        "pick w4",
        "place w4 into general bin",
        "detect",
        "place w4 into plastic bin",
        "an option not listed here",
    ]
    prompt04_generation = planner04.build_generation_prompt(state04)
    prompt04_score = planner04.build_score_prompt(state04, options04)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rendered = {
        "02_generation_prompt.txt": prompt02_generation,
        "02_score_prompt.txt": prompt02_score,
        "04_generation_prompt.txt": prompt04_generation,
        "04_score_prompt.txt": prompt04_score,
    }
    for name, prompt in rendered.items():
        (OUTPUT / name).write_text(prompt + "\n", encoding="utf-8")
        print("WROTE", OUTPUT / name, "chars=", len(prompt))


if __name__ == "__main__":
    main()
