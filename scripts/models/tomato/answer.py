from __future__ import annotations

import numpy as np


def answer_question(mode: str, target_fact: str, action_name: str | None, true_init_facts: set[str]) -> bool:
    target_fact = str(target_fact).replace(" ", "")
    action = "" if action_name is None else str(action_name)

    if mode == "random":
        return bool(np.random.random() < 0.5)

    if mode in {"oracle", "human-proxy", "auto"}:
        return _answer_human_proxy(target_fact, action, true_init_facts)

    raise ValueError(f"Unsupported tomato answer mode: {mode}")


def _answer_human_proxy(target_fact: str, action: str, true_init_facts: set[str]) -> bool:
    if action.startswith("pick") and target_fact.startswith("at("):
        return False

    if action.startswith("detect("):
        stem = action[action.rfind(",") + 1:-1].strip()
        if target_fact.startswith("ripe("):
            tomato = target_fact[len("ripe("):-1]
            q_fact = f"at({tomato},{stem})"
            if q_fact in true_init_facts:
                if f"rotten({tomato})" in true_init_facts:
                    return True
                return target_fact in true_init_facts
            return False

        if target_fact.startswith("unripe("):
            tomato = target_fact[len("unripe("):-1]
            q_fact = f"at({tomato},{stem})"
            if q_fact in true_init_facts:
                return target_fact in true_init_facts
            return False

        if target_fact.startswith("at("):
            return target_fact in true_init_facts

    if action.startswith("scan("):
        return target_fact in true_init_facts

    if (action.startswith("place") or action.startswith("discard")) and target_fact.startswith("handempty("):
        return True

    return bool(np.random.random() < 0.5)
