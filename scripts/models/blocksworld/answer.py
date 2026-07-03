from __future__ import annotations

import numpy as np


def answer_question(mode: str, target_fact: str, action_name: str | None, true_init_facts: set[str]) -> bool:
    target_fact = str(target_fact).replace(" ", "")

    if mode in {"oracle", "human-proxy", "auto"}:
        return target_fact in true_init_facts

    if mode == "random":
        return bool(np.random.random() < 0.5)

    raise ValueError(f"Unsupported blocksworld answer mode: {mode}")
