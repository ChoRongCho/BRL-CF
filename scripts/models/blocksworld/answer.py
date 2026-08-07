from __future__ import annotations

import random


NOISY_ORACLE_ERROR_RATE = 0.1


def answer_question(mode: str,
                    target_fact: str,
                    action_name: str | None,
                    true_init_facts: set[str],
                    current_facts: set[str] | None = None,
                    noisy_oracle_error_rate: float = NOISY_ORACLE_ERROR_RATE) -> bool:

    target_fact = str(target_fact).replace(" ", "")
    action = "" if action_name is None else str(action_name)

    if mode == "oracle":
        return _answer_oracle(target_fact, action, true_init_facts, current_facts)

    elif mode == "noisy-oracle":
        return _answer_noisy_oracle(
            target_fact,
            action,
            true_init_facts,
            current_facts,
            noisy_oracle_error_rate,
        )

    elif mode == "human-proxy":
        return _answer_human_proxy(target_fact, action, true_init_facts, current_facts)

    elif mode == "human":
        return _answer_human(target_fact, action, true_init_facts, current_facts)

    else:
        raise ValueError(f"Unsupported blocksworld answer mode: {mode}")


def _answer_oracle(target_fact: str,
                   action: str,
                   true_init_facts: set[str],
                   current_facts: set[str] | None = None) -> bool:

    return target_fact in true_init_facts


def _answer_noisy_oracle(target_fact: str,
                         action: str,
                         true_init_facts: set[str],
                         current_facts: set[str] | None = None,
                         error_rate: float = NOISY_ORACLE_ERROR_RATE) -> bool:

    answer = _answer_oracle(target_fact, action, true_init_facts, current_facts)
    if random.random() < error_rate:
        return not answer
    return answer


def _answer_human_proxy(target_fact: str,
                        action: str,
                        true_init_facts: set[str],
                        current_facts: set[str] | None = None) -> bool:

    return target_fact in true_init_facts


def _answer_human(target_fact: str,
                  action: str,
                  true_init_facts: set[str],
                  current_facts: set[str] | None = None) -> bool:

    raise NotImplementedError("human mode must be handled by FeedbackManager.query_human().")
