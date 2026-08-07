from __future__ import annotations

import math

import numpy as np


NOISY_ORACLE_ERROR_RATE = 0.2

CATEGORY_PREDICATES = ("plastic", "can", "paper", "general")
PLACE_ACTIONS = (
    "place_gw_bin",
    "place_paper_bin",
    "place_can_bin",
    "place_plastic_bin",
)

# Human-proxy response model.
# difficulty: 0~10. Larger means harder and closer to random answering.
# m_max: margin when difficulty is 0. m_max=4.0 gives sigmoid(4.0) ~= 0.98.
# criterion: yes-bias. Positive values bias ambiguous answers toward True.
HUMAN_PROXY_M_MAX = 4.0
HUMAN_PROXY_CRITERION = 0.0

# Correct-answer probability by difficulty when m_max=4.0 and criterion=0.0.
# difficulty | margin | P(correct)
#      0     |  4.0   |   0.9820
#      1     |  3.6   |   0.9734
#      2     |  3.2   |   0.9608
#      3     |  2.8   |   0.9427
#      4     |  2.4   |   0.9168
#      5     |  2.0   |   0.8808
#      6     |  1.6   |   0.8320
#      7     |  1.2   |   0.7685
#      8     |  0.8   |   0.6900
#      9     |  0.4   |   0.5987
#     10     |  0.0   |   0.5000
HUMAN_PROXY_DIFFICULTY_BY_SITUATION = {
    # Visual perception during detect_waste.
    "detect_visible_category": 7,  # 보이는 waste의 실제 category를 묻는 경우
    "detect_visible_wrong_category": 8,  # 보이는 waste에 대해 틀린 category를 묻는 경우
    "detect_occluded_category": 9,  # 가려진 waste의 category를 묻는 경우
    "detect_absent_category": 9,  # category 정보가 없는 waste의 category를 묻는 경우
    "detect_visible_presence": 5,  # 보이고 category가 알려진 waste의 detected 여부를 묻는 경우
    "detect_occluded_presence": 8,  # 가려진 waste의 detected 여부를 묻는 경우
    "detect_absent_presence": 8,  # category 정보가 없는 waste의 detected 여부를 묻는 경우
    # Physical state after robot manipulation.
    "pick_handempty": 1,
    "pick_holding": 1,
    "place_handempty": 2,
    "place_holding": 2,
    "place_in_bin": 2,
    "place_detected": 3,
    # Fallback when the question type is not explicitly modeled.
    "default": 5,
}


def answer_question(mode: str,
                    target_fact: str,
                    action_name: str | None,
                    true_init_facts: set[str],
                    current_facts: set[str] | None = None,
                    noisy_oracle_error_rate: float = NOISY_ORACLE_ERROR_RATE) -> bool:

    target_fact = str(target_fact).replace(" ", "")
    action = "" if action_name is None else str(action_name)
    current_facts = set() if current_facts is None else {
        str(fact).replace(" ", "") for fact in current_facts
    }

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
        raise ValueError(f"Unsupported wastesorting answer mode: {mode}")


def _answer_oracle(target_fact: str,
                   action: str,
                   true_init_facts: set[str],
                   current_facts: set[str]) -> bool:
    
    if action.startswith("detect_waste("):
        if target_fact.startswith("detected("):
            waste = target_fact[len("detected("):-1]
        elif target_fact.startswith(tuple(f"{category}(" for category in CATEGORY_PREDICATES)):
            waste = target_fact[target_fact.find("(") + 1:-1]
        else:
            waste = None

        if waste is not None:
            is_occluded = False
            for fact in true_init_facts:
                if not fact.startswith("on("):
                    continue

                top_waste, bottom_waste = fact[fact.find("(") + 1:-1].split(",")
                if bottom_waste != waste:
                    continue

                top_is_cleared = False
                for current_fact in current_facts:
                    if current_fact.startswith("holding("):
                        holding_args = current_fact[current_fact.find("(") + 1:-1].split(",")
                        if len(holding_args) >= 2 and holding_args[1] == top_waste:
                            top_is_cleared = True
                            break
                    elif current_fact.startswith("in_bin("):
                        in_bin_args = current_fact[current_fact.find("(") + 1:-1].split(",")
                        if in_bin_args and in_bin_args[0] == top_waste:
                            top_is_cleared = True
                            break

                if not top_is_cleared:
                    is_occluded = True
                    break

            if is_occluded:
                return False

            if target_fact.startswith("detected("):
                return any(
                    f"{category}({waste})" in true_init_facts
                    for category in CATEGORY_PREDICATES
                )

            return target_fact in true_init_facts

    if action.startswith("pick("):
        if target_fact.startswith("handempty("):
            return np.random.random() >= 0.9
        if target_fact.startswith("holding("):
            return np.random.random() < 0.9

    if action.startswith(PLACE_ACTIONS):
        action_waste, action_bin = _place_action_target(action)

        if target_fact.startswith("in_bin("):
            return target_fact == f"in_bin({action_waste},{action_bin})"
        if target_fact.startswith("handempty("):
            return np.random.random() < 0.9
        if target_fact.startswith("holding("):
            return np.random.random() >= 0.9
        if target_fact.startswith("detected("):
            target_waste = target_fact[len("detected("):-1]
            return target_waste != action_waste

    return target_fact in true_init_facts


def _answer_noisy_oracle(target_fact: str,
                         action: str,
                         true_init_facts: set[str],
                         current_facts: set[str],
                         error_rate: float = NOISY_ORACLE_ERROR_RATE) -> bool:

    answer = _answer_oracle(target_fact, action, true_init_facts, current_facts)
    if np.random.random() < error_rate:
        return not answer
    return answer


def _answer_human_proxy(target_fact: str,
                        action: str,
                        true_init_facts: set[str],
                        current_facts: set[str]) -> bool:
    """
    Human-proxy 의미론적 오류 메모:

    - 기본적으로는 oracle answer를 따른다.
    - visual classification 또는 occlusion에 의존하는 질문에서 주로 틀리게 만든다.
    - wastesorting 도메인에서 사람이 헷갈릴 만한 경우는 다음과 같다.
      1. detect_waste 단계의 category 질문: paper(W), can(W), plastic(W), general(W).
      2. occlusion 때문에 아래 waste가 아직 실제로 보이지 않는 경우.
      3. 시각적으로 비슷한 category 혼동: paper/general, can/plastic.
      4. waste가 부분적으로 가려져 있거나 방금 이동된 상황에서 detected(W)를 묻는 경우.
    - pick/place 결과 질문은 우선 noise 대상에서 제외한다.
      manipulation 상태 혼동을 명시적으로 모델링할 때만 포함한다.
    """

    oracle_answer = _answer_oracle(target_fact, action, true_init_facts, current_facts)
    difficulty = _human_proxy_difficulty(
        target_fact,
        action,
        true_init_facts,
        current_facts,
    )
    return _sample_human_proxy_answer(oracle_answer, difficulty)


def _sample_human_proxy_answer(oracle_answer: bool,
                               difficulty: float,
                               m_max: float = HUMAN_PROXY_M_MAX,
                               criterion: float = HUMAN_PROXY_CRITERION) -> bool:
    difficulty = min(max(float(difficulty), 0.0), 10.0)
    oracle_direction = 1.0 if oracle_answer else -1.0
    margin = m_max * (1.0 - difficulty / 10.0)
    logit = oracle_direction * margin + criterion
    p_true = 1.0 / (1.0 + math.exp(-logit))
    return np.random.random() < p_true


def _human_proxy_difficulty(target_fact: str,
                            action: str,
                            true_init_facts: set[str],
                            current_facts: set[str]) -> float:
    situation = _human_proxy_situation(
        target_fact,
        action,
        true_init_facts,
        current_facts,
    )
    return HUMAN_PROXY_DIFFICULTY_BY_SITUATION[situation]


def _human_proxy_situation(target_fact: str,
                           action: str,
                           true_init_facts: set[str],
                           current_facts: set[str]) -> str:
    if action.startswith("detect_waste("):
        if target_fact.startswith("detected("):
            waste = target_fact[len("detected("):-1]
            if _is_occluded(waste, true_init_facts, current_facts):
                return "detect_occluded_presence"
            if _has_known_category(waste, true_init_facts):
                return "detect_visible_presence"
            return "detect_absent_presence"

        if _is_category_fact(target_fact):
            args = _fact_args(target_fact)
            if not args:
                return "default"

            waste = args[0]
            if _is_occluded(waste, true_init_facts, current_facts):
                return "detect_occluded_category"
            if not _has_known_category(waste, true_init_facts):
                return "detect_absent_category"
            if target_fact in true_init_facts:
                return "detect_visible_category"
            return "detect_visible_wrong_category"

    if action.startswith("pick("):
        if target_fact.startswith("handempty("):
            return "pick_handempty"
        if target_fact.startswith("holding("):
            return "pick_holding"

    if action.startswith(PLACE_ACTIONS):
        if target_fact.startswith("in_bin("):
            return "place_in_bin"
        if target_fact.startswith("handempty("):
            return "place_handempty"
        if target_fact.startswith("holding("):
            return "place_holding"
        if target_fact.startswith("detected("):
            return "place_detected"

    return "default"


def _answer_human(target_fact: str,
                  action: str,
                  true_init_facts: set[str],
                  current_facts: set[str]) -> bool:

    raise NotImplementedError("human mode must be handled by FeedbackManager.query_human().")


def _is_category_fact(fact: str) -> bool:
    return fact.startswith(tuple(f"{predicate}(" for predicate in CATEGORY_PREDICATES))


def _has_known_category(waste: str, true_init_facts: set[str]) -> bool:
    return any(f"{predicate}({waste})" in true_init_facts for predicate in CATEGORY_PREDICATES)


def _is_occluded(waste: str,
                 true_init_facts: set[str],
                 current_facts: set[str]) -> bool:
    for fact in true_init_facts:
        if not fact.startswith("on("):
            continue

        args = _fact_args(fact)
        if len(args) < 2:
            continue

        top_waste, bottom_waste = args[:2]
        if bottom_waste != waste:
            continue

        if not _waste_is_cleared(top_waste, current_facts):
            return True

    return False


def _waste_is_cleared(waste: str, current_facts: set[str]) -> bool:
    for fact in current_facts:
        args = _fact_args(fact)
        if fact.startswith("holding(") and len(args) >= 2 and args[1] == waste:
            return True
        if fact.startswith("in_bin(") and args and args[0] == waste:
            return True

    return False


def _fact_args(fact: str) -> list[str]:
    if "(" not in fact or not fact.endswith(")"):
        return []
    return [arg.strip() for arg in fact[fact.find("(") + 1:-1].split(",")]


def _place_action_target(action: str) -> tuple[str, str]:
    args = action[action.find("(") + 1:-1].split(",")
    if len(args) < 3:
        return "", ""

    waste = args[1].strip()
    bin_name = args[2].strip()
    return waste, bin_name


if __name__ == "__main__":
    true_init_facts = {
        "waste(waste1)",
        "paper(waste1)",
        "waste(waste2)",
        "can(waste2)",
        "on(waste1,waste2)",
    }
    current_facts = set(true_init_facts)
    action = "detect_waste(brl_robot)"
    target_fact = "paper(waste1)"
    trials = 1000

    oracle_answer = answer_question(
        "oracle", target_fact, action, true_init_facts, current_facts
    )
    noisy_answers = [
        answer_question("noisy-oracle", target_fact, action, true_init_facts, current_facts)
        for _ in range(trials)
    ]
    human_proxy_answers = [
        answer_question("human-proxy", target_fact, action, true_init_facts, current_facts)
        for _ in range(trials)
    ]

    noisy_flip_count = sum(answer != oracle_answer for answer in noisy_answers)
    human_proxy_flip_count = sum(answer != oracle_answer for answer in human_proxy_answers)

    print(f"oracle: {target_fact} after {action} -> {oracle_answer}")
    print(
        f"noisy-oracle: flips={noisy_flip_count}/{trials} "
        f"rate={noisy_flip_count / trials:.3f} "
        f"configured_error_rate={NOISY_ORACLE_ERROR_RATE:.3f}"
    )
    print(
        f"human-proxy: flips={human_proxy_flip_count}/{trials} "
        f"rate={human_proxy_flip_count / trials:.3f}"
    )
