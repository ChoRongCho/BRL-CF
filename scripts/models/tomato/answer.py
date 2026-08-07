from __future__ import annotations

import math

import numpy as np


NOISY_ORACLE_ERROR_RATE = 0.2

# Keep these action-result probabilities aligned with TransitionTomato.
NAVIGATE_SUCCESS_RATE = 0.90
PICK_SUCCESS_RATE = 0.95
PLACE_SUCCESS_RATE = 0.99
DISCARD_SUCCESS_RATE = 0.99

DYNAMIC_PREDICATES = (
    "located(",
    "holding(",
    "handempty(",
    "at(",
    "loaded(",
    "discarded(",
    "holded(",
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
    # Visual perception during detect.
    "detect_visible_ripe": 8,  # detect 대상 stem에 실제로 있는 토마토가 ripe인지 묻는 경우
    "detect_visible_unripe": 8,   # detect 대상 stem에 실제로 있는 토마토가 unripe인지 묻는 경우
    "detect_visible_rotten_as_ripe": 8, # detect 대상 stem에 있는 rotten 토마토를 ripe로 볼 수 있는 경우
    "detect_absent_ripeness": 9,  # detect 대상 stem에 없는 토마토의 ripe/unripe 여부를 묻는 경우
    "detect_visible_location": 5, # 실제로 존재하는 at(tomato, stem) 위치 사실을 묻는 경우
    "detect_absent_location": 5,  # 실제로 존재하지 않는 at(tomato, stem) 위치 사실을 묻는 경우
    
    # Direct scan of a known tomato.
    "scan_ripeness": 4,
    "scan_state": 4,
    # Physical state after robot manipulation.
    "pick_handempty": 1,
    "pick_holding": 1,
    "place_discard_handempty": 2,
    "place_discard_holding": 2,
    "navigate_location": 3,
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
        raise ValueError(f"Unsupported tomato answer mode: {mode}")


def _answer_oracle(target_fact: str, 
                   action: str, 
                   true_init_facts: set[str],
                   current_facts: set[str]) -> bool: 
    
    if action.startswith("detect("):
        # Extract the tomato name
        stem_name = action[action.rfind(",") + 1:-1].strip()
        if target_fact.startswith("ripe("): # ripe(tomatoN)
            tomato_name = target_fact[len("ripe("):-1]
            tomato_name_loc = f"at({tomato_name},{stem_name})"
            if tomato_name_loc in true_init_facts:
                if f"rotten({tomato_name})" in true_init_facts:
                    return True
                return target_fact in true_init_facts
            return False

        if target_fact.startswith("unripe("): # unripe(tomatoN)
            tomato_name = target_fact[len("unripe("):-1]
            tomato_name_loc = f"at({tomato_name},{stem_name})"
            if tomato_name_loc in true_init_facts:
                return target_fact in true_init_facts
            return False

        if target_fact.startswith("at("):
            return target_fact in true_init_facts

    if action.startswith("scan("):
        return target_fact in true_init_facts

    if action.startswith("navigate("):
        robot, _, target_location = _action_args(action)
        target_located = f"located({robot},{target_location})"
        if target_fact == target_located:
            return _sample_true(NAVIGATE_SUCCESS_RATE)
        if target_fact.startswith(f"located({robot},"):
            return _sample_true(1.0 - NAVIGATE_SUCCESS_RATE)

    if action.startswith("pick("):
        robot, tomato, stem = _action_args(action)
        if target_fact in {
            f"holding({robot},{tomato})",
            f"holded({tomato},{robot})",
        }:
            return _sample_true(PICK_SUCCESS_RATE)
        if target_fact in {
            f"handempty({robot})",
            f"at({tomato},{stem})",
        }:
            return _sample_true(1.0 - PICK_SUCCESS_RATE)

    if action.startswith("place("):
        robot, tomato = _action_args(action)
        if target_fact in {
            f"loaded({tomato},{robot})",
            f"handempty({robot})",
        }:
            return _sample_true(PLACE_SUCCESS_RATE)
        if target_fact in {
            f"holding({robot},{tomato})",
            f"holded({tomato},{robot})",
        }:
            return _sample_true(1.0 - PLACE_SUCCESS_RATE)

    if action.startswith("discard("):
        robot, tomato = _action_args(action)
        if target_fact in {
            f"discarded({tomato})",
            f"handempty({robot})",
        }:
            return _sample_true(DISCARD_SUCCESS_RATE)
        if target_fact in {
            f"holding({robot},{tomato})",
            f"holded({tomato},{robot})",
        }:
            return _sample_true(1.0 - DISCARD_SUCCESS_RATE)

    if target_fact.startswith(DYNAMIC_PREDICATES):
        return target_fact in current_facts

    return target_fact in true_init_facts


def _action_args(action: str) -> list[str]:
    start = action.find("(")
    end = action.rfind(")")
    if start < 0 or end <= start:
        return []
    return [arg.strip() for arg in action[start + 1:end].split(",")]


def _sample_true(probability: float) -> bool:
    return bool(np.random.random() < probability)
    
    
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
    - 명확한 task state보다 visual perception에 의존하는 질문에서 주로 틀리게 만든다.
    - tomato 도메인에서 사람이 헷갈릴 만한 경우는 다음과 같다.
      1. detect 단계의 숙성도 질문: ripe(T), unripe(T).
      2. 실제로 queried stem에 없는 tomato가 약한 detection처럼 보이는 경우.
      3. rotten tomato를 detect 단계에서 ripe로 해석할 수 있는 경우.
      4. tomato identity 또는 location이 낮은 confidence로 애매하게 관측된 경우.
    - pick/place/discard 이후의 물리 상태 질문은 우선 noise 대상에서 제외한다.
      gripper 상태 혼동을 명시적으로 모델링할 때만 포함한다.
    """
    
    oracle_answer = _answer_oracle(target_fact, action, true_init_facts, current_facts)
    difficulty = _human_proxy_difficulty(target_fact, action, true_init_facts)
    answer = _sample_human_proxy_answer(oracle_answer, difficulty)
    return answer


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
                            true_init_facts: set[str]) -> float:
    situation = _human_proxy_situation(target_fact, action, true_init_facts)
    return HUMAN_PROXY_DIFFICULTY_BY_SITUATION[situation]


def _human_proxy_situation(target_fact: str,
                           action: str,
                           true_init_facts: set[str]) -> str:
    """
    Determine the situation for human proxy difficulty calculation.
    """
    
    if action.startswith("detect("):
        stem_name = action[action.rfind(",") + 1:-1].strip()

        if target_fact.startswith(("ripe(", "unripe(")):
            tomato_name = target_fact[target_fact.find("(") + 1:-1]
            tomato_at_detected_stem = f"at({tomato_name},{stem_name})" in true_init_facts
            if not tomato_at_detected_stem:
                return "detect_absent_ripeness"
            
            if target_fact.startswith("ripe(") and f"rotten({tomato_name})" in true_init_facts:
                return "detect_visible_rotten_as_ripe"
            if target_fact.startswith("ripe("):
                return "detect_visible_ripe"
            return "detect_visible_unripe"

        if target_fact.startswith("at("):
            if target_fact in true_init_facts:
                return "detect_visible_location"
            return "detect_absent_location"

    if action.startswith("scan("):
        if target_fact.startswith(("ripe(", "unripe(", "rotten(")):
            return "scan_ripeness"
        return "scan_state"

    if action.startswith(("place", "discard")):
        if target_fact.startswith("handempty("):
            return "place_discard_handempty"
        if target_fact.startswith("holding("):
            return "place_discard_holding"

    if action.startswith("pick"):
        if target_fact.startswith("handempty("):
            return "pick_handempty"
        if target_fact.startswith("holding("):
            return "pick_holding"

    if action.startswith("navigate") and target_fact.startswith("at("):
        return "navigate_location"

    return "default"


def _answer_human(target_fact: str, 
                  action: str, 
                  true_init_facts: set[str],
                  current_facts: set[str]) -> bool: 
    
    raise NotImplementedError("human mode must be handled by FeedbackManager.query_human().")


if __name__ == "__main__":
    true_init_facts = {
        "ripe(tomato1)",
        "at(tomato1,stem1)",
        "unripe(tomato2)",
        "at(tomato2,stem2)",
    }
    current_facts = set(true_init_facts)
    action = "detect(brl_robot,stem1)"
    target_fact = "ripe(tomato1)"
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
