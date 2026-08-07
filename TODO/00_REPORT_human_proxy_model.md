# Human-Proxy 응답 모델 구현 보고서

## 목적

이 보고서는 로봇이 부분 관측 환경에서 인간에게 선제적으로 질의하고, 그 응답을 planning에 반영하는 연구에서 사용할 human-proxy 응답 모델의 구현 방향을 정리한다. 여기서 human-proxy는 oracle처럼 항상 정답을 아는 자동 응답자가 아니고, noisy-oracle처럼 일정 확률로 정답을 뒤집는 단순 노이즈 모델도 아니다. 목표는 실제 인간이 yes/no 질문에 답할 때 보이는 오류 패턴을 최대한 가깝게 모사하는 것이다.

현재 질의는 크게 두 종류로 나뉜다. 하나는 “이 토마토가 익었나요?”처럼 세계 상태나 객체 속성에 대한 상태 질의이고, 다른 하나는 “줄기2로 이동할까요?”처럼 특정 행동의 적절성을 묻는 행동 질의다. 두 질문은 모두 yes/no 이진 응답을 요구하지만, 인간이 답을 생성하는 방식은 다르다. 상태 질의는 지각적 판단에 가깝고, 행동 질의는 현재 목표와 상황을 해석한 뒤 특정 행동의 유용성을 판단하는 의사결정 문제에 가깝다. 따라서 가장 합리적인 구현은 하나의 전역 오류율을 모든 질문에 적용하는 방식이 아니라, 질문 유형에 따라 서로 다른 응답 생성 모델을 사용하는 방식이다.

## 기존 비교축의 의미

oracle은 숨겨진 true state와 goal까지 가는 heuristic을 알고 있으며, 항상 planning success에 가장 유리한 정답을 말하는 모델이다. 이 모델은 인간을 모사하기 위한 모델이라기보다 상한선 역할을 한다. 즉, 인간 응답이 완벽하다면 질의 기반 planning이 어느 정도까지 좋아질 수 있는지를 보여주는 upper bound다.

noisy-oracle은 oracle의 답을 일정 확률로 뒤집는 모델이다. 예를 들어 오답률이 10%라면 모든 질문에서 동일하게 10% 확률로 반대 답을 낸다. 이 모델은 단순하고 비교축으로는 유용하지만, 인간 응답의 구조를 잘 설명하지 못한다. 실제 인간은 모든 질문에서 같은 확률로 틀리지 않는다. 쉬운 질문에서는 거의 맞히고, 애매한 질문에서는 더 자주 틀리며, 어떤 사람은 yes 쪽으로 더 쉽게 기울고, 어떤 사람은 더 보수적으로 no를 선택할 수 있다.

human-proxy는 이 차이를 반영해야 한다. 즉, human-proxy는 “oracle answer에 랜덤 flip noise를 넣은 모델”이 아니라, “인간이 실제로 볼 수 있는 정보로부터 내부 evidence를 만들고, 그 evidence의 강도와 질문 난이도에 따라 확률적으로 yes/no를 선택하는 모델”이어야 한다. 이 관점은 HOP-POMDP 계열에서 사람을 관측 제공자로 모델링하면서 사람의 availability, accuracy, interruption cost를 고려한 흐름과도 맞닿아 있다 [1,2]. 또한 crowdsourcing과 noisy annotator 문헌에서 인간 응답자를 confusion matrix나 sensitivity/specificity로 모델링하는 접근과도 연결된다 [3,4].

## 핵심 설계 원칙

human-proxy의 가장 중요한 원칙은 인간이 true state를 직접 보지 않는다는 것이다. 시뮬레이터 내부에는 true state가 존재하더라도, human-proxy가 그 값을 직접 참조하면 oracle과의 경계가 무너진다. 대신 human-proxy는 실제 user study에서 인간에게 제공되는 정보, 예를 들어 카메라 이미지, bounding box, 탐지 결과, 현재 위치, 로봇의 최근 행동, 인터페이스에서 강조된 객체, 질문 문장만을 기반으로 응답해야 한다.

두 번째 원칙은 질문 난이도를 명시적으로 반영하는 것이다. 같은 yes/no 질문이라도 난이도는 다르다. 토마토가 명확히 빨갛고 가까이 보이면 “익었나요?”는 쉬운 질문이다. 반대로 색이 애매하거나 가려져 있거나, 탐지 confidence가 낮거나, ripe와 rotten이 시각적으로 비슷하면 어려운 질문이다. Item Response Theory는 응답자의 능력과 문항 난이도에 따라 정답 확률이 달라진다는 관점을 제공하며, 이 구조는 human-proxy에서 질문별 difficulty를 넣는 근거가 된다 [5,6].

세 번째 원칙은 사람별 편향과 민감도를 분리하는 것이다. 어떤 사람은 조금만 빨개도 익었다고 답하고, 어떤 사람은 확실히 익은 경우에만 yes라고 답한다. Signal Detection Theory는 yes/no 판단을 sensitivity와 response criterion으로 분해한다는 점에서 상태 질의 모델링에 특히 적합하다 [7,8]. 이를 적용하면 human-proxy는 단순히 “정확도 90%인 사람”이 아니라, “잘 구분하지만 보수적인 사람”, “구분 능력은 낮지만 yes를 쉽게 말하는 사람”처럼 더 현실적인 응답자를 생성할 수 있다.

네 번째 원칙은 상태 질의와 행동 질의를 분리하는 것이다. “이 토마토가 익었나요?”는 지각적 evidence가 기준을 넘는지 판단하는 문제다. 반면 “줄기2로 이동할까요?”는 특정 행동이 현재 목표 달성에 얼마나 좋아 보이는지 판단하는 문제다. 행동 질의에는 Boltzmann-rational 또는 noisy-rational choice model이 더 자연스럽다. 이 계열은 인간이 항상 최적 행동을 고르는 완전 합리적 agent가 아니라, 더 좋아 보이는 행동을 더 높은 확률로 선택하는 bounded-rational agent라고 본다 [9,10].

## 상태 질의 모델

상태 질의는 객체 속성이나 predicate의 참거짓을 묻는다. 예를 들어 “이 토마토가 익었나요?”, “이 물체가 캔인가요?”, “이 객체가 이미 처리되었나요?” 같은 질문이다. 이 경우 human-proxy는 해당 fact에 대한 내부 evidence를 만들고, 그 evidence가 인간의 판단 기준을 넘는지에 따라 yes/no를 생성한다.

가장 기본적인 형태는 다음과 같다.

```text
P_h(Yes | q_state) = sigmoid(α_h · e(q) - c_h)
```

여기서 `e(q)`는 질문 대상에 대해 인간이 관찰할 수 있는 evidence다. 토마토 도메인에서는 색상, ripeness score, detector confidence, 거리, occlusion 정도, 조명, bounding box 품질 등이 evidence가 될 수 있다. 폐기물 분류 도메인에서는 객체의 시각적 형태, 카테고리 classifier confidence, occlusion, object size, view angle 등이 evidence가 될 수 있다.

`α_h`는 인간의 sensitivity다. 값이 클수록 evidence 차이에 민감하게 반응한다. 즉, 조금만 단서가 강해져도 yes/no 확률이 빠르게 바뀐다. 값이 작으면 evidence가 달라져도 응답 확률이 완만하게 변하고, 어려운 질문에서 50:50에 가까운 답을 더 많이 낸다.

`c_h`는 response criterion이다. 값이 높으면 보수적인 사람이다. 즉, 충분히 강한 evidence가 있어야 yes라고 답한다. 값이 낮으면 yes 쪽으로 쉽게 기운다. 이 criterion은 인간 개인차를 표현하는 데 중요하다. 특히 “익었나요?” 같은 질문에서는 false positive와 false negative의 비용을 사람이 어떻게 느끼는지에 따라 criterion이 달라질 수 있다.

이 모델은 Signal Detection Theory의 구조와 잘 맞는다. SDT에서는 인간의 관측이 내부 decision variable로 변환되고, 그 값이 criterion을 넘으면 “signal present”, 즉 yes라고 답한다. sensitivity와 criterion을 분리하면, 실제 응답률 차이가 지각 능력 때문인지, 응답 편향 때문인지 구분할 수 있다 [7,8].

## 질문 난이도 반영

위 모델에 질문 난이도를 명시적으로 넣으면 더 human-like한 proxy가 된다. 가장 단순한 형태는 다음과 같다.

```text
P_h(Correct | q_state) = sigmoid(a_h - d(q))
```

여기서 `a_h`는 응답자의 ability이고, `d(q)`는 질문 난이도다. 이 구조는 IRT 또는 Rasch model과 유사하다. IRT에서는 문항 난이도가 높아질수록 같은 능력을 가진 응답자의 정답 확률이 낮아진다 [5,6].

로봇 질의에서는 `d(q)`를 다음과 같은 feature로 계산할 수 있다.

```text
d(q) = w_1 · entropy(q) + w_2 · occlusion(q) + w_3 · distance(q)
     + w_4 · visual_ambiguity(q) - w_5 · interface_highlight(q)
```

여기서 entropy는 로봇의 belief entropy나 classifier entropy일 수 있다. occlusion과 distance는 사람이 화면에서 대상을 식별하기 어려운 정도를 나타낸다. visual ambiguity는 ripe/unripe/rotten처럼 시각적으로 비슷한 클래스 간 혼동 가능성을 나타낸다. interface_highlight는 bounding box, 확대 이미지, 설명 텍스트처럼 사용자가 질문 대상을 더 쉽게 이해하도록 돕는 요소다.

이 방식의 장점은 noisy-oracle과 명확히 구분된다는 것이다. noisy-oracle은 모든 질문에서 같은 확률로 틀리지만, difficulty-aware human-proxy는 쉬운 질문에서는 oracle에 가깝고 어려운 질문에서는 50:50에 가까워진다. 따라서 “질문의 난이도와 지각적 모호성 때문에 인간 응답 품질이 달라진다”는 현상을 직접 반영할 수 있다.

## 비대칭 오류와 개인차

상태 질의에서 yes와 no 오류는 대칭적이지 않을 수 있다. 예를 들어 어떤 사용자는 애매한 토마토를 익었다고 과하게 판단할 수 있고, 다른 사용자는 익은 토마토도 확실하지 않으면 아니라고 답할 수 있다. 이런 경우 단일 오답률보다 confusion matrix가 더 적절하다.

이진 질문에서는 다음과 같이 모델링할 수 있다.

```text
P_h(Yes | true = 1) = sensitivity_h
P_h(No  | true = 0) = specificity_h
```

Dawid-Skene 모델은 true label이 직접 관측되지 않는 상황에서 관찰자별 error rate를 추정하는 고전적 접근이고, Raykar et al.은 binary classification에서 annotator의 sensitivity와 specificity를 명시적으로 모델링한다 [3,4]. 이 구조를 human-proxy에 적용하면, 사람별로 false positive 성향과 false negative 성향을 다르게 줄 수 있다.

다만 구현에서 주의할 점은 human-proxy가 true state를 직접 보고 sensitivity/specificity만으로 답하게 만들면 너무 oracle-like해질 수 있다는 것이다. 따라서 confusion matrix는 최종 응답 샘플링 단계의 보정항으로 쓰는 것이 좋다. 즉, 먼저 interface observation으로부터 evidence와 difficulty를 계산하고, 그 결과로 나온 yes 확률에 사용자별 bias를 반영하는 방식이 더 자연스럽다.

실제 구현에서는 사용자 타입을 몇 개의 latent profile로 나눌 수 있다. 예를 들어 cautious user는 yes criterion이 높고 false negative가 많다. liberal user는 criterion이 낮고 false positive가 많다. uncertain user는 sensitivity가 낮아서 어려운 질문에서 50:50에 가까워진다. expert user는 sensitivity가 높고 difficulty 증가에도 정답률이 덜 떨어진다.

## 행동 질의 모델

행동 질의는 “줄기2로 이동할까요?”, “이 물체를 먼저 집을까요?”, “지금 스캔할까요?”처럼 특정 action이 현재 상황에서 적절한지 묻는 질문이다. 이 질문은 상태 질의와 다르게 명확한 perceptual evidence만으로 답하기 어렵다. 인간은 현재 목표, 관측된 상태, 로봇의 최근 행동, 질문 대상, 가능한 대안 행동을 종합해서 판단한다.

행동 질의에는 다음과 같은 bounded-rational choice model을 사용할 수 있다.

```text
P_h(Yes | q_action) = sigmoid(β_h · ΔQ_h(b_h, a) + b_h^0)
```

여기서 `a`는 질문된 action이고, `b_h`는 인간이 interface를 보고 형성한 approximate belief다. `ΔQ_h(b_h, a)`는 인간 관점에서 해당 action이 대안보다 얼마나 좋아 보이는지를 나타낸다. 예를 들어 다음과 같이 정의할 수 있다.

```text
ΔQ_h(b_h, a) = Q_h(b_h, a) - max_{a' ≠ a} Q_h(b_h, a')
```

`β_h`는 rationality 또는 inverse temperature다. 값이 크면 인간이 action value 차이에 민감하게 반응하고, 가장 좋아 보이는 행동에 거의 deterministic하게 yes라고 답한다. 값이 작으면 행동 가치 차이가 있어도 응답이 더 noisy해진다. `b_h^0`는 yes/no 응답 편향이다.

중요한 점은 `Q_h`를 robot planner의 실제 Q-value로 두면 안 된다는 것이다. 그러면 human-proxy가 planner 내부 heuristic을 아는 oracle에 가까워진다. human-proxy의 `Q_h`는 인간이 볼 수 있는 정보만으로 계산한 approximate action value여야 한다. 예를 들어 인간이 화면에서 본 현재 위치, 탐지된 객체 수, 질문 대상의 상태, 최근 실패 여부, goal 설명 정도만 이용해야 한다.

Boltzmann-rational 또는 noisy-rational 모델은 인간이 항상 최적 행동을 고르는 것이 아니라, 더 높은 가치의 행동을 더 높은 확률로 선택한다고 가정한다 [9,10]. 다만 인간 행동이 항상 noisy optimality로만 설명되는 것은 아니며, 위험 회피나 편향 때문에 expected value가 낮은 선택을 선호할 수 있다는 비판도 있다 [11]. 따라서 행동 질의에서 risk나 failure cost가 큰 경우에는 risk-aware term을 추가할 수 있다.

```text
Q_h(b_h, a) = expected_progress(a) - λ_h · perceived_risk(a) - κ_h · effort(a)
```

이렇게 하면 사람은 단순히 goal progress가 큰 행동을 고르는 것이 아니라, 실패 위험이 커 보이는 행동을 피하거나, 복잡한 행동보다 안전하고 단순한 행동을 선호하는 경향을 표현할 수 있다.

## 통합 human-proxy 구조

최종 human-proxy는 다음과 같이 구성하는 것이 가장 합리적이다.

```text
Input:
  query q
  query type τ(q) ∈ {state, action}
  interface observation o_UI
  robot-visible features f(q, o_UI)
  user parameters θ_h

If τ(q) = state:
  evidence e(q) 계산
  difficulty d(q) 계산
  P_h(Yes) = sigmoid(α_h · e(q) - c_h - γ_h · d(q))

If τ(q) = action:
  human approximate belief b_h 구성
  approximate action value Q_h(b_h, a) 계산
  ΔQ_h 계산
  P_h(Yes) = sigmoid(β_h · ΔQ_h + b_h^0)

Return:
  answer ~ Bernoulli(P_h(Yes))
```

이 구조는 oracle, noisy-oracle, human-proxy의 차이를 명확히 만든다. oracle은 true state와 optimal heuristic을 직접 알고 답한다. noisy-oracle은 oracle 답을 일정 확률로 뒤집는다. human-proxy는 true state가 아니라 interface observation을 보고, 질문 난이도와 인간 개인차를 반영해 확률적으로 답한다.

## 파라미터 설정 방법

user study 데이터가 없다면, 처음에는 합리적인 prior를 두고 ablation을 하는 방식이 좋다. 예를 들어 sensitivity `α_h`는 log-normal 또는 normal distribution에서 샘플링하고, criterion `c_h`는 사용자 타입별로 다르게 둔다. cautious profile은 criterion을 높게, liberal profile은 낮게 둔다. difficulty coefficient `γ_h`는 질문 난이도가 증가할 때 정답률이 얼마나 빠르게 떨어지는지를 조절한다.

user study 데이터를 일부 확보할 수 있다면, 상태 질의에 대해서는 logistic regression 또는 hierarchical Bayesian logistic model로 `P(Yes | features, user)`를 fit할 수 있다. 이때 feature는 classifier confidence, entropy, occlusion, distance, target class, interface condition 등이 될 수 있다. 행동 질의에 대해서는 `P(Yes | ΔQ_h, risk, progress, user)` 형태로 fit하면 된다.

데이터가 적다면 완전히 복잡한 모델보다 profile-based model이 낫다. 예를 들어 사용자 12명의 응답 데이터를 직접 개인별 모델로 학습하기에는 부족할 수 있다. 이 경우 cautious, balanced, liberal, uncertain 같은 profile을 정의하고, 각 profile의 파라미터를 bootstrap 또는 maximum likelihood로 맞추는 것이 안정적이다.

## 구현 예시

상태 질의의 간단한 구현은 다음과 같다.

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def state_query_answer(features, user):
    # features: robot/user interface에서 관측 가능한 값만 사용
    # confidence: classifier confidence in [0, 1]
    # entropy: normalized entropy in [0, 1]
    # occlusion: occlusion ratio in [0, 1]
    # distance: normalized distance in [0, 1]
    # visual_score: yes 쪽 evidence. 예: ripeness score, category score

    evidence = features["visual_score"]
    difficulty = (
        user["w_entropy"] * features["entropy"] +
        user["w_occlusion"] * features["occlusion"] +
        user["w_distance"] * features["distance"] -
        user["w_highlight"] * features.get("highlight", 0.0)
    )

    logit_yes = user["alpha"] * evidence - user["criterion"] - user["gamma"] * difficulty
    p_yes = sigmoid(logit_yes)
    return np.random.rand() < p_yes, p_yes
```

행동 질의의 간단한 구현은 다음과 같다.

```python
import numpy as np


def action_query_answer(action_features, user):
    # action_features는 human-proxy가 볼 수 있는 정보에서 계산한 값만 사용
    # progress: 해당 행동이 goal progress에 기여하는 정도
    # risk: 실패하거나 잘못된 상태로 갈 위험
    # effort: 인간이 보기에 복잡하거나 불필요해 보이는 정도
    # best_alt_value: 대안 행동 중 가장 좋아 보이는 값

    q_action = (
        action_features["progress"]
        - user["lambda_risk"] * action_features["risk"]
        - user["kappa_effort"] * action_features["effort"]
    )

    delta_q = q_action - action_features["best_alt_value"]
    p_yes = sigmoid(user["beta"] * delta_q + user["action_bias"])
    return np.random.rand() < p_yes, p_yes
```

이 예시는 단순하지만, 핵심 원칙을 지킨다. 상태 질의는 perceptual evidence와 difficulty로 답하고, 행동 질의는 human-visible approximate action value로 답한다. 두 경우 모두 true hidden state나 planner 내부 optimal heuristic을 직접 사용하지 않는다.

## 실험에서의 비교 방식

실험에서는 oracle, noisy-oracle, human-proxy, human을 같은 query policy에 연결해서 비교할 수 있다. 중요한 것은 success rate만 보는 것이 아니라, human-proxy가 실제 인간과 비슷한 응답 분포를 만드는지도 확인하는 것이다.

비교 지표는 다음과 같이 잡을 수 있다. 첫째, query-level accuracy다. human-proxy와 실제 human이 같은 질문에서 oracle answer와 얼마나 다르게 답하는지 비교한다. 둘째, yes-rate다. human-proxy가 실제 인간처럼 yes를 더 많이 또는 더 적게 말하는 편향을 재현하는지 본다. 셋째, difficulty-accuracy curve다. 질문 난이도가 높아질수록 실제 인간의 정답률이 떨어지는 경향이 human-proxy에서도 재현되는지 확인한다. 넷째, planning success와 query count다. human-proxy를 붙였을 때의 planning 결과가 실제 human 조건과 비슷한지 본다. 다섯째, condition-level ranking이다. 예를 들어 oracle에서는 특정 policy가 가장 좋지만 human에서는 다른 policy가 더 안정적일 수 있다. human-proxy가 이 순위를 얼마나 잘 재현하는지가 중요하다.

## 추천 구현 단계

처음부터 복잡한 cognitive model을 완성하려고 하기보다, 세 단계로 구현하는 것이 좋다.

첫 단계는 difficulty-aware noisy responder다. 기존 noisy-oracle을 확장해서 질문별 difficulty에 따라 오답률이 달라지게 한다. 이 단계만으로도 fixed x% flip noise보다 훨씬 현실적이다.

두 번째 단계는 상태 질의와 행동 질의를 분리하는 것이다. 상태 질의에는 SDT-style sigmoid model을 쓰고, 행동 질의에는 bounded-rational action model을 쓴다. 이 단계가 실제 논문에서 human-proxy의 핵심 구현으로 가장 적절하다.

세 번째 단계는 user study 데이터로 보정하는 것이다. 실제 인간 응답을 수집한 뒤, user profile 또는 hierarchical model을 fit해서 human-proxy의 파라미터를 조정한다. 이 단계까지 가면 human-proxy가 단순 시뮬레이션 가정이 아니라, 실제 인간 응답 분포를 근사하는 모델이라고 주장할 수 있다.

## 결론

가장 합리적인 human-proxy는 단일 noise rate를 갖는 모델이 아니다. 상태 질의와 행동 질의의 생성 과정을 분리하고, 각 질문에서 인간이 볼 수 있는 정보만으로 evidence 또는 approximate action value를 계산한 뒤, 질문 난이도와 개인차를 반영해 yes/no를 확률적으로 샘플링하는 모델이 가장 적절하다.

상태 질의는 Signal Detection Theory와 IRT를 결합한 형태가 자연스럽다. 즉, 인간은 perceptual evidence를 보고 criterion을 넘으면 yes라고 답하며, 질문 난이도가 높을수록 정답률이 낮아진다. 행동 질의는 Boltzmann-rational 또는 noisy-rational choice model이 자연스럽다. 즉, 인간은 특정 행동이 대안보다 좋아 보일수록 yes라고 답하지만, 그 판단은 planner의 hidden Q-value가 아니라 인간이 interface에서 볼 수 있는 정보에 기반해야 한다.

이 구조를 사용하면 oracle, noisy-oracle, human-proxy, human의 차이가 명확해진다. oracle은 완전 정보 기반 upper bound이고, noisy-oracle은 단순 오류 비교축이며, human-proxy는 실제 인간 응답의 조건부 오류와 판단 편향을 모사하는 모델이다. 따라서 논문에서는 human-proxy를 “human-like automatic responder”로 제시하면서, fixed-noise baseline보다 실제 human condition을 더 잘 예측하는지를 검증하는 방향으로 설계하는 것이 가장 설득력 있다.

## References

[1] S. Rosenthal and M. Veloso, “Modeling Humans as Observation Providers using POMDPs,” RO-MAN, 2011. https://www.rosenthalphd.com/papers/Rosenthal_ROMAN11.pdf

[2] S. Rosenthal, M. Veloso, and A. K. Dey, “Learning Accuracy and Availability of Humans Who Help Mobile Robots,” AAAI, 2011. https://ojs.aaai.org/index.php/AAAI/article/view/7980

[3] A. P. Dawid and A. M. Skene, “Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm,” Applied Statistics, 1979. https://www.jstor.org/stable/2346806

[4] V. C. Raykar et al., “Learning From Crowds,” Journal of Machine Learning Research, 2010. https://jmlr.csail.mit.edu/papers/v11/raykar10a.html

[5] R. D. Hays, L. S. Morales, and S. P. Reise, “Item Response Theory and Health Outcomes Measurement in the 21st Century,” Medical Care, 2000. https://pmc.ncbi.nlm.nih.gov/articles/PMC1815384/

[6] Columbia University Mailman School of Public Health, “Item Response Theory.” https://www.publichealth.columbia.edu/research/population-health-methods/item-response-theory

[7] N. O. Macmillan and C. D. Creelman, *Detection Theory: A User’s Guide*, 2nd ed., 2004.

[8] M. S. Landy, “Signal Detection Theory,” NYU lecture notes. https://www.cns.nyu.edu/~eero/math-tools24/Handouts/sdtchapter.pdf

[9] A. D. Dragan, “Robot Planning with Mathematical Models of Human State and Action,” arXiv, 2017. https://arxiv.org/abs/1705.04226

[10] C. Laidlaw and A. Dragan, “The Boltzmann Policy Distribution: Accounting for Systematic Suboptimality in Human Models,” ICLR, 2022. https://openreview.net/pdf?id=_l_QjPGN5ye

[11] M. Kwon, E. Biyik, A. Talati, K. Bhasin, D. P. Losey, and D. Sadigh, “When Humans Aren’t Optimal: Robots that Collaborate with Risk-Aware Humans,” HRI, 2020. https://arxiv.org/abs/2001.04377
