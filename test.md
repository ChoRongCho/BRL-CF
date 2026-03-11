
# 부분 관측 가능 마르코프 결정 과정에서의 실행 모니터링 및 신뢰도 기반 능동적 정보 수집과 계획 수정 연구 보고서

## 1. 서론

부분 관측 가능 마르코프 결정 과정(**Partially Observable Markov Decision Process, POMDP**)은 환경의 상태를 직접적으로 알 수 없는 상황에서 최적의 의사결정을 내리기 위한 수학적 프레임워크를 제공한다.

전통적으로 POMDP 연구는 **누적 보상을 최대화하는 정책(policy)**을 도출하는 데 집중해 왔다. 그러나 현실의 복잡한 환경에서는 사전에 정의된 계획(**pre-defined plan**)을 실행하는 도중 발생하는 불확실성을 관리하는 것이 필수적이다.

특히 에이전트가 **계획된 액션 시퀀스를 수행하면서 필터링을 통해 신념 상태(belief state)를 업데이트하고**, 특정 시점에서 신뢰도가 낮다고 판단될 때 능동적으로 **정보 수집 행동(information gathering action)**을 삽입하여 신뢰도를 높인 뒤 계획을 수정하는 메커니즘은 로보틱스와 인공지능 분야에서 매우 중요한 연구 주제로 다뤄지고 있다.

본 보고서는 이러한 **신뢰도 기반 능동적 정보 수집 및 계획 수정**에 관한

* 이론적 배경
* 관련 연구
* 주요 알고리즘

을 상세히 분석한다.

---

# 2. 의사결정 모델로서의 POMDP와 신념 상태의 역할

POMDP는 다음의 7개 요소로 정의된다.

[
(S, A, T, R, Z, O, \gamma)
]

| 구성 요소    | 설명       |
| -------- | -------- |
| (S)      | 상태 공간    |
| (A)      | 액션 집합    |
| (T)      | 상태 전이 함수 |
| (R)      | 보상 함수    |
| (Z)      | 관측값      |
| (O)      | 관측 확률 함수 |
| (\gamma) | 할인 인자    |

에이전트는 실제 상태 (s)를 직접 알 수 없기 때문에, 과거의 **액션과 관측 이력**을 기반으로 현재 상태에 대한 확률 분포인 **신념 상태(belief state)**를 유지한다.

신념 상태는 다음과 같은 형태를 갖는다.

[
b(s) = P(s | history)
]

신념 상태는 의사결정에 필요한 모든 정보를 요약하는 **충분 통계량(Sufficient Statistic)**으로 작용하며, 에이전트는 이 신념 상태를 기반으로 다음 액션을 선택한다.

---

# 3. 신념 상태 업데이트와 필터링 메커니즘

에이전트가 액션 (a)를 수행하고 관측값 (z)를 얻었을 때 새로운 신념 상태는 **베이즈 정리(Bayes rule)**에 따라 업데이트된다.

[
b'(s') =
\eta \cdot O(s', a, z)
\sum_{s \in S} T(s,a,s') b(s)
]

여기서

* (O(s',a,z)) : 관측 모델
* (T(s,a,s')) : 상태 전이 모델
* (\eta) : 정규화 상수

이다.

이러한 **belief filtering 과정**은 에이전트가 실행 중인 계획의 유효성을 지속적으로 평가할 수 있게 한다.

예를 들어

* 관측값이 예상과 크게 다르거나
* belief 분포의 분산이 증가하면

이는 에이전트의 **현재 신뢰도(confidence)**가 낮아졌음을 의미한다.

이러한 상황은 **계획 수정(replanning)**의 트리거가 된다.

---

# 4. 신념 MDP(Belief MDP)로의 변환

POMDP 문제는 **신념 상태를 새로운 상태로 간주함으로써 MDP로 변환**될 수 있다.

이를 **Belief MDP**라고 한다.

Belief MDP는 다음과 같이 정의된다.

[
(B, A, \tau, r, \gamma)
]

| 구성 요소    | 설명              |
| -------- | --------------- |
| (B)      | belief 상태 공간    |
| (A)      | 액션 집합           |
| (\tau)   | belief 전이 함수    |
| (r)      | belief 기반 보상 함수 |
| (\gamma) | 할인 인자           |

보상 함수는 다음과 같이 정의된다.

[
\rho(b,a) = \sum_{s} b(s)R(s,a)
]

즉 **belief 분포에 대한 기대 보상**이다.

하지만 이러한 방식은

* 상태 기반 보상만 고려하기 때문에
* 순수한 **정보 수집 행동(information gathering)**을 장려하기 어렵다는 한계가 있다.

---

# 5. 능동적 정보 수집과 신념 의존적 보상 체계

## 능동적 정보 수집 (Active Sensing)

정보 수집 행동은

* 물리적 행동을 잠시 멈추고
* 상태의 불확실성을 줄이기 위해 수행되는 행동이다.

예

* look
* inspect
* scan
* ask question

---

## ρ-POMDP

최근 연구에서는 보상 함수가 belief에 직접 의존하도록 확장된 **ρ-POMDP** 모델이 제안되었다.

[
\rho(b,a)
]

이 보상 함수는 **belief의 임의 함수**가 될 수 있다.

예를 들어

### 음의 엔트로피 보상

[
R(b) = -H(b)
]

이 경우

* belief가 확실해질수록
* 높은 보상을 얻게 된다.

즉 에이전트는 **정보 수집 행동을 자연스럽게 선택**하게 된다.

ρ-POMDP에서도 가치 함수는 여전히 **Piecewise Linear and Convex (PWLC)** 구조를 유지한다.

따라서 기존의

* Point-Based Value Iteration

같은 알고리즘을 그대로 사용할 수 있다.

---

# 6. 능동적 감지의 트리거 메커니즘

정보 수집 행동은 다음 기준에 따라 트리거된다.

## 1. Information Gain

엔트로피 감소

[
IG = H(b) - H(b')
]

또는

**Mutual Information**

---

## 2. Confidence Threshold

[
\max_s b(s) < \tau
]

일 때 sensing action 수행

---

## 3. Value of Information

정보 수집의 가치가 행동 비용보다 클 때

---

# 7. 실행 모니터링과 계획 수정 연구

## Execution Monitoring

계획 실행 중

* belief 상태 감시
* 실패 가능성 평가

---

## Deterministic Approximation

POMDP 대신

* deterministic planner 사용
* execution 중 uncertainty 관리

---

# 8. Kaelbling의 HPN (Hierarchical Planning in the Now)

HPN은

* belief planning
* execution

을 계층적으로 통합한다.

각 액션에는 **pre-image**가 정의된다.

즉

액션이 성공하기 위한 belief 조건이다.

실행 중

[
b \notin Pre(a)
]

이면

* 실행 중단
* replanning 수행

---

# 9. Minlue Wang의 실행 모니터링 연구

논문

**Improving Robot Plans for Information Gathering Tasks through Execution Monitoring**

핵심 아이디어

* approximate policy 실행
* belief 상태 모니터링
* 필요시 policy 수정

---

# 10. 주요 알고리즘 비교

| 연구          | 모니터링 방식      | 계획 수정              | 정보 수집          |
| ----------- | ------------ | ------------------ | -------------- |
| HPN         | Pre-image 검사 | 계층적 재계획            | sensing action |
| Minlue Wang | belief 비교    | policy 수정          | sampling       |
| RBSR        | belief 업데이트  | open-loop planning | info-gain      |
| Act-Reason  | entropy 기반   | reasoning 전환       | sensing        |

---

# 11. Randomized Belief-Space Replanning (RBSR)

RBSR은

* belief 업데이트
* forward simulation
* replanning

을 반복하는 알고리즘이다.

특징

* QMDP heuristic
* particle filter
* receding horizon control

---

# 12. 신뢰도 기반 메타 인지 제어

## System 1 / System 2 구조

### System 1

* 높은 신뢰도
* 빠른 실행

### System 2

* 낮은 신뢰도
* reasoning 수행

---

## 신뢰도 기반 정책

| 영역                  | 정책                |
| ------------------- | ----------------- |
| (C_t > \tau_{high}) | aggressive action |
| 중간                  | exploration       |
| (C_t < \tau_{low})  | replanning        |

---

# 13. 적용 사례

## 로봇 내비게이션

위치 불확실

→ active sensing

---

## 물체 조작

occlusion 발생

→ peeking 행동

---

## Shared Autonomy

사용자 의도 불확실

→ 질문 수행

---

# 14. 결론

본 보고서는

* POMDP 기반 실행 모니터링
* 신뢰도 기반 정보 수집
* 계획 수정 전략

을 분석하였다.

특히

* **HPN**
* **Minlue Wang**
* **ρ-POMDP**

연구들은 신뢰도 기반 계획 수정 문제에 직접적인 해답을 제공한다.

향후 연구는

* LLM 기반 reasoning
* semantic uncertainty

와 결합되어 더욱 발전할 것으로 예상된다.

에이전트가

> **자신이 무엇을 모르는지 인식하고(Know when they don't know)**

정보를 수집하며 계획을 수정하는 능력은 **현실 세계에서 신뢰 가능한 AI**를 구현하는 핵심 요소가 될 것이다.

