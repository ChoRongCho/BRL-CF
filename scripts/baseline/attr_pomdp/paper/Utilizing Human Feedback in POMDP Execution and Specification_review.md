# Utilizing Human Feedback in POMDP Execution and Specification 리뷰

## 1. 논문 정보

```text
Janine Hoelscher, Dorothea Koert, Jan Peters, and Joni Pajarinen,
“Utilizing Human Feedback in POMDP Execution and Specification,”
IEEE-RAS 18th International Conference on Humanoid Robots,
2018, pp. 104–111.
```

- 원문: [Utilizing Human Feedback in POMDP Execution and Specification.pdf](<./Utilizing Human Feedback in POMDP Execution and Specification.pdf>)
- 핵심 주제: 사용자 목표 명세와 실행 중 targeted query를 결합한 POMDP
- 실험 환경: Hidden weight가 있는 box stacking
- 실제 로봇: 7-DOF KUKA LWR arm

## 2. 초록 번역

많은 환경에서 로봇은 부분적인 관측, 가려짐, 불확실성을 처리해야 한다. 이러한
환경에서는 행동을 계획하기 위한 방법으로 부분 관측 마르코프 의사결정 과정
(POMDP)이 널리 사용된다. 그러나 특히 비전문 사용자가 존재하는 환경에서는
POMDP를 인간 생활 환경에 광범위하게 적용하는 데 방해가 되는 여러 미해결 문제가
여전히 존재한다.

이를 해결하기 위해 본 논문에서는 상호 정보 교환을 가능하게 하는 새로운 접근법을
제안한다. 이 접근법은 다음 두 가지를 모두 다룬다.

1. 작업을 명세하는 과정에서 사용자의 목표를 반영하는 것
2. 작업을 수행하는 동안 사람에게 구체적인 정보를 질문하는 것

POMDP에서는 일반적으로 보상 함수를 사용하여 작업을 명세한다. 그러나 보상 함수를
적절하게 정의하는 일은 전문가에게도 어렵고, 비전문가에게는 더욱 부담스럽다. 이에
본 논문에서는 직관적인 논리 문장 형태로 정의된 작업의 성공 확률을 최대화하는
새로운 POMDP 알고리즘을 제시한다.

또한 로봇이 구체적인 정보를 요청할 수 있도록 POMDP 모델에 표적 질문(targeted
query)을 도입한다. 기존 접근법 대부분은 전체 상태 정보를 한꺼번에 요청하는데,
이는 사용자에게 큰 부담을 줄 수 있다. 이와 달리 제안 방법은 필요한 특정 정보만
질문하며, 기존 접근법보다 큰 상태공간에도 적용할 수 있다.

제안 방법은 시뮬레이션과 7자유도 KUKA LWR 로봇 팔을 사용한 상자 쌓기 작업에서
평가되었다. 실험 결과는 표적 질문을 사용하는 것이 작업 성능을 크게 향상하며,
로봇이 사용자가 정의한 작업 목표를 충족하면서 작업 성공 확률을 성공적으로
최대화한다는 것을 보여준다.

## 논문 Section II-B 번역: Related Work

로봇의 planning 성능을 향상하는 한 가지 전략은 사람에게 정보를 요청하는 것이다.
이 접근법은 예를 들어 강화학습 분야에서 이미 널리 확립되어 있다 [5]. POMDP
모델에서는 reward function [3], observation function [4] 또는 transition model [8]을
갱신하는 방식으로 새로운 정보를 포함할 수 있다. 하지만 이러한 방법들은 많은 양의
human feedback을 요구한다. 그 대신 새로운 정보를 현재 belief state에 직접 반영할
수도 있다. Armstrong et al. [1]은 시스템의 현재 상태를 묻는 추가 action
`a_oracle`을 도입했으며, 이 action으로 얻은 정보를 사용해 belief state를
초기화한다. 이 belief update에는 oracle의 정확성에 대한 uncertainty도 반영할 수
있다 [23].

그러나 사람에게 완전한 state information을 제공하도록 요구하는 것은 번거롭거나
심지어 불가능할 수 있다. 특정 state property를 대상으로 하는 query action을
추가하여 POMDP model을 확장하는 것 자체는 비교적 간단하지만, 그렇게 만들어진
POMDP policy를 최적화하는 것은 어렵다. Targeted question을 사용하려면 여러 개의
information-gathering action이 필요하므로 action space가 커지기 때문이다. 본
논문에서는 적절한 POMDP algorithm을 사용하면 실제로 targeted query를 계획할 수
있으며, 사용자의 수고를 최소화하면서도 robotic application의 성능을 크게 향상할
수 있음을 보인다.

사용자와 로봇 사이의 interaction을 향상하는 두 번째 방향은 사용자가 robotic task에
대한 자신의 objective를 자연스러운 방식으로 표현할 수 있게 하는 것이다. 기존
연구 [29], [13], [9], [17]는 사용자의 intention을 학습하기 위해 task execution 중에
제공되는 feedback으로부터 human goal specification을 수집한다. 그러나 이러한
방법에서는 사용자가 task execution에 계속 주의를 기울이면서 여러 번 개입해야 한다.
사용자가 reward function을 직접 정의하게 하는 방법 역시 잠재적으로 훈련받지 않은
사용자에게 여러 state의 reward와 cost를 정의하고 이들 사이의 균형을 조정하도록
요구한다.

사용자가 task objective를 지정하는 더 자연스러운 방법은 원하는 state property를
설명하는 logic sentence를 사용하는 것이며, 로봇은 해당 objective들을 만족할
확률을 최대화할 수 있다. 기존의 logic-based POMDP formulation은 reward를
최대화하거나 [24] logic sentence를 완전히 만족시키는 것을 목표로 한다 [6]. 이와
달리 본 논문의 접근법은 probabilistic Markovian dynamics와 observation model을
가정하면서, 모든 logic sentence를 만족할 확률을 최대화하는 것을 목표로 한다.
연구 [7], [14]는 POMDP system이 원하는 state-space 영역에 항상 머물 확률을
최대화한다.

본 논문에서는 사용자가 임의의 time-step 부분집합에 대해 다양한 종류의 task
requirement를 지정할 수 있는 더 일반적인 문제를 다룬다. 이를 위해 Policy Graph
Improvement(PGI) [20]와 Particle-based PGI(PPGI) [19]에 기반한 새로운 algorithm을
제안한다. PGI [20]는 고정된 크기의 policy graph를 반복적으로 개선하여 POMDP를
최적화하며, optimization time이 선형적으로 증가하도록 한다. PPGI [19]는 robotic
task에서 흔히 나타나는 large state space에도 확장할 수 있다 [18]. Section III에서는
PPGI를 기반으로 large state space에서 logic sentence의 성공 확률을 최대화하는
새로운 algorithm을 소개한다. 이에 비해 성공 확률 최대화를 다룬 기존 연구 [7],
[14]는 small-to-moderate state space에서만 동작하는 algorithm에 의존한다.

## 3. 연구 문제

논문은 인간 환경에서 POMDP를 사용하는 데 필요한 두 가지 정보 교환 방향을 다룬다.

```text
Human → Robot, planning 전:
  사용자가 원하는 task objective 전달

Robot → Human → Robot, execution 중:
  로봇이 불확실한 object property를 질문하고 answer를 observation으로 사용
```

기존 POMDP의 scalar reward는 비전문 사용자가 직접 설계하기 어렵다. 또한 기존의
oracle action은 전체 world state를 요구하므로 사람에게 불필요하게 많은 정보를
요청한다. 논문은 각각을 logic objective와 targeted query로 해결한다.

## 4. Targeted query

### 4.1 Object-property 표현

Object 하나를 여러 property의 곱으로 표현한다.

```text
obj = p1 × p2 × ... × pJ
```

예를 들어 box는 길이, 높이, hidden weight 위치, 현재 배치 위치 등의 property를
가진다.

### 4.2 Query action

Targeted query는 특정 object의 특정 property 하나를 요청한다.

```text
a_query = <obj_n, p_j>
```

예:

```text
<box1, weight_position>
<box2, weight_position>
```

전체 state를 반환하는 oracle과 달리 현재 task에 필요한 정보만 요청하므로 사람의
부담을 줄인다.

### 4.3 Transition과 observation

질문은 world state를 물리적으로 변경하지 않는 정보 획득 action이다.

```text
T(s' | s, a_query) = 1  if s' = s
```

질문 후 선택한 property의 실제 값이 observation으로 주어지고, planner는 이를 이용해
belief를 갱신한다.

```text
b'(s') ∝ O(answer | s', a_query)b(s')
```

논문 실험에서는 사람이 항상 정확하게 답한다고 가정한다. 하지만 POMDP observation
model을 사용하므로 noisy answer로 확장할 수 있다.

### 4.4 Query budget

Episode에서 사용할 수 있는 최대 질문 횟수를 설정할 수 있다. 한도를 소진한 뒤에는
사람에게 묻지 못하고 robot의 자체 information-gathering action만 사용해야 한다.

엄밀한 POMDP로 구현하려면 remaining query budget을 state에 포함해야 한다.

```text
augmented state = (world state, remaining query budget)
```

## 5. 사용자 목표 명세

비전문 사용자가 scalar reward를 직접 조정하는 대신 object property에 대한 직관적인
logic sentence를 선택한다.

예:

```text
파란 상자는 높이 3보다 위에 있어야 한다.
특정 상자는 다른 상자보다 아래에 있어야 한다.
모든 상자가 완성된 tower에 포함되어야 한다.
```

논문은 logic sentence의 만족 여부를 boolean 함수로 나타내고, task trajectory에서
이 조건들을 만족할 확률을 최대화한다.

```text
C_t(s_t, a_t) ∈ {0, 1}
```

조건은 전체 task 동안 유지되거나 특정 time interval 또는 terminal state에만
적용될 수 있다.

## 6. Solver

논문은 offline policy graph planning을 사용한다.

```text
PPGI:
  Particle-based Policy Graph Improvement

LPPGI:
  Logical Particle-based Policy Graph Improvement
```

LPPGI는 일반적인 누적 scalar reward 대신 사용자가 지정한 logic objective의 만족
확률을 최대화한다. Discount factor를 비전문 사용자가 조정하게 하는 대신, 이해하기
쉬운 최대 task step 수를 지정하도록 한다.

현재 Active Search의 online particle-belief expectimax와 solver는 다르지만 targeted
query의 action, transition, observation 정의는 독립적으로 적용할 수 있다.

## 7. Box stacking 실험

### 7.1 불확실성

종이 상자 안에 동전을 넣어 hidden weight를 만든다. 로봇은 상자 외부에서 무게 위치를
직접 관측할 수 없다.

```text
box_i = <length, height, hidden weight position, x, z>
```

5개 box 조건에서는 가능한 state가 약 933,120개다. Initial belief는 가능한 hidden
weight configuration에 대한 uniform distribution이다.

### 7.2 Physical action

로봇은 특정 box를 tower의 특정 위치에 놓는다.

```text
a = <box_i, target_position>
```

낮은 위치에 시험적으로 box를 놓고 넘어지는지 관찰하면 로봇 스스로 hidden weight에
대한 정보를 얻을 수 있다. 하지만 허용 높이보다 높은 곳에서 box가 떨어지면 task가
실패한다.

### 7.3 Human-query action

로봇은 특정 box의 hidden weight 위치를 사람에게 물을 수 있다.

```text
a_query = <box_i, weight>
```

따라서 planner는 다음 세 action 유형을 함께 비교한다.

```text
1. 사람에게 property 질문
2. 로봇의 exploratory physical action
3. Tower 완성을 위한 task action
```

## 8. 비교 실험과 결과

논문은 다음 조건들을 비교한다.

- MDP
- QMDP
- POMDP with PPGI
- Human interaction 없음
- 전체 state를 알려주는 oracle action
- 특정 property만 알려주는 targeted query

주요 결과:

- MDP는 partial observability를 추적하지 못해 불필요한 행동이 많았다.
- QMDP는 미래 information gathering의 가치를 충분히 평가하지 못했다.
- 실패 위험이 큰 낮은 낙하 허용 높이에서 POMDP의 장점이 커졌다.
- Oracle과 targeted query 모두 위험한 조건의 task success rate를 향상했다.
- Oracle은 전체 정보를 얻기 위해 모든 box에 대한 interaction을 요구했다.
- Targeted query는 task 난이도에 따라 필요한 질문 수를 조절했다.
- Planner는 사람에게 묻는 것과 robot 자체 exploratory action을 함께 사용했다.
- Targeted query가 action space를 늘렸지만 planning time 증가는 크지 않았다.
- 7-DOF KUKA LWR 실제 로봇에서도 계획과 질문을 실행했다.

## 9. 기여

1. 전체-state oracle 대신 object-property targeted query를 POMDP에 도입했다.
2. Human query, robot sensing, task action을 하나의 policy에서 선택하게 했다.
3. 비전문 사용자가 logic sentence로 task objective를 지정할 수 있게 했다.
4. Logic objective 만족 확률을 최대화하는 LPPGI를 제안했다.
5. Simulation과 실제 manipulator에서 human feedback의 효과를 검증했다.

## 10. 장점

- 현재 symbolic predicate query와 매우 잘 맞는 action 표현을 제공한다.
- 사람이 전체 state를 설명해야 하는 부담을 줄인다.
- 필요한 경우에만 질문하도록 planning한다.
- 질문과 자체 탐색 사이의 trade-off를 명시적으로 처리한다.
- Large state space에서 particle policy graph를 사용한다.
- 실제 로봇 평가가 포함되어 있다.

## 11. 한계

### 11.1 정확한 human answer 가정

사람이 항상 올바르게 답한다고 가정한다. 실제 적용에는 다음 확장이 필요하다.

```text
P(correct answer | s, query)
P(wrong answer | s, query)
P(no answer | s, query)
```

### 11.2 Human availability 부재

사람이 자리에 없거나 응답을 거부하는 상황을 모델링하지 않는다. 이 문제는
HOP-POMDP의 availability와 `o_null` observation을 결합해 처리할 수 있다.

### 11.3 단순화된 dynamics

실험의 transition과 observation은 주로 deterministic하다. Stochastic physical action과
noisy robot observation을 사용하는 domain에서는 추가 검증이 필요하다.

### 11.4 Offline planning

PPGI/LPPGI는 policy graph를 미리 계산한다. 실행 중 grounded action과 state가 크게
변하는 symbolic domain에는 online planner가 더 적합할 수 있다.

### 11.5 제한적인 사용자 평가

질문 수와 task 성능은 분석하지만 실제 사용자의 피로도, 응답 오류, interruption
cost에 대한 본격적인 user study는 future work로 남아 있다.

## 12. Tomato/WasteSorting 대응

논문의 targeted query를 현재 domain에 대응시키면 다음과 같다.

### Tomato

```text
<tomato, ripeness>  → ask_tomato_ripe_or_unripe
<tomato, condition> → ask_tomato_ripe_or_rotten
<tomato, location>  → ask_tomato_location
<tomato, loaded>    → ask_tomato_loaded
<tomato, discarded> → ask_tomato_discarded
```

### WasteSorting

```text
<waste, category> → ask_waste_plastic / can / paper / general
<waste, detected> → ask_waste_detected
<waste, bin>      → ask_waste_in_bin
```

## 13. 구현 요구사항

현재 YAML의 `type: ask`와 빈 effect만으로는 targeted-query POMDP가 완성되지 않는다.
다음 요소를 연결해야 한다.

```text
1. Action 객체에 ask/query type 보존
2. Query 후보를 physical action과 함께 planner에서 평가
3. T(s, query, s) = 1인 no-op transition
4. P(answer | s, query) observation model
5. 실제 human/oracle/noisy answer 실행기
6. Answer-conditioned Bayesian belief update
7. Query cost 또는 remaining query budget
8. 동일 질문 반복 방지
```

ρ-POMDP와 결합할 경우 reward는 다음처럼 구성할 수 있다.

```text
ρ(b, a)
  = expected task reward
  + λ_info × information reward
  - λ_query × query cost
```

## 14. 최종 평가

이 논문은 현재 구현하려는 “사람에게 특정 predicate를 물어보는 POMDP planner”의
가장 직접적인 기준 논문이다. 핵심적으로 차용할 부분은 다음 표현이다.

```text
a_query = <object, property>
```

다만 현재 목표는 논문의 완전한 재현과는 다르다.

```text
원 논문:
  Standard POMDP
  + targeted query
  + 정확한 human answer
  + offline PPGI/LPPGI

현재 프로젝트:
  ρ-POMDP
  + targeted symbolic query
  + oracle/noisy/human answer
  + online particle-belief planning
```

따라서 현재 방법은 다음처럼 설명하는 것이 정확하다.

```text
Human-in-the-loop ρ-POMDP with targeted query actions,
inspired by Hoelscher et al. (2018)
```
