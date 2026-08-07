# Learning Accuracy and Availability of Humans Who Help Mobile Robots 리뷰

## 1. 논문 정보

```text
Stephanie Rosenthal, Manuela Veloso, and Anind K. Dey,
“Learning Accuracy and Availability of Humans Who Help Mobile Robots,”
Proceedings of the Twenty-Fifth AAAI Conference on Artificial Intelligence,
pp. 1501–1506, 2011.
```

- 원문: [Learning Accuracy and Availability of Humans Who Help Mobile Robots.pdf](<./Learning Accuracy and Availability of Humans Who Help Mobile Robots.pdf>)
- 제안 모델: `HOP-POMDP` (Human Observation Provider POMDP)
- 학습 알고리즘: `LM-HOP` (Learning the Model of Humans as Observation Providers)

## 2. 한 문장 요약

HOP-POMDP는 주변 사람을 로봇에게 현재 상태에 관한 관측을 제공하는 센서로
모델링하고, 각 사람의 위치, 응답 가능성, 정확도와 질문 비용을 고려해 로봇이
자율 행동과 질문 중 무엇을 선택할지 계획한다. LM-HOP는 미리 알기 어려운 사람의
응답 가능성과 정확도를 task 실행 중에 학습한다.

## 3. 해결하려는 문제

로봇은 센서와 transition의 불확실성 때문에 자신의 현재 위치를 확신하지 못할 수
있다. 전용 supervisor에게 항상 도움을 받는 대신 건물 안에 원래 존재하는 사람에게
현재 위치를 물을 수 있지만, 이 사람들에게는 다음과 같은 제약이 있다.

- 정해진 위치에 있을 때만 접근할 수 있다.
- 바빠서 답하지 않을 수 있다.
- 답하더라도 틀릴 수 있다.
- 질문하고 방해하는 데 비용이 든다.

따라서 로봇은 최단 경로만 계산해서는 안 되고, 불확실해졌을 때 도움받을 가능성이
높은 사람을 만날 수 있는 경로까지 고려해야 한다.

이 논문에서 사람은 로봇에게 다음 행동을 지시하는 supervisor가 아니다. 로봇의
현재 상태를 관측해 알려주는 `observation provider`이고, 로봇은 받은 관측을 belief
update에 사용한 뒤 스스로 행동을 선택한다.

## 4. HOP-POMDP

논문은 HOP-POMDP를 다음과 같이 정의한다.

```text
HOP-POMDP = {Λ, S, α, η, A, O, Ω, T, R}
```

| 요소 | 의미 |
|---|---|
| `S` | 로봇의 가능한 상태 |
| `Λ={λ_s}` | 각 위치에 있는 사람에게 질문하는 비용 |
| `α={α_s}` | 각 사람의 availability |
| `η={η_s}` | 각 사람의 accuracy |
| `A` | 기존 자율 행동과 `ask` action |
| `O` | 기존 관측, 사람이 제공하는 상태 관측, `null` 관측 |
| `Ω` | 사람의 availability와 accuracy가 반영된 observation model |
| `T` | domain transition model |
| `R` | 기존 task reward와 질문 비용 |

기존 POMDP의 state, task action, transition과 reward를 유지하면서 사람에게 묻는
action과 그 observation model을 추가한다. 따라서 특정 navigation task에만 사용할
수 있는 원리는 아니며, 기존 POMDP에 human observation provider를 결합하는 일반
formulation으로 볼 수 있다.

### 4.1 사람의 위치

각 상태 `s`에는 사람 `h_s`가 있다고 가정한다. 로봇은 그 사람과 같은 위치에 있을
때만 물을 수 있다.

```text
robot location = s
→ accessible human = h_s
```

질문 action 자체가 `ask(human_1)`처럼 사람 ID를 직접 선택하는 것은 아니다.
어느 사람에게 물을지는 어떤 위치로 이동했는지를 통해 간접적으로 결정된다.

### 4.2 Availability

`α_s`는 `h_s`가 질문에 응답할 확률이다.

```text
P(non-null response | s, ask) = α_s
P(null response | s, ask)     = 1 - α_s
```

`null`은 사람이 없거나, 바쁘거나, 정해진 시간 안에 답하지 않은 경우다.

### 4.3 Accuracy

`η_s`는 사람이 응답했을 때 올바른 상태 관측을 줄 조건부 확률이다.

```text
η_s
  = P(correct observation | s, ask)
    / P(non-null observation | s, ask)
```

availability가 “답을 하는가?”라면 accuracy는 “답했을 때 맞는가?”다. 두 값을
분리했기 때문에 자주 답하지만 부정확한 사람과, 드물게 답하지만 정확한 사람을
구분할 수 있다.

### 4.4 질문 비용

사람이 답하면 `λ_s`만큼 질문 비용을 받는다.

```text
R(s, ask, s, non-null) = -λ_s
```

사람이 답하지 않으면 논문에서는 비용을 0으로 둔다.

```text
R(s, ask, s, null) = 0
```

이는 응답하지 않은 사람은 실제로 방해받지 않았다고 가정한 것이다. 이 설계에서는
availability가 낮은 사람에게 질문을 시도하는 것이 상대적으로 싸게 평가될 수 있다.
현실에서 질문 시도 자체가 시간이나 사회적 비용을 발생시킨다면 `null`에도 비용을
주는 편이 적절하다.

### 4.5 질문의 transition

질문은 물리적 world state를 바꾸지 않는다.

```text
T(s, ask, s) = 1
```

대신 human response가 observation으로 들어오고 Bayes rule을 통해 belief가
갱신된다.

## 5. 누구에게, 언제 물을지에 대한 대답

### 5.1 누구에게 물을까?

논문의 답은 다음과 같다.

> 현재 belief에서 앞으로 얻을 task reward까지 고려했을 때, 접근 가능하고,
> 답할 가능성과 정확도가 높으며, 질문 비용이 합리적인 사람에게 물을 수 있는
> 경로를 선택한다.

하지만 이 표현을 “현재 위치에서 여러 사람을 비교하여 한 명을 고른다”로 해석하면
안 된다. 논문은 각 위치 `s`에 한 사람 `h_s`가 있다고 가정한다. 따라서 planner는
다음을 함께 비교한다.

```text
어느 위치로 이동할 것인가?
그 위치의 사람 h_s에게 물을 것인가?
질문하지 않고 자율 행동을 계속할 것인가?
```

결과적으로 사람 선택은 경로 선택에 포함된다. 실내 navigation 실험에서
HOP-POMDP는 최단 경로 대신 조금 더 길더라도 availability가 높은 사무실이 많은
경로를 선택했다.

사람을 평가할 때 사용되는 요소는 다음과 같다.

| 요소 | 높거나 낮을 때의 영향 |
|---|---|
| Availability `α_s` | 높으면 실제 답을 받을 가능성이 커진다. |
| Accuracy `η_s` | 높으면 답을 통해 올바른 belief를 얻을 가능성이 커진다. |
| Query cost `λ_s` | 높으면 그 사람에게 묻는 action의 가치가 낮아진다. |
| Location | 멀리 있으면 그 사람에게 접근하기 위한 이동 비용이 증가한다. |
| Information value | 그 답이 이후 행동 선택과 task 성공을 얼마나 개선하는지 결정한다. |

따라서 단순히 accuracy가 가장 높은 사람을 고르는 것이 아니다. 정확한 사람이 너무
멀리 있거나 잘 응답하지 않거나 질문 비용이 크다면, 더 가깝고 충분히 유용한 다른
사람을 만나는 경로가 선택될 수 있다.

### 5.2 언제 물을까?

논문의 답은 다음과 같다.

> 질문 후 얻는 관측이 미래의 행동 선택과 task 성공을 개선하는 기대가치가 질문
> 비용보다 클 때 묻는다.

POMDP policy는 현재 belief `b`에서 `ask`와 자율 행동의 expected return을 비교한다.

```text
Q(b, ask)
  = expected query reward
  + γ Σ_o P(o | b, ask) V(b_ask,o)

Q(b, autonomous action)
  = expected task reward
  + γ Σ_o P(o | b, action) V(b_action,o)
```

따라서 “belief confidence가 0.8보다 낮으면 질문한다”와 같은 별도 threshold가
있는 것이 아니다. 다음 조건이 함께 작용한다.

- 현재 belief가 여러 state에 퍼져 있어 다음 행동 선택이 위험함
- 사람의 답에 따라 좋은 행동과 나쁜 행동을 구분할 수 있음
- 그 사람이 응답할 가능성이 충분함
- 응답이 충분히 정확함
- 질문 비용이 정보의 가치보다 작음

반대로 belief가 이미 충분히 명확하거나, 어떤 답을 받아도 다음 행동이 같거나,
사람이 거의 답하지 않거나, 부정확하거나, 질문 비용이 크면 묻지 않는다.

사람에게 물었지만 `null`을 받으면 같은 질문을 계속 반복하지 않는다. 논문은 이
경우 OPOMDP의 `Q_MDP` 방식처럼 현재 belief에서 가장 좋은 비질문 action을 실행한다.
사람은 센서와 달리 같은 순간에 반복해서 물어도 독립적인 새 관측을 준다고 보기
어렵기 때문이다.

### 5.3 Benchmark에서의 구체적인 예

Benchmark에는 중간 상태 2와 3에 각각 사람 `h_2`, `h_3`가 있다. 로봇은 시작
상태에서 action `B` 또는 `C`를 선택해 어느 사람 쪽으로 갈 가능성을 높일지
결정한다. 이후 상태 2 또는 3에서 다시 `B/C`를 선택하는데, 잘못 선택하면 `-10`,
올바르게 선택하면 `+10`이다.

```text
시작 상태
→ h_2 쪽 경로 또는 h_3 쪽 경로 선택
→ 현재 위치가 불확실하면 ask 여부 결정
→ 응답으로 belief update
→ +10을 얻을 가능성이 높은 B/C 선택
```

질문 비용은 응답했을 때 `-1`이다. 따라서 `-1`을 지불해 상태를 더 정확히 알고
`-10`의 실패를 피할 가치가 있을 때 질문한다.

## 6. LM-HOP: 사람 모델의 온라인 학습

실제 배치 전에는 각 사람의 `α_s`, `η_s`를 정확히 알기 어렵다. LM-HOP는 task를
실행하면서 관측 횟수를 누적해 두 값을 추정한다.

### 6.1 Null response

`null`을 받으면 로봇은 실제 어느 상태에 있었는지 모르므로, 현재 belief `b(s)`에
비례해 각 사람의 unavailable count를 갱신한다.

### 6.2 Non-null response

답을 받으면 belief에 비례해 availability count를 갱신하고, 답의 내용과 belief를
이용해 accuracy 추정치를 갱신한다.

### 6.3 Policy를 다시 푸는 시점

매 응답마다 POMDP policy를 다시 계산하지 않는다. Pearson `χ²` test로 현재
HOP-POMDP에 사용 중인 사람 모델과 새 추정치가 유의하게 달라졌는지 검사한다.

```text
χ² > 3.84
→ 95% confidence에서 모델이 달라졌다고 판단
→ availability/accuracy 갱신
→ HOP-POMDP policy 재계산
```

이 threshold는 “언제 사람에게 질문하는가?”를 결정하는 threshold가 아니다.
사람 모델이 충분히 변했을 때 “언제 policy를 다시 풀 것인가?”를 결정한다.

### 6.4 Explore/exploit

현재 최적 policy만 실행하면 초기 policy에 질문이 포함되지 않았을 때 특정 사람의
availability와 accuracy를 영원히 학습하지 못할 수 있다. LM-HOP는 일부 random
action을 실행해 사람 모델을 탐색하고, 나머지에는 현재 policy를 활용한다.

논문의 Algorithm 1에 적힌 `ρ > 1/t` 조건은 시간이 지날수록 random action의
확률이 커지는 형태로 읽힐 수 있어, 일반적인 감소형 exploration 설명과 다소
불명확하다. 재현 구현에서는 저자가 의도한 exploration schedule을 추가로
확인하거나 명시적으로 정의할 필요가 있다.

## 7. 실험

### 7.1 5-state benchmark

- 상태 5개
- 중간 상태에 사람 2명
- 성공 terminal reward `+10`
- 실패 terminal reward `-10`
- 사람이 응답했을 때 query cost `-1`

LM-HOP는 availability만 학습할 때 약 20–30회, availability와 accuracy를 함께
학습할 때 약 30–40회 policy를 재계산했다. 사람 모델은 약 1,000–2,000번의 실행
안에서 실제 값에 가까워졌다. 논문이 비교한 기존 hypothesis-POMDP 학습 방식의
`10³–10⁶`회 policy 계산보다 적다.

평균 reward는 다음과 같이 보고되었다.

| 방법 | 평균 reward |
|---|---:|
| Explore only | `-0.215` |
| LM-HOP | `3.021` |
| Exploit only | `4.742` |
| True human model을 아는 optimal policy | `5.577` |

Exploit-only의 평균은 높지만, 초기에 availability가 낮은 경로를 선택하면 성능이
크게 나빠지는 양극화가 있었다. LM-HOP는 탐색을 통해 이런 초기 모델 오류에 더
강건하다는 것이 저자의 해석이다.

### 7.2 실제 건물 데이터

- graph node 60개
- 사무실 37개
- 78개 사무실에서 수집한 availability 데이터 활용
- 목표 도착 reward `+100`
- 질문 비용 `-1`

HOP-POMDP policy는 단순 최단 경로보다 길더라도 availability가 높은 사람이 더
많이 있는 경로를 선택했다. 이는 이동 비용과 미래의 도움 가능성을 하나의 policy
안에서 비교한 결과다.

## 8. 기여

1. 사람을 항상 정확한 oracle이 아니라 availability와 accuracy를 가진 observation
   provider로 모델링했다.
2. 이동, task action과 질문을 같은 POMDP policy에서 선택한다.
3. 사람의 특성을 task 실행 중에 온라인으로 학습한다.
4. 매번 policy를 다시 풀지 않고 사람 모델이 유의하게 변했을 때만 재계산한다.

## 9. 한계

1. 각 위치에 한 명의 고정된 사람이 있다고 가정한다.
2. 한 위치에서 여러 사람 중 누구에게 물을지 직접 선택하지 않는다.
3. 사람의 위치와 질문 비용은 미리 알려져 있다고 가정한다.
4. 질문 내용은 사실상 “현재 상태가 무엇인가?”로 고정되어 있다.
5. `null` response의 시간 비용과 질문 시도 비용을 0으로 둔다.
6. 같은 사람에게 즉시 다시 묻지 않는 처리는 heuristic한 `Q_MDP` fallback이다.
7. belief가 불확실한 상태에서는 정확한 사람의 응답도 다른 사람의 통계에 일부
   배분되므로 추정치가 실제 값에 완전히 수렴하지 않을 수 있다.
8. 실제 건물 실험은 수집한 availability를 이용한 policy 비교이며, 장기간 실제
   로봇이 사람의 accuracy까지 온라인 학습한 완전한 현장 실험은 아니다.

## 10. 현재 Attr-POMDP baseline과의 관계

| 항목 | HOP-POMDP | 현재 adapted Attr-POMDP |
|---|---|---|
| 불확실성 | 로봇의 domain state | domain fact를 포함한 world state |
| 질문 대상 | 현재 상태 전체 | 특정 grounded attribute fact |
| Provider | 위치별 human | Oracle, VLM, Human interface |
| Provider model | availability, accuracy, cost | 현재 Oracle은 항상 응답하고 결정적 |
| 질문 선택 | `ask`와 task action의 return 비교 | fact query와 task action의 return 비교 |
| 누구에게 묻는가 | 경로를 통해 위치별 사람 선택 | provider 종류가 실행 전 고정됨 |
| 학습 | 사람의 availability/accuracy 온라인 학습 | 현재 미구현 |

현재 구현에 HOP-POMDP의 아이디어를 적용하려면 query action에 다음 값을 추가할 수
있다.

```text
provider identity
provider availability
provider accuracy
query-attempt cost
response cost
null observation
```

그러면 planner가 `무엇을 물을지`뿐 아니라 `Oracle/VLM/Human 중 누구에게 물을지`,
`지금 물을지 task action을 계속할지`를 함께 비교할 수 있다.

다만 현재 실험처럼 feedback source를 사전에 고정하고 Oracle만 실제 구현한
상태에서는 HOP-POMDP 전체를 baseline으로 구현할 필요는 없다. HOP-POMDP는 향후
provider 선택과 human availability를 실험 변수로 만들 때 직접적인 근거가 된다.
