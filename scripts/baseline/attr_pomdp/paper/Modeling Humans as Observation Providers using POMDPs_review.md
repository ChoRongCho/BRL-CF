# Modeling Humans as Observation Providers using POMDPs 리뷰

## 1. 논문 정보

- 제목: **Modeling Humans as Observation Providers using POMDPs**
- 저자: Stephanie Rosenthal, Manuela Veloso
- 학회: 20th IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN)
- 연도: 2011
- 페이지: 53–58
- DOI: `10.1109/ROMAN.2011.6005272`
- 제안 방법: Human Observation Provider POMDP (HOP-POMDP)

## 2. 한 문장 요약

HOP-POMDP는 사람에게 물으면 항상 답을 얻는다고 가정하지 않고, **사람마다 다른
응답 가능성과 질문 비용**을 POMDP에 넣어 로봇이 어디로 이동하고 누구에게 언제
도움을 요청할지 계획한다.

## 3. 연구 문제

불확실한 환경을 이동하는 로봇은 현재 위치나 주변 상태를 정확히 알기 위해 사람에게
질문할 수 있다. 그러나 현장에 있는 사람은 전담 supervisor나 oracle과 다르다.

- 자리에 없을 수 있다.
- 자리에 있어도 통화나 업무 때문에 답하지 않을 수 있다.
- 질문을 받으면 시간이 들고 방해받을 수 있다.
- 사람마다 응답 가능성과 방해 비용이 다르다.

따라서 단순히 정보량이 큰 시점에 질문하는 것만으로는 충분하지 않다. 로봇은
**답을 받을 가능성**, **사람을 방해하는 비용**, **그 사람에게 가는 이동 비용**을
함께 고려해야 한다.

## 4. 기존 Oracular POMDP와의 차이

Oracular POMDP(OPOMDP)는 로봇이 비용 \(\lambda\)를 지불하면 언제든 정확한 관측을
주는 oracle에게 질문할 수 있다고 가정한다.

HOP-POMDP는 이 가정을 다음과 같이 바꾼다.

| 비교 기준 | OPOMDP | HOP-POMDP |
|---|---|---|
| 응답자 | 항상 대기 중인 oracle | 환경에 있는 일반 사람 |
| 응답 가능성 | 항상 응답 | 사람과 위치에 따라 다름 |
| 질문 비용 | 고정 비용 | 사람마다 다른 방해 비용 |
| 무응답 | 고려하지 않음 | `null observation`으로 모델링 |
| 이동 계획 | 가장 짧은 경로에서 도움을 기대 | 도움을 받기 쉬운 사람 쪽으로 우회 가능 |

OPOMDP는 최단 경로에 있는 사람이 답할 것이라고 기대한다. HOP-POMDP는 그 사람이
응답하지 않을 가능성이 높다면 더 멀더라도 도움을 받을 가능성이 높은 경로를
선택할 수 있다.

## 5. 사람을 Observation Provider로 모델링

논문은 사람의 복잡한 내부 상태나 행동을 POMDP state에 직접 추가하지 않는다.
대신 사람을 특별한 observation source로 모델링한다.

이 선택은 중요한 계산상의 장점이 있다. 사람의 상태를 로봇과 환경의 state에
결합하면 state space가 크게 증가하지만, observation model에 넣으면 기존 POMDP
구조를 비교적 작게 유지할 수 있다.

### 5.1 Availability

상태 \(s\)에 있는 사람의 availability를 \(\alpha_s\)로 정의한다.

```text
α_s = 상태 s에서 질문했을 때 사람이 답할 확률
0 ≤ α_s ≤ 1
```

availability에는 두 상황이 함께 포함된다.

- 사람이 그 장소에 존재하는가?
- 존재한다면 현재 질문을 받을 수 있는가?

논문은 둘을 구분하지 않는다. 로봇 입장에서는 답을 받았는지가 중요하기 때문이다.

사람이 답하면 현재 상태에 대한 정확한 관측 \(o_s\)를 제공한다고 가정한다.

```text
P(o_s | s, ask) = α_s
P(o_null | s, ask) = 1 - α_s
```

여기서 \(o_{null}\)은 사람이 답하지 않았다는 관측이다.

### 5.2 Cost of Asking

상태 \(s\)에 있는 사람에게 질문하는 비용을 \(\lambda_s\)로 정의한다.

```text
사람이 답함:
  R(s, ask, s, o_s) = -λ_s

사람이 답하지 않음:
  R(s, ask, s, o_null) = 0
```

비용은 답변 시간과 interruption cost를 나타낸다. 사람마다 업무 상황과 도움을
줄 의향이 다르므로 값도 달라질 수 있다.

이 reward 설계에서는 무응답에 비용이 없다. 따라서 응답 확률이 조금이라도 있으면
한 번 질문을 시도하는 행동이 지나치게 유리해질 수 있다. 실제 시스템에서는
질문 전달, 대기 시간, 사회적 방해가 무응답에도 발생하므로 주의가 필요한 가정이다.

### 5.3 Answer Accuracy

HOP-POMDP는 다음 두 가지를 구분한다.

- **availability uncertainty:** 사람이 답할지 알 수 없음
- **answer uncertainty:** 사람이 틀린 답을 할 수 있음

이 논문은 첫 번째만 모델링한다. 사람이 답하기만 하면 관측은 정확하다고 가정한다.
즉, 낮은 \(\alpha_s\)는 “가끔 틀린 답을 준다”가 아니라 “가끔 답하지 않는다”는
뜻이다.

## 6. HOP-POMDP 구성

논문은 HOP-POMDP를 다음 요소로 정의한다.

```text
{Λ, S, α, A, O, Ω, T, R}
```

| 요소 | 의미 |
|---|---|
| \(S\) | 로봇과 환경의 상태 |
| \(\Lambda\) | 사람별 질문 비용 |
| \(\alpha\) | 사람별 availability |
| \(A \cup \{a_{ask}\}\) | 원래 행동과 질문 행동 |
| \(O \cup \{o_s\} \cup \{o_{null}\}\) | 센서 관측, 사람의 답, 무응답 |
| \(\Omega\) | 질문 시 답 또는 무응답을 받을 확률 |
| \(T\) | 상태 전이 |
| \(R\) | 이동, task 및 질문의 reward |

질문은 정보를 얻을 뿐 환경 상태를 바꾸지 않는 pure information-gathering action이다.

```text
T(s, ask, s) = 1
```

로봇은 다음을 하나의 policy 안에서 결정한다.

1. task를 위해 어느 경로로 이동할 것인가?
2. 도움을 받을 가능성이 높은 사람 쪽으로 우회할 것인가?
3. 그 사람에게 질문할 가치가 있는가?
4. 답이 없으면 어떤 자율 행동을 수행할 것인가?

## 7. 계획과 실행의 차이

### 7.1 계획 단계

계획에서는 availability를 확률로 사용한다. 예를 들어 어떤 사람이 70%의 확률로
답한다면 질문 action에서 정확한 관측을 받을 branch와 무응답 branch를 모두
평가한다.

HOP-POMDP는 pure information-gathering action을 다룰 수 있는 일반 POMDP solver로
풀 수 있다. 논문은 optimal policy에 Witness algorithm을 사용한다.

QMDP처럼 완전 관측을 가정하는 근사 solver는 정보 획득만 하는 질문 행동의 가치를
제대로 표현하지 못하므로 적합하지 않다고 지적한다.

### 7.2 실행 단계

센서 noise와 사람의 availability는 성격이 다르다. noisy sensor는 같은 위치에서
여러 번 측정하면 다른 관측을 얻을 수 있지만, 지금 바쁜 사람에게 즉시 반복해서
물어도 갑자기 응답 가능해질 가능성은 낮다.

따라서 논문은 한 번 질문해서 \(o_{null}\)을 받으면 같은 자리에서 질문을 반복하지
않고 다른 자율 행동을 실행하도록 한다. 실제 관련 실험에서는 30초 이내 답이 없으면
무응답으로 취급했다.

이 규칙은 중요한 현실적 보완이다. 확률 모델만 그대로 실행하면 policy가 같은
사람에게 계속 질문할 수 있기 때문이다.

## 8. 비교한 두 Policy

### 8.1 Adapted OPOMDP policy

JIV heuristic을 사용한다. 기본적으로 가장 좋은 자율 행동을 계산하고, 현재
사람에게 질문해서 얻는 정보 가치가 행동 가치보다 높으면 질문한다.

그러나 경로를 계획할 때 사람의 availability를 충분히 반영하지 않기 때문에,
최단 경로에 있는 사람이 응답할 것이라고 기대하는 경향이 있다.

### 8.2 Optimal HOP-POMDP policy

사람별 availability, 질문 비용, 이동 비용을 처음부터 policy 계산에 포함한다.
따라서 다음과 같은 선택이 가능하다.

```text
최단 경로:
  짧지만 사람을 만나거나 답을 받을 가능성이 낮음

우회 경로:
  더 길지만 도움을 받을 가능성이 높은 사람이 있음

→ 정보가 task success에 중요하면 우회 경로 선택
```

## 9. 실험

### 9.1 Benchmark 환경

benchmark는 다음과 같이 구성된다.

- 상태 5개
- 두 개의 이동 경로
- 상태 2와 3에 각각 사람 한 명
- 성공 terminal reward `+10`
- 실패 terminal reward `-10`
- 사람별 availability와 질문 비용, 이동 비용을 변화

availability는 0에서 1까지 0.1 간격으로 바꾸고, 비용은 0.125에서 8까지 바꿨다.
총 5,929개 조건을 만들고 각 policy를 조건마다 1,000회 실행했다.

### 9.2 Policy 차이

Adapted OPOMDP와 optimal HOP-POMDP의 첫 이동 선택은 전체 조건의 **39.67%**에서
달랐다. OPOMDP는 availability와 관계없이 짧은 경로를 선호하지만, HOP-POMDP는
어떤 사람이 답할 수 있고 질문 비용이 얼마인지에 따라 경로를 바꿨다.

### 9.3 Reward

| Policy | 전체 조건 평균 reward |
|---|---:|
| Adapted OPOMDP | 6.01 |
| Optimal HOP-POMDP | **6.43** |

전체 평균 차이는 크지 않다. 두 사람의 availability가 같거나, 최단 경로의 사람이
비용 측면에서도 유리한 경우에는 두 policy가 동일하기 때문이다.

차이가 큰 조건에서는 결과가 달랐다. 한 사람은 항상 응답하고 다른 사람은 전혀
응답하지 않는 극단적 조건에서 HOP-POMDP의 평균 reward는 약 8.55였고,
OPOMDP는 조건에 따라 평균 3.54까지 하락했다. 논문이 말하는 “거의 두 배”는
전체 평균이 아니라 이 worst-case 영역을 가리킨다.

### 9.4 실제 건물 지도

저자들은 건물 내 78개 office를 세 날짜의 9개 시점에 조사하여 사람의 availability를
측정했다. 그중 다음 환경으로 policy를 비교했다.

- graph node 60개
- office 37개
- 이동 성공 확률 0.9, 제자리에 남을 확률 0.1
- 질문 비용은 모두 동일
- 연구실 도착 reward `+100`

OPOMDP는 연구실까지의 최단 경로를 선택했다. HOP-POMDP는 더 길지만 응답 가능한
사람을 만날 확률이 높은 경로를 선택했다.

다만 이 부분은 실제 로봇의 end-to-end 사용자 실험이 아니라, 실제 건물에서 수집한
availability를 사용한 policy 계산 사례다.

## 10. 강점

1. 사람을 항상 응답하는 oracle로 보는 비현실적인 가정을 완화한다.
2. 사람의 존재 여부와 interruptibility를 availability로 표현한다.
3. 사람별 질문 비용을 task planning에 직접 반영한다.
4. 도움을 얻기 위한 이동과 task 이동을 함께 계획한다.
5. 사람을 state에 모두 추가하지 않아 Multi-Agent POMDP보다 상태 증가를 줄인다.
6. 계획의 확률 모델과 실제 사람의 무응답 특성이 다름을 인식하고 실행 규칙을
   별도로 제안한다.

## 11. 한계

### 11.1 답변은 항상 정확하다는 가정

사람이 응답하기만 하면 정확한 상태를 알려준다고 가정한다. 착각, 잘못된 관찰,
질문의 오해, 서로 다른 전문성은 모델링하지 않는다.

### 11.2 Availability가 미리 알려져 있다는 가정

\(\alpha_s\)와 \(\lambda_s\)가 planning 전에 알려져 있다고 가정한다. 실제 환경에서는
시간, 장소, 업무 상황에 따라 변하며 온라인 추정이 필요하다.

### 11.3 무응답 비용이 0

답이 없어도 질문 전달과 대기에는 비용이 든다. 무응답을 무료로 두면 availability가
0.1보다 큰 사람에게 일단 질문하는 policy가 만들어진다. 이는 실제 HRI에서 질문을
과도하게 만들 수 있다.

### 11.4 Availability의 원인을 구분하지 않음

사람이 자리에 없는 것과 자리에 있지만 바쁜 것을 같은 무응답으로 처리한다.
두 상황은 재질문 시점이나 이동 전략이 달라야 할 수 있다.

### 11.5 사람의 상태가 시간에 따라 변하지 않음

같은 위치에서 즉시 반복 질문을 막는 실행 규칙은 있지만, availability 자체의
시간 변화는 POMDP state transition으로 모델링하지 않는다.

### 11.6 실제 사용자 평가 부족

건물 사례는 실제 availability 자료를 사용하지만, 최종 policy를 로봇이 실행했을
때의 만족도, annoyance, task success를 직접 비교하지 않았다.

## 12. Oracle, VLM, Human feedback source와의 연결

이 논문은 BRL-CF에서 feedback source를 구분해야 하는 이유를 잘 보여준다.

| Source | Availability | 응답 비용 | 응답 정확도 | HOP-POMDP 관점 |
|---|---|---|---|---|
| Oracle | 항상 가능 | 작거나 고정 | 정확 | 기존 OPOMDP에 가까움 |
| VLM | 시스템이 켜져 있으면 가능 | 계산·API 비용 | 확률적 오류 | noisy sensor에 가까움 |
| Human | 없거나 바쁠 수 있음 | 시간·방해 비용 | 개인별로 다름 | HOP-POMDP의 대상 |

중요한 점은 availability와 accuracy가 다른 개념이라는 것이다.

```text
Human:
  답을 받을 수 있는가?
  답을 받았다면 얼마나 믿을 수 있는가?

VLM:
  호출할 수 있는가?
  결과가 얼마나 정확한가?

Oracle:
  항상 호출 가능하고 정확하다고 가정
```

원 논문의 HOP-POMDP는 human availability는 모델링하지만 human answer accuracy는
모델링하지 않는다. BRL-CF에서 human과 VLM을 실제로 구현할 때는 source별로
다음 두 값을 분리하는 확장이 필요하다.

```text
P(response received | source, context)
P(answer | true state, source, question)
```

## 13. Active Search baseline으로서의 해석

이 논문은 **무엇을 물을지**보다 **누구에게 언제 물을지**에 초점이 있다. 질문
내용은 현재 상태를 정확히 알려달라는 단일 `ask` action으로 추상화된다.

따라서 다음 문제의 직접적인 baseline이다.

- feedback source 선택
- 질문 가능성 또는 timeout 모델링
- 사람에게 질문하는 사회적 비용
- 도움을 받기 위한 경로 계획

반면 다음 문제의 직접적인 baseline은 아니다.

- 여러 질문 문장 중 가장 좋은 질문 선택
- attribute 또는 fact별 query selection
- 자유로운 자연어 답변 이해
- 질문 자체의 semantic quality 비교

현재 구현에서 oracle만 동작하고 VLM과 human을 placeholder로 둔다면 HOP-POMDP의
핵심 차별점은 아직 실험되지 않는다. oracle은 항상 응답하고 정확하므로 OPOMDP
가정에 더 가깝기 때문이다.

HOP-POMDP-style baseline을 구현한다면 최소한 다음 구조가 필요하다.

```text
FeedbackSource:
  oracle:
    availability = 1.0
    query_cost = fixed
    answer_accuracy = 1.0

  vlm:
    placeholder

  human:
    placeholder
    availability
    interruption_cost

Observation:
  answer
  no_response
```

human placeholder는 단순히 `NotImplemented`로 끝내기보다, 향후 availability와
interruption cost가 들어갈 interface를 남기는 것이 논문의 모델과 잘 맞는다.

## 14. BRL-CF에 주는 핵심 시사점

1. `feedback source`는 답변 함수만으로 정의하면 부족하다.
2. source마다 응답 가능성, 비용, 정확도를 분리해야 한다.
3. 무응답도 observation의 한 종류로 취급해야 한다.
4. 질문이 실패했을 때 같은 source에 즉시 반복 질문하지 않는 정책이 필요하다.
5. oracle 결과만으로 평가하면 human-aware active search의 효과를 주장할 수 없다.
6. human 실험 전에는 human source를 placeholder로 명확히 표시해야 한다.

## 15. 최종 평가

HOP-POMDP의 핵심은 사람의 답을 단순한 정확한 센서값으로 사용하는 것이 아니라,
**그 답을 실제로 받을 수 있는가와 사람을 방해하는 비용까지 planning에 포함한
것**이다.

현재 BRL-CF의 oracle-only 단계에서는 직접적인 성능 baseline보다는 feedback-source
설계를 정당화하는 선행연구로 더 적합하다. 이후 VLM과 human source를 구현하면
source별 availability, accuracy, cost를 비교하는 baseline으로 확장할 수 있다.
