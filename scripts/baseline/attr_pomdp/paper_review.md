# Active Search 관련 방법 통합 비교

## 1. 비교 대상

이 문서는 다음 일곱 리뷰를 바탕으로 각 연구의 역할과 차이를 정리한다.

1. *A Survey of Knowledge-Based Sequential Decision-Making under Uncertainty*
2. *INGRESS: Interactive Visual Grounding of Referring Expressions*
3. *Interactive Robotic Grasping with Attribute-Guided Disambiguation*
4. *INVIGORATE: Interactive Visual Grounding and Grasping in Clutter*
5. *LLMs for Robotic Object Disambiguation*
6. *Modeling Humans as Observation Providers using POMDPs*
7. *Utilizing Human Feedback in POMDP Execution and Specification*

주의할 점은 일곱 논문이 모두 같은 문제를 푸는 경쟁 방법은 아니라는 것이다.

- **Survey**는 방법을 분류하는 taxonomy다.
- **INGRESS, Attr-POMDP, INVIGORATE, LLM 방법**은 주로 목표 물체의 모호성을
  해소한다.
- **HOP-POMDP**는 질문 내용보다 누구에게 도움을 요청할지를 다룬다.
- **Utilizing Human Feedback**은 목표 물체가 아니라 task 수행에 필요한 특정
  object-property를 질문한다.

따라서 하나의 정확도 표로 순위를 매기기보다, 각 방법이 active search의 어느
부분을 해결하는지 비교해야 한다.

## 2. 먼저 보는 핵심 비교

| 비교 기준 | Survey | INGRESS | Attr-POMDP | INVIGORATE | LLM 방법 | HOP-POMDP | Utilizing Human Feedback |
|---|---|---|---|---|---|---|---|
| 역할 | 연구 분류 | 물체 확인 | 물체 확인 | 물체 확인과 clutter grasp | 질문 생성 | 응답자 선택 | task fact 확인 |
| 숨은 것 | 방법별로 다름 | 목표 물체 | 목표 물체 | 목표 물체와 가림 관계 | 명시적 확률 상태 없음 | 로봇/환경 상태 | task의 world state |
| 무엇을 묻는가? | 직접 질문하지 않음 | “왼쪽 빨간 컵인가?” | “무슨 색인가?” | “위쪽 컵인가?” | LLM이 만든 open-ended 질문 | “현재 상태가 무엇인가?” | “물체 \(o\)의 속성 \(p\)는 무엇인가?” |
| 누구에게 묻는가? | 여러 source 분류 | 사람 | 사람 | 사람 | 사람을 가정 | 위치한 사람 | 사람 |
| 답변 형태 | 해당 없음 | `yes/no`와 교정 문장 | 속성값 또는 `yes/no` | 계획에서는 주로 `yes/no` | decision-tree branch | 정확한 답 또는 무응답 | 해당 property의 값 |
| 질문과 task 행동의 관계 | taxonomy로 설명 | 질문과 pick을 함께 선택 | 질문과 grasp를 함께 선택 | 질문·방해물 제거·grasp를 함께 선택 | 질문 tree만 생성 | 이동과 질문을 함께 선택 | sensing·질문·manipulation을 함께 선택 |
| 핵심 한계 | 직접 알고리즘 없음 | 생성 문장 오류, clutter 취약 | 고정된 속성 vocabulary | 큰 모델과 구현 복잡도 | belief·reward·실행 모델 부족 | 답변 오류 미모델링 | 정확하고 항상 가능한 사람 가정 |

## 3. 차이를 이해하기 위한 여섯 가지 축

### 3.1 무엇이 불확실한가?

#### 목표 물체가 불확실

INGRESS, Attr-POMDP, INVIGORATE, LLM 방법은 사용자가 어떤 물체를 뜻했는지
모호한 상황에서 출발한다.

```text
“컵을 집어줘”
→ 여러 컵 중 어떤 컵이 목표인가?
```

이 중 INVIGORATE는 목표 identity뿐 아니라 물체 간 가림 관계도 불확실한 것으로
모델링한다.

#### Task state가 불확실

HOP-POMDP와 Utilizing Human Feedback은 더 일반적인 task state uncertainty를
다룬다.

- HOP-POMDP: 로봇이 현재 어떤 상태에 있는지 확실하지 않음
- Utilizing Human Feedback: 물체의 위치나 속성 등 manipulation state가 불확실함

현재 BRL-CF의 symbolic world-state uncertainty에는 후자 두 연구가 더 직접적으로
가깝다.

#### 명시적인 uncertainty가 없음

LLM 방법은 후보를 나누는 질문 tree를 생성하지만 후보에 대한 Bayesian belief를
유지하지 않는다. 따라서 “현재 어느 후보를 얼마나 믿는가”와 “답을 받은 뒤
확률이 어떻게 바뀌는가”가 명시되어 있지 않다.

### 3.2 무엇을 질문하는가?

| 질문 방식 | 방법 | 설명 |
|---|---|---|
| 후보 확인 | INGRESS, INVIGORATE | 특정 후보를 설명하고 그 물체가 맞는지 묻는다. |
| 공통 속성 | Attr-POMDP | 색이나 위치를 물어 여러 후보를 한 번에 나눈다. |
| Open-ended feature | LLM 방법 | 장면에 맞는 새로운 구분 기준과 질문을 생성한다. |
| 전체 상태 관측 | HOP-POMDP | 사람이 로봇의 현재 상태를 알려주는 것으로 추상화한다. |
| 특정 fact | Utilizing Human Feedback | `<object, property>` 한 항목만 질문한다. |

예를 들어 컵 네 개를 구분한다면 다음과 같다.

```text
INGRESS / INVIGORATE:
  “왼쪽 그릇 옆의 빨간 컵인가?”

Attr-POMDP:
  “목표는 무슨 색인가?”

LLM:
  “손잡이가 테이블 바깥쪽을 향한 컵인가?”

Utilizing Human Feedback:
  Query(<cup_2, location>)
```

후보 확인 질문은 이해하기 쉽지만 후보를 하나씩 확인할 수 있다. 속성 질문은 여러
후보를 동시에 제거할 수 있지만 사용 가능한 속성이 미리 정해져야 한다. LLM
질문은 표현 범위가 넓지만 질문이 실제로 관측 가능하고 정확한지 보장하기 어렵다.
Targeted query는 symbolic task에 잘 맞지만 자연스러운 문장 생성 자체는 다루지
않는다.

### 3.3 질문을 어떻게 선택하는가?

#### 확률적 순차 계획

INGRESS-POMDP, Attr-POMDP, INVIGORATE, HOP-POMDP, Utilizing Human Feedback은
질문을 비용이 있는 action으로 모델링한다. 현재 질문의 답을 받은 뒤 belief가
어떻게 바뀌고 최종 task 결과가 어떻게 달라지는지 평가한다.

단, 계획 범위는 서로 다르다.

| 방법 | 질문과 함께 계획하는 행동 |
|---|---|
| INGRESS-POMDP | 후보 물체 pick |
| Attr-POMDP | 목표 물체 grasp |
| INVIGORATE | 방해 물체 제거와 목표 grasp |
| HOP-POMDP | 어느 경로로 이동하고 누구에게 물을지 |
| Utilizing Human Feedback | robot sensing과 manipulation |

#### LLM decision tree

LLM 방법은 prompt를 통해 질문 sequence를 tree로 만든다. 후보를 균형 있게 나누는
질문을 선호할 수 있지만, task reward나 질문 비용에 대한 명시적인 최적화와
Bayesian belief update는 없다.

#### Survey

Survey는 질문 선택기를 제안하지 않는다. 대신 이러한 방법을 **online active
knowledge acquisition**으로 분류하고, symbolic knowledge와 probabilistic planner를
어떻게 연결할지 설명하는 틀을 제공한다.

### 3.4 답변을 어떻게 처리하는가?

| 방법 | 답변 처리 |
|---|---|
| INGRESS | 계획에서는 `yes/no`로 줄이고, 실제 belief update에는 교정 설명도 사용 |
| Attr-POMDP | 미리 정한 attribute vocabulary와 binary pointing response 사용 |
| INVIGORATE | 추가 설명을 허용하지만 계획에서는 주로 긍정/부정으로 단순화 |
| LLM 방법 | 생성된 tree의 branch로 이동하며 확률적 belief update는 없음 |
| HOP-POMDP | 정확한 답 또는 `no response` |
| Utilizing Human Feedback | 사람이 알려준 object-property 값을 정확한 observation으로 사용 |

여기서 HOP-POMDP의 특징은 **틀린 답과 무응답을 구분한다는 것**이다. 사람이
답하기만 하면 정확하다고 가정하고, availability가 낮으면 무응답이 발생한다고
본다.

```text
Availability:
  답을 받을 수 있는가?

Accuracy:
  받은 답이 맞는가?
```

원 논문들은 두 요소를 동시에 충분히 모델링하지 않는다. 실제 oracle/VLM/human
비교에서는 source마다 availability와 accuracy를 별도로 정의해야 한다.

### 3.5 언제 질문을 멈추는가?

POMDP 기반 방법은 질문 action과 최종 task action의 expected return을 비교한다.

```text
한 번 더 질문해서 얻는 이득
        vs.
질문 비용
        vs.
지금 행동했을 때의 성공 또는 실패 위험
```

- INGRESS-POMDP: 질문할지 후보를 pick할지 선택
- Attr-POMDP: 질문할지 grasp할지 선택
- INVIGORATE: 질문할지 방해물을 제거할지 목표를 grasp할지 선택
- HOP-POMDP: 사람에게 물을지 자율적으로 이동할지 선택
- Utilizing Human Feedback: 사람에게 물을지 robot sensing/task action을 할지 선택

LLM 방법은 명시적인 reward와 stopping policy가 없다. 생성한 tree가 끝나거나
후보가 하나 남는 것을 사실상의 종료 조건으로 사용한다.

### 3.6 실제 task의 범위는 어디까지인가?

```text
질문 생성만:
  LLM method

목표 물체 확인 + pick:
  INGRESS

목표 물체 확인 + grasp:
  Attr-POMDP

목표 확인 + 방해물 제거 + clutter grasp:
  INVIGORATE

Navigation + 도움 요청:
  HOP-POMDP

Manipulation + sensing + targeted query:
  Utilizing Human Feedback

연구 전반의 분류:
  Knowledge-based SDM Survey
```

이 범위 차이 때문에 INVIGORATE의 성공률과 Attr-POMDP의 질문 수를 직접 비교하는
식의 해석은 적절하지 않다. 장면, state, action, perception module과 최종 task가
서로 다르기 때문이다.

## 4. 물체 Disambiguation 방법끼리의 차이

### 4.1 INGRESS

INGRESS의 핵심은 **grounding by generation**이다. 각 물체를 설명하는 문장을
생성하고 사용자 지시와 비교한다. 같은 생성 모델을 질문에도 사용한다.

장점은 물체 자체의 속성과 다른 물체와의 관계를 포함한 자연스러운 질문이다.
단점은 생성된 문장이 잘못될 수 있고, 후보마다 질문 action이 생기며, 가려진
물체에는 취약하다는 것이다.

### 4.2 Attr-POMDP

Attr-POMDP는 특정 후보를 긴 문장으로 묘사하기보다 색과 위치 같은 공통 속성을
질문한다. 답 하나로 후보 여러 개를 제거할 수 있고 response 종류도 제한되어
observation model과 계획이 단순하다.

대신 color/location으로 구분되지 않는 장면에서는 유용한 질문을 만들 수 없다.
크기, 재질, 상태와 같은 새 속성은 미리 모델에 추가해야 한다.

### 4.3 INVIGORATE

INVIGORATE는 INGRESS식 object-specific question을 clutter 환경으로 확장한다.
목표가 가려졌다면 단순히 질문하는 데서 끝나지 않고 방해 물체를 제거하는 순서까지
POMDP에서 계획한다.

가장 넓은 robotic task를 다루지만 object detector, grounding model, question
generator, blocking-relation model과 grasp detector가 모두 필요하다. 따라서
질문 선택 baseline만을 위해 재현하기에는 가장 무거운 방법이다.

### 4.4 LLM 방법

LLM 방법은 미리 정한 color/location vocabulary를 넘어 장면에 맞는 새로운 구분
기준을 생성할 수 있다. 그러나 생성된 질문이 실제 장면에서 관측 가능한지,
응답이 task belief에 어떻게 반영되는지, 질문 비용과 실패 비용을 어떻게 비교하는지
명확하지 않다.

따라서 현재 단계에서는 POMDP baseline보다 open-ended query generation을 위한
related work 또는 확장 모듈로 보는 편이 정확하다.

## 5. Human Feedback 방법끼리의 차이

### 5.1 HOP-POMDP: 누구에게 물을 것인가

HOP-POMDP의 주된 관심은 질문 내용이 아니다. 환경에 있는 사람마다 다음 값이
다르다는 점을 모델링한다.

- 그 사람이 답할 가능성
- 그 사람을 방해하는 비용
- 그 사람에게 이동하는 비용

따라서 더 짧은 경로에 사람이 있더라도 응답 가능성이 낮으면, 더 멀지만 도움을
받기 쉬운 사람 쪽으로 이동할 수 있다.

### 5.2 Utilizing Human Feedback: 무엇을 물을 것인가

이 방법은 전체 상태를 설명해 달라고 하지 않고 `<object, property>` 형태의
targeted query를 사용한다.

```text
<tomato_1, ripeness>
<waste_2, category>
<object_3, location>
```

로봇은 자체 sensing, human query, manipulation action을 같은 policy에서 선택한다.
현재 BRL-CF에서 grounded Boolean fact를 질문 action으로 만드는 구조와 가장
직접적으로 대응한다.

### 5.3 두 방법의 상호 보완성

| 질문 | HOP-POMDP | Utilizing Human Feedback |
|---|---:|---:|
| 무엇을 물을까? | 추상화됨 | object-property로 명시 |
| 누구에게 물을까? | 핵심 문제 | 한 명의 정확한 사람을 가정 |
| 사람이 답하지 않을 수 있는가? | 모델링 | 모델링하지 않음 |
| 사람이 틀릴 수 있는가? | 모델링하지 않음 | 모델링하지 않음 |
| 질문의 사회적 비용 | 사람마다 다름 | 일반 query cost |

두 방법을 결합하면 “어떤 fact를 어떤 source에게 물을 것인가”를 함께 다룰 수 있다.

## 6. Survey가 다른 방법들과 다른 이유

Knowledge-based sequential decision-making survey는 새로운 질문 policy나 solver를
제안하는 baseline이 아니다. 각 시스템을 다음 축으로 설명하는 taxonomy다.

- knowledge representation이 symbolic인지 probabilistic인지
- 두 표현이 하나로 통합되었는지 연결되어 있는지
- knowledge가 정적인지 action에 따라 변하는지
- model-based인지 model-free인지
- fully observable인지 partially observable인지
- knowledge를 offline 또는 online으로 얻는지
- agent가 능동적으로 정보를 얻는지
- knowledge source가 사람, 센서, expert 중 무엇인지

BRL-CF는 이 taxonomy에서 다음처럼 해석할 수 있다.

```text
ASP symbolic knowledge
  +
particle-based probabilistic belief
  =
linked representation

실행 중 질문 선택
  =
online active knowledge acquisition

oracle / VLM / human
  =
multiple knowledge sources
```

즉, Survey는 직접 비교할 알고리즘이 아니라 전체 architecture와 연구 위치를
설명하기 위한 이론적 근거다.

## 7. BRL-CF baseline 관점의 분류

### 7.1 직접적인 메인 baseline

#### Adapted Attr-POMDP

Attr-POMDP의 attribute question을 grounded Boolean fact query로 바꾼다.

```text
원 논문:
  AskAttr(color/location)

Adaptation:
  QueryFact(symbolic predicate)
```

질문과 task action을 같은 POMDP action space에서 선택한다는 구조를 비교하기에
적합하다.

#### Targeted-query POMDP

*Utilizing Human Feedback*의 `<object, property>` query가 현재 symbolic fact
질문과 가장 직접적으로 맞는다. robot sensing, query, task action 사이의 선택을
비교할 수 있다.

### 7.2 조건부 또는 축소 baseline

INGRESS와 INVIGORATE는 원래 vision-language 및 robot manipulation system이다.
question generator와 perception을 구현하지 않고 질문 정책만 가져온다면 다음처럼
표기해야 한다.

```text
INGRESS-style object-confirmation baseline
INVIGORATE-style object-confirmation baseline
```

논문 전체를 재현했다고 표현해서는 안 된다.

HOP-POMDP는 human availability와 cost를 실제 실험 변수로 사용할 때 직접 baseline이
된다. oracle만 사용하는 현재 단계에서는 항상 응답하는 OPOMDP 가정과 더 가깝다.

### 7.3 Related work 또는 확장 방향

- LLM 방법: open-ended question proposal
- Knowledge-based SDM Survey: architecture taxonomy

두 연구는 현재 동일 simulator에서 성능을 직접 비교할 메인 baseline으로는
적합하지 않다.

## 8. 현재 구현에 필요한 공통 인터페이스

일곱 논문을 종합하면 질문 action에는 최소한 다음 정보가 필요하다.

```text
QueryAction:
  target_fact_or_hypothesis
  question_text
  expected_answers
  feedback_source
  query_cost

FeedbackSource:
  availability
  answer_accuracy
  response_cost

Observation:
  answer
  no_response

Planner:
  task action value
  query action value
  answer-conditioned belief update
  stopping decision
```

현재 계획대로 oracle만 구현할 경우에는 다음처럼 두는 것이 명확하다.

```text
oracle:
  implemented
  availability = 1
  accuracy = 1
  scripts/models/{domain}/answer.py를 그대로 사용

vlm:
  placeholder

human:
  placeholder
  future fields:
    availability
    interruption_cost
    answer_accuracy
```

## 9. 최종 정리

각 연구가 답하는 질문은 다음처럼 구분된다.

```text
Survey:
  이 시스템은 knowledge-based decision making의 어디에 속하는가?

INGRESS:
  특정 후보를 어떤 자연어 설명으로 확인할 것인가?

Attr-POMDP:
  어떤 공통 속성을 물으면 후보를 효율적으로 나눌 수 있는가?

INVIGORATE:
  질문, 방해물 제거, 목표 grasp를 어떤 순서로 수행할 것인가?

LLM method:
  미리 정하지 않은 구분 기준과 질문을 생성할 수 있는가?

HOP-POMDP:
  응답 가능성과 비용이 다른 사람 중 누구에게 물을 것인가?

Utilizing Human Feedback:
  task 수행에 필요한 어떤 object-property를 물을 것인가?
```

BRL-CF의 가장 직접적인 비교 대상은 **Adapted Attr-POMDP**와
**targeted-query POMDP**다. HOP-POMDP는 feedback source modeling의 근거이며,
INGRESS와 INVIGORATE는 object-specific confirmation 계열의 선행연구다. LLM
방법은 향후 VLM/LLM question proposal 확장에 가깝고, Survey는 전체 구조를
설명하는 taxonomy로 사용하는 것이 적절하다.
