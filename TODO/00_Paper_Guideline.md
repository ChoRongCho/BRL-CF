# Paper Guideline

이 문서는 논문 작성 방향을 잡기 위한 rough guideline이다. 세부 문장, 실험 구성, related work 범위는 계속 수정한다.

## 1. Abstract

현실 환경에서 동작하는 로봇은 부분 관측성 때문에 현재 상태를 완전히 알 수 없다. 작은 상태 오판은 잘못된 행동 선택으로 이어지고, 누적되면 전체 task failure를 유발할 수 있다. 사람은 로봇이 직접 관측하기 어려운 정보를 제공할 수 있지만, 지속적인 질문은 사용자 부담을 증가시킨다. 따라서 로봇은 단순히 질문할 수 있는 능력뿐 아니라, 언제 질문해야 하는지, 무엇을 질문해야 하는지, 그리고 어떤 형식으로 질문해야 하는지를 결정할 수 있어야 한다.

본 논문은 부분 관측 환경에서 로봇이 planning에 필요한 정보를 스스로 식별하고, task-relevant uncertainty를 줄이기 위해 상태 단위 질문을 생성하며, 그 응답을 belief update와 replanning에 반영하는 planning-centered feedback framework를 제안한다. 제안 방법은 현재 action selection에 영향을 주는 belief uncertainty를 기반으로 질의 시점을 결정하고, action frontier에서 필요한 hidden predicate를 질의 대상으로 선택하며, 사용자의 응답이 특정 state variable에 대응되도록 state-level query를 구성한다.

우리는 tomato harvesting, waste sorting 등 symbolic robot planning domain에서 제안 방법을 평가한다. 실험은 질의 시점, 질의 내용, 질의 형식의 세 축으로 구성한다. 결과적으로 제안 방법이 매번 질문하는 방식보다 적은 질문으로 높은 task success를 유지하고, 질문하지 않는 방식보다 failure를 줄이며, action-level 또는 option-level query 방식보다 belief correction과 replanning에 더 직접적으로 연결될 수 있음을 보이는 것을 목표로 한다.

## 2. 문제 정의

### 2.1 배경

로봇은 실제 환경에서 noisy observation, imperfect action execution, hidden object property, occlusion 등으로 인해 완전한 상태 정보를 갖지 못한다. 예를 들어 tomato harvesting에서는 tomato의 ripeness나 위치를 잘못 관측할 수 있고, waste sorting에서는 occlusion 때문에 아래에 있는 waste가 아직 보이지 않을 수 있다. 이러한 hidden state uncertainty는 planner가 잘못된 action을 선택하게 만들 수 있다.

기존 planning system은 partial observability를 belief update로 다루지만, observation만으로 uncertainty가 충분히 줄어들지 않는 경우가 있다. 이때 외부 feedback source, 특히 사람은 로봇이 직접 알기 어려운 정보를 제공할 수 있다. 그러나 사람에게 질문하는 것은 공짜가 아니다. 질문이 많아질수록 응답 시간, 시스템 지연이 증가한다.

### 2.2 핵심 문제

본 논문의 문제는 다음과 같이 정의한다.

```text
부분 관측 환경에서 로봇이 task를 수행할 때,
로봇은 planning에 필요한 불확실성을 식별하고,
필요한 경우에만 사람 또는 feedback source에게 질문하며,
```

이를 위해 로봇은 다음 네 가지를 결정해야 한다.

- 언제 질문할 것인가?
- 무엇을 질문할 것인가?
- 어떤 형식으로 질문할 것인가?
- 응답을 belief / knowledge update와 replanning에 어떻게 연결할 것인가?

이 중 논문 contribution은 `When / What / How` 세 축으로 정리한다.

### 2.3 문제의 어려움

첫째, 모든 불확실성이 질문할 가치가 있는 것은 아니다. task와 무관한 hidden state에 대해 질문하면 query count는 증가하지만 task success에는 기여하지 않는다.

둘째, 질문 시점이 너무 이르면 불필요한 질문이 많아지고, 너무 늦으면 이미 잘못된 action을 실행해 dead-end에 도달할 수 있다.

셋째, 질문 형식이 planning update와 잘 연결되어야 한다. action-level query는 "이 행동을 해도 되는가?"를 묻지만, 로봇이 어떤 state predicate를 잘못 알고 있는지는 명확히 수정하지 못한다. 반면 state-level query는 특정 predicate의 truth value를 묻기 때문에 belief correction에 직접 사용될 수 있다.

## 3. 관련 연구

### 3.1 Planning under Partial Observability

POMDP는 hidden state와 noisy observation을 갖는 sequential decision problem을 다루는 대표적인 framework이다. 로봇은 observation history를 바탕으로 belief를 유지하고, belief state 위에서 expected reward를 최대화하는 action을 선택한다. 그러나 정확한 POMDP solving은 계산적으로 어렵고, 실제 robot planning에서는 online planning, sampling-based planning, symbolic abstraction 등을 결합하는 경우가 많다.

본 논문은 POMDP의 belief update 관점을 따르되, observation만으로 충분히 줄어들지 않는 task-relevant uncertainty를 사람의 feedback으로 보정하는 방향에 초점을 둔다.

### 3.2 Active Perception / Information Gathering

Active perception과 information gathering 연구는 로봇이 추가 sensing action을 선택하여 uncertainty를 줄이는 문제를 다룬다. 이러한 접근은 uncertainty reduction을 planning objective에 포함한다는 점에서 본 연구와 관련이 있다.

다만 본 논문이 다루는 질문은 단순 sensing action이 아니다. 우리는 robot action space에 query action을 추가하여 planner가 직접 고르게 하는 방식과 달리, physical planning은 유지하면서 action selection에 영향을 주는 hidden predicate를 별도로 식별하고 질문한다. 이 차이는 search space 증가와 query control 방식에서 중요하다.

### 3.3 Human-in-the-loop Robot Planning

Human-in-the-loop planning에서는 사람이 로봇에게 preference, correction, instruction, answer 등을 제공한다. 이때 중요한 문제는 사람이 언제 개입해야 하는지, 어떤 정보를 제공해야 하는지, 그리고 로봇이 그 정보를 어떻게 계획에 반영하는지이다.

본 논문은 사람이 제공하는 feedback을 state-level predicate answer로 제한하여, 응답이 belief update에 직접 연결되도록 한다. 이를 통해 자유로운 자연어 instruction이나 action-level approval보다 더 구조적인 belief correction을 목표로 한다.

### 3.4 LLM Planner Uncertainty and KnowNo-style Query

KnowNo는 LLM planner의 uncertainty를 conformal prediction으로 calibration하고, next action 선택이 불확실할 때 사용자에게 help를 요청하는 framework이다. 이는 option-level uncertainty query로 볼 수 있다. 즉 사용자는 여러 action option 중 무엇이 맞는지를 선택한다.

본 논문의 방식은 action option 자체를 묻기보다, action selection을 불확실하게 만드는 hidden state predicate를 묻는다. 따라서 KnowNo-style query는 본 논문의 `How` 실험에서 action-level 또는 option-level query baseline으로 사용할 수 있다.

## 4. 방법 제안

### 4.1 Overview

제안 방법은 partial observability 아래에서 closed-loop planning, query, belief update, replanning을 반복한다.

전체 흐름:

```text
1. 현재 belief를 기반으로 POMCP / planner가 action을 선택한다.
2. action frontier 또는 candidate action set의 uncertainty를 평가한다.
3. confidence가 threshold보다 낮으면 query가 필요하다고 판단한다.
4. 현재 action selection에 영향을 주는 predicate를 query target으로 선택한다.
5. state-level query를 feedback source에게 전달한다.
6. 응답을 belief / knowledge update에 반영한다.
7. update된 belief로 replanning한다.
```

### 4.2 When: Query Timing

질의 시점은 현재 planner가 선택하려는 action에 대한 confidence 또는 belief uncertainty를 기반으로 결정한다. planner가 충분히 확신할 수 있으면 질문하지 않고 action을 실행한다. 반대로 action frontier가 여러 가능성으로 갈라져 있고 confidence가 낮으면 질문한다.

목표는 다음이다.

```text
질문이 필요할 때만 물어보고,
필요하지 않을 때는 autonomy를 유지한다.
```

### 4.3 What: Query Content

질의 내용은 현재 planning decision에 영향을 주는 hidden predicate를 대상으로 한다. 단순히 belief에서 entropy가 높은 predicate를 아무거나 묻는 것이 아니라, action selection 또는 failure prevention에 실제로 영향을 줄 수 있는 predicate를 선택한다.

예:

```text
tomato:
  ripe(tomato3)
  at(tomato2, stem_01)

wastesorting:
  paper(waste1)
  can(waste3)
  detected(waste2)
```

핵심은 task-relevant uncertainty를 묻는 것이다.

### 4.4 How: State-level Query Format

본 논문에서 `How`는 belief update의 구현 방식이 아니라 질문의 형식이다.

비교:

```text
Action-level query:
  "이 행동을 해도 될까요?"
  예: "pick tomato3를 해도 됩니까?"

Option-level query:
  "A/B/C/D 중 어떤 행동이 맞습니까?"

State-level query:
  "이 predicate가 참입니까?"
  예: "ripe(tomato3)=True?"
```

State-level query의 장점은 응답이 특정 state variable 또는 symbolic predicate에 대응된다는 점이다. 따라서 feedback이 belief update에 직접 들어가고, 잘못된 hidden state 인지를 구조적으로 수정할 수 있다.

### 4.5 Feedback Source

실험에서는 feedback source를 다음과 같이 구분한다.

```text
Oracle:
  GT hidden state를 알고 항상 정답을 제공.
  system upper bound 확인용.

Noisy-oracle:
  oracle answer를 일정 확률로 틀리게 제공.
  answer quality에 대한 robustness 확인용.

Human-proxy:
  실제 사람이 보일 법한 일부 오류 특성을 반영하는 중간 feedback source.
  현재는 내부 분석 또는 appendix 성격.

Human:
  실제 사용자가 같은 query에 응답.
```

## 5. 실험

### 5.1 Experimental Setup

도메인:

```text
tomato harvesting
waste sorting
watering
rover
kitchen
blocksworld
```

초기 main experiment는 tomato와 wastesorting에 집중한다. 나머지 도메인은 확장성 확인 또는 후속 실험으로 둔다.

Metrics:

- task success rate
- dead-end rate
- query count
- query ratio
- planning length
- operation time / elapsed time
- search time
- belief confidence after feedback

### 5.2 Experiment 1: When to Ask

목적:

```text
query timing이 task success와 query burden에 어떤 영향을 주는지 확인한다.
```

비교 조건:

```text
No query
All query
Random query
Ours threshold-based query
```

기대 결과:

```text
Ours는 No보다 높은 성공률을 보이고,
All보다 적은 질문 수로 비슷한 성공률을 유지하며,
Random보다 안정적인 성능을 보여야 한다.
```

### 5.3 Experiment 2: What to Ask

목적:

```text
query content selection이 task performance에 기여하는지 확인한다.
```

비교 후보:

```text
Ours task-relevant predicate selection
random predicate selection
entropy-only predicate selection
search ablation
```

기대 결과:

```text
현재 planning decision에 영향을 주는 predicate를 묻는 방식이
불필요한 질문을 줄이고 failure를 더 잘 방지해야 한다.
```

### 5.4 Experiment 3: How to Ask

목적:

```text
state-level query가 action-level 또는 option-level query보다 효과적인지 확인한다.
```

비교:

```text
Ours:
  state-level predicate query

KnowNo-style:
  option-level query

Action-level:
  selected action에 대한 approval query
```

기대 결과:

```text
state-level query는 hidden predicate를 직접 수정하므로
belief update와 replanning에 더 직접적으로 연결된다.
```

### 5.5 Experiment 4: Feedback Source Robustness

목적:

```text
feedback source의 정확도에 따라 성능이 어떻게 변하는지 확인한다.
```

비교:

```text
oracle
noisy-oracle
human-proxy
human
```

주의:

```text
baseline comparison은 oracle answer로 고정한다.
answer accuracy 실험은 main contribution이 아니라 robustness analysis로 둔다.
```

### 5.6 Optional: Scale Experiment

목적:

```text
scene size가 커질 때 query count, planning length, search time이 어떻게 변하는지 확인한다.
```

이 실험은 결과가 명확할 때 appendix 또는 짧은 scalability subsection으로 둔다.

## 6. 토의

### 6.1 기대되는 주장

본 논문의 핵심 주장은 다음이다.

```text
부분 관측 환경에서 로봇은 모든 불확실성을 물어볼 필요가 없다.
현재 planning decision에 영향을 주는 uncertainty만 식별하고,
필요한 시점에 state-level query로 물어보면,
적은 질문으로 task failure를 줄일 수 있다.
```

### 6.2 KnowNo와의 차이

KnowNo는 action option의 uncertainty를 보고 사용자에게 어떤 option이 맞는지 묻는다. 반면 본 논문은 action option을 직접 묻는 대신, action uncertainty의 원인이 되는 hidden predicate를 묻는다.

정리하면:

```text
KnowNo:
  What action should I take?

Ours:
  What state information is missing for planning?
```

이 차이가 본 논문의 `How` contribution이다.

### 6.3 Active Search / Sensing Action과의 차이

Information gathering action을 action space에 넣는 방식은 POMDP 관점에서 자연스럽다. 그러나 query action을 일반 action으로 추가하면 search space가 커지고, physical action과 query action이 같은 planning budget을 경쟁한다.

본 논문은 query를 physical action과 같은 action space에 직접 넣기보다, planning uncertainty를 평가한 뒤 별도의 feedback layer에서 state-level query를 생성한다. 따라서 physical planning과 human feedback을 역할적으로 분리한다.

### 6.4 Human-facing System으로서의 의미

사람에게 묻는 시스템에서 중요한 것은 단순히 질문을 많이 하는 것이 아니라, 사람이 답할 수 있는 질문을 필요한 순간에만 하는 것이다. State-level predicate query는 사용자가 명확히 판단할 수 있는 symbolic statement로 질문을 제한하므로, belief update와 사용자 응답 사이의 연결을 명확하게 만든다.

### 6.5 한계

현재 예상되는 한계:

- predicate query가 symbolic domain representation에 의존한다.
- feedback source가 틀릴 경우 belief update가 오히려 잘못될 수 있다.
- human-proxy가 실제 human behavior를 충분히 반영하는지는 별도 검증이 필요하다.
- POMCP / online planning의 computational cost가 domain scale에 따라 증가할 수 있다.
- state-level query가 항상 사람에게 가장 쉬운 질문 형식인지는 user study가 필요하다.

### 6.6 다음 수정 포인트

앞으로 이 문서에서 같이 수정할 부분:

- abstract를 더 짧고 논문식으로 다듬기
- related work에 실제 citation 추가하기
- method section을 algorithm 중심으로 재작성하기
- experiment table 초안 만들기
- 각 contribution별 expected figure 정리하기
