# Interactive Robotic Grasping with Attribute-Guided Disambiguation 리뷰

## 1. 논문 정보

```text
Yang Yang, Xibai Lou, and Changhyun Choi,
“Interactive Robotic Grasping with Attribute-Guided Disambiguation,”
IEEE International Conference on Robotics and Automation (ICRA),
2022, pp. 8914–8920.
DOI: 10.1109/ICRA46639.2022.9812360
```

- 원문: [Interactive Robotic Grasping with Attribute-Guided Disambiguation.pdf](<./Interactive Robotic Grasping with Attribute-Guided Disambiguation.pdf>)
- 제안 방법: `Attr-POMDP`
- 핵심 문제: 모호한 자연어 지시에서 target object를 식별하고 grasp하기
- 핵심 질문: 어떤 질문을 할 것인가, 그리고 계속 질문할 것인가 grasp할 것인가
- 실험: RefCOCO와 Franka Emika Panda 실제 로봇

## 2. 한 문장 요약

Attr-POMDP는 candidate object에 대한 belief와 color/location attribute score를
사용하여 attribute 질문, pointing 질문, grasp action의 expected return을 함께
계산하는 interactive object-disambiguation POMDP다.

## 3. 문제 정의

입력은 RGB-D image와 target을 설명하는 자연어 표현이다. 로봇은 표현에 해당하는
물체를 찾아 grasp해야 한다.

```text
Input:
  RGB-D image I
  natural-language query q

Output:
  target object identification
  target grasp
```

문제는 다음 이유로 target이 하나로 결정되지 않을 수 있다는 데서 시작한다.

- 사용자의 표현이 너무 일반적이거나 불완전함
- 외형이 비슷한 물체가 여러 개 존재함
- novel object가 존재함
- 가림 또는 불완전한 visual observation이 존재함
- vision-language grounder가 여러 물체에 비슷한 matching score를 부여함

논문은 이 모호성을 단일 confidence threshold나 greedy question으로 처리하지 않고,
질문과 grasp를 sequential decision problem으로 모델링한다.

## 4. 전체 시스템

```text
RGB-D image + language query
→ objectness detection
→ vision-language grounding
→ target matching scores
→ color/location attribute scores
→ Attr-POMDP planning
→ attribute question / pointing question / grasp
→ human response
→ belief update
→ replanning
```

### 4.1 Object detection

실제 로봇에서는 unknown object에도 대응할 수 있도록 UOIS-Net을 사용한다. 검출된
object proposal이 POMDP의 candidate object가 된다.

### 4.2 Language grounding

MAttNet을 사용하여 자연어 query와 각 candidate object의 matching score
`s(x_i | q)`를 계산한다. MAttNet은 subject, location, relation module을 결합한다.

```text
s(x | q) = Σ_j ω_j q_j · x_j

j ∈ {subject, location, relation}
```

이 matching score는 초기 object belief를 구성하는 데 사용된다.

### 4.3 Attribute grounding

논문 구현은 다음 두 attribute concept을 사용한다.

- Color: 자주 등장하는 10개 color value
- Location: image를 3×3으로 나눈 9개 relative location

각 object에 대해 attribute value의 probability distribution을 만든다. 이
distribution은 attribute 질문을 했을 때 받을 human response의 observation
likelihood로 사용된다.

## 5. Attr-POMDP 정식화

논문은 disambiguation 문제를 다음 6-tuple로 정의한다.

```text
<X, T, A, R, Ω, O>
```

### 5.1 State

```text
X = {x_1, x_2, ..., x_n}
```

state는 candidate object이고, 사용자가 실제로 원하는 target `x_d`가 hidden
state다. 따라서 belief `b_t(x_i)`는 “candidate `x_i`가 target일 확률”이다.

이 논문의 belief는 전체 physical world state에 대한 belief가 아니다. target
identity에 대한 categorical belief라는 점이 중요하다.

### 5.2 Transition

```text
T(x, a, x') = 1 if x' = x
```

사용자가 원하는 target은 dialogue 도중 변하지 않는다고 가정한다. 질문은 hidden
target state를 바꾸지 않는다.

### 5.3 Action

세 종류의 action을 사용한다.

```text
AskAttr(α)
AskPoint(x_b)
Grasp(x_b)
```

`AskAttr(α)`:

- attribute concept `α`에 대해 질문
- 논문 구현의 `α`는 color 또는 location
- 예: “What is the color of your target, red or yellow?”

`AskPoint(x_b)`:

- 현재 belief가 가장 높은 object를 가리키며 확인
- `x_b = argmax_x b_t(x)`
- 예: “Do you mean this one?”

`Grasp(x_b)`:

- 현재 belief가 가장 높은 object를 target으로 결정하고 grasp

즉, 질문할지 행동할지뿐 아니라 attribute question과 pointing question 중 무엇을
사용할지도 planning으로 결정한다.

### 5.4 Reward

논문의 reward는 경험적으로 다음처럼 설정된다.

| Action | Condition | Reward |
|---|---|---:|
| `AskAttr(α)` | 모든 state | `-0.1` |
| `AskPoint(x_b)` | 모든 state | `-0.3` |
| `Grasp(x_b)` | `x_b = x_d` | `+1` |
| `Grasp(x_b)` | `x_b ≠ x_d` | `-1` |

두 질문의 비용이 다른 이유는 실제 로봇에서 소요되는 interaction time을 반영하기
위해서다. Attribute question은 여러 candidate를 한 번에 제거할 수 있고 비용도
작아서 pointing question보다 선호될 수 있다.

이 reward는 별도의 entropy bonus를 주지 않는다. 질문의 information value는 질문
후 posterior에서 올바른 grasp를 선택할 미래 가치로 나타난다.

```text
Q(b, ask)
  = question cost
  + γ Σ_o P(o | b, ask) V(b_ask,o)
```

이는 BRL-CF의 Active Search baseline reward를 설계할 때 가장 직접적으로 재사용할
수 있는 부분이다.

### 5.5 Observation

observation space는 human response다.

Attribute question:

- color/location vocabulary에서 response를 sampling
- 각 object의 attribute score를 conditional likelihood로 사용

```text
p(o | x, AskAttr(α))
```

Pointing question:

```text
                 yes    no
x = target       0.99   0.01
x ≠ target       0.01   0.99
```

사용자가 99% cooperative하고 truthfully answer한다고 가정한다. 따라서 실제
Human/VLM의 다양한 오류나 no-answer는 모델링하지 않는다.

### 5.6 Initial belief

vision-language matching score를 0 이상으로 자르고 정규화한다.

```text
b_0(x_i)
  = max(s(x_i | q), 0)
    / Σ_k max(s(x_k | q), 0)
```

완전히 모호한 query에서는 uniform belief를 사용할 수 있다.

### 5.7 Belief update

질문 후 response `o`를 받으면 Bayes rule로 update한다.

```text
b_(t+1)(x)
  = (1 / η) p(o | x, a) b_t(x)
```

관측된 response를 생성할 가능성이 높은 candidate의 posterior가 증가한다.

### 5.8 Solver

- belief-tree search
- search depth `d = 3`
- 각 node에서 action과 observation을 sampling
- expected return이 가장 높은 현재 action 선택
- 논문은 DESPOT을 belief-tree planning의 참고문헌으로 인용

논문의 전체 loop는 다음과 같다.

```text
detect objects
→ compute matching and attribute scores
→ initialize belief
→ plan from current belief
→ if Grasp: execute and terminate
→ if Ask: receive response and update belief
→ repeat
```

## 6. 기존 방법과의 차이

네 방법의 차이는 다음 네 가지 질문으로 보면 된다.

1. **무엇을 묻는가?**  
   물체 하나를 직접 확인하는지, 여러 물체를 나눌 수 있는 속성을 묻는지 본다.
2. **질문을 어떻게 고르는가?**  
   지금 당장 가장 유용한 질문만 고르는지, 그 답을 들은 뒤의 다음 행동까지
   고려하는지 본다.
3. **어떤 답을 받을 수 있는가?**  
   `yes/no`처럼 답이 제한되는지, 자유로운 자연어 답변까지 다루는지 본다.
4. **언제 질문을 멈추고 물체를 집는가?**  
   미리 정한 confidence 기준으로 멈추는지, 질문 비용과 잘못 집을 위험을 함께
   비교하여 결정하는지 본다.

| 비교 기준 | Greedy | FETCH-POMDP | INGRESS-POMDP | Attr-POMDP |
|---|---|---|---|---|
| 무엇을 묻는가? | 현재 가장 애매한 단어나 속성 | 물체를 하나씩 가리키며 “이것인가?” | 특정 물체를 묘사하며 “왼쪽의 빨간 컵인가?” | “무슨 색인가?”, “어디에 있는가?”와 같은 공통 속성 |
| 질문을 어떻게 고르는가? | **이번 질문 하나**가 주는 정보만 비교 | 답을 들은 뒤의 상황까지 계산 | 답을 들은 뒤의 상황까지 계산 | 답을 들은 뒤의 상황까지 계산 |
| 받을 수 있는 답 | 구현에 따라 다름 | `yes/no` | 자연어 답을 단순화하여 처리 | 정해진 속성값 또는 `yes/no` |
| 언제 집는가? | 보통 confidence가 기준을 넘으면 집음 | 더 묻는 것과 지금 집는 것 중 유리한 쪽 선택 | 더 묻는 것과 지금 집는 것 중 유리한 쪽 선택 | 더 묻는 것과 지금 집는 것 중 유리한 쪽 선택 |

여기서 **답을 들은 뒤의 상황까지 계산한다**는 것은, 예를 들어 “색을 물었을 때
빨강이라고 답하면 후보가 몇 개 남고, 그다음 바로 집을 수 있는가?”까지 미리
따져본다는 뜻이다. 또한 **더 묻는 것과 지금 집는 것을 함께 비교한다**는 것은
질문 횟수와 오답 위험을 같은 의사결정 안에서 평가한다는 뜻이다.

### 6.1 Greedy method와 차이

Greedy method는 현재 가장 불확실한 단어나 후보를 가장 잘 나누는 속성을 질문한다.
예를 들어 빨간 사과와 초록 사과가 남았다면 바로 색을 묻는다. 질문 한 번만 보면
합리적이지만, 그 답을 들은 뒤 어떤 질문이 또 필요할지까지는 고려하지 않는다.

Attr-POMDP는 질문의 답마다 후보가 어떻게 달라지는지, 이후 다시 질문할지 또는
집을지를 미리 비교한다. 따라서 당장 정보량이 가장 큰 질문이 아니라, **최종적으로
적은 질문으로 안전하게 집게 만드는 질문**을 선택할 수 있다.

또한 Greedy method는 보통 “확신이 80%를 넘으면 집는다”와 같은 별도 기준이
필요하다. Attr-POMDP는 질문 비용과 잘못 집었을 때의 손실을 비교하여 질문을
멈출 시점도 결정한다. 다만 논문에서는 미리 계산하는 단계 수가 제한되어 있어,
아주 긴 질문 과정까지 완벽하게 고려하는 것은 아니다.

### 6.2 FETCH-POMDP와 차이

FETCH-POMDP는 후보 물체를 하나 가리키며 “이것인가?”라고 묻고 `yes/no` 답을
받는다. 답을 처리하기 쉽다는 장점이 있지만, `no`라면 대개 후보 하나만 제외된다.
후보가 여덟 개일 때 첫 물체가 아니라고 해도 일곱 개가 남는 식이다.

Attr-POMDP는 먼저 “무슨 색인가?” 또는 “어디에 있는가?”를 물어 여러 후보를 한꺼번에
제외한다. 그래도 애매하면 가장 가능성 높은 물체를 가리켜 확인한다. 따라서 후보가
많고 색이나 위치로 잘 나뉘는 장면에서는 필요한 질문 수를 줄일 수 있다.

즉, FETCH-POMDP의 핵심 한계는 **물체를 하나씩 확인한다는 것**이고,
Attr-POMDP의 핵심 변화는 **속성으로 후보 집합을 먼저 나눈다는 것**이다.

### 6.3 INGRESS-POMDP와 차이

INGRESS-POMDP는 후보마다 “왼쪽 그릇 옆의 빨간 컵인가?”와 같은 설명을 만들어
질문한다. 단순히 물체를 가리키는 FETCH-POMDP보다 구체적이고 자연스럽다는 장점이
있다.

그러나 생성된 설명이 틀리거나 새 물체를 제대로 묘사하지 못하면 질문 자체가
혼동을 만든다. 사람도 `yes/no` 대신 “아니, 그 뒤의 컵”처럼 자유롭게 답할 수
있으므로 가능한 답을 확률 모델로 표현하기도 어렵다.

Attr-POMDP는 문장 전체를 새로 생성하지 않고, 색과 위치처럼 미리 정한 속성만
질문한다. 답의 종류가 제한되므로 계산과 갱신이 단순하고, 물체마다 별도의 질문을
만들 필요도 없다. 대신 색이나 위치로 구분할 수 없는 물체에는 약하며, 크기·형태·
재질과 같은 새로운 속성을 사용하려면 모델에 별도로 추가해야 한다.

결국 INGRESS-POMDP는 **표현력이 높지만 생성 오류와 다양한 답변을 다루기 어렵고**,
Attr-POMDP는 **표현 범위를 줄이는 대신 안정적이고 계산하기 쉬운 질문을 사용한다**.

## 7. 실험

### 7.1 RefCOCO ablation

약 1,500개의 RefCOCO validation image를 사용한다.

| Query | Random select | MAttNet | MAttNet + Attr-POMDP |
|---|---:|---:|---:|
| Unambiguous | 15.60% | 85.54% | 98.85%, 1.42 questions |
| Ambiguous | 15.60% | 27.34% | 92.69%, 1.71 questions |

Ambiguous query는 category만 주거나 prior를 전혀 주지 않는 형태로 생성한다.

### 7.2 RefCOCO baseline 비교

| Method | Accuracy | Questions | Planning/disambiguation time |
|---|---:|---:|---:|
| RandAsk | 80.28% | 3.46 | 0.12 ms |
| REG | 78.53% | 3.62 | 30.2 ms |
| FETCH-POMDP | 87.24% | 2.61 | 4023.8 ms |
| INGRESS-POMDP | 87.42% | 2.25 | 321.5 ms |
| Attr-POMDP | **92.69%** | **1.71** | 33.1 ms |

Attr-POMDP는 attribute 질문으로 여러 candidate를 먼저 제거하고 필요할 때 pointing
질문으로 마무리한다.

논문은 FETCH/INGRESS-POMDP가 object별 action으로 `O(n^d)`의 search complexity를
갖는 반면, Attr-POMDP의 attribute concept 수는 상수이므로 `O(c^d)`라고 설명한다.
다만 실제 action 수에는 attribute value와 구현 세부가 영향을 줄 수 있으므로 이
복잡도 주장은 제한된 action 설계를 전제로 이해해야 한다.

### 7.3 실제 로봇

- Franka Emika Panda
- Intel RealSense D415
- household object 28개
- scene/query configuration 70개
- 한 scene에 object 8–10개

| Method | Success | Accuracy | Questions |
|---|---:|---:|---:|
| MAttNet | 32/70 | 45.71% | - |
| RandAsk | 55/70 | 78.57% | 3.74 |
| REG | 52/70 | 74.29% | 3.84 |
| FETCH-POMDP | 57/70 | 81.43% | 3.20 |
| INGRESS-POMDP | 60/70 | 85.71% | 2.69 |
| Attr-POMDP | **64/70** | **91.43%** | **2.03** |

## 8. 강점

1. `when to ask`와 `what to ask`를 하나의 decision model로 처리한다.
2. 질문과 최종 task action을 동일 reward 아래에서 비교한다.
3. 질문의 information value가 posterior의 미래 task value로 나타난다.
4. attribute question과 pointing question을 함께 사용한다.
5. 제한된 response vocabulary 덕분에 observation model을 명시적으로 정의할 수 있다.
6. 실제 로봇과 human response를 포함한다.
7. 다른 vision-language grounder에 결합 가능한 modular disambiguation layer다.
8. method name, POMDP tuple, reward 값, belief update가 구체적이어서 재구현 가능성이
   비교적 높다.

## 9. 한계

### 9.1 Hidden state가 target identity에 한정됨

Attr-POMDP의 belief는 candidate target object에 대한 categorical distribution이다.
robot location, object state, manipulation outcome 등 전체 world state uncertainty를
다루지 않는다.

### 9.2 Attribute vocabulary가 고정됨

실제 구현은 color 10개와 location 9개만 사용한다. size, shape, material, texture,
task state 등의 새로운 attribute는 model에 직접 추가해야 한다.

### 9.3 Human model이 단순함

- pointing answer accuracy 99%
- cooperative user 가정
- no-answer/timeout 없음
- participant별 accuracy 차이 없음

### 9.4 Reward가 경험적으로 결정됨

`-0.1`, `-0.3`, `±1`은 실제 interaction time을 대략 반영한 값이며 이론적으로
도출된 값은 아니다. 다른 domain에서는 calibration 또는 sensitivity analysis가
필요하다.

### 9.5 질문 생성보다 선택에 초점

질문 template과 vocabulary는 미리 정해져 있다. open-ended grounded fact generation
문제를 해결하지 않는다.

### 9.6 Search depth가 짧음

`d = 3`이므로 장기 task execution과 여러 manipulation step이 포함되는 domain에서는
질문의 장기 가치가 horizon 밖으로 밀릴 수 있다.

## 10. BRL-CF와의 대응

| Attr-POMDP | BRL-CF adaptation |
|---|---|
| hidden target object `x_d` | hidden symbolic world state / belief particle |
| color/location attribute | grounded Boolean predicate |
| `AskAttr(α)` | `QueryFact(f)` |
| `AskPoint(x_b)` | 필요하면 full-hypothesis confirmation |
| `Grasp(x_b)` | navigate/detect/pick/scan/place/discard |
| matching-score belief | symbolic particle belief |
| human response | Oracle/Human/VLM Boolean answer |
| correct grasp `+1` | task success/task reward |
| wrong grasp `-1` | failure/dead-end penalty |
| question cost | feedback interaction cost |

가장 중요한 공통점:

```text
TaskAction ∪ QueryAction
→ same POMDP action space
→ expected return으로 질문과 task action 공동 선택
```

가장 중요한 차이:

```text
Attr-POMDP:
  target identity disambiguation

BRL-CF:
  task execution 중 변하는 symbolic world-state uncertainty
```

따라서 구현은 논문의 코드 재현이 아니라 다음처럼 표기해야 한다.

```text
Adapted Attr-POMDP with grounded Boolean fact queries
```

## 11. Baseline 구현 시 가져올 요소

직접 가져올 것:

1. QueryAction과 task action의 공동 planning
2. query no-op transition
3. answer observation likelihood
4. Bayesian belief update
5. task reward와 query cost
6. 별도 entropy bonus 없는 value-of-information
7. 질문 비용 sensitivity analysis

그대로 가져오지 않을 것:

1. target-object categorical state
2. MAttNet/UOIS-Net
3. color/location 전용 attribute matrix
4. pointing gesture action
5. 고정 reward 수치

## 12. 논문에서의 사용

메인 baseline 이름:

```text
Adapted Attr-POMDP
```

권장 설명:

```text
We adapt Attr-POMDP (Yang et al., ICRA 2022) to our symbolic
domains by replacing attribute-based object-disambiguation
questions with grounded Boolean fact queries. Unlike our
corrective layer, the adapted baseline treats query actions and
physical task actions within a unified POMDP action space.
```

비교가 검증하는 질문:

```text
정보 획득을 task planning 안에서 공동 최적화하는 방식과
planner 외부 corrective query layer 중 어느 방식이
success-query-computation trade-off에서 더 효율적인가?
```

## 13. 최종 평가

이 논문은 현재 `active_search` baseline의 가장 적절한 직접 근거다. 구체적인 method
name이 있고, 질문이 action이며, observation/reward/belief update가 명확하다.

다만 BRL-CF는 target object 선택보다 훨씬 큰 symbolic world-state와 multi-step
execution을 다루므로 “Attr-POMDP reproduction”이라고 부르면 안 된다. 핵심
decision structure를 grounded-fact domain에 맞게 옮긴 `adapted baseline`으로
정직하게 보고해야 한다.
