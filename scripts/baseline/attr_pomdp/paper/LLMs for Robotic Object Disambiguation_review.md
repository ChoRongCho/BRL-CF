# LLMs for Robotic Object Disambiguation 리뷰

## 1. 논문 정보

```text
Connie Jiang, Yiqing Xu, and David Hsu,
“LLMs for Robotic Object Disambiguation,”
arXiv:2401.03388v1 [cs.RO], 7 January 2024.
```

- 원문: [LLMs for Robotic Object Disambiguation.pdf](<./LLMs for Robotic Object Disambiguation.pdf>)
- 출판 상태: arXiv v1 preprint
- 핵심 방법: few-shot prompting으로 disambiguation decision tree 생성
- 핵심 비교 대상: Enumeration, Human, POMDP-ATTR, theoretical Optimal Split
- 실험 입력: RGB-D image가 아니라 사람이 작성한 scene text description

## 2. 한 문장 요약

이 논문은 LLM이 scene description에 명시되지 않은 새로운 distinguishing feature를
추론하고, target object를 특정하기 위한 multi-turn question decision tree와
occlusion-aware action plan을 few-shot prompting으로 생성할 수 있는지 탐색한다.

## 3. 연구 동기

기존 object-disambiguation 방법은 크게 두 문제가 있다.

```text
Enumeration:
  candidate object를 하나씩 확인
  → 완전하지만 질문 수가 많음

Attr-POMDP:
  predefined color/location attribute로 정보량 높은 질문 선택
  → 효율적이지만 미리 정의하지 않은 feature를 질문할 수 없음
```

논문은 pretrained LLM의 commonsense knowledge를 이용하면 다음을 동시에 달성할 수
있다고 주장한다.

- 여러 candidate를 한 번에 나누는 질문 생성
- scene description에 없던 새로운 distinguishing feature 추론
- sequential question decision tree 생성
- target object 위의 occluder를 제거한 뒤 deliver하는 action plan 생성
- “something to eat” 같은 broad request 해석

## 4. 논문이 다루는 세 기능

### 4.1 Generalizing user requests

정확한 object name 대신 기능이나 선호를 표현한 요청을 해석한다.

```text
“Get me something to eat.”
“Get me something to write with.”
```

LLM은 scene의 여러 object 중 request에 부합하는 candidate category를 추론한다.

### 4.2 Maneuvering occluding objects

target object가 다른 object 아래에 있으면 먼저 occluder를 옮기는 action sequence를
생성한다.

```text
move away occluding object
→ deliver target object
```

논문은 이를 full manipulation planner로 검증하지 않고 language action-plan
generation 예시로 제시한다.

### 4.3 Target-object disambiguation

여러 object가 request에 부합하면 질문 sequence를 생성한다. 목표는 가능한 target
set을 질문마다 크게 나누고, 최종적으로 하나의 target만 남기는 것이다.

```text
scene candidates
→ question
→ answer-dependent subsets
→ additional question
→ one target object
```

## 5. Zero-shot prompting

논문의 zero-shot prompt는 LLM에게 다음 action format을 요구한다.

```text
<ask> <question>
<move away> <object>
<deliver> <object>
```

그리고 다음 두 JSON-like 결과를 생성하도록 한다.

```text
Action Planner
Decision Tree
```

단순 scene에서는 zero-shot prompting도 color, size 등의 명시된 feature를 이용해
효율적으로 candidate를 나눈다.

예:

```text
four cups:
  two blue, two green
  different sizes

Q1: preferred color?
Q2: large or small?
```

### 5.1 Zero-shot의 실패

scene description에 필요한 구별 기준이 모두 쓰여 있지 않으면 LLM이 새로운
feature를 충분히 만들지 못한다.

논문의 plum pyramid 예에서는 layer는 명시되어 있지만 같은 layer 안의 relative
position은 명시되어 있지 않다. Zero-shot LLM은 layer를 물은 뒤 하나의 plum이
정해졌다고 잘못 가정한다.

즉:

```text
explicit feature 사용: 가능
unstated feature 추론: 불안정
모든 leaf가 unique object인지 검증: 불완전
```

## 6. Few-shot 방법

논문은 zero-shot prompt에 object-disambiguation 예시를 추가한다. 예시에는 다음이
포함된다.

- broad request에서 candidate category 분리
- 동일 category 내 relative position 질문
- answer-dependent nested options
- occluding object 제거
- 최종 target deliver
- 완전한 decision tree

few-shot example을 본 LLM은 scene description에 직접 없던 distinguishing feature를
생성한다.

plum pyramid 예:

```text
Q1: bottom / middle / top layer?
Q2: front / middle / back row?
Q3: left / middle / right?
```

논문은 row와 row 내부 위치를 LLM이 새로 만든 feature로 해석한다.

## 7. POMDP가 아닌 decision-tree generation

논문은 POMDP로 다루어지던 문제를 LLM이 풀 수 있음을 강조하지만, 제안 방법 자체는
POMDP가 아니다.

구현되지 않은 요소:

- explicit belief state
- transition probability
- observation likelihood
- reward function
- question cost
- Bellman backup
- online posterior update
- stochastic human-answer model

LLM이 한 번에 question decision tree와 action plan을 생성한다. 따라서 실제
interaction 중 observation을 받을 때 POMDP belief를 update하는 구조와는 다르다.

이 논문에서 `POMDP-ATTR`은 제안 방법이 아니라 비교 baseline이다.

## 8. 실험

### 8.1 Scene 구성

- tabletop scene 12개
- 다양한 object configuration과 feature 조합
- stacked object를 포함한 3D arrangement
- 각 scene-inquiry pair에 대해 3 trials
- root는 모든 candidate object
- leaf는 최종 target object

각 방법이 생성한 decision tree를 따라 target object를 식별할 수 있는지와 질문
수를 측정한다.

### 8.2 Baseline

`Optimal Split`:

- scene의 object와 attribute를 모두 알고 있을 때의 theoretical minimum
- 절대적인 query-count lower bound

`Enumeration`:

- candidate를 하나씩 가리키며 target인지 확인
- target 수가 `k`이면 평균 질문 수 `(k+1)/2`
- 논문은 이를 INVIGORATE가 사용하는 방식으로 설명

`Human Performance`:

- participant가 scene image와 description을 보고 sequential question 생성
- target을 찾을 때까지 질문하므로 accuracy는 100%
- 불필요하거나 candidate를 나누지 못하는 질문을 할 수 있음

`POMDP-ATTR`:

- 논문의 표기이며 원 논문의 `Attr-POMDP`를 의미
- color와 9개 relative location feature에 제한
- stacked 3D configuration에서 feature representation이 부족하다고 평가

### 8.3 결과

제안 방법의 보고된 success rate:

```text
95.79%
```

질문 수 기준 개선율:

| 비교 대상 | LLM method의 query-count improvement |
|---|---:|
| Enumeration | 61.91% |
| Human | 18.37% |
| POMDP-Attr | 26.00% |
| Optimal Split | -18.39% |

Optimal Split보다 `-18.39%`라는 것은 이론적 최소 질문 수보다 제안 방법이 더 많은
질문을 사용한다는 뜻이다.

stacked object가 없는 flat scene 4, 5, 12만 비교하면 LLM method는 Attr-POMDP보다
질문 수가 4.88% 적었다. 전체 26% 차이의 상당 부분은 stacked scene에서
Attr-POMDP의 고정 location representation이 맞지 않았기 때문에 발생한다.

## 9. 강점

1. predefined attribute에 제한되지 않는 질문 생성을 시도한다.
2. complex 3D arrangement에 맞는 hierarchical spatial feature를 만들 수 있다.
3. broad request interpretation과 disambiguation을 연결한다.
4. 질문 하나가 여러 candidate를 나누도록 decision tree를 구성한다.
5. 질문뿐 아니라 occluder removal과 delivery action도 동일 output에 표현한다.
6. Attr-POMDP를 최신 LLM 방법의 concrete baseline으로 사용했다는 점에서 관련
   연구 흐름을 보여준다.

## 10. 핵심 한계

### 10.1 사용한 LLM이 명시되지 않음

PDF 본문에는 실험에 사용한 구체적인 model name, version, API가 명확히 제시되지
않는다. 따라서 같은 prompt를 사용해도 결과를 재현하기 어렵다.

### 10.2 Inference 설정이 없음

다음 정보가 보고되지 않는다.

- temperature
- top-p
- random seed
- number of sampled completions
- retry/repair policy
- invalid JSON 처리
- prompt 전체 shot 수와 selection 절차
- model output을 decision tree로 변환하고 검증하는 parser

Appendix에는 zero-shot prompt와 few-shot example 하나가 있지만 완전한 reproducible
pipeline을 구성하기에는 부족하다.

### 10.3 실제 vision pipeline이 없음

논문도 next step에서 현재 input이 natural-language scene description임을 인정한다.
RGB-D image를 scene text로 변환하는 visual module은 구현되지 않았다.

따라서 실험은 다음을 검증한다.

```text
given a hand-authored textual scene description,
can an LLM generate a useful disambiguation tree?
```

다음은 검증하지 않는다.

```text
raw visual input
→ perception
→ grounded scene
→ question
→ physical execution
```

### 10.4 Real robot 실험이 없음

실제 robot execution, perception error, speech recognition error, response latency,
manipulation failure를 평가하지 않는다.

### 10.5 Human 실험 정보가 부족함

Human baseline의 participant 수, 모집 방식, trial allocation, 통계 검정 등이
본문에서 충분히 보고되지 않는다. Human accuracy를 질문을 계속하면 100%라고
가정하는 것도 실제 interaction error를 반영하지 않는다.

### 10.6 Success 정의가 약함

success는 generated decision tree에 root에서 target leaf까지 valid path가 있는지로
정의된다. 실제 user response를 받아 online으로 정확히 target을 선택하고 task를
성공하는 것과 동일하지 않다.

### 10.7 비교 공정성

Attr-POMDP는 color/location으로 제한된 원래 representation을 그대로 적용하고,
LLM은 임의의 새 feature를 생성할 수 있다. stacked scene은 Attr-POMDP에 불리하도록
representation mismatch가 크다.

flat scene만 보면 질문 수 차이가 4.88%로 줄어든다는 결과가 이를 보여준다.

### 10.8 “arbitrarily large” 주장의 근거 부족

논문은 arbitrarily large/complex scene을 강조하지만 실험은 12개 scene으로
제한된다. object 수 증가에 따른 latency, token cost, context length, parsing failure,
decision-tree 크기 증가를 체계적으로 평가하지 않는다.

### 10.9 LLM hallucination과 consistency

새 feature를 생성하는 능력은 장점이지만 해당 feature가 scene에서 실제로 관측 또는
질문 가능한지 검증하는 grounding layer가 없다. 존재하지 않는 relation이나
구별할 수 없는 feature를 생성할 위험이 있다.

## 11. Attr-POMDP 논문과 비교

| 항목 | Attr-POMDP | LLM disambiguation |
|---|---|---|
| 방법 | explicit POMDP | few-shot prompted LLM |
| Hidden state | target object | 명시적 확률 state 없음 |
| 질문 생성 | color/location template | open-ended generation |
| 질문 선택 | expected return | generated decision tree |
| Belief update | Bayesian | 없음 |
| Human model | observation likelihood | tree branch assumption |
| Reward/query cost | 명시적 | 없음 |
| Novel feature | 어려움 | 생성 가능 |
| Reproducibility | 비교적 높음 | 낮음 |
| Raw vision | RGB-D pipeline | 미구현 |
| Real robot | 있음 | 없음 |

두 논문은 상호 보완적이다.

```text
Attr-POMDP:
  decision-theoretic correctness와 uncertainty tracking

LLM method:
  open-ended feature와 question generation
```

## 12. BRL-CF와의 관련성

### 12.1 직접 가져올 수 있는 아이디어

- predefined predicate만으로 candidate를 충분히 구분할 수 없는지 검사
- candidate partition이 균형적인 질문을 선호
- 질문 sequence를 tree 형태로 시각화
- LLM/VLM provider가 향후 새로운 grounded fact candidate를 제안하도록 확장
- Optimal Split을 What-selection의 theoretical reference로 사용

### 12.2 직접 baseline으로 부적절한 이유

현재 BRL-CF의 E3는 동일한 symbolic domain에서 task success와 query efficiency를
비교해야 한다. 이 논문의 방법은:

- belief update가 없음
- task execution simulator가 없음
- reward가 없음
- raw perception이 없음
- 사용 model과 decoding 조건이 불명확함

따라서 지금 바로 메인 baseline으로 구현하면 논문 재현이 아니라 새로운 LLM
pipeline을 설계하는 작업이 된다.

### 12.3 적절한 논문 내 위치

메인 baseline보다는 최신 related work 또는 future extension으로 적합하다.

```text
Recent work uses few-shot prompted LLMs to generate
open-ended object-disambiguation trees beyond predefined
attribute vocabularies. In contrast, our method restricts
queries to grounded predicates whose truth values can be
incorporated into a maintained symbolic belief.
```

## 13. 향후 결합 가능성

Attr-POMDP와 이 논문의 장점을 결합하면 다음 구조가 가능하다.

```text
LLM/VLM:
  propose grounded candidate features/questions

Symbolic validator:
  reject ungrounded or non-observable predicates

POMDP/Ours selector:
  evaluate expected posterior/task value

Provider:
  answer selected grounded question

Belief manager:
  Bayesian update
```

중요한 원칙은 LLM이 질문을 생성하더라도 최종 action space에는 검증된 grounded
fact만 넣는 것이다.

## 14. 최종 평가

이 논문은 Attr-POMDP 이후 object disambiguation의 attribute vocabulary 한계를
LLM으로 넘으려는 흥미로운 exploratory study다. 특히 2024년 논문이 Attr-POMDP를
직접적인 state-of-the-art baseline으로 사용한다는 점은 BRL-CF가 adapted
Attr-POMDP를 비교 대상으로 선택하는 근거를 강화한다.

그러나 method specification과 experimental reporting이 부족하고 실제 perception 및
robot execution이 빠져 있다. 따라서 BRL-CF의 메인 baseline으로 사용하기보다는,
open-ended query generation을 다루는 최신 related work와 VLM/LLM 확장 방향으로
인용하는 것이 가장 타당하다.
