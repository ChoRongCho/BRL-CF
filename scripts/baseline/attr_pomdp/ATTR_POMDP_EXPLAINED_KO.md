# Attr-POMDP를 쉽게 이해하기

## 한 문장으로 설명하면

Attr-POMDP는 로봇이 현재 가진 후보 중 하나를 바로 선택할지, 사람에게
질문해서 후보를 더 줄일지를 **질문의 정보 가치와 질문 비용을 함께 계산해
결정하는 방법**이다.

이 폴더의 코드는 다음 논문을 현재 BRL 시스템에 맞게 다시 구현한 것이다.

- Yang Yang, Xibai Lou, Changhyun Choi
- *Interactive Robotic Grasping with Attribute-Guided Disambiguation*
- ICRA 2022, arXiv:2203.08037
- 로컬 PDF: [`attr_pomdp_icra2022.pdf`](attr_pomdp_icra2022.pdf)
- 웹: <https://arxiv.org/abs/2203.08037>

중요한 점은 저자의 공식 소스 코드를 복사한 것이 아니라는 것이다. 공개된
공식 저장소를 찾지 못했기 때문에, 논문에 적힌 수식과 설정을 바탕으로 만든
**paper-based reimplementation**이다.

## 원 논문에서는 무슨 문제를 푸는가?

예를 들어 탁자에 다음 물체가 있다고 하자.

1. 빨간 사과
2. 초록 사과
3. 초록 배

사람이 로봇에게 “사과를 집어 줘”라고 말하면 빨간 사과와 초록 사과 중
어느 것을 뜻하는지 알 수 없다. 로봇에게는 세 가지 선택이 있다.

- `AskAttr(color)`: “원하는 사과는 빨간색인가요, 초록색인가요?”
- `AskPoint(object)`: 물체를 가리키며 “이 물체인가요?”
- `Grasp(object)`: 가장 가능성이 높은 물체를 바로 집기

질문을 많이 하면 정답을 고르기 쉬워지지만 사람을 계속 귀찮게 하고 시간이
든다. 반대로 너무 일찍 집으면 틀린 물체를 집을 수 있다. Attr-POMDP는 이
두 위험을 하나의 계획 문제로 계산한다.

## POMDP 구성 요소

원 논문의 Attr-POMDP는 다음 요소로 구성된다.

| 요소 | 쉬운 의미 | 원 논문에서의 의미 |
| --- | --- | --- |
| 숨은 상태 `X` | 실제 정답 | 사람이 원하는 target object |
| belief `b(x)` | 각 후보가 정답일 확률 | object별 target 확률 |
| action `A` | 지금 할 선택 | `AskAttr`, `AskPoint`, `Grasp` |
| observation `Ω` | 질문에 대한 답 | 색상·위치 단어 또는 yes/no |
| observation model `O` | 후보별로 그 답이 나올 확률 | attribute score와 사용자 응답 확률 |
| reward `R` | 행동의 가치와 비용 | 올바른 grasp 보상 및 질문 비용 |

대화하는 동안 사람이 원하는 target은 바뀌지 않는다고 가정하므로 숨은
상태의 transition은 deterministic identity이다.

## 논문의 reward

논문은 다음 값을 사용한다.

| 행동 | Reward |
| --- | ---: |
| 올바른 후보 선택 | `+1.0` |
| 잘못된 후보 선택 | `-1.0` |
| attribute 질문 | `-0.1` |
| pointing 질문 | `-0.3` |

따라서 질문은 무료가 아니다. 질문으로 얻을 것으로 예상되는 이익이 `0.1`
보다 작다면 바로 현재 최선의 후보를 선택하는 편이 낫다.

예를 들어 현재 두 후보의 확률이 `[0.99, 0.01]`이라면 첫 후보를 바로
선택해도 거의 틀리지 않는다. 이때 planner는 일반적으로 질문하지 않고
commit한다. 반대로 `[0.5, 0.5]`인데 두 후보를 잘 나누는 attribute가 있다면
질문 비용을 내더라도 물어보는 편의 기대 reward가 높다.

## 질문을 언제 하고, 무엇을 묻는가?

Attr-POMDP는 When과 What을 따로 규칙으로 정하지 않는다. 다음 action들을
같은 후보 목록에 넣고 expected return을 비교한다.

```text
지금 가장 유력한 후보로 commit
attribute A 질문
attribute B 질문
attribute C 질문
...
```

각 질문 뒤에 가능한 답을 가상으로 펼치고, 답을 받았다고 가정한 posterior
belief에서 다시 질문하거나 commit한다. 이 폴더의 기본 search depth는
논문과 같은 `3`이다.

```text
현재 belief
├─ 지금 commit
├─ 질문 A
│  ├─ True를 받았을 때의 다음 최선 행동
│  └─ False를 받았을 때의 다음 최선 행동
└─ 질문 B
   ├─ True를 받았을 때의 다음 최선 행동
   └─ False를 받았을 때의 다음 최선 행동
```

- 질문 action의 가치가 commit보다 크면 질문한다. 이것이 **When**이다.
- 여러 질문 중 expected return이 가장 큰 질문을 고른다. 이것이 **What**이다.

즉 Attr-POMDP는 질문 시점과 질문 내용을 하나의 POMDP action selection으로
동시에 결정한다.

## 답을 받은 뒤 belief는 어떻게 바뀌는가?

각 후보가 질문의 답을 만들어 낼 likelihood를 사용해 Bayes update를 한다.

```text
posterior(candidate)
  ∝ prior(candidate) × P(answer | candidate, question)
```

기본 사용자 응답 모델은 논문과 같이 `0.99`의 정확도를 사용한다. 예를 들어
두 후보가 `[0.5, 0.5]`이고 첫 후보만 `red`라는 attribute를 가진 상태에서
`red=True`라는 답을 받으면 posterior는 약 `[0.99, 0.01]`이 된다.

여기서 `ANSWER_ACCURACY=0.99`는 Attr-POMDP가 계획과 belief update에 사용하는
**내부 observation model**이다. 실제 답을 새로 생성하는 설정이 아니다. 실제
답은 기존 BRL Oracle이 생성한다.

## BRL 시스템에는 어떻게 대응시켰는가?

원 논문은 “여러 물체 중 어느 물체가 target인가?”를 다룬다. 현재 BRL은
action을 실행한 뒤 “여러 symbolic successor state 중 어느 상태가
실제인가?”라는 belief를 가진다. 따라서 다음처럼 대응시켰다.

| 원 논문의 Attr-POMDP | 현재 BRL 구현 |
| --- | --- |
| candidate target object | belief의 candidate frontier state |
| object matching score | frontier weight |
| object attribute | frontier 사이에서 truth value가 갈리는 symbolic fact |
| `AskAttr(attribute)` | “이 symbolic fact가 true인가?” |
| 사용자 답변 | 기존 BRL Oracle의 Boolean 답변 |
| attribute observation model | fact 포함 여부와 `0.99` 응답 모델 |
| `Grasp(candidate)` | 가장 확률이 높은 frontier state를 knowledge로 commit |
| depth-3 object belief tree | depth-3 symbolic belief tree |

### 간단한 Waste Sorting 예시

`detect_waste` 뒤의 frontier가 다음과 같다고 하자.

```text
state 1: paper(waste1), general(waste2), plastic(waste3)
state 2: paper(waste1), plastic(waste2), general(waste3)
state 3: can(waste1), general(waste2), plastic(waste3)
```

다음 fact들은 state들을 서로 구분할 수 있다.

```text
paper(waste1)
general(waste2)
plastic(waste2)
general(waste3)
```

Attr-POMDP는 각 fact 질문 뒤에 belief가 얼마나 좋아지는지와 질문 비용을
계산한다. `general(waste2)`가 후보를 가장 효과적으로 나누고 그 가치가
commit보다 크다면 다음 질문을 선택한다.

```text
Is general(waste2) true?
```

Oracle 답을 받은 뒤 posterior를 계산하고, 다시 질문할지 MAP state를
commit할지 계획한다.

## 기존 Oracle과의 관계

Oracle은 Attr-POMDP 폴더에 새로 구현하지 않았다. 기존
`scripts/models/feedback_manager.py`의 Oracle을 그대로 호출한다.

- `detect`, `scan`, `detect_waste`처럼 실제 관측 GT가 있는 action은 기존
  domain rule과 GT에 따라 답한다.
- `navigate`, `pick`, `place`처럼 직접적인 질문 GT가 없는 action은 기존
  transition table의 확률로 virtual successor outcome을 한 번 뽑는다.
- 같은 robot action 뒤에 여러 질문을 하더라도 모두 동일한 virtual successor
  outcome을 기준으로 답한다.

따라서 baseline 사이에서 Oracle 의미가 달라지지 않는다. 차이는 질문할지와
어떤 fact를 질문할지를 Attr-POMDP가 선택한다는 점뿐이다.

## 원 논문과 완전히 같은 부분과 다른 부분

### 같은 부분

- 질문과 최종 선택을 expected return으로 함께 비교한다.
- 질문 비용과 성공·실패 reward를 사용한다.
- observation likelihood로 Bayesian belief update를 한다.
- depth-3 belief tree를 사용한다.
- 기본 reward와 응답 정확도가 논문의 값과 같다.

### 다른 부분

1. 원 논문의 candidate는 물체이지만 BRL의 candidate는 symbolic state이다.
2. 원 논문의 attribute는 color/location concept이지만 BRL에서는 symbolic
   fact가 attribute 역할을 한다.
3. BRL에는 물체를 물리적으로 가리키는 질문 인터페이스가 없으므로
   `AskPoint`는 실제 adapter action에서 제외했다. 수학 모델 자체는
   `planner.py`에 구현되어 있다.
4. 원 논문의 action space는 color/location처럼 작다. BRL frontier에는
   수십 개 fact가 생길 수 있으므로, expected posterior entropy가 가장 낮은
   상위 8개 fact만 depth-3 tree에 넣는다. 이 값은 설정으로 바꿀 수 있고
   로그에도 기록된다.
5. task action 자체는 기존 BRL POMCP가 선택한다. Attr-POMDP는 각 task action
   이후의 질문과 state commit만 담당한다.

이 차이 때문에 논문에서는 결과를 단순히 “official Attr-POMDP”라고 쓰면 안
된다. 다음과 같이 표현하는 것이 정확하다.

> Attr-POMDP, reimplemented and adapted to symbolic successor-state beliefs.

## 코드 구조

| 파일 | 역할 |
| --- | --- |
| `planner.py` | 논문의 generic finite-horizon Attr-POMDP 계산 |
| `controller.py` | BRL frontier를 candidate와 attribute로 변환하고 Oracle 호출 |
| `run_experiment.py` | 기존 environment/POMCP와 결합한 실험 loop |
| `run.sh` | 한 episode 실행 |
| `iterate.sh` | 두 domain, 다섯 scene, scene당 40회 실행 |
| `README.md` | 간단한 실행 안내와 구현 출처 |
| `attr_pomdp_icra2022.pdf` | 원 논문 PDF |

## 주요 설정

`run.sh` 위쪽 또는 환경변수로 바꿀 수 있다.

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `ATTR_DEPTH` | `3` | 질문 belief-tree 깊이 |
| `ATTRIBUTE_COST` | `0.1` | attribute 질문 한 번의 비용 |
| `ANSWER_ACCURACY` | `0.99` | 내부 사용자 응답 likelihood |
| `MAX_CANDIDATE_QUESTIONS` | `8` | tree에 넣을 symbolic fact 수 |
| `MAX_QUESTIONS_PER_ACTION` | `10` | robot action 하나 뒤의 안전 질문 상한 |
| `MAX_STEP` | `50` | task episode 최대 step |

## 실행 방법

저장소 root에서 한 episode를 실행한다.

```bash
scripts/baseline/attr_pomdp/run.sh
```

Waste Sorting scene 3을 고정 seed로 실행하려면 다음과 같다.

```bash
DOMAIN=wastesorting SCENE=03 SEED=1234 \
scripts/baseline/attr_pomdp/run.sh
```

전체 400회 실험은 다음과 같다.

```bash
scripts/baseline/attr_pomdp/iterate.sh
```

로그는 다음 위치에 저장된다.

```text
experiments_logs/system_log/<domain>/scene_<NN>_step50/attr_pomdp/
```

로그의 `[Meta]`에는 논문 설정과 BRL adapter 설정이 함께 기록된다.

```text
implementation: paper_based_reimplementation
attr_depth: 3
attribute_cost: 0.1
answer_accuracy: 0.99
max_candidate_questions: 8
```

## 결과를 해석할 때 주의할 점

- 이 baseline의 핵심 비교 대상은 **언제 질문하고 무엇을 묻는가**이다.
- 질문 횟수가 적다는 사실만으로 좋은 방법은 아니다. task success rate와 함께
  봐야 한다.
- Attr-POMDP 내부의 `0.99` likelihood와 실제 Oracle transition 확률은 서로
  다른 개념이다.
- 원 논문과 BRL의 hidden state 의미가 다르므로 원 논문에 보고된 성공률과
  BRL 성공률을 직접 비교하면 안 된다.
- 논문 표에는 `Attr-POMDP (adapted)` 또는 `Attr-POMDP (reimplementation)`처럼
  표기하는 것이 안전하다.
