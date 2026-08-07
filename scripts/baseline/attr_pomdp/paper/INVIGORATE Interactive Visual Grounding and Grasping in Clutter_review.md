# INVIGORATE: Interactive Visual Grounding and Grasping in Clutter 리뷰

## 1. 논문 정보

- 제목: **INVIGORATE: Interactive Visual Grounding and Grasping in Clutter**
- 저자: Hanbo Zhang, Yunfan Lu, Cunjun Yu, David Hsu, Xuguang Lan, Nanning Zheng
- 학회: Robotics: Science and Systems (RSS)
- 연도: 2021
- DOI: `10.15607/RSS.2021.XVII.020`
- 핵심 주제: visual grounding, interactive disambiguation, cluttered grasping, POMDP

## 2. 한 문장 요약

INVIGORATE는 사용자의 자연어 지시에서 목표 물체를 찾고, 필요하면 확인 질문을 하며,
목표를 가린 물체를 순서대로 제거한 뒤 최종 목표를 집는 전 과정을 POMDP로 계획한다.

## 3. 연구 문제

사용자가 “파란 노트를 가져다줘”라고 말해도 로봇은 다음 문제를 동시에 해결해야 한다.

1. 어떤 물체가 사용자가 말한 목표인가?
2. 목표가 보이지 않거나 다른 물체에 가려져 있는가?
3. 여러 후보가 남으면 사람에게 질문해야 하는가?
4. 목표를 집기 전에 어떤 방해 물체를 먼저 치워야 하는가?

기존 visual grounding 모델은 현재 이미지에서 지시와 가장 잘 맞는 물체를 고를 수
있지만, 가림 때문에 목표를 검출하지 못하거나 표현이 모호한 경우에 취약하다.
INVIGORATE는 신경망의 출력을 정답으로 확정하지 않고 **불완전한 관측**으로 취급한다.
이 관측을 여러 단계에 걸쳐 누적하고, 질문과 grasp를 선택하는 역할은 POMDP가 맡는다.

## 4. 전체 시스템

INVIGORATE는 네 개의 학습 모듈과 하나의 POMDP planner로 구성된다.

| 모듈 | 역할 | 출력 |
|---|---|---|
| O-Net | 물체 검출 및 단계 간 추적 | 물체 후보 |
| G-Net | 자연어 지시와 각 물체의 일치도 계산 | 목표일 가능성 |
| Q-Net | 후보 물체를 구분하는 설명 생성 | 확인 질문 |
| R-Net | 물체 간 가림 관계와 grasp pose 추정 | 제거 순서와 grasp |
| POMDP | 불확실성을 갱신하고 다음 행동 선택 | 질문 또는 grasp |

전체 흐름은 다음과 같다.

```text
자연어 지시 + RGB 이미지
        ↓
물체 검출, 목표 일치도, 가림 관계 추정
        ↓
각 물체가 목표일 가능성과 물체 간 가림 관계를 갱신
        ↓
질문 / 방해 물체 제거 / 목표 grasp 중 하나 선택
        ↓
새 이미지 또는 사용자 답변을 받은 후 다시 계획
```

핵심은 perception과 planning의 분리다. 신경망은 물체와 관계에 대한 점수를 제공하고,
POMDP는 그 점수가 틀릴 수 있다는 전제에서 행동을 결정한다.

## 5. 질문 방식

Q-Net은 후보 물체 하나를 다른 물체와의 관계로 설명한다.

```text
“Do you mean the cup on top?”
“Do you mean the apple on the right of the cup?”
```

질문은 다음 순서로 만들어진다.

1. 확인하려는 후보와 주변 물체의 모든 조합을 만든다.
2. 각 조합에 대한 relational caption을 생성한다.
3. 생성 확률이 가장 높은 설명을 선택한다.
4. `Do you mean [생성된 설명]?` 형식으로 질문한다.

색이나 모양만 묘사하는 방식은 clutter에서 물체가 가려지면 잘못된 설명을 만들 수
있다. 저자들은 이를 줄이기 위해 주변 물체와의 관계를 포함한 설명을 사용한다.

사용자는 `yes/no`뿐 아니라 “No, the left one”처럼 추가 설명을 줄 수 있다. 그러나
계획할 때는 계산량을 줄이기 위해 답을 사실상 긍정과 부정으로 단순화한다. 추가
설명은 다음 단계의 visual grounding 입력 문장에 합쳐 사용한다.

## 6. INVIGORATE POMDP

### 6.1 State: 로봇이 알 수 없는 것

상태는 두 부분으로 구성된다.

- **목표 상태:** 각 물체가 사용자의 목표인지 여부
- **가림 관계 상태:** 어떤 물체를 먼저 제거해야 다른 물체를 집을 수 있는지

로봇은 참 상태를 직접 알 수 없으므로 각 가능성에 대한 확률인 belief를 유지한다.

### 6.2 Observation: 로봇이 실제로 얻는 정보

관측은 세 종류다.

- G-Net이 출력한 물체별 language matching score
- R-Net이 출력한 물체 쌍별 가림 관계 score
- 질문에 대한 사용자의 자연어 답변

G-Net과 R-Net의 점수가 실제 정답일 때 어떤 분포를 보이는지는 clutter 데이터로부터
kernel density estimation을 사용해 학습한다. 따라서 높은 신경망 점수를 곧바로
정답으로 간주하지 않고, 학습된 오차 분포를 통해 belief를 갱신한다.

### 6.3 Action: 로봇이 선택할 수 있는 것

행동은 질문과 grasp로 나뉜다.

#### 질문

각 후보 물체 \(i\)에 대해 다음 질문을 만들 수 있다.

```text
“Do you mean [물체 i에 대해 생성된 설명]?”
```

#### Grasp macro

grasp 행동은 물체 하나를 집는 동작이 아니라 목표를 얻기 위한 순서 전체를 나타낸다.

- **Goal-directed macro:** 방해 물체를 제거하고 후보 \(i\)를 최종적으로 집는다.
- **Clearing macro:** 검출된 후보 중 목표가 없다고 판단하고 물체를 치워 아래를 탐색한다.

실행 시에는 전체 순서를 한꺼번에 수행하지 않는다. 첫 grasp만 실행하고 새 이미지를
받은 뒤 다시 belief를 갱신하고 계획한다. 이 방식은 perception 오류로 잘못된 제거
순서를 끝까지 실행하는 위험을 줄인다.

### 6.4 Reward: 무엇을 좋은 행동으로 보는가

reward는 다음 행동을 유도하도록 설계되어 있다.

- 질문 한 번: `-2`
- 잘못된 목표를 집음: `-10`
- 목표가 있는데 clearing을 수행함: `-10`
- 목표 후보가 여러 개인 상태에서 성급하게 grasp: 후보 수에 따라 penalty
- 확실한 목표를 올바르게 grasp: penalty 없음

즉, 질문을 무조건 많이 하는 것도 좋지 않고, 불확실한 상태에서 바로 집는 것도
좋지 않다. 질문 비용과 실패 위험을 비교해 최종 누적 reward가 높은 행동을 고른다.

이 reward는 이론적으로 도출된 값이 아니라 실험적으로 정한 값이다. 따라서 다른
작업에 적용할 때는 질문 비용과 실패 비용을 다시 정해야 한다.

### 6.5 Planning

planner는 현재 행동만 비교하지 않고, 질문에 `yes` 또는 `no`가 돌아온 뒤 어떤
행동을 하게 될지도 미리 계산한다. 논문에서는 최대 세 번의 질문까지 살펴본다.

예를 들어 두 리모컨이 후보라면 다음을 비교한다.

```text
지금 가장 가능성 높은 리모컨을 집기

vs.

“오른쪽 리모컨인가?”라고 질문하기
  ├─ yes → 오른쪽 리모컨 grasp
  └─ no  → 다른 후보 또는 clearing 행동 선택
```

계획 결과가 질문이면 질문 하나만 실행하고, 답변을 받은 뒤 다시 계산한다. 계획
결과가 grasp macro이면 첫 번째 grasp만 실행하고 새 이미지에서 다시 계산한다.

## 7. 기존 방법과의 차이

| 비교 기준 | INGRESS-POMDP | Attr-POMDP | INVIGORATE |
|---|---|---|---|
| 주된 문제 | 언어적으로 모호한 목표 확인 | 속성 질문으로 목표 확인 | 모호성, 가림, grasp 순서를 함께 처리 |
| 무엇을 묻는가? | 특정 물체를 생성 문장으로 확인 | 색·위치와 같은 공통 속성 | 특정 물체를 관계 표현으로 확인 |
| 물리 행동 | 목표 물체 pick | 목표 물체 grasp | 방해 물체 제거와 목표 grasp |
| 기억하는 불확실성 | 목표 물체 | 목표 물체와 속성 응답 | 목표 물체와 물체 간 가림 관계 |
| 주요 장점 | 자유로운 물체 설명 | 질문과 답의 종류가 단순함 | 질문과 clutter manipulation을 연결 |
| 주요 한계 | 생성 문장과 답변 공간이 복잡함 | 미리 정의한 속성에 제한됨 | 상태·행동·관측 모델이 크고 구현 의존성이 높음 |

### 7.1 INGRESS-POMDP와 차이

INVIGORATE의 질문 방식은 INGRESS-POMDP의 object-specific question을 계승한다.
차이는 질문으로 목표를 정한 뒤 바로 집는 데서 끝나지 않는다는 것이다. 목표가
가려졌다면 어떤 방해 물체를 먼저 제거할지도 같은 상태와 계획 안에서 결정한다.

### 7.2 Attr-POMDP와 차이

Attr-POMDP는 “무슨 색인가?”처럼 하나의 질문으로 여러 후보를 나눌 수 있다.
INVIGORATE는 “오른쪽 컵인가?”처럼 후보 하나를 묘사하여 확인한다. 따라서
INVIGORATE의 질문은 물체 수에 따라 증가하지만, Attr-POMDP는 정해진 속성 수에
따라 질문 후보가 정해진다.

반대로 Attr-POMDP는 주로 목표 확인과 grasp 시점을 다루지만, INVIGORATE는 목표가
가려진 상황에서 방해 물체를 제거하는 순서까지 계획한다.

## 8. 실험

### 8.1 환경

- Fetch robot
- Intel RealSense D435
- NVIDIA Titan X GPU
- clutter scene 10개
- 참가자 10명
- 총 test case 100개

두 가지 조건으로 나누어 평가한다.

- **Test 1:** 참가자가 장면을 보기 전에 목표를 선택
- **Test 2:** 참가자가 clutter를 본 뒤 어려운 목표를 선택

Test 2는 참가자가 의도적으로 가려졌거나 집기 어려운 물체를 선택할 수 있어 더
어렵다.

### 8.2 전체 성공률

| 방법 | Test 1 | Test 2 | 전체 |
|---|---:|---:|---:|
| MAttNet + VMRN | 76% | 60% | 68% |
| INVIGORATE | **86%** | **80%** | **83%** |

INVIGORATE는 평균 0.65번 질문했고, test case당 평균 0.5회의 추가 grasp step을
사용했다. 순수 신경망 baseline보다 전체 성공률이 15%p 높았다.

특히 baseline은 처음부터 목표가 보이지 않거나 검출되지 않으면 거의 실패했다.
INVIGORATE는 보이는 후보 중 목표가 없을 가능성을 유지하므로, 위의 물체를 제거해
숨은 목표를 찾을 수 있었다.

### 8.3 Ablation

논문은 다음 요소를 제거해 비교한다.

- 질문하지 않음
- 이전 이미지와 질의응답을 모두 기억하지 않음
- 이전 이미지는 기억하지 않고 질의응답만 기억
- 여러 단계 계산 대신 heuristic 사용

질문을 제거하면 성공률이 약 17%p 감소했다. 이력 정보를 제거하면 같은 내용을
반복해서 확인해야 하므로 질문 수가 증가했다.

여러 단계를 계산하는 planner는 heuristic보다 질문을 약간 더 했지만 실패는 줄이고
누적 reward는 높였다. 다만 저자들은 200회 실험에서도 일부 차이가 통계적으로
명확하지 않았다고 보고한다. 따라서 tree search 자체의 우월성은 전체 성공률
결과만큼 강하게 입증되지는 않았다.

### 8.4 Visual grounding

| 방법 | 전체 mean accuracy | 전체 mean L1 loss |
|---|---:|---:|
| ViLBERT | 0.831 | 0.099 |
| INVIGORATE | **0.875** | **0.050** |

이 비교에서 INVIGORATE의 장점은 단일 이미지 모델의 성능이라기보다, 여러 단계의
관측을 Bayesian filter로 누적한 결과로 해석해야 한다.

## 9. 강점

1. 언어의 모호성과 물리적 가림을 하나의 작업에서 다룬다.
2. 학습 모델의 출력을 확정값이 아닌 noisy observation으로 처리한다.
3. 질문, 방해물 제거, 목표 grasp를 같은 의사결정 안에서 비교한다.
4. 이미지와 질의응답 이력을 belief에 누적한다.
5. 긴 grasp 순서를 한꺼번에 실행하지 않고 매 단계 다시 계획한다.

## 10. 한계

### 10.1 질문 모델

질문은 물체별로 생성되므로 후보 수가 많으면 행동 수도 증가한다. 생성된 relational
caption이 틀리거나 후보를 구분하지 못할 수도 있다. 또한 자유로운 답을 허용한다고
설명하지만 실제 계획에서는 주로 긍정과 부정으로 단순화하며, 실험 참가자도 대부분
추가 설명을 제공하지 않았다.

### 10.2 Perception 오류

POMDP는 무작위적인 신경망 오류에는 견고할 수 있지만 체계적인 오류를 없애지는
못한다. 실제 실패 사례에는 다음이 포함된다.

- 방해 물체를 검출하지 못해 잘못된 순서로 grasp
- 가려진 목표의 grounding score가 지나치게 낮아 다른 물체를 목표로 선택
- 잘못된 object blocking relationship을 믿고 grasp

관측 모델이 잘못 calibration되어 있으면 belief와 planning도 함께 잘못된다.

### 10.3 실험 규모

실험은 10개 장면과 100개 test case로 제한된다. grasp 실패는 각 방법이 처리하지
않고 실험자가 물체를 직접 제거했다. 따라서 실제 end-to-end grasp robustness를
완전히 평가한 결과는 아니다.

### 10.4 수작업 reward

질문 `-2`, 실패 `-10` 등의 값은 경험적으로 정해졌다. 또한 세 번 이상 질문하면
사람이 불편해한다고 가정한다. 사람, 작업 위험도, 도메인이 달라지면 이 가정과
reward scale은 그대로 사용할 수 없다.

## 11. Active Search baseline으로서의 해석

INVIGORATE 전체를 baseline으로 재현하려면 다음이 모두 필요하다.

- object detection과 tracking
- referring-expression grounding
- relational question generation
- object blocking relationship 추정
- grasp pose와 clearing sequence 생성
- 학습된 observation model
- belief update와 POMDP tree search

따라서 현재 active search 실험에서 **질문 선택만 비교**하려는 경우 INVIGORATE
전체를 그대로 baseline으로 부르는 것은 정확하지 않다. 물리적 clearing과 grasp
planning을 제외하면 논문의 핵심 상태와 action 일부를 제거한 변형이기 때문이다.

질문 baseline으로 축소한다면 다음처럼 명시해야 한다.

```text
INVIGORATE-style query baseline

State:
  각 후보가 목표일 확률

Query action:
  “Do you mean [후보를 설명하는 문장]?”

Response:
  oracle의 yes/no

Task action:
  현재 가장 가능성 높은 후보 선택

Not implemented:
  object blocking relationship
  clearing grasp
  learned visual observation model
  real relational caption generator
```

이 경우 질문 생성기가 구현되지 않았다면 고정 template이나 oracle description을
사용하는 placeholder임을 반드시 밝혀야 한다. 그렇지 않으면 INVIGORATE의 Q-Net을
재현한 것으로 오해될 수 있다.

## 12. BRL-CF와의 관련성

BRL-CF와 가장 직접적으로 연결되는 부분은 **질문을 task action과 같은 action
space에 넣었다는 점**이다. INVIGORATE에서 질문은 부가적인 대화 기능이 아니라
grasp와 경쟁하는 정보 획득 행동이다.

다만 두 방법의 범위는 다르다.

| 항목 | INVIGORATE | BRL-CF active search |
|---|---|---|
| 불확실성 | 목표 물체와 가림 관계 | task에 필요한 fact 또는 hypothesis |
| 질문 대상 | 후보 물체 | 선택된 질문 action |
| 응답 제공자 | 사람 | oracle, VLM, human |
| 물리 행동 | clearing 및 grasp | domain task action |
| 핵심 목적 | clutter에서 올바른 목표 회수 | feedback으로 task decision 개선 |

INVIGORATE에서 가져올 수 있는 핵심 원칙은 다음과 같다.

1. 질문도 비용이 있는 action으로 취급한다.
2. 질문 후의 belief 변화가 최종 task success에 미치는 영향을 평가한다.
3. 이전 질문과 답을 기억하여 같은 정보를 반복해서 묻지 않는다.
4. perception 또는 policy output을 확정값이 아닌 noisy evidence로 취급한다.

반면 OBR, clearing grasp, relational captioning은 BRL-CF의 일반 active search
baseline에 그대로 필요한 요소는 아니다.

## 13. 최종 평가

INVIGORATE의 핵심 기여는 “가장 애매하면 질문한다”는 규칙이 아니다. 목표가
보이지 않을 가능성, 물체 사이의 가림 관계, 사용자 답변, 방해물 제거 순서를 하나의
belief와 action model에 연결했다는 점이다.

논문은 active search의 중요한 선행연구지만, 범용 질문 선택기라기보다 cluttered
robotic grasping을 위한 통합 시스템에 가깝다. 따라서 baseline으로 사용할 때는
논문 전체를 재현하는지, 아니면 object-specific yes/no 질문 전략만 가져오는지
명확하게 구분해야 한다.
