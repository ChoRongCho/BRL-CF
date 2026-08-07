# INGRESS: Interactive Visual Grounding of Referring Expressions 리뷰

## 1. 논문 정보

- 제목: **INGRESS: Interactive Visual Grounding of Referring Expressions**
- 저자: Mohit Shridhar, Dixant Mittal, David Hsu
- 학술지: *The International Journal of Robotics Research*
- 연도: 2020
- 권·호·페이지: 39(2–3), 217–232
- DOI: `10.1177/0278364919897133`
- 핵심 주제: referring-expression grounding, question generation, human–robot interaction, POMDP

이 논문은 2018년 RSS에 발표된 INGRESS를 확장한 저널판이다. 저널판의 중요한
추가는 질문 선택과 종료 결정을 위한 **INGRESS-POMDP**다.

## 2. 한 문장 요약

INGRESS는 이미지 속 각 물체를 자연어로 설명해 사용자의 지시와 비교하고, 목표가
모호하면 그 설명을 확인 질문으로 재사용하여 가능한 한 적은 질문으로 목표 물체를
선택한다.

## 3. 연구 문제

사용자가 로봇에게 “컵을 집어줘”라고 말했을 때 해결해야 할 문제는 두 가지다.

1. 자연어 표현이 이미지의 어떤 물체를 가리키는가?
2. 여러 물체가 같은 표현에 해당하면 무엇을 물어야 하는가?

논문은 다음 조건을 목표로 한다.

- 미리 정한 물체 category에 제한되지 않을 것
- 색, 모양, 위치, 물체 간 관계를 포함한 자유로운 표현을 받을 것
- 모호한 지시는 사람에게 질문하여 해소할 것
- 같은 질문을 반복하지 않고 적은 질문으로 목표를 찾을 것

다만 물체가 서로 심하게 가리지 않는 **uncluttered scene**을 가정한다. 가려진
물체를 찾거나 방해 물체를 제거하는 문제는 다루지 않는다.

## 4. 핵심 아이디어: Grounding by Generation

일반적인 grounding 모델은 이미지와 문장을 입력받아 각 물체의 일치도를 직접
출력한다. INGRESS는 반대 방향을 이용한다.

```text
각 물체를 설명하는 문장을 생성
        ↓
생성 가능한 문장과 사용자의 지시를 비교
        ↓
사용자의 지시를 가장 잘 생성할 수 있는 물체 선택
```

예를 들어 이미지에 빨간 컵과 파란 컵이 있고 사용자가 “파란 컵”이라고 말했다면,
각 물체에서 생성되는 단어 확률과 입력 문장을 비교하여 파란 컵의 점수를 높인다.

이 설계의 가장 큰 장점은 **grounding에 사용한 모델을 질문 생성에도 재사용할 수
있다는 것**이다. 후보 물체가 여러 개라면 해당 물체가 생성한 설명을 다음 template에
넣는다.

```text
“Do you mean [물체에 대해 생성된 설명]?”
```

## 5. 두 단계 Visual Grounding

INGRESS는 referring expression을 두 종류로 나누어 처리한다.

### 5.1 Self-referential expression

물체 자체의 특징을 나타내는 표현이다.

```text
“the blue cup”
“the red round object”
```

S-LSTM은 각 image region을 입력받아 그 물체를 설명하는 단어들의 확률을 생성한다.
사용자의 표현과 생성 분포 사이의 cross-entropy loss가 작은 물체가 우선 후보가 된다.

### 5.2 Relational expression

다른 물체와의 관계를 나타내는 표현이다.

```text
“the cup on the left of the bowl”
“the bottle next to the teddy bear”
```

R-LSTM은 후보 물체와 context 물체의 모든 쌍을 검사하고 relational description을
생성한다. 입력 표현과 가장 잘 맞는 물체 쌍을 찾아 최종 목표를 정한다.

두 단계를 나눈 이유는 self-referential grounding으로 후보를 먼저 줄인 뒤 관계를
검사하면 모든 물체 쌍을 비교하는 비용을 줄일 수 있기 때문이다.

## 6. 질문 생성

INGRESS는 두 종류의 질문을 생성한다.

| 질문 종류 | 확인하는 정보 | 예시 |
|---|---|---|
| Self-referential question | 물체 자체의 색·모양·종류 | “Do you mean the blue cup?” |
| Relational question | 위치 또는 다른 물체와의 관계 | “Do you mean the cup on the left?” |

질문은 후보마다 생성된다. 따라서 후보가 세 개라면 각 후보에 대한 self-referential
질문과 relational 질문이 각각 존재할 수 있다.

사용자는 다음과 같이 답할 수 있다.

```text
“Yes.”
“No.”
“No, the red cup.”
“No, the cup on the left.”
```

단순한 `yes/no`뿐 아니라 교정 설명도 받을 수 있다는 점이 generic pointing
question과의 차이다.

## 7. 두 가지 질문 선택 방법

### 7.1 INGRESS-Heuristic

Heuristic 방법은 현재 가능성이 높은 후보부터 순서대로 확인한다.

1. 후보마다 self-referential description을 생성한다.
2. 다른 후보와 충분히 다른 설명이면 그 설명으로 질문한다.
3. 설명이 후보를 잘 구분하지 못하면 relational question을 사용한다.
4. 사용자가 `no`라고 하면 다음 후보로 넘어간다.

이 방법은 질문 하나를 고르는 규칙은 있지만 이전 대화의 의미를 명시적으로
기억하지 않는다. 따라서 “아니, 다른 컵”과 같은 답에서 `다른`이 이전에 물어본
컵과 어떤 관계인지 제대로 해석하기 어렵다.

### 7.2 INGRESS-POMDP

INGRESS-POMDP는 각 후보가 목표일 확률을 belief로 유지한다. 질문의 답에 따라
belief가 어떻게 바뀌는지 계산하고, 다음 질문 또는 pick까지 고려하여 행동을
고른다.

쉬운 예시는 다음과 같다.

```text
후보: 파란 컵 3개, 빨간 컵 2개

1. “파란 컵인가?” 질문
2. 사용자: “아니, 빨간 것”
3. belief가 빨간 컵 2개에 집중
4. “왼쪽 컵인가?” 질문
5. 사용자: “아니, 다른 것”
6. 오른쪽 빨간 컵 선택
```

POMDP가 이전 답을 belief에 반영하기 때문에 마지막의 `다른 것`을 현재 남아 있는
빨간 컵 후보 안에서 해석할 수 있다.

## 8. INGRESS-POMDP 구성

### 8.1 State

state는 \(N\)개 후보 물체 중 실제 목표가 무엇인지 나타낸다.

```text
X = {후보 물체 1, 후보 물체 2, ..., 후보 물체 N}
```

사용자가 대화 도중 목표를 바꾸지 않는다고 가정하므로 state는 변하지 않는다.

### 8.2 Action

각 후보 \(x\)에 대해 세 가지 행동이 있다.

- `ASK_SELF(x)`: 후보 자체의 특징을 이용해 질문
- `ASK_REL(x)`: 후보의 관계를 이용해 질문
- `PICK(x)`: 해당 후보를 최종 목표로 선택

후보가 \(N\)개이면 총 action 수는 \(3N\)이다.

질문은 음성으로만 수행한다. Heuristic 방법과 달리 로봇 팔로 물체를 가리키지
않으므로 pointing에 필요한 동작 계획과 실행 시간을 피한다.

### 8.3 Observation

observation은 사용자의 전체 답변이다. 이를 두 부분으로 나눈다.

- **Response utterance:** `yes`, `no`와 같은 긍정 또는 부정
- **Description utterance:** `the red one`, `the cup on the left`와 같은 추가 설명

이론상 자유로운 문장이 가능하므로 observation space는 매우 크다. 논문은 계산을
가능하게 하기 위해 다음과 같이 나누어 처리한다.

- 미래 질문을 미리 계산할 때는 답을 `yes/no` 두 종류로 단순화
- 실제 답을 받은 뒤 belief를 갱신할 때는 추가 description까지 사용

즉, **planning model은 단순화되어 있지만 실제 belief update는 더 많은 언어 정보를
사용한다.**

### 8.4 Observation probability

긍정과 부정에 대한 확률은 수작업으로 설정한다.

| 상황 | 긍정 응답 | 부정 응답 |
|---|---:|---:|
| 질문한 물체가 실제 목표 | 0.99 | 0.01 |
| 질문한 물체가 목표가 아님 | 0.01 | 0.99 |

이는 사용자가 99%의 확률로 협조적이고 정확하게 답한다고 가정한 것이다.
추가 description의 확률은 S-LSTM 또는 R-LSTM이 해당 물체에서 그 문장을 생성할
확률로 계산한다.

### 8.5 Reward

| 행동 결과 | Reward |
|---|---:|
| 올바른 물체 pick | `+10` |
| 잘못된 물체 pick | `-10` |
| self-referential 질문 | `-1` |
| relational 질문 | `-1` |

질문에는 작은 비용이 있고 잘못 집는 데에는 큰 비용이 있다. 따라서 로봇은 질문을
무한히 반복하지 않으면서, 잘못 집을 위험이 높으면 추가 질문을 선택한다.

reward 값은 실험적으로 정한 값이며 다른 작업에서 그대로 정당화되지는 않는다.

### 8.6 Planning

belief tree의 최대 깊이는 4다. planner는 최대 네 단계의 질문과 답을 미리
검토하고 기대 reward가 가장 높은 첫 행동을 실행한다.

원래 자연어 observation space는 매우 크지만 계획할 때 `yes/no`로 줄이므로
tree search가 가능하다. 질문 또는 pick을 한 번 실행한 뒤 새 답을 반영해 다시
계획한다.

## 9. 실험

### 9.1 RefCOCO grounding

정답 bounding box와 예측 box의 IoU가 0.5보다 크면 성공으로 평가한다.

| Dataset | UMD Refexp HGT | INGRESS HGT | UMD Refexp MCG | INGRESS MCG |
|---|---:|---:|---:|---:|
| Val | 75.5% | **77.0%** | 56.5% | **58.3%** |
| TestA | 74.1% | **76.7%** | 57.9% | **60.3%** |
| TestB | 76.8% | **77.7%** | **55.3%** | 55.0% |

대부분의 조건에서 INGRESS가 우수하지만 차이는 크지 않으며, MCG TestB에서는
오히려 0.3%p 낮다. 저자도 RefCOCO 이미지의 후보 수가 적어 두 단계 filtering의
장점이 크게 드러나지 않았다고 설명한다.

### 9.2 실제 로봇 grounding

- 참가자 16명
- 참가자당 household-object scene 15개
- 장면당 평균 물체 8개
- Kinova MICO 6-DOF arm 및 Kinect2 사용

| 방법 | Grounding accuracy |
|---|---:|
| UMD Refexp | 31.3% |
| S-INGRESS | 53.3% |
| INGRESS | **76.7%** |

관계 표현을 제외한 S-INGRESS보다 전체 INGRESS가 크게 우수했다. 실제 사용자의
표현에서는 색과 종류뿐 아니라 다른 물체와의 관계가 중요하다는 결과다.

### 9.3 질문을 통한 disambiguation

- 참가자 24명
- 방법별 참가자 8명
- 참가자당 scenario 10개
- 총 240 trials
- 장면당 평균 후보 4.6개

| 방법 | 평균 질문 수 | 성공률 |
|---|---:|---:|
| Generic pointing question | 2.75 | 논문 본문에 직접 비교값 미제시 |
| INGRESS-Heuristic | 1.99 | 88% |
| INGRESS-POMDP | **1.45** | **89%** |

INGRESS-POMDP는 Heuristic과 비슷한 성공률을 유지하면서 질문 수를 줄였다.
Generic baseline과 비교하면 평균 질문 수가 2.75회에서 1.45회로 감소했다.

사용자 설문에서 “로봇이 추가로 필요한 정보를 효과적으로 전달한다”는 평가도
object-specific 질문이 4.25점, generic 질문이 1.63점으로 차이가 컸다.

다만 generic baseline은 pointing 동작을 사용하고 INGRESS-POMDP는 음성 질문만
사용한다. 따라서 실행 시간 차이에는 질문 선택뿐 아니라 로봇 팔을 움직이는
1–4초의 비용도 포함된다.

## 10. 강점

1. grounding 모델과 질문 생성 모델을 동일한 구조로 연결한다.
2. 물체 자체의 특징과 물체 사이의 관계를 모두 이용한다.
3. 미리 정의한 소수 category에 제한되지 않는 표현을 처리한다.
4. 질문 이력을 belief로 유지해 교정 응답을 누적한다.
5. 질문 비용과 잘못 집는 비용을 비교해 질문 종료 시점을 결정한다.
6. 실제 사용자와 로봇 실험에서 질문 수 감소를 확인한다.

## 11. 한계

### 11.1 생성 문장의 오류

질문 품질이 S-LSTM과 R-LSTM이 생성한 description에 의존한다. 데이터 편향 때문에
실제 장면과 맞지 않는 “ball in the air” 같은 문장이 생성되기도 한다.

### 11.2 관계 표현의 제한

INGRESS는 기본적으로 두 물체 사이의 binary relation을 처리한다. 다음 표현에는
취약하다.

- “오른쪽에서 세 번째 컵”
- “윗줄 가운데 물체”
- 여러 물체를 하나의 group으로 묶어야 하는 표현
- 제품의 글자나 브랜드 이름

### 11.3 Clutter와 occlusion

물체가 분리되어 있고 잘 보이는 장면을 가정한다. 부분적으로 가려진 물체에서는
false positive가 발생하며, 물체를 치워 숨은 목표를 찾는 행동은 없다. 이 문제를
확장한 후속 연구가 INVIGORATE다.

### 11.4 단순화된 사람 모델

사용자가 99% 정확하게 답하고 대화 중 목표를 바꾸지 않는다고 가정한다.
계획에서는 자유로운 답변을 `yes/no`로 축약하므로 추가 설명이 미래 결정에 주는
가치를 정확히 계산하지 못한다.

### 11.5 Action 수

각 후보마다 self question, relational question, pick이 존재하므로 후보 수가
늘수록 action space도 선형으로 증가한다. 깊이 4의 전체 tree search는 후보가 많은
장면에서 빠르게 커질 수 있다.

## 12. Attr-POMDP와의 차이

| 비교 기준 | INGRESS-POMDP | Attr-POMDP |
|---|---|---|
| 질문 단위 | 특정 후보 물체에 대한 생성 문장 | 색·위치 같은 공통 속성 |
| 예시 | “왼쪽의 빨간 컵인가?” | “목표는 무슨 색인가?” |
| 후보 분리 | 질문한 후보를 직접 확인 | 하나의 답으로 여러 후보를 나눔 |
| 답변 | 자유로운 교정 표현 허용 | 정해진 속성값 또는 `yes/no` |
| 언어 모델 | S-LSTM/R-LSTM 질문 생성기 필요 | 고정 attribute question template |
| 주요 위험 | 잘못된 문장 생성과 큰 언어 공간 | 정의되지 않은 속성을 질문하지 못함 |

INGRESS-POMDP는 더 자연스럽고 구체적인 질문을 만들 수 있지만 질문 생성기의
정확도에 의존한다. Attr-POMDP는 표현력은 제한되지만 답의 종류가 작고 여러 후보를
한꺼번에 나눌 수 있어 계획이 단순하다.

## 13. Active Search baseline으로서의 해석

INGRESS-POMDP를 그대로 재현하려면 다음 요소가 필요하다.

- image-region proposal
- S-LSTM self-referential grounding/generation
- R-LSTM relational grounding/generation
- relevancy clustering
- 전체 자연어 답변의 likelihood
- belief update와 depth-4 tree search

질문 생성기를 구현하지 않고 후보 이름을 template에 넣는다면 원 논문의 INGRESS를
재현한 것이 아니다. 이 경우 다음과 같이 명시하는 편이 정확하다.

```text
INGRESS-style baseline

구현:
  후보별 object-specific confirmation question
  oracle의 yes/no response
  질문 비용과 최종 선택 reward

Placeholder:
  S-LSTM/R-LSTM question generator
  unrestricted corrective response
  learned language observation likelihood
```

특히 oracle이 정답을 `yes/no`로만 제공한다면 INGRESS의 장점 중 하나인 “No, the
red one”과 같은 rich correction은 실험에서 빠진다. 결과를 논문과 비교할 때 이
차이를 밝혀야 한다.

## 14. BRL-CF와의 관련성

BRL-CF 관점에서 중요한 부분은 질문을 최종 task action과 경쟁하는 비용 있는
action으로 정의했다는 점이다.

| 항목 | INGRESS-POMDP | BRL-CF active search |
|---|---|---|
| 숨은 상태 | 목표 물체 | task hypothesis 또는 필요한 fact |
| 질문 | 후보별 생성 문장 | action에 포함된 질문 |
| 답변 제공자 | 사람 | oracle, VLM, human |
| 최종 행동 | 물체 pick | domain task action |
| 이력 | 언어 응답을 belief에 누적 | feedback을 belief/state에 누적 |

가져올 수 있는 핵심은 다음과 같다.

1. 질문에는 명시적인 비용을 둔다.
2. 질문의 유용성을 최종 task 성공과 연결한다.
3. 이전 답을 기억해 반복 질문을 막는다.
4. 질문과 task 종료를 같은 기준으로 비교한다.

반면 S-LSTM/R-LSTM과 image-region grounding은 범용 BRL-CF baseline의 필수
요소가 아니라 object-disambiguation domain에 특화된 구현이다.

## 15. 최종 평가

INGRESS의 가장 중요한 기여는 자연어를 이해하는 모델과 질문을 만드는 모델을
`grounding by generation`으로 통합한 것이다. INGRESS-POMDP는 그 위에서 어떤
후보를 어떤 방식으로 확인할지와 언제 물체를 선택할지를 결정한다.

Active Search의 대표적인 선행 baseline으로 인용할 수 있지만, 전체 방법은
자유로운 referring expression과 image region을 전제로 한다. 질문 생성기와
자연어 observation model을 제외한 구현은 반드시 **INGRESS-style simplified
baseline**으로 구분해야 한다.
