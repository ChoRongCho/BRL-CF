# When to Seek Help 논문과 우리 논문의 차별점

## 요약

`When to Seek Help: Trust-Aware Assistance-Seeking in Human-Supervised Autonomy`는 인간의 **trust belief**를 추정하여 로봇이 언제 인간에게 도움을 요청할지 결정하는 논문이다. 반면 우리 논문은 로봇의 **symbolic world belief**와 task-relevant uncertainty를 기반으로, 로봇이 언제 인간 피드백을 요청하고 무엇을 물어볼지 결정하는 knowledge-based POMDP framework이다.

따라서 두 논문은 모두 "선택적 인간 개입"을 다루지만, 핵심 불확실성의 대상, 질문의 목적, 정책 설계의 단위가 다르다.

## 핵심 차별점 한 문장

기존 논문은 인간 신뢰 상태를 추론하여 도움 요청이 인간 신뢰와 팀 보상에 미치는 영향을 최적화하는 반면, 우리 논문은 로봇의 지식 상태와 planning belief의 불확실성을 줄이기 위해 task-relevant hypothesis를 선택적으로 질의한다.

## 1. 불확실성의 대상이 다르다

### When to Seek Help

이 논문에서 주요 hidden state는 **human trust**이다. 로봇은 인간이 자신을 얼마나 신뢰하는지 직접 관측할 수 없으므로, 인간의 rely/intervene 행동을 통해 신뢰 상태를 추정한다.

중심 질문은 다음과 같다.

- 인간이 로봇을 신뢰하는가?
- 신뢰가 낮으면 로봇이 도움을 요청해야 하는가?
- 도움 요청이 인간 신뢰를 회복할 수 있는가?

### Our Work

우리 논문에서 주요 불확실성은 **robot's belief over world states**이다. 로봇은 perception error, execution uncertainty, environmental ambiguity 때문에 현재 symbolic state를 확신할 수 없다.

중심 질문은 다음과 같다.

- 현재 가능한 world state들 중 무엇이 참인가?
- 어떤 symbolic hypothesis가 planning에 중요한 불확실성을 만든는가?
- 어떤 질문이 belief entropy를 가장 많이 줄이는가?

### 차별점

상대 논문은 인간 내부 상태에 대한 불확실성을 모델링한다. 우리 논문은 로봇의 지식 기반 세계 상태에 대한 불확실성을 모델링한다. 즉, 상대 논문은 **human-state uncertainty**, 우리 논문은 **world-state/task-state uncertainty**가 중심이다.

## 2. 도움 요청의 목적이 다르다

### When to Seek Help

도움 요청은 주로 현재 task execution을 인간에게 넘기거나, 인간의 개입을 줄이고 신뢰를 유지/회복하기 위한 행동이다. 특히 고복잡도 상황에서 도움 요청은 로봇이 위험을 인식하고 신중하게 행동한다는 신호가 되어 인간 신뢰를 높일 수 있다.

즉 도움 요청은 다음 목적을 가진다.

- 실패 위험 감소
- 인간의 unsolicited intervention 방지
- trust repair 또는 trust calibration
- cumulative reward 증가

### Our Work

우리 논문에서 피드백 요청은 인간에게 작업 실행을 넘기는 것이 아니라, 로봇의 belief를 정제하기 위한 정보 획득 행동이다. 사용자는 로봇 대신 작업을 수행하는 것이 아니라, 특정 symbolic hypothesis의 참/거짓을 확인해준다.

예를 들면 다음과 같다.

- "Is this tomato ripe?"
- "Is this object plastic?"
- "Is this tomato rotten?"

즉 피드백 요청은 다음 목적을 가진다.

- perception ambiguity 해소
- symbolic belief refinement
- planning-relevant hypothesis 확인
- 잘못된 belief에 기반한 실패 행동 방지

### 차별점

상대 논문에서 도움 요청은 **control transfer / assistance-seeking action**에 가깝다. 우리 논문에서 질문은 **belief refinement / information-seeking query**에 가깝다.

## 3. "When to ask"의 기준이 다르다

### When to Seek Help

로봇은 인간의 trust belief가 낮고 task complexity가 높을 때 도움을 요청한다. 최적 정책은 threshold 구조를 가진다.

- 저복잡도: 도움 요청하지 않음
- 고복잡도: high-trust belief가 약 `0.73`보다 낮으면 도움 요청
- 고복잡도: high-trust belief가 충분히 높으면 자율 수행

즉 질문 시점은 인간 신뢰 belief와 환경 복잡도에 의해 결정된다.

### Our Work

로봇은 현재 belief distribution의 confidence가 threshold `tau`보다 낮을 때 피드백을 요청한다. confidence는 particle belief의 normalized entropy로 계산된다.

우리 논문의 기준은 다음과 같다.

- belief entropy가 높으면 질문
- belief가 한 상태에 충분히 집중되면 질문하지 않음
- threshold `tau`가 interaction cost와 task success 사이의 trade-off를 조절함

### 차별점

상대 논문의 threshold는 **human trust belief**에 걸린다. 우리 논문의 threshold는 **world-state belief confidence**에 걸린다.

## 4. "What to ask"를 다루는 정도가 다르다

### When to Seek Help

상대 논문은 주로 언제 도움을 요청할지에 집중한다. 로봇 행동 공간은 단순하다.

- 자율 수행
- 도움 요청

도움을 요청하면 인간이 teleoperation으로 작업을 수행한다. 어떤 정보를 물어볼지, 어떤 hypothesis가 가장 informative한지에 대한 explicit query selection은 핵심 기여가 아니다.

### Our Work

우리 논문은 언제 물을지뿐 아니라 **무엇을 물을지**를 명시적으로 다룬다. 로봇은 candidate hypothesis들을 구성하고, 각 hypothesis가 belief entropy를 얼마나 줄일지 계산한다.

우리 논문의 query selection은 다음과 같이 작동한다.

- 가능한 frontier states를 belief particles로 유지
- 현재 knowledge state와 frontier states 사이에서 다른 literals를 candidate hypothesis로 구성
- 각 hypothesis에 대해 true/false로 belief를 나눴을 때의 expected entropy 계산
- expected entropy가 가장 낮은 hypothesis를 질문으로 선택

### 차별점

상대 논문은 **assistance-seeking timing policy**가 중심이고, 우리 논문은 **timing + content selection for corrective feedback**이 중심이다. 특히 우리 논문은 `when to query`와 `what to query`를 함께 해결한다.

## 5. 지식 표현과 planning 구조가 다르다

### When to Seek Help

상대 논문은 인간 행동 모델을 IOHMM으로 학습하고, 이를 POMDP에 넣어 도움 요청 정책을 계산한다. 상태는 비교적 작고 추상적이다.

- trust state
- experience
- task complexity

정책 계산은 belief MDP와 value iteration을 사용한다.

### Our Work

우리 논문은 symbolic knowledge base 위에서 POMDP planning을 수행한다. KB는 domain knowledge, static information, local information으로 구성되고, symbolic facts와 numeric fluents를 함께 저장한다.

우리 시스템은 다음 모듈로 구성된다.

- Knowledge Base
- Planning Manager
- Sensing and Perception
- Feedback Manager
- User Interface

Planning Manager는 POMCP 기반 online planning을 수행하고, symbolic action precondition/effect를 이용해 feasible action과 reachable frontier를 제한한다.

### 차별점

상대 논문은 신뢰 모델 기반의 작은 POMDP 정책 문제에 가깝다. 우리 논문은 perception, symbolic KB, POMCP planning, feedback manager, UI까지 연결한 **end-to-end knowledge-based robotic architecture**에 가깝다.

## 6. 인간의 역할이 다르다

### When to Seek Help

인간은 감독자이며, 로봇이 요청하면 직접 teleoperation으로 로봇 팔을 조작한다. 또한 로봇이 실패할 것 같다고 판단하면 먼저 개입할 수 있다.

인간의 행동은 모델의 observation이 된다.

- rely
- intervene

### Our Work

인간은 execution을 대신 수행하는 operator라기보다, 로봇의 불확실한 symbolic hypothesis를 검증해주는 feedback provider이다. 인간은 UI를 통해 특정 질문에 답하고, 이 답변은 belief filtering에 사용된다.

인간의 행동은 다음 역할을 한다.

- predicate truth verification
- perceptual ambiguity correction
- knowledge state refinement

### 차별점

상대 논문에서 인간은 **control authority를 가진 supervisor**이다. 우리 논문에서 인간은 **knowledge correction을 제공하는 semantic feedback source**이다.

## 7. 실험 도메인과 평가 지표가 다르다

### When to Seek Help

실험은 ROS-Gazebo 기반 물체 수집 과제 하나를 중심으로 한다. 주요 평가는 다음이다.

- cumulative reward
- trust survey score
- reliance/intervention behavior
- trust-aware vs trust-agnostic policy 비교

주요 결과:

- trust-aware median reward: `84`
- trust-agnostic median reward: `69`
- 성능 차이 유의: `p = 0.0007`
- trust-aware 후 trust score: `72.14`
- trust-agnostic 후 trust score: `66.39`

### Our Work

우리 논문은 두 가지 도메인에서 평가한다.

- simulated waste sorting
- real-world tomato harvesting

주요 평가는 다음이다.

- task success rate
- average number of queries
- average execution steps
- user workload
- SAGAT accuracy
- SART
- fatigue
- task time / response accuracy

시스템 평가 결과:

- waste sorting에서 Ours는 `100.0%` success rate 달성
- tomato harvesting에서 Ours는 `98.0%` success rate 달성
- tomato harvesting에서 All은 평균 `37.3` queries, Ours는 `20.5` queries
- Ours는 All보다 적은 query로 비슷하거나 더 높은 success rate를 달성

### 차별점

상대 논문은 신뢰 기반 정책의 팀 보상과 신뢰 회복을 검증한다. 우리 논문은 다중 도메인에서 belief-based selective querying이 task success, query efficiency, execution efficiency, user workload에 미치는 영향을 검증한다.

## 8. 논문 기여의 포지셔닝 차이

### When to Seek Help의 기여

- 인간 신뢰를 IOHMM으로 모델링
- 행동 데이터만으로 추정한 hidden trust state를 설문과 대응시켜 검증
- trust-aware assistance-seeking policy를 POMDP로 계산
- 도움 요청이 고복잡도에서 trust repair로 작동할 수 있음을 보임

### Our Work의 기여

- symbolic KB와 POMDP/POMCP planning을 결합한 uncertainty-aware feedback architecture 제안
- belief entropy 기반으로 피드백 요청 시점을 결정
- expected entropy reduction 기반으로 가장 informative한 hypothesis를 선택
- FM/UI를 통해 symbolic hypothesis를 사용자 친화적 질문으로 변환
- waste sorting과 tomato harvesting에서 selective query가 high success와 reduced query cost를 동시에 달성함을 보임

## 비교표

| 항목 | When to Seek Help | Our Work |
|---|---|---|
| 중심 문제 | 로봇이 언제 인간 도움을 요청해야 하는가 | 로봇이 언제/무엇을 인간에게 물어봐야 하는가 |
| 주요 hidden/belief state | 인간 신뢰 | 로봇의 world/task state belief |
| 불확실성 대상 | human trust uncertainty | symbolic state/perception uncertainty |
| 인간 역할 | supervisor, teleoperator, intervention source | semantic feedback provider |
| 로봇 query/action | 도움 요청 또는 자율 수행 | hypothesis-level corrective query |
| query 목적 | trust repair, intervention 방지, reward 향상 | belief refinement, planning ambiguity 해소 |
| 정책 기준 | trust belief threshold | entropy/confidence threshold |
| what-to-ask | 명시적 query content selection 없음 | expected entropy reduction으로 hypothesis 선택 |
| 모델 | IOHMM + POMDP | KB + particle belief + POMCP + FM |
| 평가 | cumulative reward, trust survey | success rate, query count, steps, workload, SAGAT/SART |
| 도메인 | Gazebo object collection | waste sorting, tomato harvesting |

## Related Work에 넣을 수 있는 문장

Mangalindan et al. studied assistance seeking in human-supervised autonomy by modeling human trust as a latent state and computing a trust-aware POMDP policy for deciding when the robot should request human assistance. Their work shows that assistance requests can serve not only to avoid failures but also to repair human trust in high-complexity tasks. In contrast, our work focuses on uncertainty in the robot's symbolic world belief rather than uncertainty in human trust. We use belief entropy to decide when feedback is needed and select what to ask by choosing the hypothesis expected to maximally reduce ambiguity among candidate world states.

## Introduction/Contribution에 넣을 수 있는 차별화 문장

While prior trust-aware assistance-seeking approaches optimize whether the robot should transfer control to a human supervisor based on inferred human trust, our framework treats human feedback as an information source for refining the robot's symbolic belief state. This distinction allows the robot not only to decide when feedback is necessary, but also to determine what task-relevant hypothesis should be queried to support downstream planning.

## Discussion에 넣을 수 있는 문장

The comparison with trust-aware assistance-seeking highlights a complementary view of human involvement in autonomous systems. Trust-aware policies reason about the human's latent cognitive state to avoid inappropriate autonomy, whereas our approach reasons about the robot's latent world state to avoid acting on incorrect beliefs. These perspectives are not mutually exclusive: future systems could jointly model human trust and robot belief uncertainty, enabling robots to ask questions that are both informative for planning and calibrated to the user's trust and workload.

## 최종 포지셔닝

우리 논문은 `When to Seek Help`와 같은 trust-aware assistance-seeking 연구와 같은 문제의식을 공유한다. 즉 자율 시스템이 항상 혼자 행동하거나 항상 인간에게 묻는 것은 비효율적이며, 선택적 인간 개입이 필요하다는 점이다.

하지만 우리 논문의 독립적인 기여는 다음에 있다.

- 인간 신뢰가 아니라 로봇의 symbolic belief uncertainty를 중심으로 한다.
- 도움 요청이 아니라 hypothesis-level feedback query를 다룬다.
- 언제 물을지뿐 아니라 무엇을 물을지도 결정한다.
- KB, perception, POMCP planning, FM, UI를 연결한 실제 로봇 시스템 아키텍처를 제안한다.
- task success와 query efficiency의 trade-off를 waste sorting과 tomato harvesting에서 검증한다.

따라서 이 논문은 관련 연구로 인용하되, 우리 연구는 trust-aware assistance seeking이 아니라 **belief-aware corrective feedback querying for knowledge-based robot planning**으로 포지셔닝하는 것이 가장 명확하다.
