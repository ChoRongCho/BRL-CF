# A Survey of Knowledge-based Sequential Decision Making under Uncertainty 리뷰

## 1. 논문 정보

```text
Shiqi Zhang and Mohan Sridharan,
“A Survey of Knowledge-based Sequential Decision Making under Uncertainty,”
arXiv:2008.08548v3, 30 June 2022.
```

- 원문: [A survey of knowledge-based sequential decision-making under uncertainty.pdf](<./A survey of knowledge-based sequential decision-making under uncertainty.pdf>)
- 주제: Declarative knowledge reasoning과 불확실성 아래 sequential decision making의 결합
- 주요 대상: Logic, ASP, probabilistic planning, MDP/POMDP, reinforcement learning, robotics
- 논문 성격: Taxonomy와 open problem을 제시하는 survey

## 2. 초록 번역

선언적 지식을 이용한 추론(Reasoning with Declarative Knowledge, RDK)과 순차적
의사결정(Sequential Decision Making, SDM)은 인공지능의 두 가지 핵심 연구
분야이다. RDK 방법은 사전에 제공되거나 시간에 따라 획득되는 상식 지식을 포함한
선언적 domain knowledge를 이용해 추론한다. 반면 SDM 방법인 probabilistic
planning과 reinforcement learning은 일정한 시간 범위에서 기대 누적 utility를
최대화하는 action policy를 계산한다. 두 방법군 모두 uncertainty를 다룬다.

두 영역에는 풍부한 연구가 존재하지만, 연구자들은 아직 이들의 상호 보완적인 장점을
완전히 탐구하지 못했다. 본 논문은 uncertainty 아래에서 sequential decision을
수행하면서 RDK 방법을 활용하는 algorithm을 조사한다. 또한 주요 발전, open
problem 및 향후 연구 방향을 논의한다.

## 3. 연구 질문과 범위

논문의 중심 질문은 다음과 같다.

```text
How best to reason with declarative knowledge
for sequential decision making under uncertainty?

불확실성 아래의 순차적 의사결정을 위해
선언적 지식을 어떻게 추론하고 활용하는 것이 가장 좋은가?
```

Survey는 다음 범위로 대상을 제한한다.

- RDK를 이용해 SDM을 지원하는 방법
- Dynamic domain에서 한 agent가 uncertainty 아래 sequential decision을 수행하는 문제
- Probabilistic planning과 reinforcement learning
- Logic으로 표현한 declarative domain knowledge
- Robotics 사례를 포함한 실제 agent system

개별 RL 또는 POMDP solver의 성능을 비교하는 것이 아니라, logic knowledge와
probabilistic decision making이 어떻게 연결되는지를 분류하는 것이 목적이다.

## 4. 두 연구 영역

### 4.1 Reasoning with Declarative Knowledge

RDK는 object, attribute, action, relation 및 domain axiom을 relational statement로
표현하고 추론한다. 대표적인 표현은 다음과 같다.

- Prolog
- First-Order Logic
- Answer Set Programming(ASP)
- Action language
- STRIPS/PDDL
- Ontology

Prolog rule의 기본 형태:

```text
Head :- Body
```

ASP rule의 예:

```text
a :- b, c, not d.
```

ASP의 `not`은 classical negation이 아니라 default negation이다. Literal은 true,
false뿐 아니라 unknown일 수 있으며, 새로운 정보가 들어오면 기존 결론을 철회할 수
있는 non-monotonic reasoning을 지원한다.

RDK의 강점:

- Commonsense와 default knowledge 표현
- Object와 relation에 대한 relational abstraction
- 불완전한 지식에서의 추론
- Classical planning과 diagnosis
- 사람이 읽고 수정할 수 있는 명시적인 knowledge

RDK의 한계:

- Action과 observation uncertainty를 확률적으로 처리하기 어렵다.
- Long-term expected utility를 최대화하는 policy 계산에 적합하지 않다.
- Boolean logic만으로 sensor noise와 stochastic transition을 표현하기 어렵다.

### 4.2 Sequential Decision Making

SDM은 현재 state 또는 belief state를 action에 연결하는 policy를 계산한다.

```text
π: state → action
π: belief → action
```

Survey는 SDM을 두 종류로 구분한다.

#### Probabilistic planning

World model이 주어진 경우 MDP 또는 POMDP로 policy를 계산한다.

```text
MDP = <S, A, T, R>
POMDP = <S, A, T, R, O, Ω>
```

POMDP에서는 다음 Bayesian update로 observation 이후 belief를 갱신한다.

```text
b'(s') ∝ O(o | s',a) Σ_s T(s' | s,a)b(s)
```

출력 policy는 belief에서 action으로의 mapping이다.

```text
π: b → a
```

#### Reinforcement learning

Reward 또는 transition model이 완전하지 않을 때 environment interaction 경험을 통해
policy 또는 world model을 학습한다.

- Model-based RL: `T`, `R`을 학습한 뒤 planning
- Model-free RL: 명시적인 model 없이 value 또는 policy 학습

SDM의 강점:

- Stochastic action과 noisy observation 처리
- Long-term utility 최적화
- Belief 기반 action selection
- Exploration과 exploitation의 trade-off

SDM의 한계:

- 상식이나 default rule을 수치 model로 표현하기 어렵다.
- State/action space가 커지면 계산량이 급증한다.
- 사람이 제공한 symbolic knowledge를 효율적으로 활용하기 어렵다.

## 5. RDK와 SDM을 결합해야 하는 이유

두 영역은 서로 반대되는 강점과 약점을 가진다.

| 구분 | RDK | SDM |
|---|---|---|
| 주 표현 | Logic, relation, rule | Probability, state, belief |
| 강점 | Commonsense, abstraction, explainability | Uncertainty, stochastic dynamics, long-term policy |
| 약점 | Quantitative uncertainty | Declarative/common-sense knowledge |
| Planning 결과 | Action sequence | State/belief-dependent policy |

예를 들어 다음 knowledge는 확률 하나보다 logic으로 표현하기 자연스럽다.

```text
책은 보통 도서관에 있다.
단, 요리책은 부엌에 있을 수 있다.
```

반면 다음 문제는 POMDP가 더 적합하다.

```text
센서의 정확도는 0.85이다.
Pick action은 0.9의 확률로 성공한다.
현재 tomato가 ripe일 belief는 0.62이다.
```

통합 시스템은 logic knowledge를 사용해 relevant state/action을 제한하면서 POMDP나
RL로 uncertainty와 long-term consequence를 처리할 수 있다.

## 6. Survey의 8개 characteristic factor

논문의 핵심 기여는 RDK-for-SDM 방법을 분류하는 8개 축이다.

```text
Representation:
  Factor 1. Knowledge representation
  Factor 2. Knowledge abstraction

Reasoning:
  Factor 3. Dynamics in RDK
  Factor 4. World model in SDM
  Factor 5. Observability in SDM

Knowledge acquisition:
  Factor 6. Online vs. offline
  Factor 7. Active vs. reactive
  Factor 8. Knowledge source
```

### 6.1 Factor 1: Unified vs. linked representation

#### Unified representation

Logic knowledge와 probabilistic uncertainty를 하나의 representation으로 표현한다.

예:

- Markov Logic Network
- ProbLog
- P-log
- Probabilistic relational model
- PPDDL/RDDL 기반 통합 model

장점:

- Knowledge와 uncertainty 사이의 의미적 연결이 명확하다.
- 하나의 reasoning framework를 사용할 수 있다.
- 높은 expressive power를 가진다.

단점:

- Representation과 inference가 복잡하다.
- Large domain에서 computational burden이 크다.
- Exact reasoning이 어려워 approximate method가 필요할 수 있다.

#### Linked representation

Logic과 probabilistic model을 별도로 유지하면서 정보를 전달한다.

```text
RDK component:
  ASP, classical planner, commonsense knowledge

SDM component:
  MDP, POMDP, RL

Link:
  state/action projection, prior, reward, constraint, observation
```

장점:

- 각 문제에 가장 적합한 representation과 solver를 사용할 수 있다.
- 계산 효율과 modularity가 좋다.
- Coarse symbolic planning과 fine probabilistic control을 분리할 수 있다.

단점:

- 두 표현 사이 consistency를 유지해야 한다.
- Probability가 남아 있는 정보를 true/false logic으로 commit하면 잘못된 추론이
  발생할 수 있다.
- Information/control transfer를 직접 설계해야 한다.

### 6.2 Factor 2: Knowledge abstraction

Knowledge를 한 해상도로만 표현하는지, 여러 abstraction level에서 표현하는지를
분류한다.

```text
Coarse level:
  어느 방으로 이동할지, 어떤 object를 처리할지

Fine level:
  metric location, grasp pose, motion trajectory
```

Hierarchical representation은 큰 state/action space를 줄이지만, 서로 다른 abstraction
사이에서 relevant knowledge를 찾고 consistency를 유지하기 어렵다.

### 6.3 Factor 3: Dynamics in RDK

RDK가 현재 snapshot에 대한 inference만 수행하는지, action과 시간에 따른 변화를
추론하는지를 구분한다.

```text
Static RDK:
  현재 fact로부터 새로운 fact 추론

Dynamic RDK:
  action precondition/effect
  classical planning
  execution monitoring
  diagnosis와 replanning
```

SDM의 exploration을 logic으로 안내하려면 action과 change를 표현하는 dynamic RDK가
필요하다.

### 6.4 Factor 4: World models in SDM

```text
World model available:
  Probabilistic planning, MDP/POMDP

World model unavailable:
  Reinforcement learning

Model learned explicitly:
  Model-based RL

Policy learned directly:
  Model-free RL
```

### 6.5 Factor 5: State vs. belief state

Environment가 fully observable인지 partially observable인지 구분한다.

```text
Fully observable:
  State-based MDP

Partially observable:
  Belief-based POMDP
  또는 neural implicit state representation
```

현재 observation이 불완전하거나 틀릴 수 있다면 belief state가 필요하다.

### 6.6 Factor 6: Online vs. offline knowledge acquisition

```text
Online:
  Task execution과 knowledge acquisition을 교차 수행
  Observation마다 knowledge를 갱신

Offline:
  Task 전후 별도 training/batch 단계에서 knowledge 획득
```

Knowledge acquisition을 지원하지 않는 방법도 이 survey에서는 offline 범주로 묶는다.

### 6.7 Factor 7: Active vs. reactive acquisition

```text
Active acquisition:
  지식을 얻기 위한 action을 명시적으로 계획하고 실행
  sensing, exploration, human query

Reactive acquisition:
  Task action을 수행하는 과정에서 부수적으로 관측한 정보로 knowledge 갱신
```

Active acquisition에는 다음이 포함된다.

- Unknown action outcome을 확인하는 exploration
- Sensor 방향 또는 위치 변경
- 사람에게 질문
- 필요한 data를 얻기 위한 explicit experiment

### 6.8 Factor 8: Knowledge source

Declarative knowledge의 출처를 분류한다.

- Domain expert가 직접 작성한 rule
- Robot sensor와 computer vision
- Task experience
- Demonstration
- Human dialogue
- Web 또는 외부 database
- 여러 source의 결합

Expert knowledge는 신뢰할 수 있지만 작성 비용이 크다. Interaction으로 얻은 knowledge는
사람의 부담이 적을 수 있지만 noisy하거나 불완전할 수 있다.

## 7. 대표적인 RDK-for-SDM 방법

Survey는 대표 시스템을 주요 기여에 따라 세 그룹으로 분류한다.

### 7.1 Representation-focused systems

#### Unified systems

- Statistical relational AI
- Markov Logic Network
- ProbLog과 DTProbLog
- P-log
- PPDDL
- RDDL
- First-order MDP/POMDP representation

이 방법들은 relational structure, logic rule, probability 및 utility를 하나의 formalism에
표현하려 한다. Expressivity는 높지만 inference가 복잡하다.

#### Linked systems

Logic planner와 probabilistic planner를 별도 component로 유지한다.

대표적인 방향:

- ASP 기반 coarse planning과 POMDP 기반 fine execution
- Classical plan action을 MDP/RL option으로 mapping
- Symbolic state와 geometric state의 연결
- Declarative knowledge로 POMDP state/action relevance 결정

#### REBA

REBA는 coarse-resolution action language description과 fine-resolution probabilistic
transition diagram을 연결한다.

```text
Coarse RDK:
  Goal과 domain rule을 사용해 abstract plan 계산

Refinement/zooming:
  현재 abstract action에 relevant한 state/action만 선택

Fine SDM:
  축소된 POMDP로 구체적 action 실행
```

전체 fine-resolution model을 항상 풀지 않고 현재 action에 필요한 부분만 zoom하므로
scalability를 개선한다.

#### CORPP/iCORPP

CORPP는 P-log로 commonsense와 probabilistic declarative knowledge를 추론하고,
그 결과로 POMDP의 informative prior를 생성한다. iCORPP는 contextual knowledge를
사용해 POMDP reward와 transition도 자동으로 결정한다.

### 7.2 Reasoning-focused systems

이 그룹은 declarative reasoning을 이용해 SDM 계산을 줄이거나 policy 품질을 높인다.

대표적인 방법:

- RDK로 relevant state/action space 구성
- Logic으로 POMDP prior 생성
- Logical smoothing으로 과거 belief 수정
- Classical plan으로 RL option 또는 hierarchy 생성
- Commonsense knowledge를 이용한 reward shaping
- Sparse reward 문제에 logic-derived intermediate reward 제공
- Safety rule로 위험한 exploration 제한

Survey는 Hoelscher et al.의 targeted-query POMDP도 이 문맥에서 언급한다. Declarative
knowledge는 사용자의 logic objective를 표현하고, POMDP는 uncertainty 아래 task와
query action을 계획한다.

### 7.3 Knowledge-acquisition-focused systems

Knowledge acquisition은 세 source로 나뉜다.

#### Acting 중 acquisition

- Unexpected action outcome에서 unknown constraint 학습
- Explicit exploration으로 새로운 action effect 획득
- RDK reasoning이 필요한 learning만 trigger

#### Experience에서 acquisition

- Demonstration과 labeled trial
- RL replay buffer에서 symbolic state 추출
- Real-world trial에서 precondition/effect 학습
- Deep network behavior를 설명하는 axiom 획득

#### Human, Web 및 외부 source에서 acquisition

- Dialogue로 새로운 action과 precondition 학습
- Synonym과 object entity 학습
- Human verbal description에서 action effect 추출
- Ambiguous human question을 해소하기 위한 clarification question 생성
- Web search 결과를 first-order logic knowledge로 변환

## 8. Challenges and opportunities

### 8.1 Representation choice

Unified와 linked representation 중 어느 것이 항상 우월하지 않다. Application마다
다음을 함께 평가해야 한다.

- Expressiveness
- Computational complexity
- Correctness guarantee
- Abstraction support
- Representation 사이의 consistency
- Robot behavior에 대한 formal property

### 8.2 Interactive learning

Dynamic domain의 knowledge는 불완전하며 시간이 지나면 유효하지 않을 수 있다.
모든 것을 계속 학습하면 계산량이 커지므로 reasoning을 이용해 다음을 결정해야 한다.

```text
언제 학습할 것인가?
무엇을 학습할 것인가?
현재 task에 어떤 concept가 relevant한가?
기존 knowledge와 새 knowledge를 어떻게 합칠 것인가?
```

### 8.3 Human in the loop

논문은 다음 가정이 실제로 항상 성립하지 않는다고 지적한다.

```text
사람이 initial knowledge를 정확히 작성한다.
사람이 execution 중 항상 응답 가능하다.
사람의 feedback이 항상 신뢰할 수 있다.
```

사람을 항상 이용 가능한 oracle이 아니라 필요성과 availability에 따라 consultation할
collaborator로 모델링해야 한다. 고려할 요소는 다음과 같다.

- Human expertise
- Availability
- Communication protocol
- Interaction cost
- Social context
- Human-understandable explanation

### 8.4 Reasoning, learning, control의 결합

실제 robot은 symbolic reasoning, probabilistic learning, continuous control을 모두
필요로 한다. Coarse discrete action과 fine continuous manipulation 사이의 연결이
중요한 open problem이다.

### 8.5 Scalability와 teamwork

State/action/knowledge space가 커지고 human-robot team이 추가되면 representation과
reasoning complexity가 급증한다. Relevance, persistence, non-procrastination 등의
cognitive principle을 사용해 필요한 resource와 algorithm만 선택하는 방향을
제안한다.

### 8.6 Explainability와 trust

Logic knowledge는 belief와 decision의 근거를 추적하고 여러 abstraction level에서
설명할 기반을 제공한다. 하지만 비전문 사용자가 실제로 설명을 이해하고 신뢰하는지를
엄밀히 평가해야 한다.

### 8.7 Evaluation measure와 benchmark

개별 component의 accuracy와 execution time만으로 통합 시스템을 평가하기 어렵다.
다음 연결 효과를 측정해야 한다.

- Reasoning이 knowledge acquisition을 얼마나 효율적으로 안내하는가
- 획득한 knowledge가 planning을 얼마나 개선하는가
- 더 복잡한 domain으로 확장 가능한가
- Component 사이 interaction 수와 시간이 얼마인가
- Human satisfaction과 explanation quality가 어떠한가

## 9. 현재 Active Search 구조의 taxonomy mapping

현재 프로젝트를 survey의 8개 factor에 배치하면 다음과 같다.

| Factor | 현재 구조 | 판단 |
|---|---|---|
| 1. Representation | ASP/State와 particle POMDP를 별도 유지 | Linked representation |
| 2. Abstraction | Symbolic predicate와 grounded action | 주로 단일 symbolic level, 향후 hierarchy 가능 |
| 3. Dynamics in RDK | Domain rule, action precondition/effect | Dynamic RDK |
| 4. World model | Transition/observation/reward model 제공 | Model-based probabilistic planning |
| 5. Observability | Particle belief | Partially observable/POMDP |
| 6. Acquisition timing | 실행 중 observation마다 belief update | Online acquisition |
| 7. Acquisition mode | Detect/scan/ask action을 planner가 선택 | Active acquisition |
| 8. Knowledge source | Domain expert, robot sensor, human answer | Multiple sources |

구조를 도식화하면 다음과 같다.

```text
ASP / declarative domain rules
  ├─ object와 predicate
  ├─ action applicability
  ├─ possible/consistent symbolic state
  └─ goal condition
             ↓ linked representation
Particle-belief ρ-POMDP
  ├─ stochastic transition
  ├─ noisy observation
  ├─ belief-dependent reward
  └─ physical/sensing/query action selection
             ↓
Human targeted query
  ├─ answer
  ├─ noisy answer
  └─ null/unavailable answer
```

## 10. 이 Survey가 현재 구현에 주는 핵심 시사점

### 10.1 단순 POMDP baseline이 아니라 RDK-for-SDM system

현재 코드는 ASP domain knowledge와 particle POMDP를 연결한다. 따라서 논문에서는
다음처럼 설명할 수 있다.

```text
A linked RDK-for-SDM architecture combining
ASP-based declarative reasoning with
belief-space sequential decision making.
```

### 10.2 Ask action은 active online knowledge acquisition

`ask_*`는 단순 communication utility가 아니라 Factor 6과 7에 해당하는 active online
knowledge-acquisition action이다.

```text
Planner가 필요한 질문 선택
→ 사람 answer observation
→ belief 및 symbolic knowledge 갱신
→ 갱신된 knowledge로 다음 action 계획
```

### 10.3 Linked representation의 consistency 문제

Particle belief의 MAP state를 곧바로 ASP fact로 확정하면 residual uncertainty가
사라지고 잘못된 logical inference가 발생할 수 있다. Survey가 지적한 linked
representation의 대표적 위험이다.

따라서 다음을 구분해야 한다.

```text
Certain knowledge:
  모든 유효 particle에서 참이거나 사람이 확정한 fact

Believed knowledge:
  높은 probability지만 아직 uncertainty가 남은 fact
```

### 10.4 Query relevance로 action explosion 완화

모든 object-property 조합을 grounded query로 만들면 action space가 급증한다. RDK를
사용해 현재 goal과 plan에 relevant한 predicate만 query candidate로 제한할 수 있다.

```text
전체 query action
→ ASP goal/precondition dependency 분석
→ 현재 task-relevant query만 planner에 제공
```

이는 Survey가 강조하는 reasoning-guided knowledge acquisition의 직접적인 사례가 된다.

### 10.5 Human availability와 expertise

사람을 항상 정확하고 이용 가능한 oracle로 두면 survey가 지적한 human-in-the-loop
문제를 해결하지 못한다. 최소한 다음 observation이 필요하다.

```text
correct answer
wrong answer
no answer / timeout
```

Human model에는 accuracy, availability, query cost를 포함해야 한다.

## 11. 장점

- Logic, POMDP, RL을 하나의 관점에서 비교할 수 있는 taxonomy를 제공한다.
- Unified와 linked representation의 trade-off를 명확히 한다.
- Knowledge acquisition을 online/offline 및 active/reactive로 구분한다.
- Human query를 broader knowledge-acquisition framework에 배치할 수 있다.
- ASP와 POMDP를 결합한 현재 architecture를 설명할 이론적 언어를 제공한다.
- Representation, reasoning, learning, control을 함께 평가해야 함을 강조한다.

## 12. 한계

- 개별 방법에 대한 정량적인 meta-analysis는 제공하지 않는다.
- 매우 넓은 연구 영역을 다루므로 각 algorithm의 구현 세부사항은 제한적이다.
- 8개 factor가 대부분 독립적이라고 가정하지만 실제 system에서는 강하게 결합될 수 있다.
- 최근 foundation model/LLM 기반 knowledge acquisition은 출판 시점상 거의 다루지 않는다.
- 어떤 representation을 선택해야 하는지에 대한 단일한 algorithmic solution은 없다.
- Survey 자체의 새로운 planner나 실험 baseline은 없다.

## 13. 최종 평가

이 논문은 targeted query algorithm을 직접 제안하는 논문은 아니다. 대신 현재
Active Search 연구를 다음 세 영역의 결합으로 설명할 수 있는 taxonomy를 제공한다.

```text
Declarative knowledge:
  ASP domain rule와 symbolic predicate

Sequential decision making:
  Particle-belief ρ-POMDP

Knowledge acquisition:
  Robot sensing과 human targeted query
```

현재 프로젝트에서 가장 중요한 분류는 다음이다.

```text
Linked representation
+ Dynamic RDK
+ Model-based belief SDM
+ Active online acquisition
+ Human/sensor/expert knowledge sources
```

따라서 이 survey는 구현 baseline보다는 논문의 related work와 system architecture를
정당화하는 참고문헌으로 가치가 크다. 특히 `ask_*` action을 “질문 기능”이 아니라
**reasoning-guided active online knowledge acquisition**으로 설명하는 근거를 제공한다.
