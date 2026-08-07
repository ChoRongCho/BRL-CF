# 01 수정 계획: 추가 실험 없이 원고에서 처리할 항목

작성일: 2026-06-23

대상 리뷰: `00_stanford_review.pdf`

범위: 이 문서는 추가 실험 없이 논문 수정만으로 대응 가능한 리뷰 지적을 정리한다. 핵심은 새로운 결과를 만드는 것이 아니라, 논문의 문제 설정, 방법의 범위, 한계, 관련 연구, 표현 강도를 명확히 하는 것이다.

## 1. 우선순위가 높은 수정

### 1.1 Belief approximation의 성격을 명확히 쓰기

리뷰어가 문제 삼은 점:

- 현재 설명은 standard POMDP의 full posterior belief tracking처럼 읽힐 수 있다.
- 실제 방법은 committed knowledge state와 다음 action에서 유도되는 local frontier 위에서 belief를 유지한다.
- confidence가 threshold를 넘으면 MAP frontier state로 commit하므로, 다른 global hypothesis가 잘릴 수 있다.
- 이 과정은 overconfidence나 early mis-commitment로 이어질 수 있다.

수정 위치:

- `section/04_method.tex`: `Belief representation`
- `section/05_exp_system.tex`: `System Discussion`
- `section/06_conclusion.tex`

수정 방향:

- 이 방법이 exact history-conditioned posterior tracking이 아니라 frontier-based online approximation임을 명시한다.
- 장점과 비용을 같이 쓴다. 장점은 online tractability와 interpretable query generation이고, 비용은 multi-step hypothesis dependency를 일부 잃을 수 있다는 점이다.
- early wrong commitment, later contradiction, cascading dead-end 같은 실패 가능성을 Discussion/Limitation에 넣는다.
- 단, 이것을 결함처럼만 쓰지 말고 online robot execution을 위한 의도적인 systems design choice로 포지셔닝한다.

가능한 문장:

> Our belief representation should be interpreted as a local frontier-based approximation rather than exact posterior maintenance over the full action-observation history. After the confidence criterion is satisfied, the system commits the MAP frontier state to the knowledge base and constructs the next frontier from that committed state. This improves online tractability and keeps queries grounded in reachable symbolic alternatives, but it can prune global hypotheses that might become relevant after later observations.

### 1.2 Entropy query와 VOI 지적의 범위 정리

리뷰어가 문제 삼은 점:

- query selection이 expected entropy reduction으로 정의되어 있다.
- 리뷰어는 이를 보고 "downstream task value를 직접 최적화하지 않는다"고 지적했다.
- 즉, belief uncertainty를 줄이는 질문이 항상 task return을 가장 많이 올리는 질문은 아닐 수 있다는 주장이다.

중요한 해석:

- 이 지적은 framework 전체가 myopic이라는 뜻으로 받아들이면 안 된다.
- 우리 시스템에서 long-horizon task planning은 POMCP와 reward model이 담당한다.
- query는 task action을 대체하는 planning action이 아니라, observation ambiguity 또는 symbolic state uncertainty를 해소하기 위한 verification mechanism이다.
- 따라서 VOI-optimal query planning은 관련은 있지만, 이 논문의 직접적인 문제 설정과는 다르다.

왜 리뷰어가 이 얘기를 했는가:

- 논문에 `what to ask`, `informative query`, `planning under uncertainty`, `POMDP` 같은 표현이 함께 나오기 때문이다.
- decision-theoretic 관점의 리뷰어는 "질문도 information-gathering action이라면, query cost와 future task value를 포함해서 planning해야 하지 않나?"라고 읽을 수 있다.
- 즉, 논문이 "query as state disambiguation"인지 "query as VOI-optimal planning action"인지 경계를 더 선명하게 써야 한다.

수정 위치:

- `section/04_method.tex`: Eq. `select_hypo` 설명 직후
- `section/05_exp_system.tex`: `System Discussion`
- `section/06_conclusion.tex`
- `section/02_related.tex`: VOI / belief-dependent reward 관련 문헌을 짧게 연결

수정 방향:

- Entropy query는 "현재 frontier에서 어떤 predicate를 먼저 검증할지 정하는 기준"이라고 명확히 쓴다.
- 질문은 한 번만 하고 끝나는 것이 아니라 confidence가 충분해질 때까지 반복될 수 있음을 강조한다.
- Corollary는 true state가 frontier 안에 있고, user feedback이 correct하며, query가 후보를 분리하면 repeated filtering으로 correct frontier state에 도달할 수 있음을 보인다.
- Long-horizon action optimization은 POMCP와 reward model이 담당한다고 분리해서 쓴다.
- VOI/rho-POMDP는 query action, query cost, future value를 하나의 objective 안에서 joint optimization하는 다른 formulation이라고 설명한다.
- "우리가 VOI를 못 해서 부족하다"가 아니라, "우리 scope는 symbolic belief refinement와 POMCP planning의 architecture-level integration이다"라고 포지셔닝한다.

가능한 방어 문장:

> The entropy objective is used to select predicate-level verification questions within the current frontier, not to replace reward-based task planning. Long-horizon action selection is handled by POMCP under the task reward model. The query loop refines the symbolic state used by the planner, and Corollary 1 characterizes conditions under which repeated feedback identifies the correct frontier state. A full VOI formulation would instead treat queries as planning actions with explicit query costs and future task value; this is a complementary formulation rather than the objective of this work.

주의할 점:

- "Our method is myopic"처럼 쓰면 안 된다.
- 더 정확한 표현은 "the predicate-selection rule is greedy with respect to current frontier entropy" 정도다.
- 리뷰어 대응에서는 "framework 전체가 myopic한 것이 아니라, query ordering criterion이 VOI objective가 아니다"라고 분리해야 한다.

### 1.3 Observation likelihood를 본문에 요약

리뷰어가 문제 삼은 점:

- observation likelihood가 confidence와 query decision에 직접 영향을 준다.
- 그런데 핵심 파라미터와 likelihood form이 appendix에 많이 가 있다.

수정 위치:

- `section/04_method.tex`: Eq. `belief_update` 주변
- `section/05_exp_setting.tex`

수정 방향:

- waste sorting과 tomato harvesting에서 어떤 observation likelihood가 쓰이는지 본문에 짧게 요약한다.
- detect, category/classification, pick/place, scan observation이 belief update에 들어간다는 점을 설명한다.
- 전체 parameter table은 appendix에 있다고 안내한다.
- likelihood misspecification이나 calibration robustness는 추가 실험 없이는 limitation으로 처리한다.

### 1.4 KnowNo baseline 설정과 system/WoZ 결과 차이 설명

리뷰어가 문제 삼은 점:

- KnowNo가 action-level baseline이라 state exposure가 부족했을 수 있다.
- system evaluation에서는 KnowNo 성능이 낮은데, tomato WoZ에서는 KnowNo가 가장 높게 나와서 mismatch처럼 보인다.

수정 위치:

- `section/05_exp_system.tex`: `Baseline Comparison`
- `section/05_exp_user.tex`: `Study Design`, `Task Success and Operation Time`, `Summary`
- `appendix/03_exp.tex`: `KnowNo Baseline Implementation`

수정 방향:

- system experiment의 KnowNo가 어떤 입력, action candidates, prompt, threshold를 받았는지 명확히 쓴다.
- system-level GPT/action-query baseline과 WoZ human action-choice interface를 구분한다.
- tomato WoZ에서 KnowNo가 높은 이유를 설명한다. Tomato task는 navigation-detection-pick-scan-place/discard 흐름이 비교적 순차적이어서 next action 선택이 더 쉽게 보일 수 있다.
- waste에서는 action query가 사용자가 hidden category uncertainty와 detect/grasp 필요성을 추론하게 만들기 때문에 더 어렵다고 설명한다.
- action-level baseline은 context와 interface design에 민감하다는 점을 한계로 인정한다.

### 1.5 Real-robot validation 표현 낮추기

리뷰어가 문제 삼은 점:

- real-robot 결과가 closed-loop quantitative evaluation인지 feasibility demonstration인지 불분명하다.

수정 위치:

- `section/05_real_robot.tex`
- `main.tex`: abstract
- `section/06_conclusion.tex`

수정 방향:

- closed-loop hardware success/query/runtime 결과가 없다면, real-robot implementation 또는 feasibility demonstration이라고 표현한다.
- main quantitative claims는 controlled system evaluation과 WoZ user study에서 나온 것임을 명확히 한다.
- tomato harvesting 전체가 hardware에서 정량 평가된 것처럼 읽히지 않게 한다.

가능한 문장:

> The hardware implementation demonstrates integration of the symbolic planner, perception interface, query module, and robot control stack. The quantitative success, query, and runtime results reported above are obtained from the controlled evaluation; closed-loop hardware benchmarking remains future work.

## 2. 중간 우선순위 수정

### 2.1 Frontier merging과 frontier size 설명 추가

리뷰어가 문제 삼은 점:

- duplicate successor state가 어떻게 merge되는지 충분히 명확하지 않다.
- frontier size가 실제로 어느 정도 커지는지, runtime과 어떻게 연결되는지 더 설명이 필요하다.

수정 위치:

- `section/04_method.tex`: `Belief representation`
- `section/05_exp_system.tex`: `Scalability Test`

수정 방향:

- 동일한 symbolic fact set을 갖는 successor states는 merge한다고 명시한다.
- frontier size는 fixed particle count가 아니라 action과 domain model이 유도한 unique reachable symbolic outcomes 수라고 설명한다.
- scalability table의 `Avg. belief frontier size`를 runtime 증가 해석과 연결한다.

### 2.2 User study claim을 조심스럽게 낮추기

리뷰어가 문제 삼은 점:

- user study는 small-sample WoZ이며, 각 condition/domain을 한 번씩만 경험한다.
- SAGAT 결과는 mixed이고, workload도 전반적으로 낮다.

수정 위치:

- `main.tex`: abstract
- `section/05_exp_user.tex`: summary
- `section/06_conclusion.tex`

수정 방향:

- "confirm reduced cognitive workload" 같은 강한 표현을 "provide preliminary evidence"로 낮춘다.
- user study를 main proof가 아니라 supplementary usability check로 표현한다.
- broad human-factors superiority를 주장하지 않는다.

### 2.3 Scalability limitation과 future optimization 추가

리뷰어가 문제 삼은 점:

- 6-object tomato setting에서 runtime이 크게 증가한다.
- progressive widening, DESPOT-style scenario sampling, caching/reuse 같은 개선 방향을 논의할 수 있다.

수정 위치:

- `section/05_exp_system.tex`: `Scalability Test`, `System Discussion`
- `section/06_conclusion.tex`

수정 방향:

- larger tomato scenes에서 planning과 belief maintenance cost가 드러난다고 인정한다.
- 가능한 비용 원인으로 frontier growth, POMCP tree expansion, observation-likelihood evaluation을 언급한다.
- future work로 progressive widening, DESPOT-style scenario sampling, tree reuse, caching을 넣는다.

### 2.4 VOI, rho-POMDP, active perception 관련 연구 추가

리뷰어가 문제 삼은 점:

- decision-theoretic information gathering 문헌이 관련 있다.

수정 위치:

- `section/02_related.tex`: `Planning under Uncertainty`
- `section/02_related.tex`: `Human Corrective System`

수정 방향:

- value of information, belief-dependent rewards, rho-POMDP/rho-POMCP, active perception을 짧게 연결한다.
- 단, 우리 방법과의 차이를 분명히 한다. 우리 query는 robot viewpoint/control action이 아니라 human-facing predicate check이다.
- VOI는 future work 또는 alternative formulation으로 두고, 현재 논문의 scope를 침범하지 않게 쓴다.

## 3. 낮은 우선순위 정리

### 3.1 수식과 notation 정리

리뷰어가 문제 삼은 점:

- normalization step, weight notation, entropy notation에 minor inconsistency가 있을 수 있다.

수정 위치:

- `section/04_method.tex`
- `appendix/01_algorithms.tex`

체크리스트:

- unnormalized weight `w'_k`와 normalized weight `\tilde{w}'_k`를 구분한다.
- `j`를 frontier size로 일관되게 사용한다.
- filtering 후 renormalization을 명시한다.
- confidence, entropy, MAP commit이 normalized weight 기준인지 확인한다.

## 4. Entropy/VOI 논쟁 정리

핵심 결론:

- 우리가 이상한 문제를 푸는 것이 아니다.
- 리뷰어가 VOI를 언급한 것은 논문이 `what to ask`와 `POMDP planning`을 함께 말하기 때문에, query를 information-gathering action으로 읽었기 때문이다.
- 하지만 우리 논문의 query는 task action planning을 대체하는 것이 아니라, POMCP가 사용할 symbolic state의 불확실성을 줄이는 observation-refinement mechanism이다.

정확한 구분:

- POMCP: long-horizon task action planning을 담당한다.
- Reward model: 어떤 action sequence가 task에 좋은지 결정한다.
- Entropy query: 현재 frontier에서 어떤 predicate를 검증할지 정한다.
- Feedback filtering: user answer와 맞지 않는 frontier states를 제거한다.
- Corollary: true state가 frontier 안에 있고 feedback이 correct하며 query가 후보를 분리하면, repeated filtering으로 correct frontier state에 도달할 수 있음을 보인다.

리뷰어가 여전히 지적할 수 있는 부분:

- query ordering 자체는 current frontier entropy 기준의 greedy rule이다.
- query를 POMDP action으로 넣고, query cost와 future task value를 함께 최적화하는 VOI formulation은 아니다.
- 특히 `tau < 1`에서 모든 uncertainty를 제거하기 전에 MAP commit을 하면, 어떤 질문을 먼저 했는지가 남아 있는 uncertainty에 영향을 줄 수 있다.

따라서 논문에서 취할 태도:

- "우리 방법은 myopic하다"고 쓰지 않는다.
- "The predicate-selection rule is greedy with respect to current frontier entropy"라고 좁혀서 표현한다.
- "The overall framework is not myopic with respect to task execution because POMCP handles long-horizon action selection under the reward model"이라고 방어한다.
- VOI는 다른 문제 설정이지만 관련 문헌으로 인정하고, future work 또는 alternative formulation으로 언급한다.

## 5. 추천 수정 순서

1. Belief approximation과 MAP commit의 한계를 Method/Discussion에 명확히 쓴다.
2. Entropy query의 역할을 "state disambiguation"으로 재정의하고, POMCP/reward가 long-horizon planning을 담당한다는 점을 강조한다.
3. Real-robot section을 closed-loop quantitative evaluation이 아니라 implementation/feasibility demonstration으로 정리한다.
4. KnowNo의 system experiment와 WoZ setup 차이를 설명하고, tomato/waste 결과 차이를 해석한다.
5. Abstract와 conclusion에서 workload 및 real-world validation claim을 낮춘다.
6. Related work에 VOI/rho-POMDP/active perception을 추가하되, 우리 scope와의 차이를 분명히 한다.
7. Observation likelihood 요약, frontier merging 설명, notation cleanup을 진행한다.
