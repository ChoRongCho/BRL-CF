# T-RO Review Notes

Created: 2026-06-22

## 1. 핵심 판단

리뷰어의 전반적 평가는 "잘 설계된 시스템 논문이지만, T-RO/IJRR 수준의 방법론적 깊이와 검증 강도는 아직 부족할 수 있다"에 가깝다. 긍정적으로 보는 지점은 명확하다: symbolic knowledge, online POMDP planning, state-level human query를 하나의 아키텍처로 묶은 점, entropy-based query selection, state-level query가 action-level query보다 실용적일 수 있다는 실험적 메시지다.

반대로 가장 큰 리스크는 다음 네 가지다.

1. Belief approximation이 myopic하고, MAP frontier state로 commit하는 방식이 overconfidence와 cascading failure를 만들 수 있다는 점.
2. Observation likelihood model이 query decision과 success를 좌우하는데, calibration/robustness 분석이 부족하다는 점.
3. Entropy reduction은 myopic proxy이며 downstream task value, VOI, long-horizon dependency를 직접 최적화하지 않는다는 점.
4. KnowNo baseline 설정과 system-level/WoZ 결과 차이가 커서 fairness 문제가 제기될 수 있다는 점.

## 2. 수정 우선순위

### Priority A: 반드시 본문에서 보강

- Belief approximation의 성격을 "exact POMDP inference"가 아니라 "frontier-based online approximation"으로 명확히 포지셔닝한다.
- MAP commit이 실패할 수 있는 조건, early mis-commitment, later contradiction 처리 한계를 Discussion/Limitation에 넣는다.
- Observation likelihood model을 본문에서 짧게라도 요약하고, appendix로만 넘기지 않는다.
- KnowNo baseline이 system experiment와 WoZ study에서 어떻게 구성되었는지 명확히 적고, 성능 차이의 원인을 해석한다.
- Real-robot 결과가 closed-loop quantitative result인지 feasibility demonstration인지 명확히 구분한다.

### Priority B: 가능하면 실험/분석 추가

- Observation model sensitivity: noise/bias를 키웠을 때 success rate, query count, failure case가 어떻게 바뀌는지.
- Wrong or delayed human answer robustness: 일정 비율의 incorrect answer 또는 latency를 넣은 ablation.
- Frontier size/runtime profiling: number of unique successor states, merged duplicates, POMCP tree size, observation likelihood cost.
- VOI-style criterion과 entropy-only query criterion의 비교 또는 최소한 short discussion.

### Priority C: 문장/구조 개선

- "More queries are not always better" 주장은 실험 결과와 잘 맞으므로 유지하되, "human burden" 관련 주장은 WoZ small N 때문에 조심스럽게 쓴다.
- T-RO 투고용이면 novelty를 "new algorithm"보다 "interpretable system-level synthesis and state-level query formulation"으로 방어하는 편이 안전하다.
- Related work에 VOI, rho-POMDP, POMCPOW/DESPOT, active perception/interactive POMDP를 더 명확히 연결한다.

## 3. 예상 Response 방향

### Belief approximation

리뷰어가 지적한 부분은 타당하다. 응답에서는 "we agree"로 시작하고, 본문에 다음을 추가했다고 설명하는 방향이 좋다.

- The belief is maintained over a local symbolic frontier, not the full history posterior.
- This approximation trades global posterior fidelity for online tractability and interpretable query generation.
- Failure modes include early wrong commitment and loss of multi-step dependencies.
- The system can be extended with revalidation/backtracking or maintaining multiple committed hypotheses.

### Observation model

응답 전략:

- Main text에 likelihood form과 domain-specific parameters를 요약했다.
- Appendix에는 더 자세한 calibration procedure, parameter table, sensitivity result를 둔다.
- 추가 실험이 가능하면 biased/noisy observation에 대한 성능 degradation plot/table을 넣는다.

### Query objective and VOI

응답 전략:

- Entropy reduction은 computationally cheap, task-agnostic, predicate-level query에 적합한 criterion이라고 방어한다.
- 다만 downstream value를 직접 최적화하지 않는 한계는 인정한다.
- VOI/rho-POMDP는 future extension 또는 small comparison으로 넣는다.

### KnowNo baseline fairness

응답 전략:

- System-level experiment에서 KnowNo가 받은 input/context, action candidates, state exposure, LLM prompt 또는 oracle setting을 명확히 기술한다.
- WoZ에서는 human operator context가 달라져 성능이 바뀌었을 가능성을 설명한다.
- 동일 정보량 조건 또는 additional baseline을 넣을 수 있으면 가장 좋다.

### Real robot

응답 전략:

- closed-loop이면 success/query/runtime을 table로 제시한다.
- closed-loop가 아니면 "hardware feasibility demonstration"으로 표현을 낮추고, main quantitative claims는 controlled experiment에 기반한다고 명확히 한다.

## 4. Reviewer Questions To Track

- How sensitive are results to misspecified observation likelihoods?
- What happens with wrong or delayed human feedback?
- How large can the frontier become, and how is it capped/pruned?
- Why not compare to VOI-style query selection?
- How exactly was KnowNo instantiated in system experiments vs. WoZ?
- What are the closed-loop real-robot quantitative results?
- Have progressive widening or DESPOT-style scenario sampling been tried?

## 5. Original Review

```text
Summary
The paper proposes a knowledge-based POMDP framework that decides when to ask for help and what to ask by maintaining a belief over symbolic world hypotheses, triggering queries when belief entropy is high, and selecting state-level yes/no queries that maximize expected entropy reduction. A POMCP-based planner is integrated with a centralized symbolic knowledge base and a feedback manager that translates predicates into user-facing questions. Experiments in simulated waste sorting and a tomato-harvesting setup (with an additional WoZ user study) show that the approach can retain high task success while substantially reducing the frequency of user queries compared to continuous supervision.

Strengths
Technical novelty and innovation
The paper presents a clear integration of symbolic task models (STRIPS-style preconditions/effects) with online POMDP planning via a particle-like “frontier over reachable successor states,” a pragmatic compromise between fully factored beliefs and global particle filtering.
The what-to-ask component is principled (expected entropy reduction over hypotheses) and tightly coupled to symbolic action feasibility, resulting in informative state-level queries rather than action-level ones.
The architecture cleanly unifies sensing, planning, user feedback, and a shared knowledge base, with predicates used both for planning and interaction.
Experimental rigor and validation
Ablations over confidence thresholds (τ) systematically show trade-offs between success and query load across two domains of differing complexity.
Baseline comparisons include Always-Ask, No-Ask, Random, and an action-query baseline (KnowNo), highlighting the distinct advantages and failure modes of state-level vs. action-level queries.
A scalability test (4→6 objects) surfaces computational effects and selective querying properties as the symbolic state space grows.
Clarity of presentation
The system architecture, planning loop, belief update, and query-selection pipeline are well-described with intuitive figures and algorithmic summaries.
Related work is broad and structured, with a useful comparison table positioning the contribution along axes of interpretability, action models, uncertainty-aware planning, and feedback.
Significance of contributions
The work targets an important HRI question—deciding when and what to query—to reduce operator workload without sacrificing task performance.
Evidence that state-level queries can help maintain high success while substantially lowering query rates relative to dense supervision is valuable for practical deployments.
Weaknesses
Technical limitations or concerns
The belief approximation is highly myopic: particles are limited to frontier states induced by the current committed knowledge state and the next action, effectively collapsing the global belief after each step. This can induce overconfidence, lose multi-step dependencies, and depart from standard POMDP inference.
Observation models and their calibration are domain-specific and delegated to an appendix, yet they critically determine likelihoods and thus both the confidence and query decisions; limited analysis is provided on their accuracy or robustness to sensor/model bias.
The entropy-based confidence trigger and one-step query objective do not explicitly optimize downstream task value (e.g., expected value of information), so there is no guarantee that reduced entropy translates to better returns under long-horizon dependencies, especially in the tomato domain.
Experimental gaps or methodological issues
The KnowNo baseline is action-level and may be disadvantaged by not receiving sufficient state exposure; its configuration is only briefly outlined. The large discrepancy between system-level and WoZ user results (KnowNo poor in system results but best in tomato WoZ success) suggests potential mismatch in how baselines were instantiated/contextualized in the two evaluations.
The system-level experiments use an oracle user; the approach is not stress-tested against human labeling errors or latency, which are common in practice.
The user study is a small-sample, WoZ-style evaluation with one exposure per condition and domain; results are mixed for SAGAT and workload is generally low across conditions, limiting strong conclusions about human factors.
Real-robot validation is only briefly mentioned; it is unclear whether the reported success/queries were measured on hardware under closed-loop execution with the same pipeline or merely feasibility was demonstrated.
Clarity or presentation issues
Some core implementation choices (e.g., exact observation-likelihood forms, how duplicate successor states are merged, and effects of wrong human answers) are deferred to appendices; a concise summary in the main text would improve reproducibility.
Minor typos in formulas (e.g., reusing the same symbol in the normalization step) and notation inconsistencies appear but do not impede understanding.
Missing related work or comparisons
The decision-theoretic VOI and ρ-POMDP literature (belief-dependent rewards for information gathering) is relevant for principled “what to ask” under long horizons; only entropy reduction is used here, with no comparison to VOI-based policies.
Prior interactive/active perception and human-in-the-loop POMDP works that reason explicitly about information value and query costs could be discussed as alternative query-selection criteria and planning formulations.
Detailed Comments
Technical soundness evaluation
The frontier-based belief approximation is practical but departs substantially from standard posterior maintenance. By committing to the MAP frontier state once confidence exceeds τ, the method implicitly prunes alternative global hypotheses that might be revived by later observations. This is acceptable as a heuristic but should be discussed as such, with analysis of failure cases (e.g., early mis-commitment cascading into later dead-ends).
The query objective is standard expected entropy reduction over a binary hypothesis; it is sensible and cheap to compute but remains a myopic proxy. Integrating VOI or belief-dependent reward optimization (e.g., ρ-POMDP) would better align “what to ask” with task value under uncertainty.
Restricting POMCP to symbolically applicable actions is good practice and likely responsible for efficiency gains; however, the runtime escalation in the 6-object tomato setting (∼16× per-step time) suggests more aggressive search control (e.g., progressive widening, DESPOT-style scenario sampling, caching/reuse across steps) may be needed for scale.
Experimental evaluation assessment
The τ-ablation is thorough and domain-sensitive (threshold effects differ markedly between waste and tomato). The conclusion that “more queries is not always better” is convincingly supported.
Baseline outcomes make sense qualitatively: Always-Ask is strong but costly; No-Ask fails; Random helps in short-horizon waste but not in long-horizon tomato; Ours strikes a useful balance. Yet, the KnowNo configuration likely needs more careful equalization of information provided to the user/LLM to ensure fairness, given its sensitivity to context.
The scalability experiment reveals important computational bottlenecks (episode and step times, belief-frontier growth), and it is commendable that these costs are reported. Additional profiling (tree size vs. frontier size vs. observation likelihood cost) would clarify where to optimize.
The WoZ user study is appropriately presented as preliminary. Given the mixed SAGAT and small N, the claims about user burden should be phrased cautiously (the paper is generally careful here). Evaluating resilience to occasional incorrect human answers and to response delays would strengthen external validity.
Comparison with related work (using the summaries provided)
Online POMDP solvers such as DESPOT and POMCPOW, as well as variants handling belief-dependent rewards (ρ-POMCP/ρPOMCPOW), provide theoretically grounded ways to optimize information acquisition. While this paper pursues an architectural/system integration angle, acknowledging and contrasting to belief-reward approaches (VOI, entropy gain as reward) would situate the “what to ask” decision more rigorously.
Recent finite-time analyses for POMCPOW-style planners and approaches that reuse historical planning information could offer avenues to tame the observed runtime growth.
The paper’s stance relative to active visual search and interactive POMDPs could be sharpened: here, queries are human-facing predicates; in AVS, queries are (robotic) viewpoint/control choices. Both are information gathering under uncertainty; connecting to that literature could broaden the impact.
Discussion of broader impact and significance
The paper tackles a consequential problem for practical autonomy—reducing supervision burden without sacrificing performance—via interpretable state-level queries. This interaction modality can indeed make the human’s task easier than selecting among opaque action candidates.
Risks include over-reliance on ad-hoc observation models and potential overconfidence from stepwise belief commitments; if deployed, systems should monitor for failure modes and possibly incorporate safeguards (e.g., periodic revalidation, backtracking mechanisms).
The architecture’s interpretability (exposing predicates as the locus of queries) is a strong point for transparency and post-hoc debugging.
Questions for Authors
How sensitive are the results to misspecified observation likelihoods O(o|s,a)? Could you report calibration curves or an ablation showing performance under biased or noisier observation models?
What happens when human feedback is wrong or delayed? Do you re-ask, degrade confidence, or allow backtracking if later evidence contradicts a committed knowledge state?
How large can the frontier become in practice, and how do you cap or prune it? Can you provide statistics on unique successor merging and its effect on runtime and success?
Could you compare your entropy-based query selection to a VOI-style criterion that explicitly considers downstream task value (e.g., via a short lookahead or a belief-dependent reward surrogate)?
How was the KnowNo baseline instantiated in system experiments vs. the WoZ study, and what context was provided to it? Can you reconcile the large difference in its performance across the two evaluations?
In the real-robot implementation, what quantitative results (success, queries, runtime) can you report under closed-loop operation? If the main numbers are from controlled environments, what gaps remain for on-hardware deployment?
Have you tried progressive widening or DESPOT-style scenario sampling to curb the growth in episode/step time for larger scenes? If so, how did they affect performance and query behavior?
Overall Assessment
This paper offers a well-engineered integration of symbolic knowledge, online POMDP planning, and state-level human queries that are both interpretable and informative. The architectural clarity, principled entropy-based query selection, and cross-domain evaluation are commendable, and the preliminary user study suggests the interface is usable without excessive burden. On the other hand, the belief approximation is myopic and may induce overconfidence; observation models are central yet under-analyzed; the user study is limited in scope; and fairness/details of the action-level baseline merit closer scrutiny. The work’s novelty is moderate—expected-entropy queries and POMCP are established—but the system-level synthesis and explicit comparison of state- vs. action-level queries provide useful insights. With additional analysis on observation-model robustness, more rigorous/fair baseline configuration, and some algorithmic measures to address scalability and long-horizon VOI, the contribution would be strengthened. As it stands, I view this as a solid systems paper with practical value and a clear interaction design, though not yet at the level of methodological depth typically required for T-RO/IJRR. I recommend borderline acceptance if positioned as a systems/architecture contribution with expanded empirical rigor; otherwise, a revise-and-resubmit focusing on robustness, fairness, and scalability would be appropriate.
```

## 6. 한글 번역

### Summary

이 논문은 symbolic world hypothesis에 대한 belief를 유지하고, belief entropy가 높을 때 query를 trigger하며, expected entropy reduction을 최대화하는 state-level yes/no query를 선택하는 knowledge-based POMDP framework를 제안한다. POMCP 기반 planner는 centralized symbolic knowledge base와 통합되며, feedback manager는 predicate를 user-facing question으로 변환한다. Simulated waste sorting과 tomato-harvesting setup, 그리고 추가 WoZ user study 실험은 제안 방법이 continuous supervision 대비 user query 빈도를 크게 줄이면서도 높은 task success를 유지할 수 있음을 보여준다.

### Strengths

#### Technical novelty and innovation

- 이 논문은 symbolic task model, 즉 STRIPS-style precondition/effect와 online POMDP planning을 명확하게 통합한다. 특히 fully factored belief와 global particle filtering 사이의 실용적 절충안으로, reachable successor state에 대한 particle-like frontier를 사용한다.
- What-to-ask component는 principled하다. Hypothesis에 대한 expected entropy reduction을 사용하고 symbolic action feasibility와 밀접하게 연결되어 있어, action-level query가 아니라 informative state-level query를 생성한다.
- Architecture는 sensing, planning, user feedback, shared knowledge base를 깔끔하게 통합한다. Predicate는 planning과 interaction 모두에서 사용된다.

#### Experimental rigor and validation

- Confidence threshold `tau`에 대한 ablation은 success와 query load 사이의 trade-off를 두 개의 복잡도가 다른 domain에서 체계적으로 보여준다.
- Baseline comparison은 Always-Ask, No-Ask, Random, action-query baseline인 KnowNo를 포함한다. 이를 통해 state-level query와 action-level query의 장점과 failure mode를 비교한다.
- Scalability test는 object 수를 4개에서 6개로 늘리며 symbolic state space가 커질 때 computational effect와 selective querying 특성을 보여준다.

#### Clarity of presentation

- System architecture, planning loop, belief update, query-selection pipeline이 직관적인 figure와 algorithm summary로 잘 설명되어 있다.
- Related work가 넓고 구조적으로 정리되어 있으며, comparison table은 interpretability, action model, uncertainty-aware planning, feedback 축에서 contribution을 잘 위치시킨다.

#### Significance of contributions

- 이 연구는 task performance를 희생하지 않으면서 operator workload를 줄이기 위해 언제, 무엇을 query할지 결정하는 중요한 HRI 문제를 다룬다.
- State-level query가 dense supervision 대비 query rate를 크게 낮추면서도 높은 success를 유지할 수 있다는 증거는 practical deployment 관점에서 가치가 있다.

### Weaknesses

#### Technical limitations or concerns

- Belief approximation이 상당히 myopic하다. Particle은 현재 committed knowledge state와 next action에서 유도된 frontier state로 제한되며, 사실상 각 step 이후 global belief를 collapse한다. 이는 overconfidence를 유발하고, multi-step dependency를 잃게 만들며, standard POMDP inference에서 벗어날 수 있다.
- Observation model과 calibration은 domain-specific이며 appendix로 넘어가 있다. 그러나 이들은 likelihood, confidence, query decision을 결정하는 핵심 요소다. Sensor/model bias에 대한 accuracy나 robustness 분석은 제한적이다.
- Entropy-based confidence trigger와 one-step query objective는 downstream task value, 예를 들어 expected value of information을 명시적으로 최적화하지 않는다. 따라서 entropy 감소가 long-horizon dependency, 특히 tomato domain에서 더 나은 return으로 이어진다는 보장은 없다.

#### Experimental gaps or methodological issues

- KnowNo baseline은 action-level이고 충분한 state exposure를 받지 못해 불리했을 수 있다. 설정이 간단히만 설명되어 있다. System-level result에서는 KnowNo가 낮은 성능을 보이지만 tomato WoZ success에서는 가장 높은 성능을 보이므로, 두 evaluation에서 baseline이 어떻게 instantiated/contextualized 되었는지 mismatch 가능성이 있다.
- System-level experiment는 oracle user를 사용한다. 실제로 흔한 human labeling error나 latency에 대해서는 stress-test가 없다.
- User study는 small-sample WoZ-style evaluation이며 condition과 domain별 exposure가 한 번뿐이다. SAGAT 결과는 mixed이고 workload는 전반적으로 낮아, human factors에 대해 강한 결론을 내리기 어렵다.
- Real-robot validation은 간단히만 언급된다. 보고된 success/query가 동일 pipeline의 hardware closed-loop execution에서 측정된 것인지, 아니면 feasibility demonstration인지 불분명하다.

#### Clarity or presentation issues

- 일부 핵심 implementation choice, 예를 들어 exact observation-likelihood form, duplicate successor state merging, wrong human answer의 effect가 appendix로 넘어가 있다. Main text에 간결한 요약이 있으면 reproducibility가 좋아질 것이다.
- Formula의 minor typo, 예를 들어 normalization step에서 같은 symbol을 재사용하는 문제와 notation inconsistency가 보인다. 이해를 크게 방해하지는 않는다.

#### Missing related work or comparisons

- Decision-theoretic VOI와 rho-POMDP literature, 즉 information gathering을 위한 belief-dependent reward literature가 long-horizon에서 principled what-to-ask와 관련 있다. 이 논문은 entropy reduction만 사용하며 VOI-based policy와의 비교는 없다.
- Information value와 query cost를 명시적으로 고려하는 prior interactive/active perception 및 human-in-the-loop POMDP work를 alternative query-selection criterion과 planning formulation으로 논의할 수 있다.

### Detailed Comments

#### Technical soundness evaluation

- Frontier-based belief approximation은 실용적이지만 standard posterior maintenance와는 상당히 다르다. Confidence가 `tau`를 넘으면 MAP frontier state에 commit하기 때문에, later observation에 의해 다시 살아날 수 있는 alternative global hypothesis를 암묵적으로 prune한다. 이는 heuristic으로는 받아들일 수 있지만, 그런 성격을 명확히 논의해야 한다. Early mis-commitment가 later dead-end로 cascade되는 failure case 분석이 필요하다.
- Query objective는 binary hypothesis에 대한 standard expected entropy reduction이다. 계산이 싸고 합리적이지만 여전히 myopic proxy이다. VOI나 belief-dependent reward optimization, 예를 들어 rho-POMDP를 통합하면 uncertainty 아래에서 what-to-ask를 task value와 더 잘 align할 수 있다.
- POMCP를 symbolically applicable action으로 제한하는 것은 좋은 practice이며 efficiency gain의 원인일 가능성이 크다. 그러나 6-object tomato setting에서 per-step time이 약 16배 증가하는 것은 scale을 위해 더 aggressive한 search control, 예를 들어 progressive widening, DESPOT-style scenario sampling, caching/reuse across steps가 필요할 수 있음을 시사한다.

#### Experimental evaluation assessment

- `tau` ablation은 thorough하고 domain-sensitive하다. Threshold effect가 waste와 tomato에서 크게 다르다. “More queries is not always better”라는 결론은 설득력 있게 뒷받침된다.
- Baseline outcome은 질적으로 타당하다. Always-Ask는 강하지만 costly하고, No-Ask는 실패하며, Random은 short-horizon waste에서는 도움이 되지만 long-horizon tomato에서는 그렇지 않다. Ours는 유용한 balance를 보인다. 그러나 KnowNo는 context에 민감하므로, user/LLM에게 제공된 정보량을 더 carefully equalize할 필요가 있다.
- Scalability experiment는 episode time, step time, belief-frontier growth 등 중요한 computational bottleneck을 보여준다. 이러한 cost를 보고한 점은 좋다. 추가 profiling, 예를 들어 tree size vs. frontier size vs. observation likelihood cost를 제시하면 어디를 optimize해야 하는지 더 명확해질 것이다.
- WoZ user study는 preliminary로 적절하게 제시되어 있다. Mixed SAGAT와 small N을 고려하면 user burden에 대한 claim은 조심스럽게 표현해야 한다. 논문은 대체로 조심스럽지만, occasional incorrect human answer와 response delay에 대한 resilience를 평가하면 external validity가 강화될 것이다.

#### Comparison with related work

- DESPOT, POMCPOW 같은 online POMDP solver와 belief-dependent reward를 다루는 rho-POMCP/rhoPOMCPOW variant는 information acquisition을 이론적으로 grounded하게 optimize하는 방법을 제공한다. 이 논문은 architecture/system integration angle을 추구하지만, belief-reward approach, 즉 VOI나 entropy gain as reward를 acknowledge하고 contrast하면 what-to-ask decision을 더 엄밀하게 위치시킬 수 있다.
- POMCPOW-style planner의 finite-time analysis와 historical planning information reuse 접근은 observed runtime growth를 줄이는 방향을 제시할 수 있다.
- Active visual search와 interactive POMDP에 대한 논문의 stance를 더 선명하게 할 수 있다. 여기서는 query가 human-facing predicate이고, AVS에서는 query가 robot viewpoint/control choice이다. 둘 다 uncertainty 아래에서 information gathering이라는 점에서 연결하면 impact가 넓어질 수 있다.

#### Discussion of broader impact and significance

- 이 논문은 practical autonomy에서 중요한 문제, 즉 performance를 희생하지 않으면서 supervision burden을 줄이는 문제를 interpretable state-level query로 다룬다. 이러한 interaction modality는 opaque action candidate 중에서 선택하게 하는 것보다 human task를 쉽게 만들 수 있다.
- Risk는 ad-hoc observation model에 대한 의존과 stepwise belief commitment에서 오는 overconfidence이다. Deployment 시에는 failure mode를 monitor하고 periodic revalidation이나 backtracking mechanism 같은 safeguard를 포함해야 한다.
- Predicate를 query의 locus로 노출하는 architecture의 interpretability는 transparency와 post-hoc debugging 측면에서 강점이다.

### Questions for Authors

1. 결과가 misspecified observation likelihood `O(o|s,a)`에 얼마나 민감한가? Calibration curve나 biased/noisier observation model 아래에서의 performance ablation을 보고할 수 있는가?
2. Human feedback이 틀리거나 지연되면 어떻게 되는가? Re-ask를 하는가, confidence를 낮추는가, 아니면 later evidence가 committed knowledge state와 모순될 때 backtracking을 허용하는가?
3. Frontier는 실제로 얼마나 커질 수 있으며, 어떻게 cap 또는 prune하는가? Unique successor merging과 이것이 runtime 및 success에 미치는 영향에 대한 statistics를 제공할 수 있는가?
4. Entropy-based query selection을 downstream task value를 명시적으로 고려하는 VOI-style criterion, 예를 들어 short lookahead나 belief-dependent reward surrogate와 비교할 수 있는가?
5. System experiment와 WoZ study에서 KnowNo baseline은 어떻게 instantiated 되었고, 어떤 context가 제공되었는가? 두 evaluation 사이의 큰 성능 차이를 설명할 수 있는가?
6. Real-robot implementation에서 closed-loop operation 아래의 quantitative result, 즉 success, queries, runtime을 보고할 수 있는가? Main number가 controlled environment 기반이라면, on-hardware deployment를 위해 어떤 gap이 남아 있는가?
7. Larger scene에서 episode/step time 증가를 억제하기 위해 progressive widening이나 DESPOT-style scenario sampling을 시도했는가? 시도했다면 performance와 query behavior에 어떤 영향을 주었는가?

### Overall Assessment

이 논문은 symbolic knowledge, online POMDP planning, state-level human query를 잘 engineering된 방식으로 통합한다. Query는 interpretable하고 informative하다. Architecture의 명확성, principled entropy-based query selection, cross-domain evaluation은 장점이며, preliminary user study는 interface가 excessive burden 없이 usable할 수 있음을 시사한다. 반면 belief approximation은 myopic하고 overconfidence를 유발할 수 있으며, observation model은 핵심적이지만 분석이 부족하다. User study는 scope가 제한적이고, action-level baseline의 fairness/detail은 더 면밀히 검토할 필요가 있다.

Novelty는 moderate하다. Expected-entropy query와 POMCP는 established되어 있지만, system-level synthesis와 state-level query vs. action-level query의 explicit comparison은 유용한 insight를 제공한다. Observation-model robustness, 더 rigorous/fair한 baseline configuration, scalability와 long-horizon VOI를 다루는 algorithmic measure가 추가되면 contribution은 강화될 것이다. 현재 상태에서는 practical value와 clear interaction design을 가진 solid systems paper로 보지만, T-RO/IJRR에서 일반적으로 요구하는 methodological depth 수준에는 아직 완전히 도달하지 못했다고 본다. Systems/architecture contribution으로 positioning하고 empirical rigor를 확장한다면 borderline acceptance를 추천한다. 그렇지 않다면 robustness, fairness, scalability에 초점을 둔 revise-and-resubmit이 적절하다.
