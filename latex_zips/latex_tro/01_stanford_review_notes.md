# T-RO Review Notes

Created: 2026-06-23

Source: `01_stanford_review.pdf`

## 1. 핵심 판단

두 번째 Stanford Agentic Reviewer의 평가는 첫 번째 리뷰와 거의 같은 방향이다. 전반적으로는 "coherent and well-implemented systems contribution with moderate novelty and clear practical value"로 본다. 즉, 논문의 시스템 통합, state-level query, threshold ablation, scalability reporting은 긍정적으로 평가하지만, T-RO급 방법론 논문으로 밀기에는 robustness, VOI/value-aware query, baseline fairness, model misspecification 분석이 아직 약하다고 본다.

이 리뷰에서 반복적으로 등장하는 핵심 리스크는 다음과 같다.

1. Belief가 persistent global posterior가 아니라 per-action frontier에 정의되어 있어 myopic하고, frontier 밖의 correct world를 잃을 수 있다는 점.
2. Confidence threshold를 넘으면 single knowledge state에 commit하지만, backtracking이나 belief recovery가 없다는 점.
3. Query selection이 expected entropy만 최적화하고, task reward 기준 VOI를 직접 고려하지 않는다는 점.
4. Human error, delayed feedback, observation/transition probability misspecification에 대한 robustness 실험이 없다는 점.
5. KnowNo baseline은 action-level query라 의미적으로 proposed state-level feedback과 다르며, system-level과 WoZ에서 성능 차이가 커서 setup mismatch/fairness 질문이 나올 수 있다는 점.
6. Query cost가 reward에 명시적으로 들어가지 않고 threshold `tau`에 의해 간접적으로만 조절된다는 점.

첫 번째 리뷰와 비교하면, 이번 리뷰는 "real-robot quantitative closed-loop result"보다 "query cost modeling", "contingent/conditional symbolic planning", "VOI/VOC/VPI query selection" 쪽을 더 강조한다.

## 2. 수정 우선순위

### Priority A: 이미 상당 부분 반영했거나 본문/appendix에서 반드시 명확히 할 부분

- Frontier belief가 full global belief가 아니라 reachable successor support에 대한 local approximation임을 명확히 유지한다.
- Commit 이후 backtracking이 없다는 점은 Method가 아니라 Discussion/Limitation에서 한계로 정리한다.
- Observation/transition probability가 confidence와 query frequency에 영향을 준다는 점을 limitation에 넣는다.
- KnowNo baseline의 prompt, prediction set, `\hat{q}`, WoZ setup 차이를 appendix와 user-study discussion에서 명확히 설명한다.
- User study는 small-N WoZ exploratory study이며, main quantitative claim은 system-level evaluation임을 유지한다.

### Priority B: 추가 실험 없이 논문 수정으로 방어할 부분

- `tau`는 query cost를 reward에 직접 넣은 것이 아니라 confidence threshold로 interaction을 조절하는 heuristic임을 정직하게 설명한다.
- Entropy query는 task reward를 최적화하는 VOI가 아니라 predicate-level disambiguation criterion임을 분명히 한다.
- State-level query vs action-level query의 의미적 차이를 "not apples-to-apples, but intentional comparison of interaction modes"로 정리한다.
- Contingent planning / knowledge-level planning / sensing actions 관련 related work를 1문단 정도 보강한다.
- Observation model portability는 domain model design cost로 limitation에 넣는다.

### Priority C: 가능하면 추가 분석/실험

- Wrong human response / delayed feedback robustness.
- Observation/transition parameter perturbation sensitivity.
- Query cost를 reward에 넣은 variant 또는 VOI-style query baseline.
- Frontier size, POMCP simulation, SP likelihood, UI latency 별 runtime profiling.
- Larger scenes에서 pruning/progressive widening/learned surrogate에 대한 future direction.

## 3. 예상 Response 방향

### Belief frontier and commitment

응답 방향:

- We agree that the method does not maintain a full persistent global posterior.
- The frontier belief is a deliberate online approximation over reachable symbolic successors.
- The corollary requires complete frontier coverage and correct feedback; we clarified these assumptions.
- We added limitation/future work on reversible commitments, low-weight alternative hypotheses, and revalidation/backtracking.

리뷰어가 원하는 건 "우리가 exact inference라고 주장하지 않는다"는 명확화다. Method에서는 정의로 설명하고, Discussion에서는 failure mode로 받아들이는 게 적절하다.

### Entropy vs VOI

응답 방향:

- Expected entropy is used because the query is predicate-level state disambiguation, not direct task-action selection.
- POMCP handles reward-based long-horizon planning; the query module resolves ambiguity in the current symbolic frontier.
- We acknowledge that reward-weighted VOI could prioritize utility-critical uncertainty better.
- We add VOI/VPI/VOC and rho-POMDP-style belief reward as future extensions or related work.

너의 기존 입장처럼, "planning 성능 자체를 올리는 것이 scope이 아니다"는 점은 유지해도 된다. 다만 reviewer에게는 "VOI가 다른 문제라는 점"과 "그 한계를 알고 있다는 점"을 동시에 보여줘야 한다.

### Human error and model misspecification

응답 방향:

- Current system-level evaluation uses oracle feedback to isolate the planning/query mechanism.
- We added discussion that noisy feedback can lead to wrong filtering and wrong commitment.
- Future work includes confidence degradation, re-asking, contradiction detection, and maintaining alternative hypotheses.
- Observation and transition probabilities are domain-model parameters; sensitivity analysis is not included in the current version but is important for deployment.

추가 실험이 없다면, 너무 세게 "robust하다"라고 말하지 말고, "not evaluated; limitation"으로 정리하는 게 맞다.

### KnowNo baseline

응답 방향:

- The KnowNo baseline intentionally represents an action-level query interface, while ours represents state-level verification.
- Therefore the comparison is not meant to isolate only algorithmic planner quality; it compares two human-query modalities.
- Appendix now specifies prompt structure, scoring, prediction set, and `\hat{q}`.
- WoZ KnowNo differs from system-level KnowNo because the WoZ condition tests whether humans can choose the correct action from controlled options, not whether the autonomous KnowNo policy selects actions correctly.

이 부분은 이미 appendix/user discussion에 꽤 반영되어 있다.

### Query cost

응답 방향:

- Query cost is controlled by threshold `\tau`, not directly encoded as a POMDP reward term.
- This was a design choice to keep the query trigger interpretable and domain-independent.
- The threshold ablation empirically shows the trade-off between success and interaction cost.
- Reward-augmented query cost or VOI-style utility is future work.

## 4. Reviewer Questions To Track

- How robust is the framework to wrong human responses?
- How robust is it to misspecified observation/transition probabilities?
- Does the system support backtracking after commitment?
- Why expected entropy instead of reward-weighted VOI?
- How is the frontier constructed for complex multi-object operators?
- Are there pruning heuristics or learned surrogates for large frontier sizes?
- What exact information was exposed to KnowNo in system experiments and WoZ?
- What dominates real-robot decision-step latency?
- How portable is the observation model specification across domains?
- Can practitioners estimate `O(o|s,a)` from data rather than manually tuning it?

## 5. Original Review

```text
Summary
This paper presents a knowledge-based decision-making framework that enables a robot to decide when to ask for help and what to ask by maintaining a belief over symbolic world hypotheses and querying humans at the state (predicate) level when ambiguity remains. The approach formulates execution as a POMDP coupled with a STRIPS-style model, uses a POMCP planner constrained to applicable symbolic actions, and selects queries to minimize expected entropy over a particle belief defined on the reachable “frontier” of successor states. The authors integrate sensing, planning, feedback management, and a user interface, and evaluate the system in waste sorting and tomato harvesting via system experiments and a small Wizard-of-Oz user study, showing improved task success versus no/naïve querying and reduced interactions versus dense supervision.

Strengths
Technical novelty and innovation
• Integrates symbolic STRIPS models with POMDP-style online planning (POMCP) and state-level human queries, tying “when” and “what” to ask directly to the planning belief.
• Introduces a frontier-based symbolic belief approximation that focuses inference on reachable successor worlds, reducing combinatorial blow-up typical of global symbolic beliefs.
• Proposes an expected-entropy query selection at the predicate level, producing interpretable, user-verifiable questions grounded in the knowledge base.
• Provides a clean execution loop that commits high-confidence hypotheses to the KB and requests targeted feedback otherwise, closing the loop between belief, planning, and interaction.

Experimental rigor and validation
• Systematic threshold ablation explores the trade-off between success and interaction cost across domains, offering practical guidance for setting the confidence threshold.
• Baseline comparisons include no queries, random queries, dense (“All”) supervision, and an action-level KnowNo-style baseline; results are presented with success, queries, query rate, and planning length.
• Scalability analysis increases object count and reports not only performance but also computational indicators (frontier size, nodes expanded, episode time).
• A Wizard-of-Oz user study (n=12) measures task success, operation time, workload (NASA TLX), and situation awareness (SAGAT), complementing system-level results.

Clarity of presentation
• The system architecture and data flow are clearly explained, with consistent notation and intuitive figures linking KB, planner, sensing, and feedback.
• Preliminaries concisely recap STRIPS, POMDPs, and POMCP and explain how symbolic constraints restrict search/action feasibility.
• The uncertainty/confidence formulation and expected-entropy-based query selection are well-motivated and described with precise equations.

Significance of contributions
• Addresses a central HRI problem—when and what to ask—at execution time, demonstrating fewer queries than dense supervision while maintaining high success in two distinct domains.
• Presents a practical, interpretable alternative to action-level queries that shifts the cognitive burden from action selection to fact verification, likely improving operator workload and calibration.

Weaknesses
Technical limitations or concerns
• The belief is defined on a per-action “frontier” rather than a persistent global belief over all latent symbolic facts, which may be myopic and can fail if the correct world falls outside the enumerated frontier; the corollary assumes complete frontier coverage and perfect feedback.
• The framework commits to a single knowledge state once confidence exceeds a threshold but lacks mechanisms for backtracking or belief recovery if observations/models are misspecified or human feedback is noisy.
• Query selection optimizes expected entropy (information gain) rather than expected value-of-information relative to task reward; this can favor information that is not utility-critical.
• Human error is not modeled; no analysis of robustness to noisy or delayed feedback, despite known variability in operator inputs.

Experimental gaps or methodological issues
• The KnowNo action-query baseline differs semantically from the proposed state-level feedback; the comparison is informative but not apples-to-apples, and implementation details (e.g., prompting, access to internal state) may affect fairness.
• The user study is small (n=12), Wizard-of-Oz, and exploratory (no multiple-comparison correction), limiting strength of user-facing claims; also, tomato KnowNo achieved 100% success in WoZ versus low system-level success, suggesting condition or setup mismatch.
• No explicit modeling of query costs in the reward; reliance on a confidence threshold tau may conflate interaction cost with belief sharpness.
• Sensitivity to observation/transition probability misspecification is acknowledged but not experimentally studied; these parameters strongly affect confidence and query frequency.

Clarity or presentation issues
• Minor inconsistencies in figure captions/axis labels (e.g., plots labeled “Query Probability per Step” under different panels) may confuse readers; some formatting artifacts (from PDF extraction) persist.
• Some implementation specifics for observation likelihoods are deferred to appendices, which are crucial to reproduce the results.

Missing related work or comparisons
• Limited discussion of contingent/conditional planning and sensing within symbolic planning (e.g., conditional/knowledge-level planning in PDDL, contingent planners, mixed observability planning frameworks) that also address uncertainty and information-gathering actions.
• Prior work on value-of-information and VOC/VPI-based query selection in planning under uncertainty is not compared, despite conceptual proximity to the proposed entropy-based strategy.

Detailed Comments
Technical soundness evaluation
• The integration of STRIPS constraints into POMCP by restricting action applicability is sound and improves search efficiency.
• The frontier-based belief approximation is a practical design that exploits symbolic structure; however, it assumes the frontier covers all materially distinct next-world hypotheses. Violations (e.g., under-modeled outcomes) can lead to erroneous commitments without recovery.
• The confidence metric (entropy normalized by log2 of frontier size) is reasonable for thresholding but may not correlate with downstream value; a VOI-based trigger could better reflect utility.
• The query selection criterion (expected entropy after querying) is standard and appropriate but myopic; tasks with long dependencies (e.g., tomato harvesting) may benefit from non-myopic or reward-weighted VOI.

Experimental evaluation assessment
• The threshold ablation is thorough and illuminates domain differences, showing that over-querying yields diminishing returns and that long-horizon tasks require higher confidence.
• Baseline comparison convincingly demonstrates that naïve or random querying underperforms targeted queries, and that dense supervision inflates interaction cost; however, including a VOI-based state-query baseline or a contingent planner with sensing actions would strengthen the case.
• Scalability tests are valuable and candid: they report increased runtime/frontier sizes and reduced success in more complex tomato scenes, clarifying computational limitations.
• The WoZ user study provides early evidence of usability and workload advantages, but results should be interpreted cautiously due to sample size, scenario scripts, and the lack of end-to-end autonomy during trials.

Comparison with related work
• Relative to KnowNo-style methods that ask action-level questions, this work’s state-level queries reduce the operator’s SA burden and empirically improve success in system tests, aligning with the claim that fact verification is easier than action selection under uncertainty.
• Compared to prior symbolic uncertainty planning and ASP/POMDP hybrids, the proposed frontier belief and online POMCP integration offer a pragmatic alternative that emphasizes interpretable facts and human-in-the-loop disambiguation rather than heavy offline model construction.
• LLM-based planning methods provide flexible task interpretations but struggle with executability and uncertainty; here, symbolic transition models ensure feasibility, and uncertainty is handled explicitly with a belief and observation model.
• The paper could better position itself with respect to contingent planning and VOI-based active perception, which aim to select informative queries/sensing actions grounded in task reward.

Discussion of broader impact and significance
• The framework promotes transparent robot behavior by exposing uncertainty as symbolic predicates, facilitating reliable human correction and potentially improving trust/calibration.
• In real deployments, modeling human error and query cost—plus adding backtracking—will be essential to prevent lock-in to incorrect knowledge states and to bound cognitive load.
• The architecture is general and could transfer beyond the two domains provided appropriate predicates, action models, and observation likelihoods are defined; however, engineering effort for model specification and calibration may be substantial.

Questions for Authors
1. How robust is the framework to erroneous human responses or misspecified observation/transition probabilities? Can you report results with injected label noise or perturbed model parameters to quantify degradation and recovery?
2. Does the system support backtracking or belief revision after committing to a knowledge state if later evidence contradicts the commitment? If not, how might you extend the approach to allow reversible commitments or maintain a low-weight alternative hypothesis set?
3. Why was expected entropy chosen over a reward-weighted value-of-information criterion? Have you evaluated a VOI trigger and/or query selection that accounts for task utility rather than pure information gain?
4. How is the frontier constructed for complex, multi-object operators where combinatorial effects (e.g., occlusions, mispicks) proliferate? Are there pruning heuristics or learned surrogates to keep frontier sizes tractable?
5. Could you clarify the KnowNo baseline setup: what information (if any) about the current belief/state was exposed to the user, and how were prompts/actions standardized across trials to ensure fairness? How do you reconcile the large discrepancy between system-level and WoZ KnowNo performance?
6. What is the runtime budget per decision step on the real robot, and which components dominate latency (POMCP simulations, SP likelihoods, UI/feedback)? Any avenues to accelerate the planner for larger scenes?
7. How portable is the observation model specification across domains? Can you provide guidance or tools to help practitioners elicit/estimate O(o|s,a) from data to reduce manual tuning?

Overall Assessment
This paper tackles a timely and important HRI problem—deciding when and what to ask under uncertainty—by tightly coupling symbolic planning with a belief-driven, state-level querying mechanism. The architecture is coherent and well-implemented, and the empirical results, especially the threshold ablations and scalability study, provide meaningful evidence that targeted state-level questions can preserve high success with substantially fewer interactions than dense supervision. The main limitations lie in the myopic frontier belief, the lack of robustness analyses (human error, model misspecification), and the absence of explicit VOI-style baselines or contingent planning comparisons. The user study is informative but small and exploratory. Overall, the work is a solid systems contribution with moderate novelty and clear practical value; addressing robustness and adding reward-aware query baselines would strengthen it further. I recommend publication at Could you recommend journals that would be a good fit for robotics-related research? after minor-to-moderate revisions focused on robustness experiments, baseline breadth, and clarifications of the baseline setups and limitations.
```

## 6. 한글 번역

### Summary

이 논문은 symbolic world hypothesis에 대한 belief를 유지하고, ambiguity가 남아 있을 때 human에게 state-level, 즉 predicate-level query를 던짐으로써 robot이 언제 도움을 요청하고 무엇을 물어볼지 결정하는 knowledge-based decision-making framework를 제시한다. 이 접근법은 execution을 STRIPS-style model과 결합된 POMDP로 공식화하고, applicable symbolic action으로 제한된 POMCP planner를 사용하며, reachable successor state의 frontier에 정의된 particle belief 위에서 expected entropy를 최소화하는 query를 선택한다. 저자들은 sensing, planning, feedback management, user interface를 통합하고, waste sorting과 tomato harvesting에서 system experiment 및 small Wizard-of-Oz user study를 통해 no/naive querying 대비 task success가 개선되고 dense supervision 대비 interaction이 줄어듦을 보인다.

### Strengths

#### Technical novelty and innovation

- Symbolic STRIPS model, POMDP-style online planning인 POMCP, state-level human query를 통합하여 when-to-ask와 what-to-ask를 planning belief에 직접 연결한다.
- Reachable successor world에 inference를 집중하는 frontier-based symbolic belief approximation을 도입하여 global symbolic belief의 combinatorial blow-up을 줄인다.
- Predicate-level expected-entropy query selection을 제안하여 knowledge base에 grounded된 interpretable하고 user-verifiable한 질문을 생성한다.
- High-confidence hypothesis는 KB에 commit하고, 그렇지 않으면 targeted feedback을 요청하는 깔끔한 execution loop를 제공한다.

#### Experimental rigor and validation

- Systematic threshold ablation은 domain별 success와 interaction cost 사이의 trade-off를 탐색하며 confidence threshold 설정에 실용적 근거를 제공한다.
- Baseline comparison은 no query, random query, dense supervision인 All, action-level KnowNo-style baseline을 포함하며, success, queries, query rate, planning length를 보고한다.
- Scalability analysis는 object count를 증가시키고 performance뿐 아니라 frontier size, expanded nodes, episode time 같은 computational indicator도 보고한다.
- Wizard-of-Oz user study는 n=12로 task success, operation time, workload, SAGAT을 측정하여 system-level result를 보완한다.

#### Clarity of presentation

- System architecture와 data flow가 명확히 설명되어 있으며, KB, planner, sensing, feedback을 연결하는 notation과 figure가 직관적이다.
- Preliminaries는 STRIPS, POMDP, POMCP를 간결히 요약하고 symbolic constraint가 search/action feasibility를 어떻게 제한하는지 설명한다.
- Uncertainty/confidence formulation과 expected-entropy query selection은 precise equation과 함께 잘 동기화되어 있다.

#### Significance of contributions

- Execution time에 언제 무엇을 물어볼지 결정하는 central HRI problem을 다루며, 두 domain에서 dense supervision보다 적은 query로 높은 success를 유지할 수 있음을 보여준다.
- Action-level query 대신 fact verification으로 cognitive burden을 옮기는 practical하고 interpretable한 대안을 제시한다.

### Weaknesses

#### Technical limitations or concerns

- Belief가 all latent symbolic facts에 대한 persistent global belief가 아니라 per-action frontier에 정의되어 있다. 이는 myopic할 수 있고 correct world가 enumerated frontier 밖에 있으면 실패할 수 있다. Corollary도 complete frontier coverage와 perfect feedback을 가정한다.
- Confidence threshold를 넘으면 single knowledge state에 commit하지만, observation/model이 misspecified되거나 human feedback이 noisy할 때 backtracking 또는 belief recovery mechanism이 없다.
- Query selection은 task reward에 대한 expected value-of-information이 아니라 expected entropy, 즉 information gain을 최적화한다. 따라서 utility-critical하지 않은 정보를 선호할 수 있다.
- Human error가 모델링되어 있지 않으며, noisy 또는 delayed feedback에 대한 robustness 분석이 없다.

#### Experimental gaps or methodological issues

- KnowNo action-query baseline은 proposed state-level feedback과 semantic이 다르다. 비교는 informative하지만 apples-to-apples는 아니며, prompting이나 internal state 접근 같은 implementation detail이 fairness에 영향을 줄 수 있다.
- User study는 n=12, Wizard-of-Oz, exploratory이고 multiple-comparison correction이 없으므로 user-facing claim의 강도는 제한적이다. 또한 tomato KnowNo가 WoZ에서는 100% success를 보인 반면 system-level에서는 낮은 success를 보였으므로 condition/setup mismatch 가능성이 있다.
- Reward에 query cost가 명시적으로 모델링되어 있지 않다. Confidence threshold `tau`에 의존하는 방식은 interaction cost와 belief sharpness를 혼동할 수 있다.
- Observation/transition probability misspecification에 대한 sensitivity는 acknowledged되어 있지만 실험적으로 분석되지 않았다. 이 parameter들은 confidence와 query frequency에 강하게 영향을 준다.

#### Clarity or presentation issues

- Figure caption이나 axis label의 minor inconsistency, 예를 들어 여러 panel에서 "Query Probability per Step" label이 혼동될 수 있으며, PDF extraction artifact 같은 formatting artifact가 남아 있다.
- Observation likelihood의 일부 implementation detail이 appendix로 넘어가 있는데, 이는 reproducibility에 중요하다.

#### Missing related work or comparisons

- Symbolic planning에서 uncertainty와 information-gathering action을 다루는 contingent/conditional planning, knowledge-level planning in PDDL, contingent planners, mixed observability planning framework 논의가 제한적이다.
- Planning under uncertainty에서 VOI, VOC, VPI 기반 query selection은 proposed entropy-based strategy와 개념적으로 가깝지만 비교되지 않았다.

### Detailed Comments

#### Technical soundness evaluation

- STRIPS constraint를 POMCP에 통합하여 applicable action만 고려하는 것은 sound하며 search efficiency를 개선한다.
- Frontier-based belief approximation은 symbolic structure를 활용하는 practical design이다. 그러나 frontier가 materially distinct한 next-world hypothesis를 모두 포함한다고 가정한다. Under-modeled outcome 같은 violation은 recovery 없는 erroneous commitment로 이어질 수 있다.
- Frontier size의 `log2`로 정규화한 entropy-based confidence metric은 thresholding에는 합리적이지만 downstream value와 상관되지 않을 수 있다. VOI-based trigger가 utility를 더 잘 반영할 수 있다.
- Expected entropy after querying criterion은 standard하고 적절하지만 myopic하다. Tomato harvesting처럼 long dependency가 있는 task는 non-myopic 또는 reward-weighted VOI가 도움이 될 수 있다.

#### Experimental evaluation assessment

- Threshold ablation은 thorough하며 domain difference를 잘 보여준다. Over-querying은 diminishing return을 만들고, long-horizon task는 더 높은 confidence를 요구한다는 점을 보여준다.
- Baseline comparison은 naive/random querying이 targeted query보다 떨어지고 dense supervision이 interaction cost를 늘린다는 점을 설득력 있게 보여준다. 다만 VOI-based state-query baseline이나 sensing action을 포함한 contingent planner를 넣으면 논지가 강화될 것이다.
- Scalability test는 valuable하고 candid하다. 더 복잡한 tomato scene에서 runtime/frontier size가 증가하고 success가 감소함을 보고하여 computational limitation을 명확히 한다.
- WoZ user study는 usability와 workload advantage에 대한 early evidence를 제공하지만, sample size, scenario script, end-to-end autonomy 부재 때문에 조심스럽게 해석해야 한다.

#### Comparison with related work

- KnowNo-style action-level question과 비교할 때, 이 논문의 state-level query는 operator의 situation-awareness burden을 줄이고 system test에서 success를 개선한다. 이는 uncertainty 아래에서 fact verification이 action selection보다 쉽다는 주장과 일치한다.
- Prior symbolic uncertainty planning 및 ASP/POMDP hybrid와 비교하면, proposed frontier belief와 online POMCP integration은 heavy offline model construction보다 interpretable fact와 human-in-the-loop disambiguation을 강조하는 pragmatic alternative이다.
- LLM-based planning은 flexible task interpretation을 제공하지만 executability와 uncertainty에 약하다. 이 논문은 symbolic transition model로 feasibility를 보장하고, belief와 observation model로 uncertainty를 명시적으로 다룬다.
- Contingent planning과 VOI-based active perception에 대한 positioning을 더 명확히 할 수 있다. 이들은 task reward에 grounded된 informative query/sensing action을 선택하는 접근이다.

#### Discussion of broader impact and significance

- 이 framework는 uncertainty를 symbolic predicate로 노출하여 transparent robot behavior를 촉진하고, reliable human correction과 trust/calibration 향상에 기여할 수 있다.
- Real deployment에서는 human error와 query cost를 모델링하고 backtracking을 추가하는 것이 중요하다. 그렇지 않으면 incorrect knowledge state에 lock-in되거나 cognitive load를 제한하기 어렵다.
- Architecture는 적절한 predicate, action model, observation likelihood가 정의되면 두 domain을 넘어 transfer될 수 있다. 그러나 model specification과 calibration을 위한 engineering effort는 상당할 수 있다.

### Questions for Authors

1. Erroneous human response나 misspecified observation/transition probability에 대해 framework는 얼마나 robust한가? Injected label noise나 perturbed model parameter를 사용한 degradation/recovery 결과를 보고할 수 있는가?
2. Knowledge state에 commit한 뒤 later evidence가 contradiction을 보이면 backtracking이나 belief revision을 지원하는가? 그렇지 않다면 reversible commitment나 low-weight alternative hypothesis set을 유지하는 방식으로 어떻게 확장할 수 있는가?
3. Reward-weighted VOI criterion 대신 expected entropy를 선택한 이유는 무엇인가? Pure information gain이 아니라 task utility를 고려하는 VOI trigger 또는 query selection을 평가했는가?
4. Occlusion, mispick처럼 combinatorial effect가 늘어나는 complex multi-object operator에서 frontier는 어떻게 구성되는가? Frontier size를 tractable하게 유지하기 위한 pruning heuristic이나 learned surrogate가 있는가?
5. KnowNo baseline setup을 명확히 설명할 수 있는가? Current belief/state에 대한 어떤 정보가 user에게 노출되었고, fairness를 보장하기 위해 prompt/action은 trial 간 어떻게 standardized 되었는가? System-level과 WoZ KnowNo performance의 큰 차이를 어떻게 설명하는가?
6. Real robot에서 decision step당 runtime budget은 얼마이고, 어떤 component가 latency를 지배하는가? POMCP simulation, SP likelihood, UI/feedback 중 무엇이 병목인가? Larger scene을 위해 planner를 가속할 방안은 무엇인가?
7. Observation model specification은 domain 간 얼마나 portable한가? Practitioner가 `O(o|s,a)`를 data에서 추정하거나 elicitation할 수 있도록 guidance나 tool을 제공할 수 있는가?

### Overall Assessment

이 논문은 uncertainty 아래에서 언제 무엇을 물어볼지 결정하는 timely하고 중요한 HRI 문제를 symbolic planning과 belief-driven state-level query mechanism을 긴밀하게 결합하여 다룬다. Architecture는 coherent하고 잘 구현되어 있으며, 특히 threshold ablation과 scalability study는 targeted state-level question이 dense supervision보다 훨씬 적은 interaction으로 높은 success를 보존할 수 있다는 의미 있는 증거를 제공한다. 주요 한계는 myopic frontier belief, human error와 model misspecification에 대한 robustness analysis 부재, explicit VOI-style baseline 또는 contingent planning comparison 부재이다. User study는 informative하지만 small and exploratory이다.

전반적으로 이 작업은 moderate novelty와 clear practical value를 가진 solid systems contribution이다. Robustness를 다루고 reward-aware query baseline을 추가하면 더 강해질 것이다. Reviewer는 robustness experiment, baseline breadth, baseline setup 및 limitation clarification에 초점을 둔 minor-to-moderate revision 이후 publication을 추천한다고 평가한다.
