## Summary

The paper proposes a planning-and-querying framework for robots operating under partial observability that decides both when to seek external feedback (e.g., from a VLM or human) and what to ask. It maintains a weighted belief over symbolic successor states (“frontier”) after each action, triggers queries when normalized belief entropy indicates low confidence, and selects the predicate to verify that maximizes expected entropy reduction. Experiments in two task domains on both simulated and physical platforms show substantial gains in task success over conformal-prediction-based and reward-based baselines, with ablations demonstrating that timing (when to query) primarily impacts success while selection (what to query) reduces query counts.

## Strengths

### Technical novelty and innovation

- Uses an explicit, structured symbolic frontier to construct a compact belief over reachable worlds and tie both query timing (normalized entropy) and selection (expected entropy reduction) directly to that belief.
- Integrates belief-based querying with POMCP for action selection while keeping queries outside the action space, avoiding search explosion from adding query actions to the planner.
- Clear decomposition of “when” vs. “what,” with selection grounded in information-theoretic criteria rather than proxy heuristics.

### Experimental rigor and validation

- Evaluated across two distinct domains (WasteSorting, TomatoHarvesting) with differing uncertainty sources, in both simulated (oracle) and real settings (VLM- and human-in-the-loop).
- Includes comparisons to multiple baselines (conformal-prediction style and reward-based query selection) and a thoughtful ablation that isolates the effects of timing vs. selection and a threshold sweep.
- Provides analysis of failure cases for VLM and human trials and qualitative interviews that support the user-centered advantages of state-verification questions.

### Clarity of presentation

- Method is well structured with equations for belief construction, confidence, and entropy-based selection; an algorithm box concisely shows the full loop.
- System diagram and task descriptions make the execution pipeline understandable and reproducible at a high level.

### Significance of contributions

- Addresses a central challenge in deploying robots with unreliable perception: deciding proactively which uncertainties to resolve before taking potentially costly actions.
- The empirical findings—that timing drives success while selection curbs query cost—are actionable design principles for practitioners.

## Weaknesses

### Technical limitations or concerns

- The “frontier” belief only enumerates successors of the committed MAP knowledge state s_t^K under the last action, which risks ignoring lingering uncertainties in parts of the state unaffected by that action; the commit-to-MAP design may prevent recovery from earlier misbeliefs that do not appear in the immediate frontier.
- External feedback is treated as noiseless for belief updates in simulation and as hard truth in physical trials; there is no model of feedback reliability, nor a mechanism to hedge against erroneous answers (e.g., soft weighting, repeated queries, or Bayesian fusion).
- Triggering is purely entropy-thresholded, without explicit cost of querying or value-of-information trade-offs; this can lead to over-querying at higher thresholds.

### Experimental gaps or methodological issues

- POMCP-Q seems at a computational disadvantage (substantially larger planning times) and includes queries in the action space; it is unclear whether compute budgets or search parameters were matched fairly across methods.
- While the paper compares to conformal-prediction–triggered approaches, it omits direct comparisons to belief-entropy or information-gain planners in ρ-POMDP form (e.g., ρ-POMCP with −H(b′) or ΔH rewards) and to task-relevant-uncertainty methods (e.g., STRUG), which are closer in spirit to the proposed belief-uncertainty criteria.
- VLM/human trials have relatively small sample sizes per condition and domain; statistical significance tests are not reported.

### Clarity or presentation issues

- The observation and transition models used to weight the frontier (Eq. 2) are not specified in enough detail to assess their fidelity or how they are learned/estimated from SP modules.
- It remains ambiguous how the frontier is enumerated in practice for multi-fact effects and how invalid symbolic combinations are ruled out at scale.

### Missing related work or comparisons

- Recent belief-aware symbolic planning with explicit VLM grounding (e.g., VAP‑TAMP’s predicate verification loop, BKLV A’s K-fluents belief with observation actions) and conformant probabilistic planning with belief over fluents (RoVLaP) should be discussed in more depth; these works already tie predicate-level uncertainty to planning/execution.
- Information-theoretic query selection is well established in ρ-POMDP and active sensing literature (e.g., ρ‑POMCP, EFE-as-ρ-POMDP); tighter positioning relative to these would clarify originality.

## Detailed Comments

### Technical soundness evaluation

The core mechanism—constructing a weighted frontier belief and triggering queries based on normalized entropy—is conceptually sound and aligns with POMDP principles. The expected-entropy-reduction criterion for selecting predicates is a principled instantiation of myopic information gain at the symbolic level.

Two design choices merit scrutiny: (i) belief over only the immediate frontier of the committed single world s_t^K may be too myopic; unmodeled uncertainties that persist across steps might not be represented, limiting corrective queries; (ii) hard filtering of belief on external answers with no noise model risks overconfidence and irreversible errors when the source is wrong (as seen in the failure analysis).

The separation of concerns—keeping queries out of the POMCP action set—trades off global optimality for tractability and speed; this is reasonable, but should be framed as a design choice with pros/cons and not purely as a shortcoming of reward-based baselines.

### Experimental evaluation assessment

The oracle experiments are appropriately sized (200 trials) and include ablations that convincingly disentangle timing from selection. The threshold sweep is particularly informative, demonstrating diminishing returns at τ=1.0.

Physical evaluations strengthen ecological validity, and the error analysis (VLM vs planner failures) plus user interviews are valuable. However, the VLM/human sample sizes are modest (n=40 and n≈10 per condition), and significance testing or confidence intervals would increase statistical rigor.

Fairness and comparability: POMCP-Q’s larger planning time suggests unequal computational budgets. If the proposed method’s advantage partly stems from a far smaller search space (no query actions), then either compute should be equalized (time- or simulation-matched) or results should be framed as a speed–accuracy trade-off rather than solely as accuracy gains.

### Comparison with related work (using the summaries provided)

Relative to VAP‑TAMP (predicate verification via VLM) and BKLV A (three-valued belief with observation actions), this paper’s contribution is tighter, entropy-based triggering and selection with a compact frontier belief. Unlike those systems, it quantifies confidence and selects specific predicates to minimize posterior entropy, which is a more explicit information-theoretic objective.

Compared to RoVLaP, which grounds fluent probabilities from a VLM and compiles an MLSS for conformant planning, the present work focuses on online, stepwise verification driven by belief entropy and maintains a particle-like frontier rather than a factored belief plus invariant constraints. Both maintain explicit symbolic uncertainty; RoVLaP emphasizes per‑fluent calibration and conformant planning, while the proposed method emphasizes myopic entropy-based verification and POMCP action selection.

In relation to ρ‑POMCP and EFE-as-ρ-POMDP approaches that optimize belief-dependent utilities based on information gain or entropy, the proposed work externalizes sensing by querying humans/VLMs and triggers based on current-state entropy, not cumulative expected value. A direct baseline with ρ‑POMCP(−H(b′)) or ΔH(b,b′) would be a compelling and fairer comparison than reward-only POMCP‑Q.

STRUG targets task-relevant uncertainty, avoiding reduction of irrelevant entropy. The proposed method queries on the most entropy-reducing predicate regardless of task relevance; whether the queried fluent is task-critical is mediated only indirectly via the frontier and upcoming actions. A task-relevance weighting could further reduce queries.

### Discussion of broader impact and significance

The paper offers a practical recipe: trigger on belief uncertainty, and select queries by expected entropy reduction on symbolic facts. The ablation-backed insight that timing mainly drives success whereas selection mainly drives cost is broadly useful for robot systems that must balance autonomy with occasional external assistance.

Limitations suggest natural future directions: integrating feedback reliability models, considering query costs explicitly (value-of-information), expanding beyond immediate frontiers to carry forward alternative hypotheses, and tighter integration with factored or learned belief models to scale.

## Questions for Authors

1. How are transition and observation probabilities T and O specified or learned in each domain? What is the sensitivity of performance to mis-specified O (especially when sensor posteriors are biased)?
2. Does the frontier belief include uncertainties that were not caused by the immediately preceding action (e.g., lingering misclassifications about other objects)? If not, how does the system recover when the committed s_t^K is wrong in a way that does not appear in the next frontier?
3. Have you considered modeling noisy external feedback (human/VLM) by soft updates (e.g., Bayesian fusion with reliability priors) rather than hard filtering? Would this mitigate the observed VLM-induced failures?
4. How were compute budgets matched across methods? For POMCP‑Q in particular, can you report results under a time-matched or simulation-count–matched regime to isolate algorithmic differences from search budget effects?
5. How large can the candidate predicate set become in more complex domains, and what is the computational cost of evaluating expected posterior entropy over all candidates each step?
6. Could you compare against a ρ‑POMCP variant that directly optimizes belief-entropy reduction (−H(b′) or ΔH) to better contextualize the benefits of decoupling queries from the action space?
7. How robust are results to the confidence threshold τ across domains with different uncertainty structures, and could τ be adapted online using, e.g., a target success probability or an explicit query-cost budget?

## Overall Assessment

This paper presents a clear, principled, and practically effective framework for proactive querying in symbolic task planning under uncertainty. The core ideas—triggering on belief entropy and selecting predicates by expected entropy reduction on a structured frontier—are well motivated and empirically validated across simulated and real robot settings. The ablation studies are a highlight, yielding actionable guidance about the roles of timing and selection.

The main weaknesses concern (i) the narrow frontier belief tied to a committed MAP state (which may under-represent persistent uncertainties), (ii) the lack of a reliability model for noisy external answers, and (iii) incomplete comparisons to closely related belief-entropy planners (ρ‑POMCP/EFE, STRUG) and potential compute-budget mismatches with POMCP‑Q. These issues do not undermine the core contribution but suggest needed clarifications and, ideally, additional experiments.

Overall, the work is solid, relevant, and likely to be useful to the Autonomous Robots community. With strengthened discussion of limitations, clearer specification of T/O models, and either a time-matched comparison or an added ρ‑POMCP-style baseline, I would support acceptance.

---

# 한글 번역본

## 요약

이 논문은 부분 관측 환경에서 작동하는 로봇을 위해, 외부 피드백(예: VLM 또는 사람)을 언제 요청하고 무엇을 질문할지 함께 결정하는 계획 및 질의 프레임워크를 제안한다. 각 행동 이후 기호적 후속 상태들(“프런티어”, frontier)에 대한 가중 신념을 유지하고, 정규화된 신념 엔트로피가 낮은 신뢰도를 나타내면 질의를 수행하며, 기대 엔트로피 감소량을 최대화하는 술어를 선택하여 검증한다. 두 작업 도메인에서 시뮬레이션 및 실제 플랫폼을 사용한 실험 결과, 컨포멀 예측 기반 및 보상 기반 베이스라인보다 작업 성공률이 크게 향상되었다. 또한 절제 실험에서는 질의 시점(언제 질문할지)이 주로 성공률에 영향을 미치는 반면, 질의 대상 선택(무엇을 질문할지)은 질의 횟수를 줄이는 것으로 나타났다.

## 강점

### 기술적 신규성과 독창성

- 명시적이고 구조화된 기호적 프런티어를 사용하여 도달 가능한 세계 상태들에 대한 간결한 신념을 구성하고, 질의 시점 결정(정규화된 엔트로피)과 질의 대상 선택(기대 엔트로피 감소)을 모두 이 신념에 직접 연결한다.
- 신념 기반 질의를 행동 선택용 POMCP와 통합하면서 질의를 행동 공간 밖에 두어, 계획기에 질의 행동을 추가할 때 발생하는 탐색 공간의 폭발적 증가를 방지한다.
- “언제”와 “무엇”을 명확하게 분리하고, 질의 대상 선택을 대리 휴리스틱이 아닌 정보이론적 기준에 근거하도록 한다.

### 실험의 엄밀성과 검증

- 불확실성의 원인이 서로 다른 두 도메인(WasteSorting, TomatoHarvesting)에서 시뮬레이션 환경(오라클)과 실제 환경(VLM 및 사람이 개입하는 방식) 모두를 대상으로 평가하였다.
- 여러 베이스라인(컨포멀 예측 방식 및 보상 기반 질의 선택)과의 비교를 포함하며, 질의 시점과 질의 대상 선택의 효과를 분리하는 세심한 절제 실험 및 임계값 변화 실험을 수행하였다.
- VLM 및 사람 대상 실험의 실패 사례를 분석하고, 상태 검증 질문의 사용자 중심적 장점을 뒷받침하는 정성적 인터뷰를 제공한다.

### 서술의 명확성

- 신념 구성, 신뢰도, 엔트로피 기반 선택을 설명하는 수식으로 방법을 체계적으로 제시하며, 알고리즘 박스에서 전체 반복 절차를 간결하게 보여 준다.
- 시스템 도식과 작업 설명을 통해 실행 파이프라인을 큰 틀에서 이해하고 재현할 수 있다.

### 기여의 중요성

- 신뢰하기 어려운 인식 기능을 갖춘 로봇을 실제로 운용할 때의 핵심 과제, 즉 잠재적으로 큰 비용을 초래하는 행동을 수행하기 전에 어떤 불확실성을 해소할지 선제적으로 결정하는 문제를 다룬다.
- 질의 시점이 성공률을 좌우하고 질의 대상 선택이 질의 비용을 줄인다는 실험적 발견은 실무자가 적용할 수 있는 설계 원칙을 제공한다.

## 약점

### 기술적 한계 또는 우려 사항

- “프런티어” 신념은 마지막 행동에 대해, 이미 확정한 MAP 지식 상태 s_t^K의 후속 상태만 열거한다. 따라서 해당 행동의 영향을 받지 않는 상태 부분에 남아 있는 불확실성을 무시할 위험이 있다. MAP 상태로 확정하는 설계 때문에, 직후의 프런티어에 나타나지 않는 과거의 잘못된 신념에서 회복하지 못할 수 있다.
- 시뮬레이션의 신념 갱신에서는 외부 피드백에 잡음이 없다고 간주하고, 실제 실험에서는 이를 확정적 사실로 취급한다. 피드백 신뢰도 모델이 없으며, 잘못된 응답에 대비하는 장치(예: 연성 가중치 부여, 반복 질의, 베이지안 융합)도 없다.
- 질의 수행 여부는 오직 엔트로피 임계값으로 결정되며, 질의 비용이나 정보 가치에 따른 상충관계를 명시적으로 고려하지 않는다. 이 때문에 임계값이 높을 때 과도한 질의가 발생할 수 있다.

### 실험의 공백 또는 방법론적 문제

- POMCP-Q는 계획 시간이 훨씬 길다는 점에서 계산 측면의 불리함이 있어 보이며, 질의를 행동 공간에 포함한다. 방법 간 계산 예산이나 탐색 매개변수를 공정하게 맞추었는지는 불분명하다.
- 논문은 컨포멀 예측으로 질의를 유발하는 접근법과 비교하지만, 제안한 신념 불확실성 기준과 취지가 더 가까운 ρ-POMDP 형태의 신념 엔트로피 또는 정보 이득 기반 계획기(예: −H(b′) 또는 ΔH를 보상으로 사용하는 ρ-POMCP), 그리고 작업 관련 불확실성을 다루는 방법(예: STRUG)과의 직접 비교는 빠져 있다.
- VLM 및 사람 대상 실험은 조건별·도메인별 표본 수가 상대적으로 적으며, 통계적 유의성 검정이 보고되지 않았다.

### 설명 또는 서술상의 문제

- 프런티어에 가중치를 부여하는 데 사용하는 관측 모델과 전이 모델(식 2)의 설명이 충분히 구체적이지 않아, 모델의 충실도나 SP 모듈로부터 이를 어떻게 학습 또는 추정하는지 평가하기 어렵다.
- 여러 사실에 영향을 미치는 효과가 있을 때 실제로 프런티어를 어떻게 열거하는지, 그리고 규모가 커질 때 유효하지 않은 기호 조합을 어떻게 배제하는지가 여전히 모호하다.

### 누락된 관련 연구 또는 비교

- 명시적 VLM 그라운딩을 사용하는 최근의 신념 기반 기호 계획 연구(예: VAP‑TAMP의 술어 검증 루프, BKLV A의 관측 행동을 포함한 K-fluents 신념), 그리고 플루언트에 대한 신념을 사용하는 컨포먼트 확률적 계획 연구(RoVLaP)를 더 깊이 논의할 필요가 있다. 이 연구들은 이미 술어 수준의 불확실성을 계획 및 실행과 연결하고 있다.
- 정보이론적 질의 선택은 ρ-POMDP 및 능동 센싱 문헌(예: ρ‑POMCP, EFE-as-ρ-POMDP)에서 잘 정립되어 있다. 이 연구들과의 관계를 더 정밀하게 정리하면 독창성이 명확해질 것이다.

## 세부 의견

### 기술적 타당성 평가

가중 프런티어 신념을 구성하고 정규화된 엔트로피에 따라 질의를 수행하는 핵심 메커니즘은 개념적으로 타당하며 POMDP 원리에 부합한다. 술어 선택에 사용하는 기대 엔트로피 감소 기준은 기호 수준에서 근시안적 정보 이득을 원칙에 맞게 구현한 것이다.

두 가지 설계 선택은 면밀한 검토가 필요하다. (i) 확정된 단일 세계 상태 s_t^K의 직후 프런티어만을 대상으로 신념을 구성하는 방식은 지나치게 근시안적일 수 있다. 여러 단계에 걸쳐 지속되지만 모델링되지 않은 불확실성이 표현되지 않아, 오류를 바로잡기 위한 질의가 제한될 수 있다. (ii) 잡음 모델 없이 외부 응답에 따라 신념을 경성 필터링하면, 정보원이 틀렸을 때 과도한 확신과 되돌릴 수 없는 오류가 발생할 위험이 있다. 이는 실패 분석에서도 확인된다.

질의를 POMCP의 행동 집합 밖에 두는 관심사 분리는 전역 최적성과 계산 가능성 및 속도 사이의 절충이다. 이는 합리적인 선택이지만, 단순히 보상 기반 베이스라인의 단점으로만 설명하기보다 장단점이 있는 설계 선택으로 제시해야 한다.

### 실험 평가에 대한 검토

오라클 실험은 적절한 규모(200회 시행)로 수행되었으며, 질의 시점과 질의 대상 선택의 효과를 설득력 있게 분리하는 절제 실험을 포함한다. 특히 임계값 변화 실험은 τ=1.0에서 추가 이득이 줄어든다는 점을 보여 주어 유용하다.

실제 환경 평가는 생태학적 타당성을 높이며, 오류 분석(VLM 실패와 계획기 실패의 구분) 및 사용자 인터뷰도 가치가 있다. 다만 VLM 및 사람 대상 실험의 표본 수는 크지 않으므로(조건별 n=40 및 n≈10), 유의성 검정이나 신뢰구간을 제시하면 통계적 엄밀성이 높아질 것이다.

공정성과 비교 가능성: POMCP-Q의 더 긴 계획 시간은 계산 예산이 같지 않았을 가능성을 시사한다. 제안 방법의 이점이 훨씬 작은 탐색 공간(질의 행동을 포함하지 않음)에서 일부 비롯된다면, 계산량을 같게 맞추거나(시간 또는 시뮬레이션 횟수 기준), 결과를 단순한 정확도 향상이 아니라 속도와 정확도 사이의 절충으로 설명해야 한다.

### 관련 연구와의 비교(제공된 요약 기준)

VAP‑TAMP(VLM을 통한 술어 검증) 및 BKLV A(관측 행동을 포함한 3값 신념)와 비교하면, 이 논문의 기여는 간결한 프런티어 신념을 사용하여 엔트로피 기반 질의 수행 결정과 질의 대상 선택을 더 긴밀하게 결합한 데 있다. 해당 시스템들과 달리, 신뢰도를 정량화하고 사후 엔트로피를 최소화하도록 특정 술어를 선택하므로 정보이론적 목적이 더 명시적이다.

VLM으로부터 플루언트 확률을 그라운딩하고 컨포먼트 계획을 위해 MLSS를 컴파일하는 RoVLaP와 비교하면, 본 연구는 신념 엔트로피에 기반한 온라인 단계별 검증에 초점을 맞추며, 인수분해된 신념과 불변 제약조건의 조합 대신 입자와 유사한 프런티어를 유지한다. 두 방법 모두 기호적 불확실성을 명시적으로 유지하지만, RoVLaP는 플루언트별 보정과 컨포먼트 계획을 강조하고 제안 방법은 근시안적 엔트로피 기반 검증 및 POMCP 행동 선택을 강조한다.

정보 이득이나 엔트로피에 기반하여 신념 의존적 효용을 최적화하는 ρ‑POMCP 및 EFE-as-ρ-POMDP 접근법과 비교할 때, 제안 연구는 사람이나 VLM에 질의하여 센싱을 외부화하며, 누적 기대 가치가 아닌 현재 상태의 엔트로피를 기준으로 질의 수행을 결정한다. ρ‑POMCP(−H(b′)) 또는 ΔH(b,b′)를 사용하는 직접적인 베이스라인은 보상만 사용하는 POMCP‑Q보다 설득력 있고 더 공정한 비교가 될 것이다.

STRUG는 작업과 관련된 불확실성을 대상으로 하여, 무관한 엔트로피를 줄이는 일을 피한다. 제안 방법은 작업 관련성과 관계없이 엔트로피를 가장 많이 줄이는 술어에 대해 질의한다. 질의 대상 플루언트가 작업에 핵심적인지는 프런티어와 이후 행동을 통해서만 간접적으로 반영된다. 작업 관련성에 따른 가중치를 부여하면 질의를 더 줄일 수 있을 것이다.

### 더 넓은 파급 효과와 중요성에 대한 논의

이 논문은 실용적인 방법을 제시한다. 신념의 불확실성을 기준으로 질의를 수행하고, 기호적 사실에 대한 기대 엔트로피 감소량에 따라 질의 대상을 선택하는 것이다. 질의 시점은 주로 성공률을 좌우하고 질의 대상 선택은 주로 비용을 좌우한다는 통찰은 절제 실험으로 뒷받침되며, 자율성과 간헐적인 외부 지원 사이의 균형을 맞춰야 하는 로봇 시스템 전반에 유용하다.

한계에서 자연스럽게 도출되는 향후 연구 방향으로는 피드백 신뢰도 모델의 통합, 질의 비용의 명시적 고려(정보 가치), 직후 프런티어를 넘어 대안 가설을 계속 유지하는 방식으로의 확장, 그리고 확장성을 위한 인수분해된 신념 모델 또는 학습된 신념 모델과의 긴밀한 통합이 있다.

## 저자에게 묻는 질문

1. 각 도메인에서 전이 확률 T와 관측 확률 O는 어떻게 명세하거나 학습하는가? O가 잘못 명세되었을 때, 특히 센서의 사후확률이 편향되어 있을 때 성능은 얼마나 민감하게 영향을 받는가?
2. 프런티어 신념은 직전 행동 때문에 발생한 것이 아닌 불확실성(예: 다른 물체에 대한 지속적인 오분류)도 포함하는가? 그렇지 않다면, 확정한 s_t^K가 다음 프런티어에 나타나지 않는 방식으로 잘못되어 있을 때 시스템은 어떻게 회복하는가?
3. 잡음이 있는 외부 피드백(사람 또는 VLM)을 경성 필터링 대신 연성 갱신(예: 신뢰도 사전분포를 사용하는 베이지안 융합)으로 모델링하는 방안을 고려했는가? 이를 통해 관찰된 VLM 유발 실패를 줄일 수 있는가?
4. 방법 간 계산 예산은 어떻게 맞추었는가? 특히 POMCP‑Q의 경우, 탐색 예산의 영향과 알고리즘 자체의 차이를 분리할 수 있도록 시간이나 시뮬레이션 횟수를 동일하게 맞춘 조건의 결과를 보고할 수 있는가?
5. 더 복잡한 도메인에서 후보 술어 집합은 얼마나 커질 수 있으며, 매 단계 모든 후보에 대해 기대 사후 엔트로피를 평가하는 계산 비용은 어느 정도인가?
6. 질의를 행동 공간에서 분리하는 이점을 더 명확한 맥락에서 평가할 수 있도록, 신념 엔트로피 감소(−H(b′) 또는 ΔH)를 직접 최적화하는 ρ‑POMCP 변형과 비교할 수 있는가?
7. 불확실성 구조가 다른 도메인들에서 결과는 신뢰도 임계값 τ에 대해 얼마나 강건한가? 또한 목표 성공 확률이나 명시적인 질의 비용 예산 등을 사용하여 τ를 온라인으로 조정할 수 있는가?

## 종합 평가

이 논문은 불확실성하의 기호적 작업 계획에서 선제적으로 질의하기 위한 명확하고 원칙에 충실하며 실용적으로 효과적인 프레임워크를 제시한다. 신념 엔트로피를 기준으로 질의를 수행하고, 구조화된 프런티어에서 기대 엔트로피 감소량에 따라 술어를 선택한다는 핵심 아이디어는 타당한 동기를 갖추었으며 시뮬레이션과 실제 로봇 환경 모두에서 실험적으로 검증되었다. 절제 실험은 특히 돋보이며, 질의 시점과 질의 대상 선택의 역할에 대해 실제로 적용할 수 있는 지침을 제공한다.

주요 약점은 (i) 확정된 MAP 상태에 묶인 좁은 프런티어 신념(지속적인 불확실성을 충분히 표현하지 못할 수 있음), (ii) 잡음이 있는 외부 응답에 대한 신뢰도 모델의 부재, (iii) 밀접하게 관련된 신념 엔트로피 기반 계획기(ρ‑POMCP/EFE, STRUG)와의 불충분한 비교 및 POMCP‑Q와의 계산 예산 불일치 가능성이다. 이러한 문제들이 핵심 기여를 훼손하지는 않지만, 설명을 보완하고 가능하다면 추가 실험을 수행할 필요가 있음을 시사한다.

종합하면, 이 연구는 견실하고 관련성이 높으며 Autonomous Robots 커뮤니티에 유용할 가능성이 크다. 한계에 대한 논의를 강화하고, T/O 모델을 더 명확하게 명세하며, 시간을 동일하게 맞춘 비교 또는 ρ‑POMCP 방식의 베이스라인을 추가한다면 게재 승인을 지지하겠다.
