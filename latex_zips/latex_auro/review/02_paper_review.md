# 논문 리뷰: *When and What to Query for Proactive Feedback in Robot Task Planning under Partial Observability*

## 1. 논문 요약

로봇이 부분관측 하에서 task를 수행할 때 **언제(when)** 외부에 질의하고 **무엇을(what)** 물을지를 결정하는 방법을 제안한다.

- **상태 표현**: STRIPS식 symbolic 모델 위에서, 직전 action `a_t`로 도달 가능한 successor 상태들(frontier)에 대한 가중 belief `B_{t+1}`을 만든다. 가중치는 `T(s'|s^K_t, a_t) · O(o|s', a_t)`를 정규화한 값이다 (Eq. 2).
- **When**: 정규화 엔트로피 기반 confidence `Conf = 1 − H(B)/log2 j`가 임계값 τ(기본 0.8) 미만이면 질의한다 (Eq. 3).
- **What**: belief를 분할했을 때 기대 posterior entropy가 최소인 atomic fact를 binary 질문("Is tomato 1 ripe?")으로 묻는다 (Eq. 4). 응답은 hard filter로 belief에 반영한다 (Eq. 5). Conf ≥ τ가 될 때까지 반복하고, MAP 상태를 다음 planning 상태로 확정한다.
- **Action 선택**: POMCP가 physical action만 대상으로 수행한다.
- **평가**: WasteSorting(WS)과 TomatoHarvesting(TH)에서 oracle 시뮬레이션(200 trials), VLM(Gemma 4 31B, 40 trials), 사람(5명, 10 trials/조건)을 사용한다. 베이스라인은 KnowNo, IntroPlan, POMCP-Q이고, ablation(T1–T3, S1–S2)과 τ sweep, 인터뷰를 포함한다.
- **핵심 결과**: oracle에서 성공률 100.0% / 98.0%(최강 baseline POMCP-Q는 87.0% / 57.5%). VLM 92.5% / 92.5%, 사람 90% / 100%. "timing은 성공률을, selection은 질의 횟수를 좌우한다"는 결론을 낸다.

---

## 2. 종합 평가

| 항목 | 평가 | 코멘트 |
|---|---|---|
| Novelty | 중간 | 개별 요소(entropy 기반 trigger, information gain 기반 질의)는 고전적이다. symbolic frontier belief와 state-fact 질의의 결합이 기여다. |
| Technical soundness | 중간 | 방법의 핵심 서술(belief 유지)과 구현(one-step frontier + MAP commit)의 간극, 응답 오류에 대한 취약성이 있다. |
| Empirical strength | 중상 | 3개 응답원, ablation, threshold sweep까지 폭넓다. 다만 baseline 공정성과 통계 처리에 의문이 남는다. |
| Clarity | 중상 | 구성은 명확하다. 미완성 흔적(`Table ??` 등)이 있다. |
| Reproducibility | 중간 | reward 설계, T/O 모델 출처, baseline 입력 정보가 불명확하다. |

**권고: Weak Accept (조건부).** 실험 범위와 일관된 방향성은 설득력 있다. 다만 §3의 M1–M3은 결론("timing이 핵심")의 해석에 직접 영향을 주므로, 보완이나 최소한 명시적 논의가 필요하다.

---

## 3. 강점

1. **문제 설정이 명확하다.** "confident한 action 예측 ≠ 정확한 state 추정"이라는 지적(KnowNo류의 맹점)이 실제 실패 사례(unmet precondition에서의 자율 선택, 질의 없이 진행된 rotten tomato 오인식)로 뒷받침된다.
2. **질의 대상을 state fact로 바꾼 설계가 설득력 있다.** 사람/VLM이 action precondition을 추론할 필요가 없다. 사람 실험에서 KnowNo 실패 중 상당수가 "선행 action을 건너뛴 선택"이었고, 인터뷰 결과(P1, P2, P4)와 일치한다.
3. **Timing × selection 분해 ablation이 깔끔하다.** Fully-Random, T1–T3, S1–S2로 두 축을 분리했고, 비교 가능한 조건에서 query count도 함께 보고했다 (Table 3, Fig. 6).
4. **Threshold sweep이 정직하다.** τ=1.0이 τ=0.8 대비 질의를 +91% / +107% 늘리면서 성공률은 +0 / +2%p만 올린다는 점, 즉 수확체감을 명시했다.
5. **Query count만으로 설명되지 않는다는 분석이 좋다.** TH에서 IntroPlan 8.96회 / 40% vs Ours 9.16회 / 98%.
6. **Failure 분해가 투명하다.** VLM 오답 vs planner 실패를 구분했다 (WS KnowNo 3 vs 17, TH 18 vs 12). 보고된 비율과 실패 건수가 서로 맞는다.
7. **Planning time 이점이 실질적이다.** LLM 호출이 없는 구조 덕분에 약 0.9–1.4 s 수준이다.

---

## 4. 주요 우려사항 (Major)

### M1. 베이스라인 비교가 "timing"과 "질의 유형"을 교란한다

Ours는 **state-fact binary query**를, KnowNo/IntroPlan은 **action-choice query**를 한다. 따라서 baseline 비교(Table 2)에서 나타난 성능 차이가 "언제 묻는가"의 차이인지 "무엇의 형태로 묻는가"의 차이인지 분리되지 않는다. 논문 스스로도 사람 실험에서 "action selection은 precondition 해석을 요구한다"고 해석하므로, 이는 timing이 아닌 **질의 modality**의 효과를 주장하는 것에 가깝다.

더 직접적인 관찰이 있다. **Ours에서 τ=0(질의 없음)일 때 성공률은 WS 55.0%, TH 52.0%** (§5.2.3)인데, 이는 KnowNo(48.5 / 43.0)나 IntroPlan(51.5 / 40.0)과 비슷하거나 더 높다. 이 수치는 다음과 같이 읽을 수 있다.

- 성능 격차의 상당 부분은 **질의 전략이 아니라 base planner**(symbolic precondition-aware POMCP + 상태 추적)에서 온다.
- KnowNo/IntroPlan은 질의를 수행하고도 Ours의 "질의 없음" 수준에 머문다.

**제안**
- 각 방법의 **no-query 성능**과 "질의로 얻은 성공률 증가량 / 질의 1회당 증가량"을 함께 보고한다.
- 공정한 baseline 추가: (a) KnowNo + symbolic applicability filter(unmet precondition action 제거), (b) KnowNo 방식의 conformal timing + **state-fact query**(= T2와 유사하나 baseline 취급), (c) belief+entropy trigger + action-choice query.
- Oracle 실험에서 LLM baseline이 어떤 상태 정보(ground-truth state description 여부)를 입력받는지, oracle이 어떤 action을 정답으로 응답하는지 명시한다.

### M2. "Belief를 유지·갱신한다"는 서술과 실제 구현 사이의 간극

Intro는 "task 전반에 걸쳐 weighted belief를 유지하고 갱신한다"고 하고, §3에서 Bayes filter를 소개한다. 그러나 실제 구현은 다음과 같다.

- `B_{t+1}`은 **현재 point estimate `s^K_t`와 `a_t`로부터의 one-step frontier**에서만 만들어진다 (Eq. 2는 `T(·|s^K_t, a_t)`).
- 질의 루프 후 **MAP으로 collapse**한다 (Alg. 1 line 19). 즉 belief는 step 간에 전달되지 않는다.
- 결과적으로 Conf ≥ τ이면서 MAP이 틀린 경우 **이후 복구 경로가 없다.** 오류는 누적된다.
- Related Work의 "particle을 현재 action 이후 reachable state에서만 샘플링한다"는 서술은 이 frontier 구성과 연결되지만, 제안 방법은 particle filter가 아니라 **one-step weighted enumeration + 매 step 재초기화**다. 이 점을 정직하게 기술하는 편이 낫다.
- POMCP의 내부 belief와 query용 frontier belief는 서로 다른 객체이고, planner는 향후 질의 가능성을 고려하지 않는다 (physical action만 탐색). 즉 "POMDP로 정식화"했다기보다 **"POMCP 위에 얹은 query wrapper"**에 가깝다.

**제안**: 위 구조를 명시하고, full-history belief를 유지하는 변형(또는 이전 step의 잔여 불확실성을 prior로 이월하는 변형)과 비교하는 소규모 ablation을 추가한다. 최소한 한계로 논의해야 한다.

### M3. 응답 오류에 취약한 hard filtering (Eq. 5)

Eq. 5는 응답을 무조건 참으로 받아들여 불일치 상태를 **제거**한다. 오답이 들어오면 정답 상태가 belief에서 사라지고, 이후 Conf=1로 **확신에 찬 오류**가 된다. 실제로 VLM 실험에서 Ours의 실패 3/3(WS, TH 각각)이 오답 피드백이고, 사람 WS의 단일 실패도 잘못된 확인 때문이다. 즉 남은 실패는 거의 전부 이 구조의 취약점이다.

- Appendix A에서 POMCP-Q에는 "assumed response accuracy 1.0"이 있어 응답 모델 개념은 이미 도입되어 있다. 그러나 Ours에는 없다.
- VLM 실험에서 log-prob(top-5)을 켰지만 Ours는 이를 쓰지 않고 Yes/No로 이진화한다. KnowNo만 token 확률을 쓴다.

**제안**: 응답 정확도 `α`를 둔 soft update(`likelihood ∝ α / (1−α)`), 또는 VLM log-prob를 evidence로 쓰는 변형, 상충 응답 시 재질의 정책을 추가한다. soft update에서는 Conf가 1에 도달하지 않으므로 **정지 규칙이 비용 인식형**이 되어야 한다. 이는 자연스러운 후속 연구이자, 현 결과(oracle 응답 100% 정확 가정)의 외적 타당성 한계로 반드시 명시해야 한다.

### M4. Ablation 결론의 해석 — "selection은 성공률에 영향이 없다"는 상당 부분 설계상 자명하다

- 모든 구성이 **trigger 이후 Conf ≥ τ가 될 때까지** 질의를 계속하고(T2 제외), oracle 응답은 항상 정답이다. 따라서 어떤 selection을 쓰든 결국 거의 같은 posterior에 도달한다. S1/S2/Ours의 성공률이 거의 같고(WS 100/100/100, TH 97.5/97.5/98.0), T1과 Fully-Random의 성공률이 **두 도메인에서 소수점까지 동일**(73.50)한 것도 이 때문이다. 즉 "timing이 성공률, selection이 질의 수"라는 분업은 이 실험 설계에서 **거의 구조적으로 결정된다.**
- **Eq. 4의 구조적 사실**: 질의 결과가 상태의 결정적 함수이고 응답이 정확하다면, 기대 entropy 감소량은 `H(S) − E[H(S|y)] = H_b(P(q))`(질의 사실의 marginal에 대한 binary entropy)와 같다. 즉 Eq. 4는 **"P(q)가 0.5에 가장 가까운 fact 선택"**과 동치다. 이 점을 밝히면 방법이 단순화되고 오히려 설명하기 쉬워지지만, 동시에 selection이 **planning relevance를 전혀 고려하지 않는다**는 점이 드러난다. 목표와 무관한 fact(예: 현재 plan에 영향이 없는 객체의 category)도 균형 잡힌 분할이면 선택된다.
- S2(reward 기반 selection)가 22.79회로 가장 많은 것은 reward 기반 selection이 열등해서라기보다 γ(0.9/0.5)와 탐색 예산의 영향일 수 있다 (M5).

**제안**
- Noisy 응답(M3) 하에서 selection ablation을 반복한다. 이 조건에서 selection의 효과가 성공률에도 드러나는지가 진짜 검증이다.
- Task-relevant VOI(value of information) 기반 selection(precondition/goal-relevant fact 한정 또는 가중)과 비교한다.
- Table 1이 완전한 factorial이 아니라는 점(여러 칸이 `–`)을 결론 문구에 반영한다 ("primarily affects" 수준의 표현).

### M5. POMCP-Q 설정이 불리할 가능성

- POMCP-Q는 γ=0.9 (WS) / 0.5 (TH), Ours는 γ=0.2이다. Appendix의 discount cutoff(ε=0.005)에 따르면 Ours의 탐색 depth는 약 4이지만, γ=0.9이면 이론적 depth가 ~50이어서 **100 simulations**로는 tree가 거의 자라지 못한다. 거기에 query action까지 action space에 들어가 branching이 커진다. 논문도 이를 "may contribute"라고 인정하지만, 이 효과를 분리하는 실험이 없다.
- "γ는 성공률이 가장 높은 값으로 탐색했다"는 설명이 Ours와 POMCP-Q 양쪽에 동일하게 적용되었는지 불분명하다 (둘 다 γ sweep 결과를 보고하면 좋다).
- POMCP-Q는 **query 비용 0**으로 설정하면서도 **질의가 50-decision 한도를 소모**한다. WS에서 평균 24.5 query + 11.9 action = 약 36 decisions로 한도(50)에 가깝다. 한도 초과 실패가 얼마나 되는지 분해해야 한다.
- **Reward function**(성공 보상, failure penalty 10 외)이 본문에 없다. γ=0.2라는 매우 근시안적 설정에서 어떤 shaping이 신호를 전달하는지, 이식성과 재현성의 핵심 정보다.

**제안**: POMCP-Q에 대해 simulations 수와 γ를 sweep한 결과(성공률/질의 수)를 부록에 제시하고, 한도 소진 실패율을 분리 보고한다.

### M6. Confidence 정의와 τ 선택의 근거

- **Calibration 미평가**: `T`와 `O`가 어떻게 주어지는지(수작업? 시뮬레이터의 true 모델?)가 본문에 없다. Oracle에서 true 모델을 쓴다면 belief가 구성상 calibrated이므로 timing 성능이 낙관적이다. 실제(VLM/사람/real sensor)에서 `Conf < τ`가 실제 상태 오류와 얼마나 상관되는지 (reliability diagram, Conf-binned MAP 정확도) 검증이 필요하다.
- **τ의 해석성**: 내가 계산한 바로는 τ=0.8은 frontier 크기 j=2~8에서 대략 `p_max ≳ 0.94–0.97`에 해당한다 (예: j=2일 때 (0.97, 0.03) → Conf≈0.806). 상당히 엄격한 기준이다. `1 − p_max`(MAP 오류 확률), 또는 margin 기반 trigger와 비교하면 τ가 "허용 오류율"로 해석되어 더 직관적이다. 논문에는 이 대안과의 비교가 없다.
- **Test set 튜닝**: τ=0.8은 §5.2.3의 sweep에서 선택되었는데, 이때 평가한 200 trials가 baseline/ablation 비교에 쓰인 바로 그 trials다 (Appendix A.4: "reuse"). 초기 configuration도 5개뿐이므로 τ 선택과 평가가 같은 분포 위에서 이뤄진다. 별도 seed/configuration에서의 hold-out 확인이 필요하다.
- **Step-function 형태**: WS는 τ 0.1–0.6에서 평탄(53.5–54%)하다가 0.7에서 100%로 점프하고, TH는 0.3→0.4에서 52→88.5%로 뛴다. belief weight가 소수의 이산 값(수작업 T/O에서 유래)을 갖는다는 신호로 보인다. 그렇다면 "도메인마다 필요한 τ가 다르다"는 결론은 새 도메인에서 τ를 어떻게 정할지에 대한 지침 없이 끝난다. Query 시점의 Conf 분포를 보이면 좋다.

### M7. 통계와 인간 실험

- **신뢰구간/검정 부재**: 성공률에 CI나 검정이 없다. oracle은 큰 차이라 문제가 적지만, VLM(40)과 사람(10 trials, 5명, 2 trials/조건)은 표본이 작다. 반복측정 구조(참가자 내)도 고려되지 않았다. 효과가 매우 커서 방향성은 신뢰할 수 있지만, **pilot/exploratory**로 명시하고 Wilson CI 등을 병기하기를 권한다.
- **생존자 편향**: 비용 지표(query count, plan length, time)는 성공 trial 평균이다. 성공이 2/10인 KnowNo(WS, 사람)와 9/10인 Ours를 비교하는 값은 조건이 다르다.
- **Oracle 다양성**: 200 trials가 5개 초기 configuration × 40회 반복이다. seed 다양성은 있지만 **환경 다양성**은 작다.
- **Workload는 측정되지 않았다**: "덜 생각한다"는 인터뷰 진술(n=5)에만 의존한다. NASA-TLX 같은 표준 척도가 없다. 또한 WS에서 Ours의 query당 interaction time(5.26 s)이 KnowNo(3.86 s)보다 길고, 총 interaction time도 약 3배(36.5 s vs 11.6 s)다. P3 제외 분석이 있으나 사후 분석이다.
- **참가자 구성과 task 지식**: 5명(남 1/여 4), task 목표와 action 목록을 사전 제공받았다. 도메인 전문가(수확/분리배출 숙련자)가 아니다. KnowNo 조건의 인간 오류가 "비전문가가 action precondition을 잘못 해석한 탓"인지 일반적 현상인지는 구분되지 않는다.
- **VLM 조건의 prompt 비대칭**: Ours는 predicate별 판단 기준이 매우 상세하다 (예: "큰 beige 부패 영역일 때만 No"). 이는 사실상 hand-engineered perception verifier이므로, KnowNo prompt와 엔지니어링 투입량이 대등한지 설명이 필요하다. 또한 이 설정에서 VLM은 "외부 human feedback"이라기보다 **보조 perception module**에 가깝다.

### M8. 일반화와 확장성

- 도메인이 2개이고 규모가 작다 (객체 ≈ 4개, plan 길이 ≈ 10–14). **Frontier 크기 j의 분포**와 runtime scaling(불확실한 fact가 k개면 successor가 최대 2^k)이 보고되지 않았다. Appendix에 frontier capacity 8,000, transition sampling budget 250이 등장하므로 truncation이 일어나는 경우가 있는지, 그때 Conf 계산에 편향이 없는지 설명이 필요하다.
- Intro는 execution failure와 환경 변화를 동기로 들지만, 실험은 **perception noise**만 다룬다.
- 질의 후보 `Q(B)`는 "frontier 상태들에서 값이 갈리는 atomic fact"인데, 사람이 **답할 수 있는** fact인지(예: `scanned`, `observed` 같은 절차 predicate)는 별도 필터가 없다. Domain-independent라는 주장은 계획 무관/응답 불가 fact에 대한 처리가 보강될 때 더 강해진다.
- **T/O 모델 민감도**(모델을 교란했을 때 성능)가 없다.

---

## 5. 관련 연구 위치 (보완 제안)

- 고전적 POMDP **ask action** 및 dialog 연구(예: Doshi & Roy의 spoken-dialog POMDP), **사람을 observation provider로 모델링**한 연구(예: Rosenthal & Veloso), 로봇 능동 질의(예: Cakmak & Thomaz, Tellex et al.의 asking for help), 그리고 Bayesian experimental design/active learning의 expected information gain 계보와 명시적으로 위치를 비교하면 novelty 주장이 더 정확해진다. (정확한 서지는 확인 후 인용 바람.)
- Qu et al. (2023, IoT DNN inference)을 normalized-entropy confidence의 근거로 인용하는 것은 어색하다. Kaelbling et al. (1998)을 "execution failure"의 근거로 인용하는 것도 부정확하다. 더 적합한 출처가 있으면 교체를 권한다.

---

## 6. 사소한 지적 및 표현 (Minor)

1. **Abstract**
   - "We evaluate the framework in two robotic task domains…"와 "We evaluate the framework on two robotic tasks against other systems…"가 중복된다. 한 문장으로 합치면 된다.
   - 첫 문장이 길다. 예: "In this paper, we present a method for deciding when and what a robot should ask during task planning."
   - 수치 표기가 `100.0%`, `98.00%`로 혼재한다.
2. **§5 intro**: "The planned human evaluation is designed for five participants…"는 계획형 문장이다. 이미 수행·보고된 실험이므로 과거형으로 고쳐야 한다.
3. **§5.2.3**: `Table ??`가 깨져 있다 (threshold 결과 표 누락 또는 참조 오류). τ별 수치(성공률, 질의 수)는 본문에만 있다.
4. **§5.2.2**: "W로 timing, Q로 selection을 표기한다"고 했지만 실제 표기는 T/S다.
5. **§5.2.2**: "However T3 requires…"는 쉼표가 빠졌다 ("However, T3 requires…").
6. **Fig. 5 caption**: "An asterisk marks…"라고 했으나 figure/table에 asterisk가 없다. Table 2의 `18.18±0.00`(WS 사람, Ours)과 `11.00±0.00` 같은 SD=0 항목은 성공 9건이 모두 동일 경로였다는 뜻으로 보이며, 설명이 필요하다.
7. **Appendix A.2 (Table A2)**: WS의 KnowNo "physical VLM calibration q̂"가 oracle(GPT-4o) q̂와 **동일한 값(0.845165)**이다 (TH는 다름). 모델이 달라졌는데 값이 같다면 복사 오류이거나 재사용일 가능성이 있다. 후자라면 conformal coverage 보장이 성립하지 않는다. 확인과 명시가 필요하다. 목표 coverage 수준도 적혀 있지 않다.
8. **Table 3**: 최저 query count에 밑줄을 쳤으나 해당 구성이 성공률이 낮은 경우(T2 TH: 4.05회/63%)가 있어 오해의 소지가 있다. **성공률–질의 수 Pareto plot**(baseline, ablation, τ sweep을 한 그림에)이 더 효과적이다.
9. **Affiliation**에 `Institution2` placeholder가 남아 있다.
10. **Planning time 비교**: Ours(로컬 탐색)와 KnowNo/IntroPlan(원격 LLM API 호출)은 하드웨어/네트워크 조건이 달라 직접 비교가 어렵다. 시스템 비용 비교라는 점을 명시하면 충분하다.
11. **수치 교차 검증 메모**: 본문의 파생 수치(T3 +125.43% / +32.71%, S1/S2 대비 감소율 48.54 / 65.64 / 27.72 / 39.92%, τ=1.0 증가율 91.19 / 107.30%)를 표의 반올림 값으로 재계산했을 때 모두 일치(±0.1%p)했다. VLM/사람의 실패 건수 분해도 성공률과 합이 맞는다.

---

## 7. 저자에게 드리는 질문

1. Oracle 실험에서 KnowNo/IntroPlan의 입력(ground-truth 상태 서술 포함 여부)과 oracle의 action 응답 규칙은 무엇인가?
2. Ours에서 `T`와 `O`는 어떻게 정해지는가? 시뮬레이터의 true 모델과 동일한가? 모델을 교란하면 성능은 어떻게 되는가?
3. τ=0일 때 KnowNo/IntroPlan(질의 off)의 성공률은 얼마인가? 질의가 각 방법에 주는 순이득은?
4. 응답 오류율 α를 인위적으로 주입(예: 5–20%)하면 Ours의 성공률이 어떻게 변하는가? soft update 변형은 시도해 보았는가?
5. MAP collapse 대신 belief를 이월하면 어떻게 되는가?
6. POMCP-Q에서 simulations 수/γ sweep 결과와, 50-decision 한도에 걸려 실패한 trial 비율은?
7. 평균 frontier 크기 j와 최대값, 그리고 capacity(8,000) 초과로 truncation이 발생한 비율은?
8. Eq. 4가 `H_b(P(q))` 최대화와 동치임을 인지하고 있는가? 그렇다면 planning-relevant fact로 제한한 변형과의 비교는?
9. Table A2의 WS VLM q̂ 값은 oracle 값 재사용인가?

---

## 8. 수정 우선순위 제안

| 우선순위 | 항목 | 비용 | 효과 |
|---|---|---|---|
| 1 | M1: no-query 성능 병기 + (a)/(b)/(c) 공정 baseline | 중 | 핵심 주장(timing의 효과)의 방어력 대폭 향상 |
| 2 | M3: noisy 응답 실험 + soft update | 중 | 외적 타당성, 실패 원인의 직접 대응 |
| 3 | M2: belief 서술 정정 + 이월 belief ablation | 소–중 | 방법론의 정직성/정합성 |
| 4 | M5: POMCP-Q sweep 및 한도 분리 | 소 | 가장 강한 baseline 대비 주장 보강 |
| 5 | M6: calibration 분석 + hold-out τ | 소–중 | trigger 신뢰성 근거 |
| 6 | M4: Eq. 4 ≡ H_b(P(q)) 명시 + relevance 변형 | 소 | 이론적 명료성 |
| 7 | M7: CI 병기, pilot 명시, NASA-TLX(가능하면) | 소 | 인간 실험의 주장 수위 조정 |
| 8 | Minor 전체 (`Table ??`, abstract 중복, placeholder 등) | 소 | 완성도 |
