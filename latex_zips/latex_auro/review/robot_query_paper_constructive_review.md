# 「Determining When and What to Ask in Robot Task Execution」 원고 검토 및 수정 제안

검토 대상: `BRL_achitecture (2).pdf` (총 32쪽, 본문 및 Appendix 포함)  
검토 기준: 2026년 10월 8일 원고. **Human evaluation과 Interviews는 미완성**이므로, 임시 참가자 수·인구통계·부분 실험 결과를 완성된 주장으로 평가하지 않음.

## 총평

원고의 핵심 연구 질문인 **언제(when) 질문할 것인가**와 **무엇(what)을 질문할 것인가**가 명확하고, 두 요소의 역할을 분리하는 ablation이 논문의 가장 강한 근거다. Table 4에서는 confidence timing을 고정하면 성공률이 비슷하게 유지되는 반면 query selection에 따라 질문 횟수가 크게 달라진다. 반대로 query selection을 고정하고 timing을 바꾸면 성공률 차이가 커진다. 이 관찰은 논문의 주요 메시지를 뒷받침한다.

Oracle → VLM → Human이라는 평가 흐름도 타당하다. 다만 현재 원고에서 시급히 해결할 문제는 문체보다 **방법론의 수학적 정의**, **비교 실험의 예산과 설정 차이**, **데이터가 직접 지지하는 범위를 넘는 해석**이다. 인간 실험과 인터뷰가 아직 진행 중인 것은 별도의 미완성 항목으로 다룬다.

## 가장 중요한 수정: Belief의 의미와 갱신 방식 (Section 3–4, pp. 4–7)

Section 3.2는 일반적인 POMDP Bayesian filtering을 정의한다.

\[
b_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)\sum_s T(s'\mid s,a_t)b_t(s).
\]

반면 Section 4.1, Eq. (4)는 선택된 단일 지식 상태 \(s_t^K\)로부터 reachable successor를 구성하고, 각 상태를 전이 확률과 관측 likelihood로 가중한다.

\[
\hat b_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)T(s'\mid s_t^K,a_t).
\]

이 두 식은 같지 않다. 현재 구현이 후자의 방식이라면, 전체 belief에 대한 Bayesian filtering이라기보다 **선택한 지식 상태를 중심으로 구성하는 action-local belief approximation**이라고 명확히 설명해야 한다. 이는 반드시 잘못된 방법이라는 뜻이 아니라, 구현과 수학적 주장 사이의 차이를 정확히 공개해야 한다는 의미다. 반대로 실제 구현이 이전 시점의 모든 weighted particles를 전파한다면 원고의 Eq. (4)를 수정해야 한다.

Appendix A (pp. 22–25)는 POMCP의 search-tree particles와 query selection을 위한 weighted frontier를 분리하여 기술한다. 그러나 query에 대한 응답 이후 POMCP root belief가 어떻게 반영·갱신되는지는 충분히 드러나지 않는다. SEARCH가 사용하는 `B`가 무엇인지, 최대확률 symbolic state \(s_t^K\)와 어떤 관계인지, 두 belief 표현이 일치해야 하는지 명시할 필요가 있다.

**확인할 구현 질문:** 실제 코드가 (가) 이전의 posterior particle 전체를 다음 단계로 전파하는지, (나) MAP state를 중심으로 한 단계 successor를 다시 만드는지 확인한다. 두 번째라면 근사의 범위와 잠재적 정보 손실을 밝힌다.

### Expected information gain과 task relevance (Section 4.1.2, p. 6)

현재 질문이 상태 predicate의 참·거짓을 오류 없이 알려주는 deterministic binary query라면, Eq. (7)의 expected entropy reduction은 해당 predicate의 Boolean entropy와 같다.

\[
IG(q)=H(B)-\mathbb E_y[H(B\mid y,q)]
= -p_q\log_2p_q-(1-p_q)\log_2(1-p_q),
\quad p_q=P(q=\mathrm{true}).
\]

따라서 이 규칙은 **belief를 가장 균형 있게 분할하는 state fact**를 선택한다. 이것이 곧 계획에 가장 중요한 사실을 선택한다는 수학적 보장은 없다. Symbolic model이 질문할 수 있는 사실을 정하고, entropy gain이 후보 중 불확실성을 가장 크게 줄이는 사실을 결정하며, 그 결과가 실제 성능에 미치는 영향은 실험으로 검증한다는 세 단계를 구분하는 편이 정확하다.

질문 후보 집합 \(Q(B)\)가 모든 구별 가능한 atom인지, action precondition이나 goal-relevant atom으로 제한되는지 구현과 원고를 일치시켜야 한다.

### 정규화 entropy 기반 confidence의 특성 (p. 5)

\[
\operatorname{Conf}(B)=1-\frac{H(B)}{\log_2 |B|}
\]

에서는 남아 있는 state 수가 변하면 분모도 바뀐다. 예컨대 belief \((0.6,0.2,0.2)\)에서 가장 확률이 높은 state를 제거한 뒤 \((0.5,0.5)\)로 정규화하면, 절대 entropy가 줄어도 정규화 confidence는 더 낮아질 수 있다. 즉 질의로 uncertainty가 줄었다고 해서 confidence가 항상 증가하는 것은 아니다. 이 성질이 query loop의 종료 조건에 어떤 영향을 주는지 확인하고 기술하는 것이 좋다.

Fig. 11의 WasteSorting은 \(\tau=0.7\)에서 성공률이 갑자기 상승한다. Query trigger 시점의 confidence 분포나 threshold별 질문 발생 지점 수를 로그로 추가 분석하면 이 현상을 더 잘 설명할 수 있다.

### 오류 있는 피드백과 상태 제거 (p. 6, Eq. (8))

현재는 외부 응답과 불일치하는 state를 모두 제거한다. 이는 oracle 응답에서는 자연스럽지만 VLM/Human의 오류가 있을 때 실제 상태를 belief에서 제거할 수 있다. 응답에 부합하는 state가 하나도 남지 않는 경우의 fallback, 이후 복구 가능성, 응답 오류를 고려하지 않는 가정을 명확하게 서술하자. 지금 바로 새로운 noisy-answer 모델을 구현할 필요는 없다. VLM 실패 분석과 연결해 limitation에 넣을 수 있다.

### 명목상 depth와 실질 탐색 깊이 (Appendix A–B)

설정은 maximum depth 20, \(\gamma=0.2\), \(\epsilon=0.005\)다. Appendix A의 \(\gamma^{\mathrm{depth}}<\epsilon\) 종료 조건이 실제로 적용된다면 \(\gamma^4=0.0016<0.005\)이므로 실질적 탐색은 약 4단계에서 종료될 수 있다. 구현상의 depth 정의까지 확인하고, 이 설정이 의도한 값인지 설명해야 한다.

## 실험의 공정성: 정확히 무엇이 문제인가? (Section 5, Appendix B)

**핵심은 두 방법에 동일한 task-completion opportunity가 주어졌는가이다.** 현재 실험에서 몇몇 차이는 알고리즘의 본질적인 차이일 수 있지만, 일부는 평가 프로토콜의 차이이므로 구분해야 한다.

### Query-Action과 Ours의 실행 예산이 다르다 (Appendix B.1, p. 25)

| 항목 | Ours | Query-Action |
|---|---|---|
| 시행 제한 | 물리적 행동 50회 | 질문과 물리적 행동을 합한 decision 50회 |
| 질문이 시행 예산을 차감하는가 | 아니오 | 예 |
| 질문 횟수의 효과 | 물리적 행동 예산과 별개 | 많이 질문하면 실행 가능한 물리적 행동 횟수 감소 |

예를 들어 질문 30회를 사용한 trial에서 Ours는 물리적 행동 50회를 더 수행할 수 있지만 Query-Action은 최대 20회만 수행할 수 있다. 따라서 Query-Action이 step limit에 걸려 실패했다면, 그 실패가 질의 선택이 불량해서인지, 질문이 물리적 행동을 위한 예산을 소모했기 때문인지 분리하기 어렵다. 두 조건에서 질문 비용을 서로 다르게 부여하는 효과가 생기는 것이다.

**단, 이 설정 차이만으로 현재 성공률 격차가 왜곡되었다고 증명되는 것은 아니다.** Table 2의 *성공한 trial* 기준으로 Query-Action은 WasteSorting에서 평균 24.52회 질문, 11.89회 물리적 행동(합계 36.41회), TomatoHarvesting에서는 11.28회 질문, 14.94회 물리적 행동(합계 26.22회)을 사용한다. 이 평균들은 모두 50보다 작다. 결정적인 확인 대상은 **실패한 trial이 종료 시점에 질문 때문에 총 decision budget 50을 소진했는가**이다. 예산 소진 실패가 거의 없다면 이 차이는 설계상 우려 사항이지 관측된 우위의 주된 원인이라고 말하기 어렵다.

**권장 검증:** 동일한 physical-action budget 50을 허용하고, 질문에 대해 양쪽에 동일한 별도 query limit 또는 비용을 부여해 Query-Action을 재평가한다. 계산 자원이 부족하면 최소한 모든 실패 trial을 `decision limit`, `physical action limit`, `planner dead end`, `wrong action`, `other` 등으로 분류하고, query count와 physical-action count의 분포를 제시한다. 이후 해당 예산 차이가 어느 정도 영향을 미쳤는지 보수적으로 해석한다.

### 다른 discount factor와 query cost (Appendix B.2–B.3, pp. 26–27)

Ours의 일반적인 설정은 \(\gamma=0.2\)이며, Query-Action은 WasteSorting에서 0.9, TomatoHarvesting에서 0.5를 쓴다. Query-Action은 query cost 0으로 설정된다. 즉 한쪽은 비슷한 수의 질문을 하더라도 미래 보상에 대한 평가 방식이 다르다. 할인율과 질문 비용은 reward-based action selection을 직접 바꾸기 때문에 실험의 해석에 영향을 미칠 수 있다.

하지만 서로 다른 알고리즘에 같은 \(\gamma\)를 무조건 강제해야 한다는 뜻은 아니다. 각 방법에 가장 적절한 설정을 충분히 튜닝했다면 **시스템 비교**로는 타당할 수 있다. 다만 `query policy만 바꿔서 비교했다`거나 성능 차이를 `state confidence의 효과`라고 직접 귀속하기는 어렵다. 왜 각 값을 선택했는지, 탐색한 후보와 선택 기준이 무엇인지 명시하자.

또한 Query-Action이 질문을 POMCP action space에 포함해 100 simulations 안에서 평가하고, Ours는 질문을 별도로 처리한 뒤 physical actions만 탐색한다는 차이는 방법 자체의 일부이기도 하다. 이 결과는 실행된 두 **시스템의 효율성 차이**를 보여줄 수 있지만, `정보 기준만의 순수한 효과`와 혼동하면 안 된다.

### KnowNo·IntroPlan과의 비교는 알고리즘 전반이 함께 바뀐다 (Section 5.2.1, Appendix B.2)

KnowNo/IntroPlan은 GPT-4o로 action candidates를 생성·평가한다. Ours는 symbolic model 및 POMCP 기반의 행동 선택을 한다. 따라서 query trigger나 질문의 형식뿐 아니라 **planning mechanism, candidate generation, failure modes, latency**가 함께 달라진다. Ours의 성공률이 더 높다는 사실은 유효한 시스템 수준 결과다. 그러나 그 차이가 오직 `state verification 질문을 했기 때문`이라고 귀속할 수는 없다.

이를 해결하는 핵심 근거는 이미 수행한 **when/what ablation**이다. 동일한 계획 구조에서 query timing과 selection을 바꾼 실험으로 각 요소의 역할을 주장하고, KnowNo/IntroPlan과의 비교는 end-to-end 시스템 성능 비교로 명시하자.

### Ablation도 완전히 한 요인만 바뀌는 것은 아니다 (Appendix B.3)

W2는 conformal action prediction set을 질문마다 다시 평가한다. 반면 다른 여러 정책은 confidence가 threshold에 도달할 때까지 추가 질문을 반복한다. 따라서 W2와 Ours의 비교는 최초 trigger뿐 아니라 **query episode의 종료 정책**도 달라진다. `timing-only` 비교라고 부르기보다는 query-triggering policy 비교로 설명하거나, 가능하면 종료 조건을 통제한 추가 실험을 제시하면 좋다.

### 질문 수·planning time 집계 방식의 해석 (Section 5.1.3, Appendix B.7)

성공률은 전체 시도한 trial을 사용하지만, query count, plan length, time 등은 **성공한 trial만** 사용한다. 성공률이 낮은 KnowNo 등의 query mean은 성공한 일부 trial의 비용이므로, 전체 시행에서 얼마나 자주 사람을 불렀는지와 동일하지 않다. 성공 trial 조건부 평균을 유지하되, 전체 attempted trials 평균 질문 횟수도 함께 제시하면 이 선택 편향을 줄일 수 있다.

Planning-time 정의 또한 다르다. Oracle 비교에서 KnowNo/IntroPlan의 language model inference 시간과 Ours/Query-Action의 POMCP search 시간이 비교된다. Physical robot 평가에서는 직접 검색 시간을 재는 대신 residual estimate를 사용한다고 Appendix에 기술되어 있다. 따라서 빠른 planning 결과를 `새 belief 알고리즘 자체가 더 빠르다`고 해석하기보다, `평가된 구현에서 반복적인 LLM action-generation을 요구하지 않아 latency가 낮다`고 표현해야 한다.

## 실험 결과 해석: 강한 주장과 제한할 주장

### Table 4의 메시지는 충분히 강하다

Confidence timing에서 WasteSorting의 Q1, Q2, Ours 성공률은 모두 100%이고, TomatoHarvesting에서는 97.5%, 97.5%, 98%다. 그러나 질문 횟수는 WasteSorting에서 15.21, 22.79, 7.83회, TomatoHarvesting에서 12.68, 15.25, 9.16회로 달라진다. 이 결과는 **성공률이 비슷한 조건에서 entropy gain selection이 질문 수를 줄인다**는 주장을 강하게 뒷받침한다.

반대로 W3의 WasteSorting 99%와 Ours 100%는 200회 중 두 trial 차이다. TomatoHarvesting Q1 97.5%와 Ours 98%는 단 한 trial 차이다. 이러한 작은 차이는 통계적 불확실성을 고려해 `유사한 수준`이라고 해석하는 편이 안전하다. 제안 방식의 진정한 강점은 해당 작은 성공률 차이가 아니라 **같거나 비슷한 성공 수준에서 질문 수를 절약한 것**이다.

### Reward-based timing의 실패 원인을 단정하지 않는다

Section 5.2.2의 `reward-based timing ... fail to verify necessary state information at the appropriate time`라는 서술은 원인을 특정한다. Ablation 덕분에 정책에 따른 성능 차이는 확인할 수 있지만, 실제로 **어느 시점의 어느 사실을 확인하지 않았는지**를 기록한 분석이 없다면 그 실패 원인까지 단언할 수는 없다.

권장 문장:

> These results indicate that reward-based query timing does not consistently provide the state verification needed for successful execution, even when it produces more queries overall.

해당 실패 원인까지 주장하려면 실패 trial의 `high uncertainty → no query → inappropriate action → failure` 사례를 추적하는 분석이 추가로 필요하다.

### Threshold는 trade-off의 관점에서 설명한다

\(\tau=0.8\)은 두 도메인에서 높은 성공률을 제공하지만 TomatoHarvesting의 최고 성공률은 더 높은 threshold에서 나온다. 따라서 `최적 threshold`보다는 **두 도메인에 공통 적용 가능한 성공률–질문 횟수 절충점**이라고 설명하자. Threshold를 선택한 데이터와 그 성능을 보고하는 데이터가 겹친다면 탐색적 분석인지, 별도 validation으로 선택한 것인지 구분한다.

### Abstract의 query-count 표현을 좁힌다

`higher task success without substantially increasing the number of queries`는 모든 baseline과의 비교에서는 성립하지 않는다. Ours는 KnowNo보다 훨씬 많은 질문을 사용한다. 대신 `(i) reward-based Query-Action 대비 더 높은 성공률과 적은 질문`, `(ii) confidence timing 안에서 entropy gain이 질문 횟수 절감`, `(iii) KnowNo 대비 더 높은 성공률이지만 더 많은 질문`을 분리해 기술하는 것이 정확하다.

## 원고의 구성과 논리

Introduction은 `기존 방법은 state uncertainty를 사용하지 않는다`고 단정하기보다, 기존 접근은 **action prediction uncertainty 또는 expected task reward를 query criterion으로 삼는 반면**, 제안 방법은 **symbolic state의 confidence와 expected entropy reduction을 명시적 기준으로 사용한다**고 구분하자. Reward-based POMDP도 state belief를 다룰 수 있으므로 이를 전면 부정하지 않는 것이 중요하다.

Related Work의 Section 2.1은 POMDP와 particle method의 일반 배경을 압축하고, Section 2.2에서 action-level clarification과 state-level verification의 차이를 더 분명히 제시하면 좋다.

현재 Section 5.2.1 안에 Oracle, VLM, Human, cross-source comparison이 함께 있고, 뒤에 ablation과 threshold가 나온다. 보다 자연스러운 대안은 `Oracle baseline → When/What ablation → Threshold → Physical robot VLM/Human → Interviews → Discussion` 순서다. 이는 필수 변경 사항은 아니지만 논문의 중심 기여를 먼저 보여준다는 장점이 있다.

Fig. 2는 when–what–feedback loop를 더 강조하고 Fig. 3은 하드웨어/시스템 구조에 집중해 중복 정보를 줄이자. Fig. 5와 Fig. 6에서는 success 및 query count에 시각적 비중을 더 두고, 나머지 지표는 표에서 보완하면 읽기 쉽다. 일반적인 POMDP 개념을 설명하는 Fig. 1의 비중도 줄일 수 있다.

미해결 표기 `Institution2`, `Appendix ??`, `Sec. ??`, confidence 설명 뒤의 `?`, 일부 실험실 표기, 인용·저자 명세 오류 등을 최종본에서 정리해야 한다. Plan sequence와 policy에서 사용하는 기호가 같은 경우도 혼동 가능성을 점검하자.

## 인간 실험과 인터뷰: 아직 완료 전이라는 전제

Human evaluation과 Interviews는 현재 **임시 결과 및 미작성 단락**이므로, 현 수치에서 일반적 결론을 도출하지 않는다. 본문에 있는 placeholder demographics는 반드시 실제 수집 자료로 교체해야 하며, 최종 참가자 수와 전체 시행 수가 Appendix D와 일치해야 한다.

인터뷰는 단순히 참가자별 발화를 나열하기보다 **질문 시점의 맥락 이해**, **state verification 대 action selection의 판단 부담**, **질문 대상의 명확성**, **작업 흐름 중단 및 반복 질문 경험**과 같이 실제 응답에서 확인된 주제별로 조직하는 것이 좋다. 실제 발화가 이를 뒷받침할 때만 결과로 서술하고, 예상과 다른 사례도 포함하자. 별도의 음성 파일 제출을 전제로 구성할 필요는 없다.

## 영문 핵심 문단 수정 초안

아래 내용은 현재 원고를 기반으로 한 **수정 방향 예시**다. 특히 Abstract의 Human 부분은 최종 결과가 나온 뒤 조정해야 하며, Method의 용어는 실제 구현 확인을 전제로 한다.

### Abstract

We present a method for determining when a robot should request external information and what it should ask during task execution under state uncertainty. Resolving every ambiguity through external queries incurs interaction costs, whereas executing actions without sufficient state information can lead to task failure. Our method combines online symbolic planning with a weighted belief over reachable successor states. After each action and observation, the uncertainty of this belief determines whether state verification is required, and expected entropy reduction identifies a Boolean state fact to query. The response is incorporated into the belief before subsequent planning and execution. We evaluate the method in waste sorting and tomato harvesting using oracle feedback and physical robot experiments with vision-language model (VLM) feedback. Ablation results show that confidence-based query timing supports high task success across different query selection strategies, while entropy-based selection reduces the number of queries at comparable success rates. Compared with a reward-based query-action strategy, our method achieves higher task success with fewer queries. Physical robot experiments with VLM feedback further demonstrate higher task success than an action-selection-based baseline, although the proposed method requires more queries. These results highlight the distinct roles of query timing and query selection in robot task execution under uncertainty.

### Introduction: Research Gap

Existing selective querying methods determine when assistance is needed using prediction confidence, calibrated action sets, or the expected utility of external information. These approaches provide different criteria for identifying decisions that may benefit from assistance. However, uncertainty over candidate actions and the expected reward of querying do not directly specify how much uncertainty remains in the robot's current symbolic task state. In tasks where action applicability depends on uncertain state facts, explicitly evaluating this uncertainty provides an alternative basis for deciding when to request information. Furthermore, selecting which state fact to verify requires identifying information that can distinguish among the remaining state hypotheses. We therefore determine when to query from confidence in the current task state and what to query from the expected reduction in state uncertainty.

### Discussion: Quantitative Findings

The ablation experiments indicate that query timing and query selection play different roles in the evaluated tasks. When query timing is determined by confidence in the current state, random, reward-based, and entropy-based query selection all achieve high task success. However, these strategies differ substantially in the number of queries required. Entropy-based selection reduces query count while maintaining comparable success rates. In contrast, entropy-based selection alone does not recover high success when query timing remains random. These results suggest that the query trigger is important for determining whether additional state information is obtained, while query selection influences how efficiently the remaining uncertainty is resolved.

The threshold experiments further demonstrate the trade-off between state verification and interaction cost. Higher confidence requirements generally improve task success, but the strictest threshold substantially increases the number of queries with limited additional improvement. The selected threshold of 0.8 provides a common operating point that achieves high success in both domains without requiring complete confidence before execution continues.

Physical robot experiments with VLM feedback show that the proposed method maintains higher task success than KnowNo in both domains. However, the improvement is accompanied by additional queries, and its effect on total execution time differs across tasks. In WasteSorting, the proposed method uses more interaction time and slightly more total time. In TomatoHarvesting, it achieves lower total time despite using more queries. These observations indicate that query frequency alone does not determine execution cost. The reasoning required to provide feedback, the subsequent action sequence, and the number of repeated decisions can also influence overall performance. Because the compared systems use different action-selection mechanisms, these results should be interpreted as system-level comparisons rather than an isolated effect of query representation.

## 수정 우선순위와 주말 작업 제안

**최우선(구현 확인 필요):** Section 4의 local weighted frontier가 진정한 posterior propagation인지 확인하고, Algorithm 1·Appendix A의 belief와 수식을 일치시킨다. 이어서 Query-Action 실패 trial의 종료 사유 및 decision budget 소진 여부를 확인한다.

**높은 우선순위(기존 결과·로그로 가능):** 전체 attempted trials 기준 query count를 보조 지표로 제시한다. W2의 stopping rule 차이를 명시한다. Reward-based timing의 실패 원인에 대한 과도한 인과적 단정을 줄인다. Discount factor와 baseline tuning protocol의 차이를 투명하게 기술한다.

**영문·구조 작업:** Abstract와 Introduction에서 query-count 주장을 비교 대상별로 구분하고, Discussion에서 시스템 수준 비교와 when/what의 개별 효과를 분리한다. 필요하다면 실험 절 순서를 재배치한다.

**Human experiment 이후:** 모집 결과, 실제 시행 수, 인터뷰에서 발견된 주제 및 근거 발화를 반영한다. 인터뷰 해석은 사전 가설이 아니라 수집된 자료에 근거하도록 한다.

## 최종 판단

핵심 방향을 바꿀 필요는 없다. 이 원고는 `언제 질문하는가`가 성공률과 밀접하고 `무엇을 질문하는가`가 높은 성공 수준에서의 질문 효율성에 영향을 준다는 실험적 메시지가 이미 선명하다. 다만 이 메시지의 설득력을 최대화하려면 **belief formulation의 정확성**과 **baseline의 실행 예산 차이**를 먼저 해결해야 한다. 비교 조건의 차이를 인정하고 필요한 추가 검증을 수행하는 것은 Ours의 성능을 약화하는 것이 아니라, 무엇이 실제로 입증되었는지를 더 분명하게 만드는 작업이다.
