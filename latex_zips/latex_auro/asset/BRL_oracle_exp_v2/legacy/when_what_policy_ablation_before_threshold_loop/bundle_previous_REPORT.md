# 실험 결과 통합 보고서

작성 기준일: 2026-10-05 (Value 정책 재실험 반영)  
상세 파라미터: [`EXPERIMENT_SETTINGS.md`](EXPERIMENT_SETTINGS.md)

## 1. 보고서 목적

이 보고서는 `analysis_recent_v2`의 네 핵심 실험을 하나의 논리로 연결한다. 검증하려는
주장은 로봇이 단순히 더 많이 질문해서 성공하는 것이 아니라, 불확실한 상황에서
**언제 질문할지(When)**와 **무엇을 질문할지(What)**를 구분해 결정함으로써 높은
task success와 낮은 human query burden을 함께 달성한다는 것이다.

실험은 다음 질문에 차례로 답한다.

1. Confidence threshold가 성공률과 질문 수를 어떻게 바꾸는가?
2. Proposed When과 What은 각각 어떤 역할을 하는가?
3. CP 또는 action-value 기반 대안도 같은 효과를 내는가?
4. 완성된 방법은 기존 query baseline보다 나은가?

## 2. 실험 구성

| 실험 | 비교 조건 | 실행 슬롯 | 유효 결과 | POMCP simulations |
|---|---|---:|---:|---:|
| Threshold sweep | threshold 0.0–1.0 | 4,400 | 4,400 | 100 |
| When–What random ablation | Random / What only / When only / Ours | 1,600 | 1,600 | 100 |
| When–What policy ablation | Ours / CP-When / Value-When / Value-What | 1,600 | 1,596 + 오류 4건 | 100 |
| Baseline comparison | Ours / Query-Action / KnowNo / IntroPlan / selected Query-Action | 2,000 | 2,000 | POMCP 조건 100 |

두 domain(Tomato, Waste Sorting), domain별 5개 scene, 조건·scene별 40회 실행을
사용한다. 성공률 오차막대는 95% Wilson interval이고 연속 지표는 mean ± SE다.
Policy ablation의 CP-When parsing 오류 4건은 task failure로 성공률 분모에 포함한다.
성공률은 전체 실행을 분모로 계산하고, 질문 수와 step 수는 **성공한 episode만**을
대상으로 집계한다.

## 3. Threshold sweep

![Threshold별 task success](00_threshold/03_figures/success.png)

![Threshold별 성공 episode의 질문 수](00_threshold/03_figures/questions_success_only.png)

질문이 거의 발생하지 않는 threshold 0.0–0.3에서는 성공률이 약 53%에 머물렀다.
Threshold가 0.7 이상이 되면 성공률이 급격히 상승하며, 질문을 허용하는 범위와 task
success가 직접 연결됨을 보여준다.

| Threshold | 성공/전체 | 성공률 | 성공 episode당 질문 수 |
|---:|---:|---:|---:|
| 0.0 | 214/400 | 53.50% | 0.00 |
| 0.7 | 385/400 | 96.25% | 7.53 |
| **0.8** | **388/400** | **97.00%** | **8.46** |
| 0.9 | 399/400 | 99.75% | 9.37 |
| 1.0 | 400/400 | 100.00% | 16.98 |

후속 실험에는 성공률과 질문 비용을 함께 고려한 절충값으로 **threshold 0.8**을 사용했다.
이 설정은 97.0%의 성공률을 달성하면서, 항상 질문을 허용하는 1.0보다 성공 episode당
질문을 8.52회 줄인다. Threshold sweep의 목적은 하나의 절대 최적값을 주장하는 것이
아니라, 0.8을 높은 성공률과 제한된 질문 비용 사이의 operating point로 정하는 것이다.

## 4. When과 What의 기능적 기여

![When–What random ablation의 task success](01_when_what_random/03_figures/success.png)

![When–What random ablation의 성공 episode 질문 수](01_when_what_random/03_figures/questions_success_only.png)

2×2 ablation은 When과 What이 서로 다른 지표에 기여함을 보여준다.

| 조건 | When | What | 성공률 | 성공 episode당 질문 수 |
|---|---|---|---:|---:|
| Random | Random | Random | 73.50% | 9.65 |
| What only | Random | Ours | 73.50% | 7.04 |
| When only | Ours | Random | 98.75% | 13.96 |
| **Ours** | **Ours** | **Ours** | **98.75%** | **8.47** |

### When의 역할

Random What을 고정하고 proposed When을 적용하면 성공률이 73.50%에서 98.75%로
**25.25%p 상승**한다. Proposed What을 고정한 비교에서도 73.50%에서 98.75%로
같은 폭만큼 상승한다. 즉 이 결과에서 When은 질문 횟수를 최소화하는 요소라기보다,
질문이 필요한 순간을 포착해 **task failure를 방지하는 요소**다.

### What의 역할

Random When에서 proposed What으로 교체하면 성공률은 73.50%로 유지되면서 성공
episode당 질문이 9.65회에서 7.04회로 감소한다. Proposed When을 고정하면 성공률은
98.75%로 유지되고 질문은 13.96회에서 8.47회로 **5.50회(39.4%) 감소**한다. 따라서 What의 기여는
추가 성공률보다 **같은 성공 수준을 더 적은 질문으로 달성하는 것**이다.

이 결과가 contribution과 가장 직접적으로 연결된다. When은 reliability를, What은
query efficiency를 담당하며, 두 결정을 결합해야 높은 성공률과 제한된 질문 비용을
동시에 얻는다.

## 5. When–What policy comparison (2026-10-05 Value 재실험)

![Task success](02_when_what_policy_ablation/03_figures/success.png)

![성공 episode 질문 수](02_when_what_policy_ablation/03_figures/questions_success_only.png)

Ours・CP-When의 기존 결과는 유지하고 Value-When・Value-What의 각 400회를 새 배치로 교체했다. 이전 Value 결과는 합산하지 않고 legacy에 보관했다.

| 조건 | 성공/전체 | 성공률 | 성공 episode당 질문 수 | 전체 평균 질문 수 |
|---|---:|---:|---:|---:|
| ours | 396/400 | 99.00% | 8.49 | 8.49 |
| cp_when | 274/400 | 68.50% | 5.69 | 4.80 |
| value_when | 280/400 | 70.00% | 10.59 | 8.90 |
| value_what | 395/400 | 98.75% | 19.07 | 18.99 |

| 도메인 | Value-When 성공률 | Value-What 성공률 |
|---|---:|---:|
| tomato | 65.50% | 97.50% |
| wastesorting | 74.50% | 100.00% |

Value 조건의 설정은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours・CP-When은 기존 gamma=0.2 결과이다. 실제 물리 행동용 POMCP의 gamma도 함께 바뀌었다.

이전 대비 Value-When 성공률은 60.25% → 70.00%(+9.75%p), Value-What은 98.25% → 98.75%(+0.50%p)이다. 같은 domain・scene・seed의 paired 검정에서 Value-When의 전체 개선은 Holm p=0.01063, Waste 개선은 Holm p=0.0009803이다. Tomato Value-When과 Value-What의 개선은 이 검정에서 유의하지 않았다.

Value-What과 Ours의 paired 성공률 차이는 0.25%p, exact McNemar p=1.0이다. 성공률 동등성을 입증하는 결과는 아니며, 관측된 성공률은 비슷했다. 성공 episode 평균 질문은 Value-What 19.07회, Ours 8.49회로 약 10.58회 차이가 난다.

Value-When은 물리 행동당 1개 질문 제한을 유지했고, 답변 후 confidence가 0.8 미만인데 제한으로 종료한 step이 760개다. CP-When도 답변마다 CP gate를 재평가한다. 따라서 이 비교는 의도한 공통 threshold 질문 반복 구조의 통제 ablation이 아니며, 성능 차이를 When/What만의 효과로 해석하지 않는다.

상세 결과, 이전 배치 paired 비교, 실제 질문 구조 점검은 [개별 report.md](02_when_what_policy_ablation/report.md)에 있다. 의도와 구현의 의사코드 비교는 [exp_set.md](02_when_what_policy_ablation/exp_set.md)를 참조한다.

## 6. Baseline comparison

![Query baseline의 task success](03_baseline/03_figures/success.png)

![Query baseline의 성공 episode 질문 수](03_baseline/03_figures/questions_success_only.png)

| 방법 | 성공/전체 | 성공률 | 성공 episode당 질문 수 |
|---|---:|---:|---:|
| **Ours** | **390/400** | **97.50%** | **8.51** |
| Query-Action original | 218/400 | 54.50% | 1.22 |
| KnowNo | 118/400 | 29.50% | 3.92 |
| IntroPlan | 183/400 | 45.75% | 7.77 |
| Query-Action selected | 289/400 | 72.25% | 19.25 |

KnowNo와 IntroPlan은 2026-09-30에 완료한 800회 통합 재실행으로 교체했다. 두
domain 모두 prompt v1, generation temperature 0.3, score temperature 5.0, exact oracle을
사용했다. Ours는 KnowNo보다 68.00%p, IntroPlan보다 51.75%p 높은 성공률을
보였다. Tomato에서는 Ours 95.0%, KnowNo 10.5%, IntroPlan 40.0%였고, Waste에서는
각각 100.0%, 48.5%, 51.5%였다.

성공 episode당 질문 수는 KnowNo가 3.92회, IntroPlan이 7.77회로 Ours의 8.51회보다
적지만, 성공률이 크게 낮다. 따라서 낮은 질문 수를 효율성으로 단독 해석할 수 없다.
실패 로그에서 KnowNo Tomato는 fallback 선택과 미관측 tomato pick이, IntroPlan Tomato는
손에 든 tomato 없이 scan을 선택한 경우가 주요 조기 실패로 관측됐다.

Domain별 gamma와 zero query cost를 적용한 selected Query-Action은 성공률을 72.25%까지
높였지만, 성공 episode당 질문이 19.25회로 증가했다. Ours는 이 조건보다 성공률이
25.25%p 높고 성공 episode당 질문은 10.74회 적었다. Paired 성공률 비교도 유의했다(Holm-adjusted
`p=2.25e-24`). 즉 parameter tuning만으로는 Ours의 성공–질문 조합을 재현하지 못했다.

## 7. 종합 해석

현재 결과가 뒷받침하는 논리적 흐름은 다음과 같다.

1. 질문을 허용하지 않으면 두 domain에서 성공률이 약 53%에 머문다.
2. Proposed When은 질문이 필요한 시점을 식별해 성공률을 약 25%p 높인다.
3. Proposed What은 성공률을 유지하면서 성공 episode의 질문을 최대 39.4% 줄인다.
4. 새 Value-What은 Ours와 비슷한 관측 성공률에서 더 많은 질문을 사용한다. Value-When은 70.0% 성공률이다. 설정과 반복 구조 차이 때문에 이 비교에서 정책만의 인과 효과는 분리하지 않는다.
5. 통합 재실행에서 Ours는 KnowNo와 IntroPlan보다 51.75–68.00%p 높은 성공률을 보였다.
6. Ours는 tuned Query-Action보다 높은 성공률을 보이며, tuned 조건보다 질문 수도 적다.

따라서 결과 중심 contribution은 다음처럼 표현할 수 있다.

> Belief-aware When decision은 불확실성으로 인한 task failure를 줄이고, EIG-based What
> decision은 성공에 필요한 human queries를 줄인다. 두 결정을 결합하면 기존 query
> strategy가 달성하지 못한 reliability–query efficiency trade-off를 얻는다.

## 8. 보고 기준
- 성공률은 전체 실행을 기준으로 계산한다. 질문 수와 step 수는 성공한 episode만을
  대상으로 계산해, 실패에 따른 조기 종료가 효율 지표에 섞이지 않도록 한다.

세부 수치와 검정은 각 실험의 `report.md`, 집계값은 `01_processed/summary.csv`, 그림에
사용된 값은 `02_graph_data/figure_data.csv`에서 확인할 수 있다.
