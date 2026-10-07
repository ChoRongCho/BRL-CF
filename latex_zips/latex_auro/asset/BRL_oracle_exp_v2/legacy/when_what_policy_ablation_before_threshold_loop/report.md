# When–What policy comparison (Value rerun: 2026-10-05) — 분석 보고서

Ours・Action-CP When은 2026-09-21~22의 각 400회를 유지하고, Value-When・Value-What은 2026-10-05 배치의 각 400회로 교체했다. Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours・CP-When은 gamma=0.2이다. gamma는 질문 가치 평가기와 실제 행동용 POMCP 모두에 적용됐다. Value-When의 물리 행동당 최대 1개 질문 제한은 유지됐다. 따라서 질문 시점만 바꾼 통제 비교로 해석하지 않는다. CP-When의 기존 형식 파싱 오류 4건은 성공률에서 실패로 집계한다. 이전 Value 배치와 새 배치는 합산하지 않고 별도 paired 비교로 제공한다.

실행 슬롯 1,600개, 유효 결과 1,596개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 핵심 결과

1. Value-When 성공률 70.00% (280/400), 전체 평균 질문 8.902회, 성공 실행 평균 10.593회.
2. Value-What 성공률 98.75% (395/400), 전체 평균 질문 18.988회, 성공 실행 평균 19.068회.
3. 설정 변경 전 대비 Value-When 성공률은 60.25%에서 70.00%로, Value-What은 98.25%에서 98.75%로 변했다. gamma와 cost를 동시에 바꿨으므로 개별 효과를 분리하지 않는다.
4. Value-When의 질문 1개 제한은 실제 모든 step에서 확인됐다. 의도했던 공통 threshold 반복 구조를 복구한 실험은 아니다.

## 전체·도메인별 결과

| Domain | Condition | 성공/실행 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 396/400 | 99 | 8.49 | 12.5 | 1.13 | 0 |
| all | cp_when | 274/400 | 68.5 | 4.8 | 11.4 | 31 | 4 |
| all | value_when | 280/400 | 70 | 8.9 | 11.7 | 4.33 | 0 |
| all | value_what | 395/400 | 98.8 | 19 | 13.6 | 6.74 | 0 |
| tomato | ours | 196/200 | 98 | 9.14 | 14.7 | 1.28 | 0 |
| tomato | cp_when | 126/200 | 63 | 3.38 | 13.3 | 33.7 | 2 |
| tomato | value_when | 131/200 | 65.5 | 7.95 | 13 | 4.33 | 0 |
| tomato | value_what | 195/200 | 97.5 | 15.2 | 15.4 | 4.89 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.982 | 0 |
| wastesorting | cp_when | 148/200 | 74 | 6.22 | 9.55 | 28.3 | 2 |
| wastesorting | value_when | 149/200 | 74.5 | 9.86 | 10.4 | 4.34 | 0 |
| wastesorting | value_what | 200/200 | 100 | 22.8 | 11.8 | 8.59 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/실행 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 39/40 | 8.7 | 15.1 |
| tomato | 02 | ours | 40/40 | 9.12 | 13.8 |
| tomato | 03 | ours | 39/40 | 9.05 | 14.8 |
| tomato | 04 | ours | 39/40 | 9.85 | 14.9 |
| tomato | 05 | ours | 39/40 | 9 | 14.8 |
| tomato | 01 | cp_when | 26/40 | 3.92 | 12.8 |
| tomato | 02 | cp_when | 20/40 | 3.15 | 14.2 |
| tomato | 03 | cp_when | 26/40 | 3.03 | 11.7 |
| tomato | 04 | cp_when | 27/40 | 3.52 | 13.5 |
| tomato | 05 | cp_when | 27/40 | 3.27 | 14.2 |
| tomato | 01 | value_when | 30/40 | 8.7 | 14.8 |
| tomato | 02 | value_when | 26/40 | 7.7 | 12.3 |
| tomato | 03 | value_when | 26/40 | 7.62 | 12.6 |
| tomato | 04 | value_when | 26/40 | 7.92 | 12.8 |
| tomato | 05 | value_when | 23/40 | 7.8 | 12.4 |
| tomato | 01 | value_what | 38/40 | 14.6 | 16.2 |
| tomato | 02 | value_what | 40/40 | 15 | 14.6 |
| tomato | 03 | value_what | 40/40 | 15.4 | 14.5 |
| tomato | 04 | value_what | 37/40 | 14.9 | 16.5 |
| tomato | 05 | value_what | 40/40 | 16 | 15.3 |
| wastesorting | 01 | ours | 40/40 | 6.42 | 9.5 |
| wastesorting | 02 | ours | 40/40 | 7.17 | 10.4 |
| wastesorting | 03 | ours | 40/40 | 8.03 | 10.5 |
| wastesorting | 04 | ours | 40/40 | 9.03 | 10.6 |
| wastesorting | 05 | ours | 40/40 | 8.5 | 10.7 |
| wastesorting | 01 | cp_when | 30/40 | 6.12 | 8.55 |
| wastesorting | 02 | cp_when | 29/40 | 5.22 | 9.8 |
| wastesorting | 03 | cp_when | 33/40 | 7.03 | 10.2 |
| wastesorting | 04 | cp_when | 26/40 | 6.1 | 9.23 |
| wastesorting | 05 | cp_when | 30/40 | 6.65 | 9.95 |
| wastesorting | 01 | value_when | 30/40 | 9.5 | 10.1 |
| wastesorting | 02 | value_when | 30/40 | 9.85 | 10.3 |
| wastesorting | 03 | value_when | 31/40 | 10.4 | 10.8 |
| wastesorting | 04 | value_when | 29/40 | 9.82 | 10.4 |
| wastesorting | 05 | value_when | 29/40 | 9.72 | 10.2 |
| wastesorting | 01 | value_what | 40/40 | 21.2 | 11.7 |
| wastesorting | 02 | value_what | 40/40 | 22.3 | 11.8 |
| wastesorting | 03 | value_what | 40/40 | 22.6 | 11.8 |
| wastesorting | 04 | value_what | 40/40 | 23.8 | 11.9 |
| wastesorting | 05 | value_what | 40/40 | 24.1 | 12 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | cp_when | 396 | 120 | 2 | 29.8 | 2.82e-33 | 8.47e-33 | 3.68 |
| all | value_when | 400 | 118 | 2 | 29 | 1.09e-32 | 2.19e-32 | -0.415 |
| all | value_what | 400 | 4 | 3 | 0.25 | 1 | 1 | -10.5 |
| tomato | cp_when | 198 | 70 | 2 | 34.3 | 1.11e-18 | 3.34e-18 | 5.78 |
| tomato | value_when | 200 | 67 | 2 | 32.5 | 8.19e-18 | 1.64e-17 | 1.2 |
| tomato | value_what | 200 | 4 | 3 | 0.5 | 1 | 1 | -6.04 |
| wastesorting | cp_when | 198 | 50 | 0 | 25.3 | 1.78e-15 | 3.55e-15 | 1.59 |
| wastesorting | value_when | 200 | 51 | 0 | 25.5 | 8.88e-16 | 2.66e-15 | -2.02 |
| wastesorting | value_what | 200 | 0 | 0 | 0 | 1 | 1 | -15 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- 이 패키지는 실행 오류를 성공률에서 실패로 집계한다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours (99.00%), 최소 평균 질문은 cp_when (4.803회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | query cost | simulations | threshold | 모델 |
|---|---|---|---|---|---|
| ours | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
| cp_when | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
| value_when | 0.5, 0.9 | 0.0 | 100 | 0.8 | 미기록 |
| value_what | 0.5, 0.9 | 0.0 | 100 | 0.8 | 미기록 |

## 실행 오류 원본

- tomato / cp_when / scene 02 / seed 2138595888: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005/00_raw/tomato/scene_02/cp_when/run_29_seed_2138595888/console.log`
- tomato / cp_when / scene 03 / seed 1447435166: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005/00_raw/tomato/scene_03/cp_when/run_12_seed_1447435166/console.log`
- wastesorting / cp_when / scene 03 / seed 1682900593: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005/00_raw/wastesorting/scene_03/cp_when/run_23_seed_1682900593/console.log`
- wastesorting / cp_when / scene 04 / seed 1445352403: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005/00_raw/wastesorting/scene_04/cp_when/run_18_seed_1445352403/console.log`

## 2026-10-05 재실험과 이전 Value 배치의 paired 비교

domain・scene・seed가 일치하는 새 실행과 이전 실행을 대응했다. 차이는 새 값 − 이전 값이다. Holm 보정은 각 domain에서 두 Value 조건에 적용했다.

| Domain | Condition | 쌍 수 | 이전 성공 | 새 성공 | 변화(%p) | 새 실행만 성공 | 이전만 성공 | exact p | Holm p | 질문 변화(전체 평균) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | value_when | 400 | 241 | 280 | 9.75 | 113 | 74 | 0.005315 | 0.01063 | 7.888 |
| all | value_what | 400 | 393 | 395 | 0.50 | 6 | 4 | 0.7539 | 0.7539 | 6.188 |
| tomato | value_when | 200 | 127 | 131 | 2.00 | 47 | 43 | 0.752 | 1 | 6.725 |
| tomato | value_what | 200 | 193 | 195 | 1.00 | 6 | 4 | 0.7539 | 1 | 2.420 |
| wastesorting | value_when | 200 | 114 | 149 | 17.50 | 66 | 31 | 0.0004902 | 0.0009803 | 9.050 |
| wastesorting | value_what | 200 | 200 | 200 | 0.00 | 0 | 0 | 1 | 1 | 9.955 |

## 실제 질문 구조 점검

Value-When은 질문 1개 제한 때문에 종료한 step이 3,561개이며, 그중 답변 후 confidence가 0.8 미만인 step은 760개이다. 모든 Value-When step에서 질문 수 ≤ 1을 확인했다.
Value-What은 confidence를 반복 평가하며, 갱신된 belief에서 질문 Q를 다시 계산한다. 고정 1개 제한은 없다.

| Condition | Stop reason | Steps |
|---|---|---:|
| value_what | no_query_candidate | 396 |
| value_what | not_triggered | 3354 |
| value_what | when_stopped:at_or_above_threshold | 1699 |
| value_when | not_triggered | 1107 |
| value_when | question_limit_reached | 3561 |

## 설정 차이와 결과 해석

새 Value 두 조건은 Tomato gamma=0.5, Waste gamma=0.9, query cost=0.0이다. 기본 POMCP와 가치 평가기 모두 동일 args.gamma를 사용하므로 행동 계획과 질문 가치가 함께 변했다. n_simulations의 기본값은 100이며 root 후보 수가 많으면 가치 평가 예산은 후보 수+1까지 증가할 수 있다.
epsilon=0.005, max_depth=20에서 Tomato는 depth 8에 할인 종료하고 Waste는 depth 20 상한으로 종료한다. 이전 gamma=0.2는 depth 4에 할인 종료했다. 따라서 이 결과는 cost 감소, 미래 보상 가중치 변화, 탐색 범위 변화가 함께 반영된 설정 비교이다.
Ours/CP는 기존 실행을 유지했다. 서로 다른 gamma와 질문 반복 규칙 때문에 이 네 조건 비교로 When/What만의 인과 효과를 주장하지 않는다. Value-When은 공통 threshold 질문 반복을 복구한 뒤 별도 재실험해야 그 설계를 평가할 수 있다.
실행 cumulative reward는 할인 없는 물리 행동 보상 합이며 질문 비용을 별도 차감하지 않는다. 시간은 각 로그의 episode total_time으로, 인간 응답 시간은 포함하지 않는다.

## 생성물

집계 및 paired 비교는 `01_processed/`, 그림 입력은 `02_graph_data/`, PNG/PDF는 `03_figures/`에 저장했다. `paper_table.tex`는 이 패키지의 최신 요약표이며, 이전 배치 전체는 `../legacy/when_what_policy_ablation_before_20261005/`에 보존했다.
