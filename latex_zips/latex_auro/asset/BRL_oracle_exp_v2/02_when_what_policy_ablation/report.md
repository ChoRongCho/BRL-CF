# When–What policy comparison (Value-When threshold-loop rerun) — 분석 보고서

Value-When은 2026-10-05 20:43 배치 400회로 교체했다. 질문 1개 제한을 제거하고 Q 기반 When은 episode 진입 시 한 번만 평가하며, 진입 후 confidence threshold=0.8로 반복・종료한다. Value-What은 같은 날 19:03 배치 400회 결과를 유지한다. 두 Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours와 CP-When은 기존 각 400회를 유지하고 gamma=0.2이다. 보관된 CP-When 결과는 수정 전 답변마다 CP 재평가 구현의 결과이므로, 현재 수정된 CP 코드의 평가로 해석하지 않는다. 기존 CP parsing 오류 4건은 성공률에서 실패로 집계한다. 이전 Value-When과 최신 배치는 합산하지 않는다.

실행 슬롯 1,600개, 유효 결과 1,596개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 핵심 결과

1. 수정된 Value-When 성공률 83.75% (335/400). 전체 평균 질문 13.870회, 성공 실행 평균 15.406회.
2. 같은 gamma/cost에서 1개 제한을 사용한 이전 Value-When 70.00%보다 13.75%p 높아졌다. Tomato는 65.50% → 68.50%, Waste는 74.50% → 99.00%이다.
3. Value-What은 재실행하지 않고 기존 395/400(98.75%) 결과를 유지했다.
4. 모든 새 Value-When step에서 When을 1회 평가했으며, 질문 1개 제한에 의한 종료는 없다.

## 전체·도메인별 결과

| Domain | Condition | 성공/실행 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 396/400 | 99 | 8.49 | 12.5 | 1.13 | 0 |
| all | cp_when | 274/400 | 68.5 | 4.8 | 11.4 | 31 | 4 |
| all | value_when | 335/400 | 83.8 | 13.9 | 12.3 | 4.79 | 0 |
| all | value_what | 395/400 | 98.8 | 19 | 13.6 | 6.74 | 0 |
| tomato | ours | 196/200 | 98 | 9.14 | 14.7 | 1.28 | 0 |
| tomato | cp_when | 126/200 | 63 | 3.38 | 13.3 | 33.7 | 2 |
| tomato | value_when | 137/200 | 68.5 | 10.1 | 12.9 | 4.51 | 0 |
| tomato | value_what | 195/200 | 97.5 | 15.2 | 15.4 | 4.89 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.982 | 0 |
| wastesorting | cp_when | 148/200 | 74 | 6.22 | 9.55 | 28.3 | 2 |
| wastesorting | value_when | 198/200 | 99 | 17.6 | 11.8 | 5.08 | 0 |
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
| tomato | 01 | value_when | 32/40 | 10.9 | 13.7 |
| tomato | 02 | value_when | 27/40 | 10.2 | 13.1 |
| tomato | 03 | value_when | 29/40 | 9.6 | 12.7 |
| tomato | 04 | value_when | 23/40 | 9.68 | 12.2 |
| tomato | 05 | value_when | 26/40 | 10.2 | 12.7 |
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
| wastesorting | 01 | value_when | 39/40 | 16.9 | 11.4 |
| wastesorting | 02 | value_when | 39/40 | 17.2 | 11.6 |
| wastesorting | 03 | value_when | 40/40 | 18.6 | 11.7 |
| wastesorting | 04 | value_when | 40/40 | 17.7 | 12.2 |
| wastesorting | 05 | value_when | 40/40 | 17.7 | 11.9 |
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
| all | value_when | 400 | 63 | 2 | 15.2 | 1.16e-16 | 2.33e-16 | -5.38 |
| all | value_what | 400 | 4 | 3 | 0.25 | 1 | 1 | -10.5 |
| tomato | cp_when | 198 | 70 | 2 | 34.3 | 1.11e-18 | 3.34e-18 | 5.78 |
| tomato | value_when | 200 | 61 | 2 | 29.5 | 4.37e-16 | 8.75e-16 | -0.98 |
| tomato | value_what | 200 | 4 | 3 | 0.5 | 1 | 1 | -6.04 |
| wastesorting | cp_when | 198 | 50 | 0 | 25.3 | 1.78e-15 | 5.33e-15 | 1.59 |
| wastesorting | value_when | 200 | 2 | 0 | 1 | 0.5 | 1 | -9.79 |
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

- tomato / cp_when / scene 02 / seed 2138595888: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005_threshold_loop/00_raw/tomato/scene_02/cp_when/run_29_seed_2138595888/console.log`
- tomato / cp_when / scene 03 / seed 1447435166: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005_threshold_loop/00_raw/tomato/scene_03/cp_when/run_12_seed_1447435166/console.log`
- wastesorting / cp_when / scene 03 / seed 1682900593: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005_threshold_loop/00_raw/wastesorting/scene_03/cp_when/run_23_seed_1682900593/console.log`
- wastesorting / cp_when / scene 04 / seed 1445352403: `experiments_logs/analysis_recent/when_what_policy_ablation_20261005_threshold_loop/00_raw/wastesorting/scene_04/cp_when/run_18_seed_1445352403/console.log`

## 질문 1개 제한 제거 전후 paired 비교

같은 gamma/cost, domain・scene・seed의 400쌍을 대응했다. 차이는 새 값 − 이전 값이다. 각 domain에서 Value-When 1개 비교의 양측 exact McNemar p를 보고한다. 전체와 domain 검정을 중복 독립 증거로 해석하지 않는다.

| Domain | 이전 성공 | 새 성공 | 변화(%p) | 새 실행만 성공 | 이전만 성공 | exact p | 전체 평균 질문 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 280/400 | 335/400 | 13.75 | 59 | 4 | 1.3821e-13 | 4.968 |
| tomato | 131/200 | 137/200 | 3.00 | 10 | 4 | 0.17957 | 2.175 |
| wastesorting | 149/200 | 198/200 | 24.50 | 49 | 0 | 3.5527e-15 | 7.760 |

## 새 Value-When의 실제 질문 구조

When은 관측 belief 갱신 후 다음 물리 행동 전에 한 번 판단한다. 시작 후 EIG로 질문을 선택하고, 답변 후 confidence < 0.8이면 추가 질문한다. 후보 없음・belief 미감소도 종료 조건이다. 같은 episode에서 같은 사실을 반복하지 않는다.

| Domain | 2개 이상 질문한 steps | step당 최대 질문 |
|---|---:|---:|
| tomato | 276 | 7 |
| wastesorting | 480 | 10 |

| Domain | 종료 이유 | Steps |
|---|---|---:|
| tomato | no_query_candidate | 81 |
| tomato | not_triggered | 978 |
| tomato | threshold_reached | 1513 |
| wastesorting | no_query_candidate | 291 |
| wastesorting | not_triggered | 119 |
| wastesorting | threshold_reached | 1949 |

## 유지한 비교군과 해석 범위

Value-What의 raw source와 SHA-256은 이전 19:03 배치 400개와 모두 동일하다. Ours・CP 참조 결과도 그대로다. CP의 보관된 결과는 현재 공통 threshold 반복으로 수정된 코드의 결과가 아니다. 최신 네 조건은 gamma와 CP 반복 구현이 다르므로 정책만의 완전한 통제 비교로 주장하지 않는다.
Value-When 전후 비교는 같은 gamma/cost에서 질문 1개 제한과 질문 반복 구조 변경을 평가한다. 초기 seed가 같아도 질문・행동 경로가 달라지면 난수 소비가 달라진다. 성공 실행 질문 평균과 전체 평균을 구분해 보고한다.
cumulative reward는 할인 없는 물리 행동 보상 합이고 질문 비용을 별도 차감하지 않는다. 새 배치의 총 성공률과 실패 원인은 task 수준 결과이며 실행 오류는 0건이다.

## 출처 및 재생성

새 실행: `experiments_logs/when_what_policy_ablation/20261005_204332_446690`.
원본 SHA-256, episode CSV와 trace, 그림 집계 일치는 `integrity_check.json`에 기록했다.
```bash
/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_threshold_loop.py
```
