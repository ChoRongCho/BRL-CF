# When–What ablation (gamma = 0.2) — 분석 보고서

2026-09-21에 다시 실행한 1,600 episodes를 분석했다. Tomato와 Waste Sorting에서 장면 1–5를 사용했고, 각 조건은 400회다. 모든 로그의 gamma는 0.2이며 누락, 중복, 실행 오류는 없다.

## 이 실험이 답하는 질문

이 ablation은 질문 정책을 **언제 물을지(When)**와 **무엇을 물을지(What)**로 나누어 각각의 역할을 확인한다.

- When의 효과는 `Random`과 `When only`, 또는 `What only`와 `Ours`의 성공률 차이로 본다.
- What의 효과는 성공률이 비슷한 조건 안에서 성공한 실행의 질문 수 차이로 본다.
- 전체 실행 질문 수도 함께 기록하지만, 실패가 일찍 끝나면 질문할 기회가 줄어들기 때문에 질의 효율의 주된 비교에는 성공 실행 질문 수를 사용한다.

## 핵심 결과

1. **질문 타이밍이 성공률을 결정했다.** Random What을 유지한 채 Ours When을 적용하면 성공률이 68.25%에서 98.50%로 **30.25 percentage points** 증가했다. Ours What을 사용한 조건에서도 69.00%에서 99.75%로 **30.75 percentage points** 증가했다.
2. **질문 내용 선택이 질문 수를 줄였다.** Ours When 조건에서 성공 실행당 질문 수가 14.11회에서 8.64회로 **5.47회, 38.8% 감소**했다. Random When 조건에서도 9.68회에서 7.10회로 **2.58회, 26.7% 감소**했다.
3. **두 구성요소를 결합한 Ours가 가장 좋은 성공–질의 trade-off를 보였다.** 전체 성공률은 99.75%이고, 성공 실행당 질문 수는 8.64회다. When only는 성공률 98.50%로 비슷하지만 질문을 14.11회 사용했다.

## 전체·도메인별 결과

| Domain | Condition | 성공/전체 | 성공률 | 전체 질문 | 성공 시 질문 | 전체 물리 step | 성공 시 물리 step |
|---|---|---:|---:|---:|---:|---:|---:|
| all | Random When + Random What | 273/400 | 68.25% | 8.21 | 9.68 | 11.45 | 12.72 |
| all | Random When + Ours What | 276/400 | 69.00% | 6.05 | 7.10 | 11.74 | 12.85 |
| all | Ours When + Random What | 394/400 | 98.50% | 14.07 | 14.11 | 12.43 | 12.37 |
| all | Ours When + Ours What | 399/400 | 99.75% | 8.68 | 8.64 | 12.55 | 12.46 |
| tomato | Random When + Random What | 136/200 | 68.00% | 7.96 | 8.98 | 13.38 | 14.82 |
| tomato | Random When + Ours What | 138/200 | 69.00% | 6.41 | 7.30 | 14.09 | 15.17 |
| tomato | Ours When + Random What | 195/200 | 97.50% | 12.76 | 12.78 | 14.26 | 14.14 |
| tomato | Ours When + Ours What | 199/200 | 99.50% | 9.44 | 9.36 | 14.70 | 14.52 |
| wastesorting | Random When + Random What | 137/200 | 68.50% | 8.47 | 10.37 | 9.52 | 10.64 |
| wastesorting | Random When + Ours What | 138/200 | 69.00% | 5.69 | 6.89 | 9.39 | 10.52 |
| wastesorting | Ours When + Random What | 199/200 | 99.50% | 15.38 | 15.41 | 10.61 | 10.62 |
| wastesorting | Ours When + Ours What | 200/200 | 100.00% | 7.92 | 7.92 | 10.41 | 10.41 |

## 장면별 반복 경향

각 셀은 `성공률 / 성공 실행 질문 수`다. When의 성공률 효과와 What의 질문 절감 효과가 두 도메인의 모든 장면에서 반복된다.

| Domain | Scene | Random | What only | When only | Ours |
|---|---:|---:|---:|---:|---:|
| tomato | 01 | 75.0% / 8.9 | 70.0% / 7.2 | 97.5% / 12.1 | 100.0% / 9.1 |
| tomato | 02 | 65.0% / 9.2 | 65.0% / 7.8 | 95.0% / 13.2 | 97.5% / 9.4 |
| tomato | 03 | 70.0% / 8.7 | 72.5% / 7.1 | 100.0% / 13.6 | 100.0% / 9.8 |
| tomato | 04 | 67.5% / 9.0 | 70.0% / 7.2 | 97.5% / 12.7 | 100.0% / 9.3 |
| tomato | 05 | 62.5% / 9.1 | 67.5% / 7.2 | 97.5% / 12.3 | 100.0% / 9.1 |
| wastesorting | 01 | 72.5% / 10.7 | 72.5% / 6.9 | 100.0% / 12.2 | 100.0% / 7.0 |
| wastesorting | 02 | 67.5% / 10.7 | 70.0% / 6.9 | 97.5% / 15.3 | 100.0% / 7.6 |
| wastesorting | 03 | 80.0% / 8.9 | 80.0% / 6.4 | 100.0% / 14.7 | 100.0% / 7.9 |
| wastesorting | 04 | 60.0% / 10.5 | 60.0% / 6.5 | 100.0% / 17.6 | 100.0% / 8.4 |
| wastesorting | 05 | 62.5% / 11.6 | 62.5% / 7.8 | 100.0% / 17.2 | 100.0% / 8.7 |

## 실패와 지표 해석

| Condition | 실패 | Plan failure | Max step |
|---|---:|---:|---:|
| Random When + Random What | 127 | 125 | 2 |
| Random When + Ours What | 124 | 120 | 4 |
| Ours When + Random What | 6 | 5 | 1 |
| Ours When + Ours What | 1 | 0 | 1 |

전체 실행 평균만 보면 Random의 질문 수가 8.21회로 Ours의 8.68회보다 작다. 이는 Random이 더 효율적이어서가 아니라 127회가 과제를 끝내기 전에 종료되어 질문 기회가 줄었기 때문이다. 성공 실행끼리 비교하면 Ours는 8.64회, Random은 9.68회다.

성공 실행의 물리 step은 Ours 12.46, When only 12.37로 거의 같다. 따라서 Ours의 질문 감소는 물리 계획이 짧아져서 생긴 결과가 아니라 What 정책이 같은 수준의 과제를 더 적은 질문으로 처리한 결과다.

## 산출물

- `00_raw/`: 이번 실행의 원본 로그 1,600개와 seed CSV
- `00_raw_sources.csv`: 각 raw 파일의 SHA-256과 경로
- `01_processed/episodes.csv`: 실행별 1차 가공 데이터
- `01_processed/summary.csv`: 전체·도메인별 집계
- `01_processed/scenes.csv`: 장면별 집계
- `02_graph_data/figure_data.csv`: 그림과 직접 대응하는 값과 오차막대
- `03_figures/`: overview와 지표별 PNG/PDF

그림의 성공률 오차막대는 95% Wilson interval이고, 연속 지표는 mean ± SE다. 가설검정이나 seed 기반 paired 검정을 이 보고서의 근거로 사용하지 않았다.

## 기록된 설정

- gamma: 0.2
- n_simulations: 200
- threshold: 0.8
- random query probability: 0.4
- max step: 50
