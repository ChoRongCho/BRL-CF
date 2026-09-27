# When–What policy ablation (gamma = 0.2) — 분석 보고서

Ours의 질문 시점·내용 정책을 고정 기준으로 두고, 질문 시점만 belief-state conformal prediction 또는 query-action value로 바꾸거나 질문 내용만 query-action value로 바꾼 통제 ablation. 두 도메인, 장면 1–5, 조건별 400회이며 동일 domain/scene/iteration의 네 조건은 같은 seed를 사용했다.

실행 슬롯 1,600개, 유효 결과 1,600개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 핵심 결과

1. CP-When은 성공률 98.25%로 Ours 99.00%와 유사했고, 성공 실행 질문 수는 7.47회로 Ours 8.49회보다 적었다. 이 결과만으로 Ours의 질문 시점이 CP보다 우월하다고 주장할 수 없으며 두 방식의 trade-off로 해석한다.
2. Value-When은 성공 실행 질문 수가 1.28회였지만 성공률이 60.25%로 하락했다. QueryAction Q-value를 질문 시점으로 사용하면 질문을 지나치게 적게 시작했다.
3. Value-What은 성공률 98.25%를 유지했지만 성공 실행 질문 수가 12.84회였다. Ours EIG는 8.49회로 같은 성공 수준에서 질문을 약 33.9% 줄였다.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 396/400 | 99 | 8.49 | 12.5 | 1.13 | 0 |
| all | cp_when | 393/400 | 98.2 | 7.42 | 12.2 | 0.973 | 0 |
| all | value_when | 241/400 | 60.2 | 1.01 | 11.2 | 2.09 | 0 |
| all | value_what | 393/400 | 98.2 | 12.8 | 12.7 | 2.27 | 0 |
| tomato | ours | 196/200 | 98 | 9.14 | 14.7 | 1.28 | 0 |
| tomato | cp_when | 193/200 | 96.5 | 7 | 14 | 1.12 | 0 |
| tomato | value_when | 127/200 | 63.5 | 1.23 | 14 | 2.75 | 0 |
| tomato | value_what | 193/200 | 96.5 | 12.8 | 14.7 | 2.25 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.982 | 0 |
| wastesorting | cp_when | 200/200 | 100 | 7.83 | 10.3 | 0.828 | 0 |
| wastesorting | value_when | 114/200 | 57 | 0.805 | 8.46 | 1.43 | 0 |
| wastesorting | value_what | 200/200 | 100 | 12.8 | 10.7 | 2.28 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 39/40 | 8.7 | 15.1 |
| tomato | 02 | ours | 40/40 | 9.12 | 13.8 |
| tomato | 03 | ours | 39/40 | 9.05 | 14.8 |
| tomato | 04 | ours | 39/40 | 9.85 | 14.9 |
| tomato | 05 | ours | 39/40 | 9 | 14.8 |
| tomato | 01 | cp_when | 40/40 | 6.8 | 14.4 |
| tomato | 02 | cp_when | 37/40 | 6.55 | 13.2 |
| tomato | 03 | cp_when | 38/40 | 7.22 | 14 |
| tomato | 04 | cp_when | 38/40 | 7.2 | 14.3 |
| tomato | 05 | cp_when | 40/40 | 7.22 | 14.2 |
| tomato | 01 | value_when | 20/40 | 1.15 | 12.5 |
| tomato | 02 | value_when | 26/40 | 1.1 | 12.4 |
| tomato | 03 | value_when | 26/40 | 1.32 | 15.2 |
| tomato | 04 | value_when | 28/40 | 1.12 | 14.9 |
| tomato | 05 | value_when | 27/40 | 1.43 | 14.8 |
| tomato | 01 | value_what | 37/40 | 12.4 | 15.7 |
| tomato | 02 | value_what | 40/40 | 12.5 | 13.8 |
| tomato | 03 | value_what | 38/40 | 12.6 | 14.6 |
| tomato | 04 | value_what | 39/40 | 13.5 | 14.9 |
| tomato | 05 | value_what | 39/40 | 12.8 | 14.7 |
| wastesorting | 01 | ours | 40/40 | 6.42 | 9.5 |
| wastesorting | 02 | ours | 40/40 | 7.17 | 10.4 |
| wastesorting | 03 | ours | 40/40 | 8.03 | 10.5 |
| wastesorting | 04 | ours | 40/40 | 9.03 | 10.6 |
| wastesorting | 05 | ours | 40/40 | 8.5 | 10.7 |
| wastesorting | 01 | cp_when | 40/40 | 6.42 | 9.5 |
| wastesorting | 02 | cp_when | 40/40 | 7.17 | 10.4 |
| wastesorting | 03 | cp_when | 40/40 | 8.03 | 10.5 |
| wastesorting | 04 | cp_when | 40/40 | 9.03 | 10.6 |
| wastesorting | 05 | cp_when | 40/40 | 8.5 | 10.7 |
| wastesorting | 01 | value_when | 22/40 | 0.65 | 7.95 |
| wastesorting | 02 | value_when | 24/40 | 0.825 | 8.97 |
| wastesorting | 03 | value_when | 24/40 | 0.8 | 8.38 |
| wastesorting | 04 | value_when | 21/40 | 0.925 | 8.47 |
| wastesorting | 05 | value_when | 23/40 | 0.825 | 8.53 |
| wastesorting | 01 | value_what | 40/40 | 11 | 9.85 |
| wastesorting | 02 | value_what | 40/40 | 12.4 | 10.8 |
| wastesorting | 03 | value_what | 40/40 | 12.4 | 10.9 |
| wastesorting | 04 | value_what | 40/40 | 14.7 | 10.9 |
| wastesorting | 05 | value_what | 40/40 | 13.6 | 11 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | cp_when | 400 | 7 | 4 | 0.75 | 0.549 | 0.549 | 1.07 |
| all | value_when | 400 | 155 | 0 | 38.8 | 4.38e-47 | 1.31e-46 | 7.47 |
| all | value_what | 400 | 3 | 0 | 0.75 | 0.25 | 0.5 | -4.31 |
| tomato | cp_when | 200 | 7 | 4 | 1.5 | 0.549 | 0.549 | 2.15 |
| tomato | value_when | 200 | 69 | 0 | 34.5 | 3.39e-21 | 1.02e-20 | 7.92 |
| tomato | value_what | 200 | 3 | 0 | 1.5 | 0.25 | 0.5 | -3.62 |
| wastesorting | cp_when | 200 | 0 | 0 | 0 | 1 | 1 | 0 |
| wastesorting | value_when | 200 | 86 | 0 | 43 | 2.58e-26 | 7.75e-26 | 7.03 |
| wastesorting | value_what | 200 | 0 | 0 | 0 | 1 | 1 | -5 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours (99.00%), 최소 평균 질문은 value_when (1.015회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | query cost | simulations | threshold | 모델 |
|---|---|---|---|---|---|
| ours | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
| cp_when | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
| value_when | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
| value_what | 0.2 | 1.0 | 100 | 0.8 | 미기록 |
