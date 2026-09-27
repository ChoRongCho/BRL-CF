# When–What–Random ablation (gamma=0.2, n_simulations=100) — 분석 보고서

2026-09-22 재실행한 When–What–Random 2×2 ablation. gamma=0.2, n_simulations=100이며 두 도메인, 장면 1–5, 조건별 400회로 총 1,600회다. 동일 domain/scene/iteration의 네 조건은 같은 seed를 사용했다. 원본 로그는 00_raw로 실제 이동했고 직전 n_simulations=200 패키지는 experiments_logs/analysis_archive/when_what_original_nsim200_20260921에 보관했다.

실행 슬롯 1,600개, 유효 결과 1,600개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | random | 294/400 | 73.5 | 8.25 | 11.9 | 1.05 | 0 |
| all | ours_what_only | 294/400 | 73.5 | 6.09 | 12 | 1.07 | 0 |
| all | ours_when_only | 395/400 | 98.8 | 13.9 | 12.7 | 1.08 | 0 |
| all | ours | 395/400 | 98.8 | 8.46 | 12.8 | 1.11 | 0 |
| tomato | random | 147/200 | 73.5 | 8.12 | 14.2 | 1.29 | 0 |
| tomato | ours_what_only | 147/200 | 73.5 | 6.57 | 14.4 | 1.34 | 0 |
| tomato | ours_when_only | 195/200 | 97.5 | 12.6 | 14.9 | 1.32 | 0 |
| tomato | ours | 195/200 | 97.5 | 9.17 | 15.2 | 1.39 | 0 |
| wastesorting | random | 147/200 | 73.5 | 8.38 | 9.64 | 0.816 | 0 |
| wastesorting | ours_what_only | 147/200 | 73.5 | 5.61 | 9.56 | 0.812 | 0 |
| wastesorting | ours_when_only | 200/200 | 100 | 15.2 | 10.6 | 0.846 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.75 | 10.3 | 0.843 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | random | 32/40 | 8.65 | 14.2 |
| tomato | 02 | random | 25/40 | 7.35 | 17.2 |
| tomato | 03 | random | 28/40 | 7.72 | 11.8 |
| tomato | 04 | random | 30/40 | 8.47 | 13.9 |
| tomato | 05 | random | 32/40 | 8.38 | 13.7 |
| tomato | 01 | ours_what_only | 31/40 | 7.08 | 15.2 |
| tomato | 02 | ours_what_only | 26/40 | 6.33 | 17.2 |
| tomato | 03 | ours_what_only | 28/40 | 5.97 | 11.9 |
| tomato | 04 | ours_what_only | 29/40 | 6.58 | 13.7 |
| tomato | 05 | ours_what_only | 33/40 | 6.9 | 13.8 |
| tomato | 01 | ours_when_only | 39/40 | 12.6 | 13.8 |
| tomato | 02 | ours_when_only | 39/40 | 12.7 | 15.8 |
| tomato | 03 | ours_when_only | 39/40 | 12.8 | 14.9 |
| tomato | 04 | ours_when_only | 39/40 | 12.8 | 15 |
| tomato | 05 | ours_when_only | 39/40 | 12.2 | 14.7 |
| tomato | 01 | ours | 39/40 | 9.22 | 15 |
| tomato | 02 | ours | 39/40 | 9.3 | 16.4 |
| tomato | 03 | ours | 39/40 | 9.32 | 14.9 |
| tomato | 04 | ours | 39/40 | 9.25 | 15.2 |
| tomato | 05 | ours | 39/40 | 8.75 | 14.8 |
| wastesorting | 01 | random | 27/40 | 7.2 | 8.45 |
| wastesorting | 02 | random | 26/40 | 7.2 | 9.47 |
| wastesorting | 03 | random | 33/40 | 8.5 | 9.9 |
| wastesorting | 04 | random | 32/40 | 9 | 10 |
| wastesorting | 05 | random | 29/40 | 10 | 10.4 |
| wastesorting | 01 | ours_what_only | 27/40 | 4.85 | 8.43 |
| wastesorting | 02 | ours_what_only | 26/40 | 4.95 | 9.35 |
| wastesorting | 03 | ours_what_only | 33/40 | 5.47 | 9.8 |
| wastesorting | 04 | ours_what_only | 32/40 | 6.33 | 9.93 |
| wastesorting | 05 | ours_what_only | 29/40 | 6.45 | 10.3 |
| wastesorting | 01 | ours_when_only | 40/40 | 12.4 | 9.78 |
| wastesorting | 02 | ours_when_only | 40/40 | 14.4 | 10.9 |
| wastesorting | 03 | ours_when_only | 40/40 | 14.9 | 10.7 |
| wastesorting | 04 | ours_when_only | 40/40 | 16.1 | 10.6 |
| wastesorting | 05 | ours_when_only | 40/40 | 18.2 | 11.1 |
| wastesorting | 01 | ours | 40/40 | 6.67 | 9.53 |
| wastesorting | 02 | ours | 40/40 | 7.22 | 10.5 |
| wastesorting | 03 | ours | 40/40 | 7.7 | 10.6 |
| wastesorting | 04 | ours | 40/40 | 8.12 | 10.5 |
| wastesorting | 05 | ours | 40/40 | 9.03 | 10.6 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | random | 400 | 105 | 4 | 25.2 | 1.78e-26 | 3.56e-26 | 0.21 |
| all | ours_what_only | 400 | 104 | 3 | 25.2 | 2.52e-27 | 7.55e-27 | 2.37 |
| all | ours_when_only | 400 | 1 | 1 | 0 | 1 | 1 | -5.46 |
| tomato | random | 200 | 52 | 4 | 24 | 1.1e-11 | 2.2e-11 | 1.05 |
| tomato | ours_what_only | 200 | 51 | 3 | 24 | 2.92e-12 | 8.76e-12 | 2.6 |
| tomato | ours_when_only | 200 | 1 | 1 | 0 | 1 | 1 | -3.46 |
| wastesorting | random | 200 | 53 | 0 | 26.5 | 2.22e-16 | 6.66e-16 | -0.635 |
| wastesorting | ours_what_only | 200 | 53 | 0 | 26.5 | 2.22e-16 | 6.66e-16 | 2.14 |
| wastesorting | ours_when_only | 200 | 0 | 0 | 0 | 1 | 1 | -7.46 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- 실행 오류는 성공률 분모에서 제외하고 별도 보고한다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours_when_only (98.75%), 최소 평균 질문은 ours_what_only (6.090회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | query cost | simulations | threshold | 모델 |
|---|---|---|---|---|---|
| random | 0.2 | 미기록 | 100 | 0.8 | 미기록 |
| ours_what_only | 0.2 | 미기록 | 100 | 0.8 | 미기록 |
| ours_when_only | 0.2 | 미기록 | 100 | 0.8 | 미기록 |
| ours | 0.2 | 미기록 | 100 | 0.8 | 미기록 |
