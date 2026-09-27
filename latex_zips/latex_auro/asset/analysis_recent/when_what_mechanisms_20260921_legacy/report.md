# When–What mechanism ablation — 분석 보고서

9월 21일 배치 1600 슬롯. cp_when Tomato scene03 iteration39의 LLM 선택지 파싱 오류 1건은 execution_error로 유지. 유효 결과 1599건.

실행 슬롯 1,600개, 유효 결과 1,599개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 396/400 | 99 | 8.49 | 12.5 | 1.13 | 0 |
| all | cp_when | 291/399 | 72.9 | 5.73 | 11.4 | 18.6 | 1 |
| all | value_when | 236/400 | 59 | 0.873 | 11.2 | 2.3 | 0 |
| all | value_what | 393/400 | 98.2 | 12.8 | 12.7 | 2.57 | 0 |
| tomato | ours | 196/200 | 98 | 9.14 | 14.7 | 1.28 | 0 |
| tomato | cp_when | 126/199 | 63.3 | 4 | 12.9 | 21.4 | 1 |
| tomato | value_when | 124/200 | 62 | 1.01 | 13.9 | 3.03 | 0 |
| tomato | value_what | 193/200 | 96.5 | 12.8 | 14.7 | 2.32 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.982 | 0 |
| wastesorting | cp_when | 165/200 | 82.5 | 7.45 | 9.89 | 15.7 | 0 |
| wastesorting | value_when | 112/200 | 56 | 0.73 | 8.39 | 1.57 | 0 |
| wastesorting | value_what | 200/200 | 100 | 12.8 | 10.7 | 2.81 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 39/40 | 8.7 | 15.1 |
| tomato | 02 | ours | 40/40 | 9.12 | 13.8 |
| tomato | 03 | ours | 39/40 | 9.05 | 14.8 |
| tomato | 04 | ours | 39/40 | 9.85 | 14.9 |
| tomato | 05 | ours | 39/40 | 9 | 14.8 |
| tomato | 01 | cp_when | 30/40 | 4.6 | 13.2 |
| tomato | 02 | cp_when | 27/40 | 3.95 | 13 |
| tomato | 03 | cp_when | 19/39 | 3.49 | 12.1 |
| tomato | 04 | cp_when | 24/40 | 4.2 | 13.1 |
| tomato | 05 | cp_when | 26/40 | 3.75 | 13 |
| tomato | 01 | value_when | 21/40 | 1 | 12.8 |
| tomato | 02 | value_when | 25/40 | 1.02 | 12.3 |
| tomato | 03 | value_when | 26/40 | 1.1 | 14.9 |
| tomato | 04 | value_when | 27/40 | 1.02 | 14.8 |
| tomato | 05 | value_when | 25/40 | 0.925 | 14.9 |
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
| wastesorting | 01 | cp_when | 31/40 | 6.6 | 9.4 |
| wastesorting | 02 | cp_when | 34/40 | 6.72 | 9.88 |
| wastesorting | 03 | cp_when | 31/40 | 7.6 | 10.1 |
| wastesorting | 04 | cp_when | 33/40 | 8.1 | 9.88 |
| wastesorting | 05 | cp_when | 36/40 | 8.22 | 10.2 |
| wastesorting | 01 | value_when | 22/40 | 0.55 | 7.92 |
| wastesorting | 02 | value_when | 24/40 | 0.95 | 8.93 |
| wastesorting | 03 | value_when | 23/40 | 0.65 | 8.4 |
| wastesorting | 04 | value_when | 20/40 | 0.9 | 8.2 |
| wastesorting | 05 | value_when | 23/40 | 0.6 | 8.5 |
| wastesorting | 01 | value_what | 40/40 | 11 | 9.85 |
| wastesorting | 02 | value_what | 40/40 | 12.4 | 10.8 |
| wastesorting | 03 | value_what | 40/40 | 12.4 | 10.9 |
| wastesorting | 04 | value_what | 40/40 | 14.7 | 10.9 |
| wastesorting | 05 | value_what | 40/40 | 13.6 | 11 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | cp_when | 399 | 106 | 2 | 26.1 | 3.63e-29 | 7.26e-29 | 2.76 |
| all | value_when | 400 | 160 | 0 | 40 | 1.37e-48 | 4.11e-48 | 7.62 |
| all | value_what | 400 | 3 | 0 | 0.75 | 0.25 | 0.25 | -4.31 |
| tomato | cp_when | 199 | 71 | 2 | 34.7 | 5.72e-19 | 1.14e-18 | 5.16 |
| tomato | value_when | 200 | 72 | 0 | 36 | 4.24e-22 | 1.27e-21 | 8.13 |
| tomato | value_what | 200 | 3 | 0 | 1.5 | 0.25 | 0.25 | -3.62 |
| wastesorting | cp_when | 200 | 35 | 0 | 17.5 | 5.82e-11 | 1.16e-10 | 0.38 |
| wastesorting | value_when | 200 | 88 | 0 | 44 | 6.46e-27 | 1.94e-26 | 7.1 |
| wastesorting | value_what | 200 | 0 | 0 | 0 | 1 | 1 | -5 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours (99.00%), 최소 평균 질문은 value_when (0.873회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | simulations | threshold | 모델 |
|---|---|---|---|---|
| ours | 0.2 | 100 | 0.8 | 미기록 |
| cp_when | 0.2 | 100 | 0.8 | 미기록 |
| value_when | 0.2 | 100 | 0.8 | 미기록 |
| value_what | 0.2 | 100 | 0.8 | 미기록 |

## 실행 오류 원본

- tomato / cp_when / scene 03 / seed 2436888965: `experiments_logs/analysis_recent/when_what_mechanisms_20260921/00_raw/tomato/scene_03/cp_when/run_39_seed_2436888965/console.log`
