# Query baseline comparison — 분석 보고서

현재 활성 폴더의 Ours, Query-Action POMCP, KnowNo GPT-4, IntroPlan. 각 방법 실행일이 다름(원본 timestamp 및 parameter 열 참조). 과거 논문 그림을 그대로 복제한 것이 아니라 현재 raw를 재집계한 스냅샷. 다른 버전·모델의 백업은 제외. 원본 파일을 이 실험의 00_raw로 실제 이동함. original_source_file은 이동 전 경로.

실행 슬롯 1,600개, 유효 결과 1,600개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 390/400 | 97.5 | 8.41 | 12.4 | 1.01 | 0 |
| all | query_action_pomcp | 218/400 | 54.5 | 1.18 | 13.3 | 1.67 | 0 |
| all | knowno | 224/400 | 56 | 2.39 | 10.2 | 15.6 | 0 |
| all | introplan | 219/400 | 54.8 | 3.51 | 9.99 | 41.2 | 0 |
| tomato | ours | 190/200 | 95 | 9 | 14.4 | 1.22 | 0 |
| tomato | query_action_pomcp | 118/200 | 59 | 1.2 | 16.7 | 1.98 | 0 |
| tomato | knowno | 110/200 | 55 | 1.71 | 11.3 | 18.2 | 0 |
| tomato | introplan | 128/200 | 64 | 3.54 | 11.9 | 54.3 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.806 | 0 |
| wastesorting | query_action_pomcp | 100/200 | 50 | 1.16 | 9.93 | 1.36 | 0 |
| wastesorting | knowno | 114/200 | 57 | 3.08 | 8.97 | 13.1 | 0 |
| wastesorting | introplan | 91/200 | 45.5 | 3.49 | 8.04 | 28.2 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 38/40 | 8.6 | 14.9 |
| tomato | 02 | ours | 40/40 | 9.12 | 13.8 |
| tomato | 03 | ours | 35/40 | 8.53 | 13.8 |
| tomato | 04 | ours | 38/40 | 9.75 | 14.7 |
| tomato | 05 | ours | 39/40 | 9 | 14.8 |
| tomato | 01 | query_action_pomcp | 28/40 | 1.55 | 18.4 |
| tomato | 02 | query_action_pomcp | 24/40 | 1.55 | 16.8 |
| tomato | 03 | query_action_pomcp | 22/40 | 1.23 | 15.4 |
| tomato | 04 | query_action_pomcp | 23/40 | 0.85 | 17.1 |
| tomato | 05 | query_action_pomcp | 21/40 | 0.8 | 15.9 |
| tomato | 01 | knowno | 18/40 | 1.57 | 10.2 |
| tomato | 02 | knowno | 22/40 | 1.48 | 11.3 |
| tomato | 03 | knowno | 21/40 | 1.65 | 11.2 |
| tomato | 04 | knowno | 25/40 | 2.25 | 12 |
| tomato | 05 | knowno | 24/40 | 1.6 | 11.8 |
| tomato | 01 | introplan | 23/40 | 3.6 | 11.1 |
| tomato | 02 | introplan | 25/40 | 3.48 | 11.9 |
| tomato | 03 | introplan | 23/40 | 3.35 | 11.6 |
| tomato | 04 | introplan | 29/40 | 3.85 | 12.6 |
| tomato | 05 | introplan | 28/40 | 3.4 | 12.5 |
| wastesorting | 01 | ours | 40/40 | 6.42 | 9.5 |
| wastesorting | 02 | ours | 40/40 | 7.17 | 10.4 |
| wastesorting | 03 | ours | 40/40 | 8.03 | 10.5 |
| wastesorting | 04 | ours | 40/40 | 9.03 | 10.6 |
| wastesorting | 05 | ours | 40/40 | 8.5 | 10.7 |
| wastesorting | 01 | query_action_pomcp | 14/40 | 0.875 | 8.45 |
| wastesorting | 02 | query_action_pomcp | 22/40 | 0.975 | 9.97 |
| wastesorting | 03 | query_action_pomcp | 22/40 | 1.05 | 10.2 |
| wastesorting | 04 | query_action_pomcp | 23/40 | 1.45 | 11 |
| wastesorting | 05 | query_action_pomcp | 19/40 | 1.43 | 10 |
| wastesorting | 01 | knowno | 25/40 | 2.55 | 8.72 |
| wastesorting | 02 | knowno | 26/40 | 3.98 | 9.8 |
| wastesorting | 03 | knowno | 20/40 | 3 | 8.47 |
| wastesorting | 04 | knowno | 19/40 | 2.9 | 8.82 |
| wastesorting | 05 | knowno | 24/40 | 2.95 | 9.05 |
| wastesorting | 01 | introplan | 22/40 | 3.75 | 8.38 |
| wastesorting | 02 | introplan | 25/40 | 3.67 | 9.53 |
| wastesorting | 03 | introplan | 17/40 | 3.35 | 8.03 |
| wastesorting | 04 | introplan | 12/40 | 3.38 | 6.97 |
| wastesorting | 05 | introplan | 15/40 | 3.3 | 7.33 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | query_action_pomcp | 400 | 174 | 2 | 43 | 3.25e-49 | 9.76e-49 | 7.24 |
| all | knowno | 400 | 170 | 4 | 41.5 | 3.15e-45 | 3.15e-45 | 6.02 |
| all | introplan | 400 | 175 | 4 | 42.8 | 1.1e-46 | 2.21e-46 | 4.9 |
| tomato | query_action_pomcp | 200 | 74 | 2 | 36 | 7.75e-20 | 1.55e-19 | 7.8 |
| tomato | knowno | 200 | 84 | 4 | 40 | 1.58e-20 | 4.74e-20 | 7.29 |
| tomato | introplan | 200 | 66 | 4 | 31 | 1.65e-15 | 1.65e-15 | 5.46 |
| wastesorting | query_action_pomcp | 200 | 100 | 0 | 50 | 1.58e-30 | 3.16e-30 | 6.67 |
| wastesorting | knowno | 200 | 86 | 0 | 43 | 2.58e-26 | 2.58e-26 | 4.75 |
| wastesorting | introplan | 200 | 109 | 0 | 54.5 | 3.08e-33 | 9.24e-33 | 4.34 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours (97.50%), 최소 평균 질문은 query_action_pomcp (1.175회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | simulations | threshold | 모델 |
|---|---|---|---|---|
| ours | 0.2 | 100 | 0.8 | 미기록 |
| query_action_pomcp | 0.2 | 100 | 미기록 | 미기록 |
| knowno | 미기록 | 미기록 | 미기록 | gpt-4o |
| introplan | 미기록 | 미기록 | 미기록 | gpt-4o |
