# Query baseline comparison — 분석 보고서

도표에서는 Ours, KnowNo, IntroPlan, Query-Action의 4조건을 비교한다. Query-Action (original)은 모든 비교 표·그래프에서 제외하며 원본과 실행별 데이터만 보존한다. KnowNo Tomato는 2026-09-16 집계의 prompt v2 200회(성공률 43.0%, qhat=0.8404)를 유지하고, Waste는 2026-09-30 prompt v1 200회(성공률 48.5%, qhat=0.845165078859924, generation temperature=0.3)를 사용한다. 두 domain 모두 GPT-4o, score temperature=5.0, exact oracle이다. Tomato의 generation temperature는 원본에 기록되지 않아 확정하지 않는다. IntroPlan은 2026-09-30 prompt v1, generation temperature=0.3, score temperature=5.0, exact oracle 재실행 결과를 유지했다(qhat Tomato=0.9855517605160237, Waste=0.980493087109584, top_k=3). Ours와 도표에 남긴 Query-Action 결과는 유지했다. 도표의 Query-Action 설정은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. 이 설정은 도표 라벨 대신 별도로 명시한다. 이는 prompt만 바꾼 통제 비교가 아니라 각 배치의 기록된 설정에 따른 결과 비교이다.

실행 슬롯 2,000개, 유효 결과 2,000개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 핵심 결과

1. KnowNo (Tomato v2 / Waste v1): all 성공 183/400 (45.75%), episode당 질문 1.86회, 성공 episode당 질문 2.47회.
2. KnowNo (v2): tomato 성공 86/200 (43.00%), episode당 질문 1.79회, 성공 episode당 질문 2.45회.
3. KnowNo (v1): wastesorting 성공 97/200 (48.50%), episode당 질문 1.92회, 성공 episode당 질문 2.48회.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 390/400 | 97.5 | 8.41 | 12.4 | 1.01 | 0 |
| all | knowno | 183/400 | 45.8 | 1.86 | 9.68 | 17 | 0 |
| all | introplan | 183/400 | 45.8 | 6.28 | 9.96 | 47.7 | 0 |
| all | query_action_pomcp_selected | 289/400 | 72.2 | 16.9 | 12.7 | 7.6 | 0 |
| tomato | ours | 190/200 | 95 | 9 | 14.4 | 1.22 | 0 |
| tomato | knowno | 86/200 | 43 | 1.79 | 10.9 | 16.3 | 0 |
| tomato | introplan | 80/200 | 40 | 6.91 | 10.9 | 57.6 | 0 |
| tomato | query_action_pomcp_selected | 115/200 | 57.5 | 9.72 | 13.8 | 5.04 | 0 |
| wastesorting | ours | 200/200 | 100 | 7.83 | 10.3 | 0.806 | 0 |
| wastesorting | knowno | 97/200 | 48.5 | 1.92 | 8.48 | 17.8 | 0 |
| wastesorting | introplan | 103/200 | 51.5 | 5.65 | 9.03 | 37.8 | 0 |
| wastesorting | query_action_pomcp_selected | 174/200 | 87 | 24 | 11.5 | 10.2 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 38/40 | 8.6 | 14.9 |
| tomato | 02 | ours | 40/40 | 9.12 | 13.8 |
| tomato | 03 | ours | 35/40 | 8.53 | 13.8 |
| tomato | 04 | ours | 38/40 | 9.75 | 14.7 |
| tomato | 05 | ours | 39/40 | 9 | 14.8 |
| tomato | 01 | knowno | 19/40 | 1.93 | 10.8 |
| tomato | 02 | knowno | 15/40 | 1.38 | 10.3 |
| tomato | 03 | knowno | 20/40 | 1.8 | 11.2 |
| tomato | 04 | knowno | 17/40 | 2.08 | 10.8 |
| tomato | 05 | knowno | 15/40 | 1.8 | 11.1 |
| tomato | 01 | introplan | 14/40 | 5.88 | 9.25 |
| tomato | 02 | introplan | 15/40 | 7.25 | 11.3 |
| tomato | 03 | introplan | 13/40 | 7.15 | 10.4 |
| tomato | 04 | introplan | 22/40 | 7.05 | 11.9 |
| tomato | 05 | introplan | 16/40 | 7.22 | 11.5 |
| tomato | 01 | query_action_pomcp_selected | 21/40 | 8.93 | 12.8 |
| tomato | 02 | query_action_pomcp_selected | 27/40 | 9.68 | 14.8 |
| tomato | 03 | query_action_pomcp_selected | 22/40 | 10.1 | 13.6 |
| tomato | 04 | query_action_pomcp_selected | 21/40 | 9.88 | 14.4 |
| tomato | 05 | query_action_pomcp_selected | 24/40 | 10.1 | 13.5 |
| wastesorting | 01 | ours | 40/40 | 6.42 | 9.5 |
| wastesorting | 02 | ours | 40/40 | 7.17 | 10.4 |
| wastesorting | 03 | ours | 40/40 | 8.03 | 10.5 |
| wastesorting | 04 | ours | 40/40 | 9.03 | 10.6 |
| wastesorting | 05 | ours | 40/40 | 8.5 | 10.7 |
| wastesorting | 01 | knowno | 21/40 | 2.27 | 8.35 |
| wastesorting | 02 | knowno | 24/40 | 2.52 | 9.62 |
| wastesorting | 03 | knowno | 17/40 | 2.02 | 7.78 |
| wastesorting | 04 | knowno | 19/40 | 1.5 | 8.4 |
| wastesorting | 05 | knowno | 16/40 | 1.27 | 8.28 |
| wastesorting | 01 | introplan | 19/40 | 5.42 | 8.65 |
| wastesorting | 02 | introplan | 24/40 | 6 | 9.47 |
| wastesorting | 03 | introplan | 20/40 | 5.4 | 8.53 |
| wastesorting | 04 | introplan | 18/40 | 5.6 | 9.2 |
| wastesorting | 05 | introplan | 22/40 | 5.83 | 9.3 |
| wastesorting | 01 | query_action_pomcp_selected | 40/40 | 22.1 | 11.2 |
| wastesorting | 02 | query_action_pomcp_selected | 36/40 | 25.1 | 11.9 |
| wastesorting | 03 | query_action_pomcp_selected | 34/40 | 24.5 | 11.5 |
| wastesorting | 04 | query_action_pomcp_selected | 32/40 | 23.6 | 11.4 |
| wastesorting | 05 | query_action_pomcp_selected | 32/40 | 24.8 | 11.3 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | knowno | 400 | 212 | 5 | 51.8 | 3.72e-56 | 7.44e-56 | 6.56 |
| all | introplan | 400 | 210 | 3 | 51.8 | 2.45e-58 | 7.34e-58 | 2.13 |
| all | query_action_pomcp_selected | 400 | 108 | 7 | 25.2 | 2.25e-24 | 2.25e-24 | -8.45 |
| tomato | knowno | 200 | 109 | 5 | 52 | 1.48e-26 | 2.96e-26 | 7.21 |
| tomato | introplan | 200 | 113 | 3 | 55 | 6.27e-30 | 1.88e-29 | 2.09 |
| tomato | query_action_pomcp_selected | 200 | 82 | 7 | 37.5 | 2.43e-17 | 2.43e-17 | -0.725 |
| wastesorting | knowno | 200 | 103 | 0 | 51.5 | 1.97e-31 | 5.92e-31 | 5.91 |
| wastesorting | introplan | 200 | 97 | 0 | 48.5 | 1.26e-29 | 2.52e-29 | 2.18 |
| wastesorting | query_action_pomcp_selected | 200 | 26 | 0 | 13 | 2.98e-08 | 2.98e-08 | -16.2 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- 실행 오류는 성공률 분모에서 제외하고 별도 보고한다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours (97.50%), 최소 평균 질문은 knowno (1.857회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | query cost | simulations | threshold | 모델 |
|---|---|---|---|---|---|
| ours | 0.2 | 미기록 | 100 | 0.8 | 미기록 |
| knowno | 미기록 | 미기록 | 미기록 | 0.8404, 0.845165078859924 | gpt-4o |
| introplan | 미기록 | 미기록 | 미기록 | 0.980493087109584, 0.9855517605160237 | gpt-4o |
| query_action_pomcp_selected | 0.5, 0.9 | 0.0 | 100 | 미기록 | 미기록 |
