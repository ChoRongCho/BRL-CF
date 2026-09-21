# Original When–What ablation — 분석 보고서

동일 9월 12일 배치를 복원: ours는 .backup_20260917_071915 원본을 명시적으로 사용. 현재 ours 폴더는 9월 19일 재실행본이라 이 배치에 섞지 않음. 원본 파일을 이 실험의 00_raw로 실제 이동함. original_source_file은 이동 전 경로.

실행 슬롯 1,600개, 유효 결과 1,600개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.

## 전체·도메인별 결과

| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ours | 398/400 | 99.5 | 14 | 15.2 | 3.89 | 0 |
| all | ours_when_only | 399/400 | 99.8 | 23.1 | 15.3 | 3.91 | 0 |
| all | ours_what_only | 284/400 | 71 | 9.06 | 14.8 | 3.79 | 0 |
| all | random | 280/400 | 70 | 12.8 | 14.7 | 3.79 | 0 |
| tomato | ours | 198/200 | 99 | 15.3 | 18.6 | 6 | 0 |
| tomato | ours_when_only | 199/200 | 99.5 | 19.8 | 18.7 | 6.03 | 0 |
| tomato | ours_what_only | 132/200 | 66 | 10.2 | 18.1 | 5.83 | 0 |
| tomato | random | 128/200 | 64 | 12.3 | 18 | 5.82 | 0 |
| wastesorting | ours | 200/200 | 100 | 12.8 | 11.9 | 1.78 | 0 |
| wastesorting | ours_when_only | 200/200 | 100 | 26.3 | 11.9 | 1.8 | 0 |
| wastesorting | ours_what_only | 152/200 | 76 | 7.93 | 11.5 | 1.75 | 0 |
| wastesorting | random | 152/200 | 76 | 13.3 | 11.5 | 1.76 | 0 |

## 장면별 결과

| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |
|---|---|---|---:|---:|---:|
| tomato | 01 | ours | 40/40 | 15.6 | 18.6 |
| tomato | 02 | ours | 40/40 | 15.5 | 18.6 |
| tomato | 03 | ours | 40/40 | 16.2 | 18.9 |
| tomato | 04 | ours | 38/40 | 15 | 18.6 |
| tomato | 05 | ours | 40/40 | 14.2 | 18.1 |
| tomato | 01 | ours_when_only | 40/40 | 19.4 | 18.3 |
| tomato | 02 | ours_when_only | 40/40 | 19.6 | 18.8 |
| tomato | 03 | ours_when_only | 40/40 | 22.3 | 19.6 |
| tomato | 04 | ours_when_only | 39/40 | 19.5 | 18.9 |
| tomato | 05 | ours_when_only | 40/40 | 18.4 | 17.8 |
| tomato | 01 | ours_what_only | 27/40 | 11.5 | 19.3 |
| tomato | 02 | ours_what_only | 30/40 | 9.8 | 17.6 |
| tomato | 03 | ours_what_only | 25/40 | 9.3 | 19.5 |
| tomato | 04 | ours_what_only | 26/40 | 9.97 | 17.1 |
| tomato | 05 | ours_what_only | 24/40 | 10.4 | 16.7 |
| tomato | 01 | random | 27/40 | 14.2 | 20.4 |
| tomato | 02 | random | 29/40 | 11 | 16.6 |
| tomato | 03 | random | 24/40 | 11.6 | 18.9 |
| tomato | 04 | random | 26/40 | 12.3 | 17 |
| tomato | 05 | random | 22/40 | 12.2 | 17.1 |
| wastesorting | 01 | ours | 40/40 | 12.1 | 11.6 |
| wastesorting | 02 | ours | 40/40 | 12.4 | 11.7 |
| wastesorting | 03 | ours | 40/40 | 12.8 | 11.8 |
| wastesorting | 04 | ours | 40/40 | 13 | 12.2 |
| wastesorting | 05 | ours | 40/40 | 13.5 | 12.4 |
| wastesorting | 01 | ours_when_only | 40/40 | 24.7 | 11.6 |
| wastesorting | 02 | ours_when_only | 40/40 | 25.6 | 11.7 |
| wastesorting | 03 | ours_when_only | 40/40 | 24.5 | 11.8 |
| wastesorting | 04 | ours_when_only | 40/40 | 28.5 | 12.2 |
| wastesorting | 05 | ours_when_only | 40/40 | 28 | 12.3 |
| wastesorting | 01 | ours_what_only | 32/40 | 7.65 | 11.3 |
| wastesorting | 02 | ours_what_only | 32/40 | 8.12 | 11.5 |
| wastesorting | 03 | ours_what_only | 30/40 | 7.65 | 11.3 |
| wastesorting | 04 | ours_what_only | 31/40 | 8.9 | 12 |
| wastesorting | 05 | ours_what_only | 27/40 | 7.33 | 11.3 |
| wastesorting | 01 | random | 32/40 | 13.1 | 11.4 |
| wastesorting | 02 | random | 32/40 | 13.8 | 11.5 |
| wastesorting | 03 | random | 31/40 | 12.8 | 11.4 |
| wastesorting | 04 | random | 30/40 | 15.3 | 11.8 |
| wastesorting | 05 | random | 27/40 | 11.5 | 11.3 |

## 동일 scene·seed paired 비교

기준 조건: `ours`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.

| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | ours_when_only | 400 | 0 | 1 | -0.25 | 1 | 1 | -9.01 |
| all | ours_what_only | 400 | 114 | 0 | 28.5 | 9.63e-35 | 1.93e-34 | 4.97 |
| all | random | 400 | 118 | 0 | 29.5 | 6.02e-36 | 1.81e-35 | 1.24 |
| tomato | ours_when_only | 200 | 0 | 1 | -0.5 | 1 | 1 | -4.51 |
| tomato | ours_what_only | 200 | 66 | 0 | 33 | 2.71e-20 | 5.42e-20 | 5.12 |
| tomato | random | 200 | 70 | 0 | 35 | 1.69e-21 | 5.08e-21 | 3.04 |
| wastesorting | ours_when_only | 200 | 0 | 0 | 0 | 1 | 1 | -13.5 |
| wastesorting | ours_what_only | 200 | 48 | 0 | 24 | 7.11e-15 | 2.13e-14 | 4.83 |
| wastesorting | random | 200 | 48 | 0 | 24 | 7.11e-15 | 2.13e-14 | -0.555 |

## 해석 및 제한

- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.
- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.
- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.
- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.
- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.

관측 결과: 전체 최고 성공률은 ours_when_only (99.75%), 최소 평균 질문은 ours_what_only (9.065회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.

## 기록된 설정

| Condition | gamma | simulations | threshold | 모델 |
|---|---|---|---|---|
| ours | 0.95 | 100 | 0.8 | 미기록 |
| ours_when_only | 0.95 | 100 | 0.8 | 미기록 |
| ours_what_only | 0.95 | 100 | 0.8 | 미기록 |
| random | 0.95 | 100 | 0.8 | 미기록 |
