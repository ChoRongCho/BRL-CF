# When–What policy comparison (Value-When threshold-loop rerun)

## 상태

**analyzed_with_reference_batch_differences** — 수정된 Value-When 400/400 정상 완료. 기존 Value-What 400회 및 Ours/CP 800개 참조 유지.

Value-When은 2026-10-05 20:43 배치 400회로 교체했다. 질문 1개 제한을 제거하고 Q 기반 When은 episode 진입 시 한 번만 평가하며, 진입 후 confidence threshold=0.8로 반복・종료한다. Value-What은 같은 날 19:03 배치 400회 결과를 유지한다. 두 Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours와 CP-When은 기존 각 400회를 유지하고 gamma=0.2이다. 보관된 CP-When 결과는 수정 전 답변마다 CP 재평가 구현의 결과이므로, 현재 수정된 CP 코드의 평가로 해석하지 않는다. 기존 CP parsing 오류 4건은 성공률에서 실패로 집계한다. 이전 Value-When과 최신 배치는 합산하지 않는다.

실행 로그 1600건. 파일명에서 확인된 실행일: 2026-09-21, 2026-09-22, 2026-10-05.

## 파일 구조

- Raw: `00_raw/`에 manifest가 지정한 원본 파일을 실제 보관. `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`: 전체·장면별 집계. `paired.csv`, `paired_episodes.csv`는 해당 분석에서 paired 비교를 생성한 경우에만 존재. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 이 패키지는 실행 오류도 성공률에서 실패로 집계하며 오류 건수를 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음.

원본 파라미터: `{"gamma": ["0.2", "0.5", "0.9"], "n_simulations": ["100"], "query_cost": ["0.0", "1.0"], "failure_penalty": ["10.0"], "answer_accuracy": ["1.0"], "threshold": ["0.8"]}`

| Domain | Condition | Status | n |
|---|---|---|---:|
| tomato | cp_when | execution_error | 2 |
| tomato | cp_when | ok | 198 |
| tomato | ours | ok | 200 |
| tomato | value_what | ok | 200 |
| tomato | value_when | ok | 200 |
| wastesorting | cp_when | execution_error | 2 |
| wastesorting | cp_when | ok | 198 |
| wastesorting | ours | ok | 200 |
| wastesorting | value_what | ok | 200 |
| wastesorting | value_when | ok | 200 |

## 재생성

프로젝트 루트에서:

```bash
/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_threshold_loop.py
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.

## 최신 배치

Value-When만 20:43 배치로 교체했다. Value-What과 Ours/CP 참조는 유지한다. 이전 1개 제한 배치와 합산하지 않는다. 상세 전후 비교와 질문 구조는 report.md에 기록했다.
