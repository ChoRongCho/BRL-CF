# When–What policy ablation (gamma = 0.2)

## 상태

**paper_ready** — Action-CP When 400회(정상 396회, 형식 파싱 실패 4회)와 Value-When, Value-What, Ours 각 400회를 합친 결과. 파싱 실패 4회는 성공률에서 실패로 집계한다.

Ours의 질문 시점·내용 정책을 고정 기준으로 두고, 질문 시점만 KnowNo action conformal prediction 또는 query-action value로 바꾸거나 질문 내용만 query-action value로 바꾼 통제 ablation. CP-When의 action prediction set은 When trigger로만 사용하고, What은 Ours EIG를 사용한다. 두 도메인, 장면 1–5, 조건별 400회이며 action-option 형식 파싱 오류 4건은 요청에 따라 task failure로 성공률 분모에 포함한다.

실행 로그 1600건. 파일명에서 확인된 실행일: 2026-09-21, 2026-09-22.

## 파일 구조

- Raw: `00_raw/`에 manifest가 지정한 원본 파일을 실제 보관. `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`: 전체·장면별 집계. `paired.csv`, `paired_episodes.csv`는 해당 분석에서 paired 비교를 생성한 경우에만 존재. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 이 패키지는 실행 오류도 성공률에서 실패로 집계하며 오류 건수를 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음.

원본 파라미터: `{"gamma": ["0.2"], "n_simulations": ["100"], "query_cost": ["1.0"], "failure_penalty": ["10.0"], "answer_accuracy": ["1.0"], "threshold": ["0.8"]}`

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
python3 experiments_logs/analysis_recent/scripts/pipeline.py --package when_what_policy_ablation_20260921 --stage all
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.
