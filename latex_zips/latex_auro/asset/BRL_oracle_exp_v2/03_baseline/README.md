# Query baseline comparison

## 상태

**paper_ready** — 기존 4조건과 Query-Action (Tomato γ=0.5, Waste γ=0.9, cq=0) 400회를 합친 5조건 baseline 비교.

Ours, 기존 Query-Action POMCP, KnowNo, IntroPlan과 도메인별 gamma를 적용한 Query-Action을 비교한다. 기존 Query-Action은 gamma=0.2/query_cost=1.0이다. 추가 Query-Action은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0이며 n_simulations=100이다. 두 Query-Action 모두 failure_penalty=10.0, answer_accuracy=1.0이다. 각 방법 실행일과 파라미터는 원본 로그와 episodes.csv에 기록했다.

실행 로그 2000건. 파일명에서 확인된 실행일: 2026-09-19, 2026-09-20, 2026-09-22.

## 파일 구조

- Raw: `00_raw/`에 manifest가 지정한 원본 파일을 실제 보관. `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`: 전체·장면별 집계. `paired.csv`, `paired_episodes.csv`는 해당 분석에서 paired 비교를 생성한 경우에만 존재. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 성공률은 유효 결과 기준으로 계산하며 오류 건수는 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음. 기존 논문 그림의 오차막대/필터와 같다고 가정하지 말 것.

원본 파라미터: `{"gamma": ["0.2", "0.5", "0.9"], "n_simulations": ["100"], "query_cost": ["0.0", "1.0"], "failure_penalty": ["10.0"], "answer_accuracy": ["1.0"], "threshold": ["0.8"]}`

| Domain | Condition | Status | n |
|---|---|---|---:|
| tomato | introplan | ok | 200 |
| tomato | knowno | ok | 200 |
| tomato | ours | ok | 200 |
| tomato | query_action_pomcp | ok | 200 |
| tomato | query_action_pomcp_selected | ok | 200 |
| wastesorting | introplan | ok | 200 |
| wastesorting | knowno | ok | 200 |
| wastesorting | ours | ok | 200 |
| wastesorting | query_action_pomcp | ok | 200 |
| wastesorting | query_action_pomcp_selected | ok | 200 |

## 재생성

프로젝트 루트에서:

```bash
python3 experiments_logs/analysis_recent/scripts/pipeline.py --package query_baselines_current_20260920 --stage all
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.
