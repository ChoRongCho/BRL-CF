# When–What policy ablation (gamma = 0.2)

## 상태

**archive_belief_state_cp** — CP-When을 belief-state prediction set으로 구현했던 대안 설계의 보존 결과. 원래 실험 정의인 action-CP When 결과가 아니므로 논문 수치와 그림에 사용하지 않는다.

이 실행에서 Value-When과 Value-What은 현재 정의와 같지만, CP-When은 폐기한 belief-state CP 정의를 사용했다. 두 도메인, 장면 1–5, 조건별 400회이며 재현과 출처 확인을 위해서만 보존한다.

실행 로그 1600건. 파일명에서 확인된 실행일: 2026-09-21.

## 파일 구조

- Raw: `00_raw/`에 manifest가 지정한 원본 파일을 실제 보관. `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`: 전체·장면별 집계. `paired.csv`, `paired_episodes.csv`는 해당 분석에서 paired 비교를 생성한 경우에만 존재. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 성공률은 유효 결과 기준으로 계산하며 오류 건수는 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음. 기존 논문 그림의 오차막대/필터와 같다고 가정하지 말 것.

원본 파라미터: `{"gamma": ["0.2"], "n_simulations": ["100"], "query_cost": ["1.0"], "failure_penalty": ["10.0"], "answer_accuracy": ["1.0"], "threshold": ["0.8"]}`

| Domain | Condition | Status | n |
|---|---|---|---:|
| tomato | cp_when | ok | 200 |
| tomato | ours | ok | 200 |
| tomato | value_what | ok | 200 |
| tomato | value_when | ok | 200 |
| wastesorting | cp_when | ok | 200 |
| wastesorting | ours | ok | 200 |
| wastesorting | value_what | ok | 200 |
| wastesorting | value_when | ok | 200 |

## 재생성

프로젝트 루트에서:

```bash
python3 experiments_logs/analysis_recent/scripts/pipeline.py --package when_what_policy_ablation_20260921 --stage all
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.
