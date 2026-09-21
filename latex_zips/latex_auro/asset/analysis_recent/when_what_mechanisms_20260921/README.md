# When–What mechanism ablation

9월 21일 배치 1600 슬롯. cp_when Tomato scene03 iteration39의 LLM 선택지 파싱 오류 1건은 execution_error로 유지. 유효 결과 1599건.

실행 로그 1600건. 파일명에서 확인된 실행일: 2026-09-21.

## 파일 구조

- Raw: `00_raw/`에 실제 원본 파일을 보관(핵심 4개 실험). `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`, `paired.csv`, `paired_episodes.csv`: 전체·장면별 집계 및 동일 scene/seed 비교. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 성공률은 유효 결과 기준으로 계산하며 오류 건수는 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음. 기존 논문 그림의 오차막대/필터와 같다고 가정하지 말 것.

원본 파라미터: `{"gamma": ["0.2"], "n_simulations": ["100"], "threshold": ["0.8"]}`

| Domain | Condition | Status | n |
|---|---|---|---:|
| tomato | cp_when | execution_error | 1 |
| tomato | cp_when | ok | 199 |
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
python3 experiments_logs/analysis_recent/scripts/pipeline.py --package when_what_mechanisms_20260921 --stage all
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.
