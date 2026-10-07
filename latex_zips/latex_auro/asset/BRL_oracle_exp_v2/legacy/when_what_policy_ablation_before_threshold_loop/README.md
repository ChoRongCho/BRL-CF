# When–What policy comparison (Value rerun: 2026-10-05)

## 상태

**analyzed_with_design_limitations** — 새 Value 실행 800/800 정상 완료; 기존 Ours・CP 800개 참조. 질문 구조 및 gamma 차이를 명시한 비교.

Ours・Action-CP When은 2026-09-21~22의 각 400회를 유지하고, Value-When・Value-What은 2026-10-05 배치의 각 400회로 교체했다. Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours・CP-When은 gamma=0.2이다. gamma는 질문 가치 평가기와 실제 행동용 POMCP 모두에 적용됐다. Value-When의 물리 행동당 최대 1개 질문 제한은 유지됐다. 따라서 질문 시점만 바꾼 통제 비교로 해석하지 않는다. CP-When의 기존 형식 파싱 오류 4건은 성공률에서 실패로 집계한다. 이전 Value 배치와 새 배치는 합산하지 않고 별도 paired 비교로 제공한다.

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
/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_20261005.py
```

이 명령은 원본 해시・800회 설정・trace와 집계 일치를 검증하고 분석과 LaTeX 패키지를 다시 생성한다. 이전 Value와 새 Value를 합산하지 않는다. PDF 보고서는 시스템 Python으로 다음 명령을 실행한다:

```bash
python3 experiments_logs/analysis_recent/scripts/render_analysis_pdf.py latex_zips/latex_auro/asset/BRL_oracle_exp_v2/REPORT.md latex_zips/latex_auro/asset/BRL_oracle_exp_v2/02_when_what_policy_ablation/report.md
```

## 재실험 상세

최신 Value 800회와 이전 Ours/CP 800개 참조만 집계한다. 이전 Value 결과는 합산하지 않는다. `report.md`에 이전 Value와의 paired 비교 및 질문 구조 점검을 기록했다. `exp_set.md`는 기존 설정 설명과 이번 실행 설정을 함께 보존한다.
