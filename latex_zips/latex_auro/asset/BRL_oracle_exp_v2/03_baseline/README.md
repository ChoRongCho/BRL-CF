# Query baseline comparison

## 상태

**paper_ready** — KnowNo Tomato는 2026-09-16 v2 43.0%, Waste는 2026-09-30 v1 48.5%. Ours·KnowNo·IntroPlan·Query-Action 4조건 비교.

도표에서는 Ours, KnowNo, IntroPlan, Query-Action의 4조건을 비교한다. Query-Action (original)은 모든 비교 표·그래프에서 제외하며 원본과 실행별 데이터만 보존한다. KnowNo Tomato는 2026-09-16 집계의 prompt v2 200회(성공률 43.0%, qhat=0.8404)를 유지하고, Waste는 2026-09-30 prompt v1 200회(성공률 48.5%, qhat=0.845165078859924, generation temperature=0.3)를 사용한다. 두 domain 모두 GPT-4o, score temperature=5.0, exact oracle이다. Tomato의 generation temperature는 원본에 기록되지 않아 확정하지 않는다. IntroPlan은 2026-09-30 prompt v1, generation temperature=0.3, score temperature=5.0, exact oracle 재실행 결과를 유지했다(qhat Tomato=0.9855517605160237, Waste=0.980493087109584, top_k=3). Ours와 도표에 남긴 Query-Action 결과는 유지했다. 도표의 Query-Action 설정은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. 이 설정은 도표 라벨 대신 별도로 명시한다. 이는 prompt만 바꾼 통제 비교가 아니라 각 배치의 기록된 설정에 따른 결과 비교이다.

실행 로그 2000건. 파일명에서 확인된 실행일: 2026-09-19, 2026-09-20, 2026-09-22.

## KnowNo 도메인별 확정 설정

| 설정 | Tomato | Waste |
|---|---|---|
| 선택 결과 | 2026-09-16 집계 | 2026-09-30 실행 |
| Prompt | v2 | v1 |
| 생성 temperature | 원본 기록 없음 | 0.3 |
| Score temperature | 5.0 | 5.0 |
| qhat | 0.8404 | 0.845165078859924 |
| 성공률 | 43.0% (86/200) | 48.5% (97/200) |

## Query-Action 설정

도표 라벨은 `Query-Action`으로 표기한다. Tomato: `gamma=0.5`, Waste: `gamma=0.9`. 공통 설정: `query_cost=0.0`, `n_simulations=100`.
비교 대상은 4조건 × 400회 = 1,600회이다. 제외한 original 400회의 원본·실행별 기록도 보관하므로 raw 및 episodes.csv에는 2,000회가 남는다.

## 파일 구조

- Raw: `00_raw/`에 manifest가 지정한 원본 파일을 실제 보관. `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.
- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.
- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.
- `01_processed/summary.csv`, `scenes.csv`: 전체·장면별 집계. `paired.csv`, `paired_episodes.csv`는 해당 분석에서 paired 비교를 생성한 경우에만 존재. 상세 해석은 `report.md`.
- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.
- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.

## 집계 기준

정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 성공률은 유효 결과 기준으로 계산하며 오류 건수는 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음. 기존 논문 그림의 오차막대/필터와 같다고 가정하지 말 것.

원본 파라미터: `{"gamma": ["0.2", "0.5", "0.9"], "n_simulations": ["100"], "query_cost": ["0.0", "1.0"], "failure_penalty": ["10.0"], "answer_accuracy": ["1.0"], "threshold": ["0.8", "0.8404", "0.845165078859924", "0.980493087109584", "0.9855517605160237"]}`

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
python3 - <<'PYTHON'
import sys
from pathlib import Path
sys.path.insert(0, 'experiments_logs/analysis_recent/scripts')
import pipeline
p = Path('latex_zips/latex_auro/asset/BRL_oracle_exp_v2/03_baseline')
for step in (pipeline.stage1, pipeline.stage2, pipeline.stage3, pipeline.docs):
    step(p)
PYTHON
```

단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.
