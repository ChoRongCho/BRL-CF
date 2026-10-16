# AURO 논문 실험 데이터

`section/05_exp_setting.tex`와 `section/05_exp_system.tex`의 실험 순서에 맞춰 baseline, ablation, threshold를 최상위 분류로 사용한다. Oracle·VLM·Human은 baseline 내부의 응답 제공 조건이며, 정리 CSV의 `environment` 열로 구분한다. Oracle은 시뮬레이션, VLM·Human은 실제 로봇 실험이다.

## 폴더 구조

```text
BRL_EXP/
├── 01_BASELINE/
│   ├── 00_raw/
│   │   ├── oracle_baselines/       # KnowNo, IntroPlan, Query-Action 원본 배치
│   │   ├── vlm/                    # VLM 시행별 결과 로그
│   │   ├── vlm_sessions/           # VLM 전체 세션 로그·recording 메타데이터
│   │   ├── human/p001~p003/        # 새 사람 실험의 전체 로그·인터뷰, 영상 제외
│   │   └── source_inputs/          # 입력 CSV, 배치 manifest, 이전 논문 수치
│   ├── 01_processed/               # 시행 단위 runs.csv, statistics.csv, 사람 CSV, 실패 분류
│   ├── 02_graph_data/              # 그림용 수치와 표용 CSV·LaTeX
│   └── 03_figures/                 # Oracle·VLM·Human·응답 제공 조건 비교, 27개 PDF
├── 02_ABLATION/
│   ├── 00_raw/random_when_what/     # Fully-Random, W1, Q1 원본 배치
│   ├── 00_raw/policy_when_what/     # W2, W3, Q2, 공통 Ours 원본 배치
│   ├── 00_raw/source_inputs/       # 배치 manifest와 원래 집계 CSV
│   ├── 01_processed/
│   ├── 02_graph_data/
│   └── 03_figures/                 # 성공률·질문 수, 도메인별 4개 PDF
├── 03_THRESHOLD/
│   ├── 00_raw/threshold_sweep/     # threshold 0.0~1.0 원본 배치
│   ├── 00_raw/source_inputs/
│   ├── 01_processed/
│   ├── 02_graph_data/
│   └── 03_figures/                 # 성공률·질문 수, 2개 PDF
├── 04_CODE/                       # 원자료 배치, 파싱, 집계, 그림 생성 코드
├── source_index.csv               # 원본 시행 목록과 패키지 내 raw_path
├── source_files.csv               # 복사한 모든 파일의 원본 경로·SHA-256
├── validation.json                # 원자료 검증과 집계 결과
└── experiment_panels_manifest.json # 생성된 33개 PDF와 사용 수치
```

원본 폴더는 보존하고 실제 파일을 복사했다. 영상은 제외했으며 symlink와 외부 raw 경로에 의존하지 않는다. 원본 배치의 비교에서 제외된 설정도 `00_raw`에 남겨 두었고, `01_processed/runs.csv`에는 논문 비교에 사용한 시행만 담았다. 따라서 raw 배치의 전체 시행 수와 논문 표본 수는 다를 수 있다. 이미 대체된 legacy 배치는 가져오지 않았다.

## 실험별 시행 선택

| 실험 | 데이터 선택 | 분석 시행 수 |
| --- | --- | ---: |
| Baseline / Oracle | KnowNo, IntroPlan, 선택된 Query-Action 배치, 공통 Ours. 도메인·방법별 200회 | 1,600 |
| Baseline / VLM | 기존 논문 그림에 사용한 최종 VLM 배치. 도메인·방법별 40회 | 160 |
| Baseline / Human | `08_Delete_Me/00_real_human/human_summary.csv`와 일치하는 p001~p003의 새 시행. 도메인·방법별 6회 | 24 |
| Ablation | Fully-Random, W1~W3, Q1~Q2, 공통 Ours. 도메인·조건별 200회 | 2,800 |
| Threshold | 0.0~1.0, 0.1 간격. 도메인·threshold별 200회. 0.8은 공통 Ours로 교체 | 4,400 |

공통 Ours는 `02_ABLATION/00_raw/policy_when_what`의 400회 배치다. WasteSorting 200/200, TomatoHarvesting 196/200이며, 세 비교의 `raw_path`와 `raw_sha256`이 동일하다. 다른 Ours 배치를 합산하지 않는다. Threshold 원본 0.8 시행은 raw로 보존하고 분석에서는 제외한다.

| 원본 condition | 논문 표기 | When | What |
| --- | --- | --- | --- |
| random | Fully-Random | Random | Random |
| ours_what_only | W1 | Random | Entropy Gain |
| cp_when | W2 | Conformal Prediction | Entropy Gain |
| value_when | W3 | Reward | Entropy Gain |
| ours_when_only | Q1 | Confidence | Random |
| value_what | Q2 | Confidence | Reward |
| ours | Ours | Confidence | Entropy Gain |

실행 오류 4건을 포함하는 W2의 성공률 분모는 도메인별 200회다. 오류 시행은 성공률에서 실패로 집계하고, 미기록 비용 지표를 0으로 채우지 않는다.

## 데이터 사용과 지표 정의

- `01_processed/runs.csv`: 시행별 정규화 자료. 응답 조건, 도메인, 방법, 원본 condition, 배치, 장면, 참여자, 성공 여부, 비용 지표와 원자료 경로·해시를 포함한다.
- `01_processed/statistics.csv`: 성공률과 성공 시행의 지표 집계. `02_graph_data/statistics.csv`는 같은 수치의 그림·표용 사본이다.
- 성공률은 실패·실행 오류를 포함한 모든 분석 시행을 분모로 한다. 비용 지표는 성공 시행의 산술평균과 표본 표준편차(`ddof=1`)를 사용한다. 성공률의 오차막대는 95% Wilson 구간이다.
- `query_probability`는 시행별 `(질문이 한 번 이상 발생한 물리 step 수 / 물리 step 수) × 100`이다. 집계는 시행별 비율의 평균이다. 그래프·표 CSV에서는 백분율 단위다.
- `questions`는 외부 요청 횟수, `steps`는 물리 행동 수다. Query-Action에서도 query action을 plan length에 더하지 않는다.
- 시간 단위는 초다. `plan_time_s`와 `measured_search_time_s`는 실제 검색 시간이며, Oracle의 KnowNo·IntroPlan에서는 기록된 추론 `elapsed_sec`의 합이다. 현재 생성하는 VLM·Human planning-time 그림도 실제 검색 시간을 사용한다.
- `planning_residual_time_s = total_time_s - interaction_time_s - execute_time_s`는 별도로 저장하는 잔여시간 추정치다. 실제 검색 시간과 합치지 않는다. 기존 그림은 이 추정치를 사용했으므로 시간 그림 일부는 새 출력과 다르다.
- 성공 시행이 1개인 경우 표준편차는 빈 값이며 그림에 `*`를 표시한다. 결과 로그의 성공 여부를 사용한다.

### 사람 데이터

`human_summary.csv`를 주 입력으로 사용하고 실제 로그에서 재계산한 24행과 전 항목의 일치를 검증했다. `human_aggregate.csv`는 함께 보존했지만 그래프를 만드는 데 필요한 표본 표준편차와 개별 시행 정보가 부족하므로 직접 입력으로 쓰지 않는다.

`01_BASELINE/01_processed/human_summary.csv`와 `human_aggregate.csv`는 재생성 결과다. 원래 두 CSV는 `00_raw/source_inputs/human`에 그대로 보존했다. 사람 집계 파일의 `query_rate`는 0~1 비율이고 step 수를 합산한 pooled ratio다. 논문 그래프에 쓰는 시행별 비율 평균과 구분한다.

기존 논문 Human 그림의 9월 30일·10월 2일 파일럿은 새 데이터에 합산하지 않았다. 기존 수치는 `00_raw/source_inputs/manuscript_snapshot`에 보존했다. 새 데이터는 10월 12~13일 p001~p003의 24회이며, 논문의 계획된 5명/40회와 구분한다.

## 그림·표 입력

| 실험 | 그래프 직접 입력 | 표 직접 입력 |
| --- | --- | --- |
| Baseline / Oracle | `01_BASELINE/02_graph_data/exp_oracle_baselines_data.csv` | `oracle_table.csv`, `oracle_table.tex` |
| Baseline / VLM·Human | `01_BASELINE/02_graph_data/vlm_human_statistics.csv` | `physical_table.csv`, `physical_table.tex` |
| 응답 제공 조건 비교 | `01_BASELINE/02_graph_data/response_comparison_statistics.csv` | baseline 표·통계 CSV의 environment 열 |
| Ablation | `02_ABLATION/02_graph_data/ablation_unified_by_domain.csv` | `ablation_table.csv`, `ablation_table.tex` |
| Threshold | `03_THRESHOLD/02_graph_data/manuscript_threshold_by_domain.csv` | `threshold_table.csv`, `threshold_table.tex` |

표 CSV는 반올림 전 수치를 제공한다. `.tex`는 소수점 두 자리와 평균 ± 표본 SD로 표시하는 독립 `tabular` 양식이며 논문의 지면 구성에 맞게 열을 선택할 수 있다. 그림 파일명은 유지하고, 원고에서는 `figures/04_ORACLE_BASELINE`부터 `09_THRESHOLD`까지 figure별 폴더를 사용한다. 배포 경로는 `04_CODE/figure_paths.py`에서 관리한다.

## 재생성

프로젝트 루트에서:

```bash
python3 02_BRL_CODE/latex_zips/latex_auro/asset/BRL_EXP/04_CODE/rebuild.py
```

`build_data.py`는 Python 표준 라이브러리만 사용한다. 그림 생성에는 `04_CODE/requirements.txt`의 matplotlib이 필요하다. 모든 스크립트는 현재 작업 디렉터리와 무관하게 실행할 수 있다.

| 코드 | 역할 |
| --- | --- |
| organize_sources.py | 원본 배치를 복사하고 source index와 해시 목록 작성. 원본이 갱신됐을 때 명시적으로 재실행 |
| build_data.py | 패키지 내 raw 검증·파싱, 시행 선택, statistics 및 그래프·표 CSV/TeX 생성 |
| summarize_human.py | 사람 세션 로그에서 human_summary.csv 재생성 |
| aggregate_human_summary.py | 사람 summary에서 human_aggregate.csv 재생성 |
| classify_vlm_failures.py | VLM 세션 로그를 사용한 기존 실패 분류 재생성 |
| plot_experiment_panels.py | 그래프 입력 CSV를 읽어 33개 개별 PDF와 manifest 생성 |
| figure_paths.py | 원고 figure별 폴더 경로와 배포 manifest 관리 |
| rebuild.py | 데이터 → 실패 분류 → 그림 생성 순서로 실행 |
| update_human.py | 사람 원자료·CSV만 새로 반영하고 Human 7개 및 응답 제공 조건 비교 4개 PDF를 논문 figures에 자동 적용 |

Oracle·VLM 원자료를 갱신할 때는 `organize_sources.py`, `rebuild.py` 순서로 실행한다. 사람 데이터 추가는 아래 자동 갱신 명령을 사용한다. `source_files.csv`에 기록된 복사 파일을 수정하면 해시 검증에서 중단한다.

일반 `rebuild.py`는 결과를 각 실험의 `03_figures`에 생성한다. 사람 데이터 자동 갱신은 Human·응답 제공 조건 비교 그림을 논문 루트 `figures`에도 적용한다. 본문 `.tex`의 수치와 표는 자동 변경하지 않는다. 생성된 그림을 논문에 적용할 때:

```bash
python3 02_BRL_CODE/latex_zips/latex_auro/asset/BRL_EXP/04_CODE/rebuild.py --publish-figures
```

이 옵션은 33개 실험 PDF를 논문 `figures`의 figure별 폴더에 복사하고, 데이터·manifest는 `figures/00_DATA`에 저장한다. 본문 수치·표·참여자 수는 별도로 갱신해야 한다.

## 검증 및 원본 출처

`validation.json`은 raw 해시 검증, 사람 summary 일치, 공통 Ours, 원래 Oracle의 40개 지표 행 및 ablation·threshold 그림 수치 일치, VLM 기존 통계 일치를 기록한다. 기존 VLM planning-time 그림의 잔여시간 추정치도 정리 데이터에 보존했다.

원본 데이터 위치:

- Oracle: `02_BRL_CODE/experiments_final/01_Oracle/{00_threshold,01_when_what_random,02_when_what_policy_ablation,03_baseline}`.
- VLM: `02_BRL_CODE/experiments_final/02_VLM/00_vlm_exp`, `04_Total_Summary/data/vlm_*_source.csv`. 재설치도 이 경로를 사용하며 이전 asset 묶음에 의존하지 않는다.
- Human: `08_Delete_Me/00_real_human`.
- 논문 기준 수치: `02_BRL_CODE/experiments_final/04_Total_Summary/data`, `latex_auro/figures`의 원래 CSV·manifest.

VLM 정리 CSV의 원본 `model` 값은 KnowNo에서 `gpt-4o`, Ours에서 미기록으로 남아 있다. `05_exp_setting.tex`의 Gemma 표기를 원본 로그 값에 덮어쓰지 않았다. 실험 조건을 본문에 확정할 때 확인할 항목이다.

## p004·p005 추가 후 자동 그림 갱신

`08_Delete_Me/00_real_human/p004`, `p005`에 기존 참여자와 같은 구조의 로그를 넣는다. 참여자 수를 고정하지 않으므로 p004만 추가된 중간 시점에도 실행할 수 있다. 빈 참여자 폴더는 시행을 추가하지 않는다.

프로젝트 루트에서 기존 두 명령만 실행한다:

```bash
python3 08_Delete_Me/00_real_human/summarize_human.py
python3 08_Delete_Me/00_real_human/aggregate_human_summary.py
```

두 번째 명령이 CSV 저장 후 `04_CODE/update_human.py`를 자동 호출한다. 현재 로그와 summary·aggregate의 일치를 먼저 검증하고, 사람 원자료 사본·해시·CSV를 갱신한 뒤 시행별 통계를 재계산한다. 사람 그림 7개(`exp_human_*.pdf`) 및 Oracle·VLM·Human 비교 그림 4개(`exp_response_*.pdf`)를 `latex_auro/figures/06_HUMAN_BASELINE`과 `07_RESPONSE_COMPARISON`에 같은 파일명으로 덮어쓴다. 논문에 이미 있는 `\includegraphics` 경로를 그대로 사용할 수 있다. Oracle·VLM·ablation·threshold 그림은 이 자동 갱신에서 덮어쓰지 않는다.

24회, 32회, 40회로 늘어나는 표본 수는 로그에서 자동 계산한다. 성공률은 모든 시행, 비용 지표는 성공 시행 기준이다. 생성된 표 CSV·TeX와 manifest도 갱신된다. 본문에 직접 적어 둔 수치·참여자 수 및 표의 빈 칸은 별도로 갱신해야 한다.

자동 그림 갱신 없이 CSV만 생성하려면 `aggregate_human_summary.py --no-publish`를 사용한다. 별도 `--input` 또는 `--output` 경로를 지정한 분석도 자동 적용하지 않는다. 이미 CSV를 생성한 경우 `04_CODE/update_human.py`만 직접 실행해도 된다.

검증용 `04_CODE/test_human_update.py`는 임시 폴더에서만 참여자 로그를 복제하여 24→32→40회 집계와 12개 그림 적용을 검사한다. 이 검증 데이터는 실제 데이터·논문 그림에 들어가지 않는다.

사람 로그를 추가한 후에는 `python3 08_Delete_Me/00_real_human/aggregate_human_summary.py` 하나만 실행해도 된다. 기본 경로에서는 최신 로그로 `human_summary.csv`를 먼저 재생성한 뒤 aggregate와 그림을 갱신한다. `--no-publish`는 두 CSV만 갱신한다. 별도 입력/출력 경로를 지정한 분석에서는 원자료 재요약과 자동 그림 적용을 수행하지 않는다.

### 전체 시행의 질문 1회당 상호작용 시간

`interaction_per_query.py`는 성공·실패를 모두 포함하여 전체 상호작용 시간 합계 / 전체 질의 수 합계를 계산한다. 질문이 0회인 시행은 제외하며, 최댓값을 제거하지 않는다. aggregate의 `avg_interaction_per_query`도 같은 정의다. `metric_n`은 개별 질의 수다. 시행 평균의 표준편차를 질의별 표준편차로 사용하지 않으므로 이 독립 그림에는 오차막대를 표시하지 않는다. 입력은 정리된 `human_summary.csv`, 그래프 데이터는 `01_BASELINE/02_graph_data/human_interaction_per_query_all.csv`다. 독립 그림은 `latex_auro/figures/exp_human_interaction_time_per_query_all.pdf`에 생성된다. 사람 데이터 갱신 시 이 그림도 자동 갱신되어 관련 PDF는 총 12개, 전체 생성 PDF는 33개다. 나머지 비용 지표의 success-only 집계 기준은 그대로다.
