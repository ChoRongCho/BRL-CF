# analysis_recent_v2

논문 결과 보고에 사용하는 최신 실험 분석 묶음이다. 모든 핵심 POMCP 실험은
`n_simulations=100`으로 통일되어 있다. 실험별 조건과 전체 파라미터는
[`EXPERIMENT_SETTINGS.md`](EXPERIMENT_SETTINGS.md)에 정리되어 있다.

전체 실험의 결과와 contribution 중심 해석은 [`REPORT.md`](REPORT.md)에서 확인한다.

## 실험

| 폴더 | 내용 | 실행 수 |
|---|---|---:|
| `00_threshold/` | confidence threshold에 따른 task success–query trade-off | 4,400 |
| `01_when_what_random/` | proposed/random When과 What의 2×2 ablation | 1,600 |
| `02_when_what_policy_ablation/` | CP/Value 기반 When·What 정책 ablation | 1,600 slots |
| `03_baseline/` | Ours와 Query-Action, KnowNo, IntroPlan 비교 | 2,000 |
| `legacy/` | 논문 결과에 사용하지 않는 이전 구현의 raw·출처 기록 | — |

`02_when_what_policy_ablation/`은 1,596개 유효 결과와 실행 오류 4개를 포함하며,
오류는 task failure로 집계한다.

## 공통 구조

- `00_raw/`: 원본 실행 로그. 수정하거나 삭제하지 않는다.
- `00_raw_sources.csv`, `manifest.json`: raw 출처, 해시, 조건 및 실행 metadata.
- `01_processed/episodes.csv`: episode 단위 분석값.
- `01_processed/summary.csv`: 조건·도메인별 핵심 집계.
- `01_processed/scenes.csv`: scene별 집계.
- `01_processed/paired*.csv`: 동일 scene·seed 기반 paired 비교.
- `02_graph_data/figure_data.csv`: 그림에 직접 사용한 수치와 오차막대.
- `03_figures/`: 결과 그림(PNG/PDF).
- `report.md`: 결과 해석과 주요 수치.
- `README.md`: 개별 실험의 조건과 집계 기준.

## 확인 순서

결과는 각 실험의 `report.md` → `01_processed/summary.csv` →
`02_graph_data/figure_data.csv` 순서로 확인한다. 개별 실행을 추적할 때만
`episodes.csv`와 `00_raw/`를 사용한다.

`legacy/`는 provenance 보존 전용이며 현재 논문 수치나 그림에 사용하지 않는다.
