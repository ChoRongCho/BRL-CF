# 논문용 최신 실험 자산

이 폴더는 논문에서 바로 사용할 가공 데이터와 그림을 모은 사본이다. 원본 로그와 분석의 기준 저장소는 `experiments_logs/analysis_recent/`이다.

## 현재 상태

| 실험 | 폴더 | 상태 | 결과 |
|---|---|---|---:|
| Threshold sweep | [`threshold_20260912`](threshold_20260912/README.md) | paper ready | 4,400/4,400 valid |
| When–What–Random 2×2 ablation | [`when_what_original_20260912`](when_what_original_20260912/README.md) | paper ready | 1,600/1,600 valid |
| When–What policy ablation | [`when_what_mechanisms_20260921`](when_what_mechanisms_20260921/README.md) | **rerun required** | 기존 결과 사용 금지 |
| Query baseline comparison | [`query_baselines_current_20260920`](query_baselines_current_20260920/README.md) | paper ready | 2,000/2,000 valid, 5 conditions |

Policy ablation 폴더는 CP-When과 Value-When 구현을 바로잡기 전 실행을 출처 확인용으로 보존한 것이다. 새 policy-ablation 실행 전까지 이 폴더의 수치와 그림을 논문 결과로 사용하지 않는다.

## Baseline 그림의 다섯 조건

1. Ours
2. Query-Action original (`gamma=0.2`, `query_cost=1.0`)
3. KnowNo
4. IntroPlan
5. Query-Action tuned (`gamma=0.5`, `query_cost=0.0`)

## 각 실험 폴더

- `01_processed/episodes.csv`: 실행별 1차 가공값
- `01_processed/summary.csv`, `scenes.csv`: 전체·도메인·scene 집계
- `02_graph_data/figure_data.csv`: 그래프와 직접 일치하는 값
- `03_figures/`: PNG와 PDF 그림
- `report.md`: 결과표와 분석
- `manifest.json`, `STATUS.md`: 입력 조건과 논문 사용 상태

`00_raw/`는 Git 제외 대상이다. 원본 확인과 재분석에는 `experiments_logs/analysis_recent/<실험>/00_raw/`를 사용한다. 전체 목록은 [`index.csv`](index.csv)에서 확인할 수 있다.
