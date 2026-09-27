# 논문 자산 안내

현재 실험 그림과 표의 기준 자산은 [`analysis_recent/`](analysis_recent/README.md)에 정리되어 있다.

- Threshold sweep: 논문 사용 가능
- When–What–Random 2×2 ablation: 논문 사용 가능
- Query baseline comparison: 논문 사용 가능, 기존 4조건과 tuned Query-as-Action을 합친 5조건
- When–What policy ablation: 논문 사용 가능, 수정된 4조건 1,600회

실행 원본의 기준 위치는 `experiments_logs/analysis_recent/`이다. 이 자산 폴더의 `00_raw/`는 Git에서 제외하며, 논문에는 `01_processed/`, `02_graph_data/`, `03_figures/`, `report.md`를 사용한다.

`experiments_20260912/`, `experiments_20260916/` 등 기존 폴더는 과거 정리본이다. 최신 논문 수치는 `analysis_recent/`을 기준으로 한다.
