# VLM 실험 최종 패키지

Tomato・Waste × POMDP・KnowNo × scene 1–4 × 각 10회 = 총 160회.
최종 결과는 [report.md](report.md), [report.pdf](report.pdf)를 참조한다.

```text
00_vlm_exp/
├── 00_raw/<domain>/scene_<N>/<method>/<session>/
│   └── 원본 로그・실험 결과・recording.json (영상 제외)
├── 01_processed/
│   ├── episodes.csv / runs.csv / failures.csv
│   ├── summary.csv / progress.csv / success_comparisons.csv
│   └── tomato/・waste/ 도메인별 CSV
├── 02_graph_data/
│   ├── figure_data.csv (aggregation으로 성공 실행/전체 실행 구분)
│   ├── success_only/ (그림 입력 수치 및 정의)
│   └── all_runs/ (그림 입력 수치・표 및 정의)
└── 03_figures/
    ├── overview.png / overview.pdf / overview.svg
    ├── success_only/ (성공 실행 평균 그림과 도메인별 그림)
    └── all_runs/ (전체 실행 평균 그림 및 요약표)
```

성공률은 전체 시행 기준이며 질문・행동・시간은 성공 실행 평균과 전체 실행 평균을 따로 제공한다.
그림 오차막대는 표본 표준편차이며, 보고서의 성공률 구간은 95% Wilson CI이다.
원본 출처와 SHA-256은 `00_raw_sources.csv` 및 `manifest.json`에 기록했다.
영상은 제외했으며 원본 session의 나머지 파일은 내용 변경 없이 복사했다.

재생성:

```bash
cd /home/fr/brl
python3 collect_exp/package_vlm_exp.py
```

원 분석 재생성은 `collect_exp/vlm_comparison_final_20261005/analyze.py`를 먼저 실행한다.
