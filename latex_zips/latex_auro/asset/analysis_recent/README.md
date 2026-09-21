# 논문 실험 데이터·분석

## 핵심 실험 4개

| 실험 | 폴더 | 실행 슬롯 | 유효 결과 |
|---|---|---:|---:|
| Threshold sweep | [threshold_20260912](threshold_20260912/README.md) | 4400 | 4400 |
| When–What–Random ablation | [when_what_original_20260912](when_what_original_20260912/README.md) | 1600 | 1600 |
| When–What mechanism ablation | [when_what_mechanisms_20260921](when_what_mechanisms_20260921/README.md) | 1600 | 1599 |
| Baseline comparison | [query_baselines_current_20260920](query_baselines_current_20260920/README.md) | 1600 | 1600 |

네 실험의 원본 로그와 부속 파일은 각 폴더의 `00_raw/`로 **실제 이동**했습니다. 심볼릭 링크나 복사본이 아닙니다. 총 16,909개 파일의 이동 전후 SHA-256을 검증했습니다. 이동 이력은 `raw_migration.json`, 검증 결과는 `raw_integrity_check.json`에 있습니다.

## 네 실험 공통 구조

```text
<experiment>/
├── 00_raw/                         # 실제 원본 로그·trace·console·seed 파일
├── 00_raw_sources.csv              # 결과 파싱 입력 경로·해시
├── manifest.json                   # 분석 입력/조건
├── 01_processed/
│   ├── episodes.csv                # 실행별 결과·raw 출처
│   ├── summary.csv                 # 전체 및 도메인별 집계
│   ├── scenes.csv                  # 장면별 집계
│   ├── paired.csv                  # 동일 scene/seed 통계 비교
│   └── paired_episodes.csv         # 실제 매칭된 실행 쌍과 차이
├── 02_graph_data/figure_data.csv    # 그림 값·표본 수·오차막대
├── 03_figures/                     # overview 및 개별 지표 PNG/PDF
├── report.md                       # 전체/장면/paired 결과와 해석 범위
├── README.md
└── scripts/                        # 공통 파이프라인 호출
```

실제 계산은 `scripts/pipeline.py`, `scripts/detailed_analysis.py`를 공통 사용합니다. 각 실험의 analyze.py, build_graph_data.py, plot_results.py는 각각 1·2·3단계를 호출합니다.

성공률은 유효 결과 기준이며 오류 포함 성공률은 summary.csv에 별도 표기합니다. 성공률 오차막대는 95% Wilson CI, 평균은 ±1 SE입니다. Paired 성공 비교는 양측 exact McNemar와 domain별 Holm 보정, 연속값 차이는 paired 정규근사 95% CI입니다. Threshold 기준은 0.8, 나머지 세 실험은 Ours입니다. 모두 원본 seed로 매칭합니다.

## 재생성 및 검증

프로젝트 루트 `02_BRL_POMDP_CODE`에서:

```bash
python3 experiments_logs/analysis_recent/scripts/pipeline.py --stage all
python3 experiments_logs/analysis_recent/scripts/validate_paper_outputs.py
```

`--package <폴더명>`으로 개별 실험 지정 가능. `--stage 1`은 raw→실행별 CSV, `--stage 2`는 집계/paired/보고서/그림 CSV, `--stage 3`은 그림 CSV→PNG/PDF입니다. 원본 해시 변경 시 중단합니다.

## 배치 경계

- Threshold: 9월 12일 0.0–1.0 sweep.
- 기존 When–What: 같은 9월 12일 배치를 사용. Ours는 9월 12일 백업 원본을 이동했으며 이후 재실행 Ours와 혼합하지 않음.
- Baseline: 방법별 최신 활성 결과 스냅샷. 실행일·gamma 등은 episodes.csv와 report.md의 설정 표 확인.
- 메커니즘: 9월 21일 1600-slot 배치. cp_when 파싱 오류 1건 유지.
- Gamma/n_simulations 9월 17일 6개 보조 배치도 동일 집계·보고서를 생성하지만 raw는 기존 위치에 유지. `index.csv`에 전체 10개 배치 목록이 있음.

## 이동 후 재개 경로

기존 raw 경로는 더 이상 존재하지 않으며 호환 링크를 만들지 않았습니다. 메커니즘 실험의 미완료 1건을 재개할 경우 새 run root를 사용합니다:

```bash
./run/iterate_when_what_mechanisms.sh --resume --run-root experiments_logs/analysis_recent/when_what_mechanisms_20260921/00_raw
```

원본 runs.csv/로그 내부의 과거 경로 문자열은 당시 기록이므로 수정하지 않았습니다. 분석 manifest와 생성 스크립트는 새 실제 경로를 사용합니다. 재실행으로 원본이 변경되면 분석 manifest의 해시와 입력 목록을 명시적으로 갱신해야 합니다.
