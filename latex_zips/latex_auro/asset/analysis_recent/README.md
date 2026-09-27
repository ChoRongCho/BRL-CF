# 논문 실험 데이터와 분석

이 폴더가 실험 분석의 **원본 저장소(canonical source)**다. 원본 실행 로그는 각 패키지의 `00_raw/`, 실행별 가공값은 `01_processed/episodes.csv`, 그림에 직접 사용한 값은 `02_graph_data/figure_data.csv`, 그림은 `03_figures/`에 둔다.

네 핵심 실험의 목적, 조건 정의 및 전체 파라미터는 `EXPERIMENT_SETTINGS.md`에 정리했다.

## 논문 핵심 실험

| 실험 | 패키지 | 상태 | 실행 결과 |
|---|---|---|---:|
| Threshold sweep | `threshold_20260912` | paper ready | 4,400/4,400 valid |
| When–What–Random 2×2 ablation | `when_what_original_20260912` | paper ready | 1,600/1,600 valid |
| When–What policy ablation | `when_what_policy_ablation_20260921` | paper ready | 1,600 slots; 1,596 valid, 4 execution errors counted as failures |
| Baseline comparison | `query_baselines_current_20260920` | paper ready | 2,000/2,000 valid, 5 conditions |

`when_what_policy_ablation_20260921`은 새 action-CP When 결과다. 이전 belief-state CP 결과는 `when_what_policy_ablation_belief_state_cp_20260921`에 분리 보존했고 논문 수치에는 사용하지 않는다. 더 오래된 구현 결과는 `when_what_mechanisms_20260921_legacy`에 보존했다.

## Baseline 5조건

`query_baselines_current_20260920`은 다음 조건을 함께 보존한다.

1. Ours
2. Query-Action original (`gamma=0.2`, `query_cost=1.0`)
3. KnowNo
4. IntroPlan
5. Query-Action (`Tomato gamma=0.5`, `Waste gamma=0.9`, `query_cost=0.0`, `n_simulations=100`)

기존 네 조건은 변경하지 않았고 Tomato `gamma=0.5`, Waste `gamma=0.9`, `query_cost=0.0`, `n_simulations=100`인 Query-Action 400회를 추가했다.

## 공통 패키지 구조

```text
<package>/
├── 00_raw/                       # 실제 원본 로그, console, trace, seed metadata
├── 00_raw_sources.csv            # 분석 입력 파일과 SHA-256
├── manifest.json                 # 조건, 라벨, source 목록, 상태
├── STATUS.md                     # paper-ready / rerun-required 표시
├── 01_processed/
│   ├── episodes.csv              # 실행별 1차 가공값
│   ├── summary.csv               # 전체·도메인별 집계
│   ├── scenes.csv                # scene별 집계
│   ├── paired.csv                # paired 분석을 사용한 패키지만 생성
│   └── paired_episodes.csv       # paired 분석을 사용한 패키지만 생성
├── 02_graph_data/figure_data.csv # 그림과 일치하는 최종 CSV
├── 03_figures/                   # PNG와 PDF
├── README.md
└── report.md
```

## 보조 실험

최상단의 `gamma_*`와 `n_simulations_*` 여섯 패키지는 9월 17일 보조 tuning 기록이다. 삭제하지 않지만 논문 핵심 네 실험과 분리해 `index.csv`에서 `auxiliary_tuning/archive`로 표시한다. Query-as-Action gamma×cost 내부 tuning은 `experiments_logs/query_action_tuning/20260921_130229_529697/analysis/`에 있다.

## 재생성

프로젝트 루트에서 실행한다.

```bash
python3 experiments_logs/analysis_recent/scripts/pipeline.py --package <package> --stage all
python3 experiments_logs/analysis_recent/scripts/validate_paper_outputs.py
```

LaTeX용 사본은 `latex_zips/latex_auro/asset/analysis_recent/`에 두며, 원본 raw는 이 폴더만을 기준으로 한다.
