# 논문 핵심 실험 설정 보고서

작성 기준일: 2026-10-05 (Value 정책 재실험 반영)  
Canonical analysis root: `analysis_recent_v2/`

## 1. 실험 구성 개요

논문에 사용하는 핵심 시스템 실험은 다음 네 가지다.

| 번호 | 실험 | 분석 패키지 | 목적 | 실행 상태 |
|---:|---|---|---|---:|
| 1 | Threshold sweep | `00_threshold` | belief confidence threshold에 따른 성공률–질문 수 변화를 확인 | 4,400/4,400 valid |
| 2 | When–What–Random 2×2 ablation | `01_when_what_random` | Ours의 When과 What이 각각 성공률과 질문 효율에 기여하는지 확인 | 1,600/1,600 valid |
| 3 | When–What policy ablation | `02_when_what_policy_ablation` | When 또는 What을 CP/value 정책으로 교체했을 때의 차이를 확인 | 1,600 slots; 1,596 valid, 4 failures |
| 4 | Query baseline comparison | `03_baseline` | Ours를 Query-Action, KnowNo, IntroPlan과 비교 | 2,000/2,000 valid |

각 패키지는 다음 구조를 갖는다.

```text
<package>/
├── 00_raw/                       # 이동하여 보존한 원본 실행 로그
├── 00_raw_sources.csv            # 분석 입력 raw 경로와 SHA-256
├── manifest.json                 # 조건, 라벨, raw source와 실행 상태
├── 01_processed/
│   ├── episodes.csv              # episode별 1차 가공값
│   ├── summary.csv               # 전체·domain별 집계
│   ├── scenes.csv                # scene별 집계
│   ├── paired.csv                # paired 비교 결과
│   └── paired_episodes.csv       # paired episode 값
├── 02_graph_data/figure_data.csv # 그림을 직접 생성한 최종 수치
├── 03_figures/                   # PNG와 PDF
├── README.md
└── report.md
```

## 2. 공통 환경과 실행 단위

| 항목 | 설정 |
|---|---|
| Domain | `tomato`, `wastesorting` |
| Scene | domain별 scene 1–5 |
| 반복 | condition/domain/scene별 40회 |
| Condition당 episode | 2 domains × 5 scenes × 40 = 400회 |
| 최대 step | 50 |
| POMCP simulations | 100 |
| 기본 discount factor | `gamma=0.2` |
| 기본 confidence threshold | `0.8` |
| 피드백 | 실제 사람이 아닌 simulation oracle |
| Seed 정렬 | 같은 domain·scene·iteration의 비교 조건에 같은 초기 seed 사용 |

Seed 정렬은 조건 사이의 초기 난수 설정을 맞추고 실행을 재현하기 위한 것이다. 서로 다른 정책이 다른 행동을 선택한 이후에는 상태 및 난수 소비 경로가 달라질 수 있다.

질문 수와 행동 수는 분리해서 기록한다. Query-as-Action의 최대 50 decision에는 물리 행동과 질문 행동이 모두 포함되지만, 결과의 `steps` 또는 plan length에는 물리 행동만 기록한다. 전체 decision 수는 원본 로그의 `total_actions`에서 확인한다.

## 3. Experiment 1: Threshold sweep

### 3.1 목적

Ours가 질문을 시작하는 belief confidence threshold를 변화시키면서 성공률과 질문 횟수의 trade-off를 확인한다. 질문 내용 선택은 모든 조건에서 Ours의 expected information gain(EIG)을 유지한다.

### 3.2 조건과 실행 수

| 항목 | 설정 |
|---|---|
| Threshold | `0.0, 0.1, ..., 1.0`의 11개 값 |
| Condition당 실행 | 400회 |
| 총 실행 | 11 × 400 = 4,400회 |
| When | belief confidence가 threshold보다 낮으면 질문 시작 |
| What | 후보 state fact 중 EIG가 가장 큰 fact |

동일 domain·scene·iteration의 11개 threshold는 같은 seed를 사용한다.

### 3.3 Planner 설정

| 파라미터 | 값 |
|---|---:|
| `gamma` | 0.2 |
| `n_simulations` | 100 |
| `max_step` | 50 |
| `max_depth` | 20 |
| UCB `c` | 1.0 |
| `epsilon` | 0.005 |
| `max_particles` | 250 |

### 3.4 완료 상태와 관측값

- 4,400/4,400 episode가 정상 파싱됐다.
- 누락, 중복, seed 불일치 및 실행 오류가 없다.
- 전체 성공률과 평균 질문 수는 threshold 0.8에서 97.0%/8.58회, 0.9에서 99.75%/9.37회, 1.0에서 100%/16.98회다.
- 1.0은 0.9 대비 성공률 증가는 0.25%p이지만 질문 수가 크게 증가한다.

## 4. Experiment 2: When–What–Random 2×2 ablation

### 4.1 목적

질문 시점인 When과 질문 내용인 What을 Ours와 random으로 교차하여 두 요소의 역할을 분리한다.

### 4.2 조건 정의

| Condition | When 정책 | What 정책 |
|---|---|---|
| Random | 각 물리 step에서 확률 0.4로 질문 시작 | 모호한 state fact 중 무작위 선택 |
| What only (`ours_what_only`) | 각 물리 step에서 확률 0.4로 질문 시작 | Ours EIG |
| When only (`ours_when_only`) | belief confidence `< 0.8` | 모호한 state fact 중 무작위 선택 |
| Ours | belief confidence `< 0.8` | Ours EIG |

Random What은 현재 belief의 후보 state 사이에서 값이 달라지는 fact 중 하나를 무작위로 선택한다. Ours What은 질문 후 기대 엔트로피 감소량이 가장 큰 fact를 선택한다.

### 4.3 파라미터와 실행 수

| 파라미터 | 값 |
|---|---:|
| `gamma` | 0.2 |
| `n_simulations` | 100 |
| confidence threshold | 0.8 |
| random-query probability | 0.4 |
| `max_step` | 50 |
| `max_depth` | 20 |
| UCB `c` | 1.0 |
| `epsilon` | 0.005 |
| `max_particles` | 250 |
| `max_belief_particles` | 8,000 |
| 총 실행 | 4 conditions × 400 = 1,600회 |

동일 domain·scene·iteration의 네 조건은 같은 seed를 사용한다.

### 4.4 완료 상태와 관측값

| Condition | 전체 성공률 | 전체 평균 질문 수 |
|---|---:|---:|
| Random | 73.5% | 8.25 |
| What only | 73.5% | 6.09 |
| When only | 98.75% | 13.92 |
| Ours | 98.75% | 8.46 |

- 1,600/1,600 episode가 정상 파싱됐다.
- Ours When을 사용한 두 조건의 성공률이 98.75%로 같았다.
- Ours는 When only와 같은 성공률을 유지하면서 평균 질문을 13.92회에서 8.46회로 줄였다.
- 이 결과는 When이 성공률과 연결되고 What이 질문 효율과 연결된다는 주장을 뒷받침한다.

## 5. Experiment 3: When–What policy comparison

Value-When은 2026-10-05 20:43 질문 반복 수정 배치 400회, Value-What은 19:03 배치 400회로 구성한다. Ours・CP-When은 기존 각 400회로 유지한다. 정상 결과 1,596개와 CP 오류 4개를 포함한다.

| 조건 | When | What | 반복·종료 | Tomato gamma | Waste gamma | query cost | simulations |
|---|---|---|---|---:|---:|---:|---:|
| Ours | confidence < 0.8 | EIG | confidence threshold | 0.2 | 0.2 | 1.0 (value 평가 미사용) | 100 |
| CP-When | action CP ambiguity | EIG | 답변마다 CP 재평가 | 0.2 | 0.2 | 1.0 (value 평가 미사용) | 100 |
| Value-When | max Q(query) > max Q(next physical) | EIG | 진입 후 confidence threshold 반복 | 0.5 | 0.9 | 0.0 | 100 |
| Value-What | confidence < 0.8 | 최고 질문 Q | confidence threshold | 0.5 | 0.9 | 0.0 | 100 |

공통 max_step=50, max_depth=20, epsilon=0.005, UCB c=1.0이다. Value 평가의 failure_penalty=10.0, answer_accuracy=1.0이며 답변은 auto Oracle이다. CP의 qhat은 Tomato 0.8404, Waste 0.8704, score_temperature=5.0이다.

Value gamma는 질문 평가와 실제 물리 행동 계획 모두에 적용한다. 가치 평가 simulation 예산은 max(100, root 후보 수+1)로 늘어날 수 있다. query cost 0은 질문 보상을 0으로 만든다.

When은 관측으로 belief를 갱신한 다음, 다음 물리 행동 전에 판단한다. 질문 시작을 판단한 최고 Q 질문을 Value-When에서 실제로 그대로 묻는 것은 아니며 What은 EIG이다. Value-What은 root에서 질문 Q끼리 비교하고, 물리 행동 Q와 비교하여 질문을 취소하지 않는다.

**참조 결과 제한:** 새 Value-When에는 질문 1개 제한이 없다. CP의 보관된 결과는 수정 전 반복 gate 구현이다. Ours/CP와 Value의 gamma도 달라 네 조건 전체를 정책만의 통제 비교로 해석하지 않는다.

| 조건 | 성공률 | 전체 평균 질문 | 성공 실행 평균 질문 |
|---|---:|---:|---:|
| ours | 99.00% | 8.488 | 8.490 |
| cp_when | 68.50% | 4.803 | 5.690 |
| value_when | 83.75% | 13.870 | 15.406 |
| value_what | 98.75% | 18.988 | 19.068 |

Value-What 유지 배치: `experiments_logs/when_what_policy_ablation/20261005_190346_519071`. 이전 Value 결과는 `legacy/when_what_policy_ablation_before_20261005/`에 보존한다. 세부 해석과 paired 검정은 개별 `report.md`를 참조한다.


Value-When 새 배치: `experiments_logs/when_what_policy_ablation/20261005_204332_446690`. 1개 제한을 사용한 직전 패키지는 `legacy/when_what_policy_ablation_before_threshold_loop/`에 보존한다.

## 6. Experiment 4: Query baseline comparison

### 6.1 목적

Ours를 세 baseline과 비교하고, Query-as-Action의 별도 parameter search에서 선택한 domain-specific gamma 설정을 추가하여 총 다섯 조건을 비교한다.

### 6.2 조건과 파라미터

| Condition | 주요 설정 |
|---|---|
| Ours | `gamma=0.2`, threshold `0.8`, `n_simulations=100`, EIG What |
| Query-Action original | `gamma=0.2`, `query_cost=1.0`, `failure_penalty=10.0`, `answer_accuracy=1.0`, `n_simulations=100` |
| KnowNo | GPT-4o, prompt v1, generation temperature 0.3, score temperature 5.0, Tomato `qhat=0.9011194251944138`, Waste `qhat=0.845165078859924` |
| IntroPlan | GPT-4o, prompt v1, generation temperature 0.3, score temperature 5.0, `top_k=3`, Tomato `qhat=0.9855517605160237`, Waste `qhat=0.980493087109584` |
| Query-Action selected | Tomato `gamma=0.5`, Waste `gamma=0.9`, `query_cost=0.0`, `failure_penalty=10.0`, `answer_accuracy=1.0`, `n_simulations=100` |

`n_simulations`는 Ours와 두 Query-Action 조건의 raw log에서 100으로 확인된다. KnowNo와 IntroPlan의 분석 로그에는 이 필드가 기록되지 않으므로 해당 두 조건의 표에는 별도 POMCP simulation 값을 주장하지 않는다.

두 Query-Action 조건의 planner 설정은 다음과 같다.

| 파라미터 | 값 |
|---|---:|
| `max_step` | 50 total decisions, including QueryAction |
| `max_depth` | 20 |
| UCB `c` | 1.0 |
| `epsilon` | 0.005 |
| `max_particles` | 250 |
| `max_belief_particles` | 8,000 |
| `max_node_particles` | 8,000 |

### 6.3 실행 수와 완료 상태

- 각 조건은 400회이며 총 2,000/2,000 episode가 valid다.
- KnowNo와 IntroPlan은 2026-09-30 통합 재실행 800회로 교체했다. Tomato와 Waste 모두
  prompt v1, generation temperature 0.3, score temperature 5.0, exact oracle로 실행했다.
- 2026-09-22에 완료한 Query-Action selected 400회는 Tomato `gamma=0.5`, Waste `gamma=0.9`로 실행됐다.
- Python 3.8로 잘못 시작해 import 전에 실패한 이전 400회는 분석에 포함하지 않았다. Python 3.10 `brl` 환경에서 완료한 400/400 결과만 사용한다.
- 기존 공통 `gamma=0.5` variant는 `experiments_logs/analysis_archive/query_baselines_pre_domain_gamma_20260920`에 보존하고 논문용 최신 패키지에서 제외했다.

### 6.4 관측값

| Condition | 전체 성공률 | 전체 평균 질문 수 |
|---|---:|---:|
| Ours | 97.5% | 8.42 |
| Query-Action original | 54.5% | 1.18 |
| KnowNo | 29.5% | 3.58 |
| IntroPlan | 45.75% | 6.28 |
| Query-Action selected | 72.25% | 16.86 |

Query-Action selected의 domain별 결과는 Tomato 57.5%/평균 질문 9.73회, Waste 87.0%/평균 질문 24.0회다. Parameter를 조정한 조건에서도 Ours의 전체 성공률이 더 높고 평균 질문 수는 더 적었다.

## 7. 결과 포함 및 제외 기준

1. `manifest.json`에 고정된 raw source와 SHA-256이 일치하는 episode만 정상 결과로 파싱한다.
2. 정상 종료한 task failure는 성공률 분모와 전체 질문·행동 평균에 포함한다.
3. 성공 실행만의 질문·행동 평균은 `*_success_only` 열과 별도 그림으로 분리한다.
4. 실행 오류는 정상 수치로 변환하지 않는다. Policy ablation의 지정된 4건만 요청된 기준에 따라 성공률에서 실패로 집계한다.
5. 오래된 `n_simulations=200`, 폐기한 belief-state CP 정의, 수정 전 policy 구현 및 이전 Query-Action variant는 archive에 보존하지만 논문용 최신 수치에 포함하지 않는다.
6. 그림은 각 패키지의 `02_graph_data/figure_data.csv`를 직접 읽어 생성한다.

## 8. 최종 무결성 상태

| 실험 | Raw/manifest | Episode CSV | Graph CSV | PNG/PDF | LaTeX mirror |
|---|---|---|---|---|---|
| Threshold sweep | 확인 완료 | 4,400 valid | 154 rows | 생성 완료 | 일치 |
| When–What–Random | 확인 완료 | 1,600 valid | 56 rows | 생성 완료 | 일치 |
| When–What policy | 확인 완료 | 1,596 valid + 4 counted failures | 56 rows | 생성 완료 | 일치 |
| Baseline comparison | 확인 완료 | 2,000 valid | 70 rows | 생성 완료 | 일치 |

정리 전 검증 스크립트로 raw hash, CSV 집계, paired episode 수와 그림 파일을 확인했다.
검증 결과는 최상단 `raw_integrity_check.json`에 보존한다.

## 9. 결과 확인 경로

논문용 수치는 `01_processed/summary.csv`와 `02_graph_data/figure_data.csv`에서
가져간다. 개별 episode 확인이 필요할 때 `episodes.csv`의 `raw_source`와
`raw_sha256`을 사용하고, 원본 내용은 `00_raw/`에서 확인한다.
