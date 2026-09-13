# Experiment assets (2026-09-12)

이 디렉터리는 현재 논문 개편 이후 완료된 POMDP 실험 결과를 정리한 것이다.
논문 본문에는 자동으로 삽입하지 않았으며, 표와 문장을 작성할 때 사용할 수
있는 데이터와 재현 자료만 모았다.

## 1. Threshold sweep

- 구성: 2 domains x 11 thresholds x 5 scenes x 40 runs
- 총 실행: 4,400
- threshold: 0.0부터 1.0까지 0.1 간격
- 비교 공정성: 같은 domain/scene/run seed를 모든 threshold가 공유
- 최대 physical step: 50
- 그림: `../../figures/system_eval/threshold/`
- 표: `threshold/threshold_paper_table.csv` 및 `.tex`
- 전체 집계: `threshold/threshold_summary.csv`
- 실행별 데이터: `threshold/raw_runs.csv`

논문에서 사용할 예정인 tau=0.8에서 Waste는 성공률 100.0%, 평균 질문 수
13.3회, step당 질문 확률 0.43이다. Tomato는 성공률 93.5%, 평균 질문 수
15.5회, step당 질문 확률 0.44이다.

tau=1.0에서는 두 domain 모두 성공률이 96.0%이지만, step당 질문 확률이
Waste 1.00, Tomato 0.96까지 증가한다. 따라서 모든 불확실성을 질문으로
해소하는 것보다 tau=0.8이 질문 비용과 성공률 사이의 균형을 보여준다.

## 2. When--What ablation

- 구성: 2 domains x 4 policies x 5 scenes x 40 runs
- 총 실행: 1,600
- 공통 설정: tau=0.8, random query probability=0.4
- policies: Random, Ours-when-only, Ours-what-only, Ours
- 비교 공정성: 네 policy가 같은 domain/scene/run seed를 공유
- 그림: `../../figures/system_eval/when_what/`
- 표: `when_what/when_what_paper_table.csv` 및 `.tex`
- paired 검정: `when_what/when_what_paired_success_contrasts.csv`
- 전체 집계: `when_what/when_what_summary.csv`
- 실행별 데이터: `when_what/raw_runs.csv`

성공률은 다음과 같다.

| Policy | Waste | Tomato |
|---|---:|---:|
| Random | 76.0% | 64.0% |
| Ours-when-only | 100.0% | 99.5% |
| Ours-what-only | 76.0% | 66.0% |
| Ours | 100.0% | 99.0% |

이 결과에서는 질문 시점을 proposed rule로 바꾸는 효과가 크다. Random 대비
Ours의 paired success 차이는 Waste +24.0%p, Tomato +35.0%p이며, 각각의
McNemar exact p-value는 7.11e-15와 1.69e-21이다. 반면 random timing에서
질문 내용만 proposed selection으로 바꾼 Ours-what-only의 개선은 작다.

## 3. Legacy scale experiment

과거 scale 결과는 삭제하지 않고 `legacy_scale/`과
`../../figures/system_eval/legacy_scale/`에 보존했다. 현재 계획에서는 핵심
본문 실험이 아니므로 threshold 및 When--What 결과와 분리했다.

## 4. 아직 포함하지 않은 결과

KnowNo와 Query-Action POMCP의 통합 baseline batch는 완료 결과가 아니다.
2026-09-12에 생성된 일부 KnowNo 실행은 scoring API 호출을 03/04 구현과
동일하게 수정하기 전에 수행되었으므로 이 묶음에 포함하지 않았다. 두 baseline
실험이 완주된 뒤 별도로 재집계해야 한다.

## Provenance

- Threshold source: `experiments/system_eval/figure/threshold/00_20260912_202158/`
- When--What source: `experiments/system_eval/figure/when_what/00_20260912_202414/`
- Threshold paired seeds: `provenance/threshold_paired_seeds.csv`
- When--What paired seeds: `provenance/when_what_paired_seeds.csv`
- 집계 및 그림 스크립트는 `provenance/`에 복사했다.
