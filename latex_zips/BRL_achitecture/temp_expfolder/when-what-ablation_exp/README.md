# Baseline 통합 비교 — 2026-09-16

IntroPlan 실행 `iterate_baseline_20260915_204042_983169.csv`의 400회를 기존 결과와 통합했다.
각 방법·도메인별 200회(5 scenes × 40 seeds), 네 방법 총 1,600회를 비교한다.
Scene·seed 매칭, 중복 없음, IntroPlan 원본 로그 400개와 seed CSV의 일치를 확인했다.
KnowNo tomato의 별도 seed=42 디버깅 실행 1개는 비교에서 제외하고 raw_runs에는 유지했다.

## 결과

질문 수와 시간은 실패를 포함한 전체 episode 평균이다. 행동 수는 기존 표와 동일하게 성공 episode만의 평균이다.

| 도메인 | 방법 | 성공률 | 평균 질문 수 | 평균 행동 수 (성공만) | 평균 시간 (초, 전체) |
| --- | --- | --- | --- | --- | --- |
| Waste | KnowNo | 36.0% | 2.54 | 10.03 | 9.46 |
| Waste | IntroPlan | 30.5% | 2.75 | 10.02 | 44.75 |
| Waste | Query-Action POMCP | 89.0% | 26.55 | 14.15 | 16.08 |
| Waste | Ours | 100.0% | 12.76 | 11.95 | 1.78 |
| Tomato | KnowNo | 43.0% | 1.79 | 14.01 | 16.26 |
| Tomato | IntroPlan | 56.5% | 3.73 | 14.73 | 73.69 |
| Tomato | Query-Action POMCP | 54.5% | 8.18 | 17.84 | 8.29 |
| Tomato | Ours | 99.0% | 15.32 | 18.47 | 6.00 |

## 해석

- IntroPlan vs KnowNo: tomato 성공률 +13.5%p (86→113/200), waste −5.5%p (72→61/200).
  Tomato는 모든 scene에서 성공률이 높았고, waste는 scene 02에서 같고 나머지에서 낮았다.
- IntroPlan 질문 수는 tomato 1.79→3.73, waste 2.54→2.75로 늘었다.
  따라서 이번 BRL 구현에서는 두 도메인 모두에 걸쳐 성공률과 질의 수가 개선됐다고 말할 수 없다.
- IntroPlan wall-clock 평균은 tomato 73.69초, waste 44.75초로 KnowNo 대비 각각 약 4.53배, 4.73배.
  설명 생성 호출을 포함한 실제 관측 시간이며, API 상태·실행 시점 차이도 섞여 있어 순수 계산 복잡도 비교는 아니다.
- Ours는 tomato 99%, waste 100%로 성공률이 가장 높다. 그러나 LLM baseline보다 질문은 많다.
  Query-Action POMCP 대비 waste는 성공률이 높으면서 질문이 적지만, tomato는 성공률과 질문 모두 높다.
- 낮은 성공률 방법은 조기 실패로 episode가 짧아지고 질문 기회가 줄 수 있다.
  전체 질문 수와 성공 episode 한정 질문 수를 함께 제공했으며, 성공 episode만 비교하는 것에도 선택 편향이 있다.

## 대응 표본 비교

동일 scene·seed 기준 IntroPlan vs KnowNo:

- Tomato: KnowNo 실패→IntroPlan 성공 37쌍, 반대 10쌍. Exact McNemar p≈0.0000985.
- Waste: 개선 5쌍, 악화 16쌍. Exact McNemar p≈0.0266.

이 p값은 다중비교 보정 전 탐색적 값이다. 전체 대응 검정은 `query_baseline_paired_success_contrasts.csv`에 기록했다.
같은 seed가 정책마다 동일한 행동·난수 소비 경로를 보장하지는 않는다.

## IntroPlan 실패 유형

Summary의 종료 사유를 분류했다. 아래 숫자는 과업 실패이며 API/프로세스 실패가 아니다.

| 도메인 | 종료 사유 | 수 |
| --- | --- | --- |
| Tomato | fresh 토마토 폐기 | 50 |
| Tomato | rotten 토마토 적재 | 20 |
| Tomato | unripe 토마토를 들고 있음 | 5 |
| Tomato | 실행 불가능한 fallback 선택 | 9 |
| Tomato | 잘못된 scan/pick | 3 |
| Waste | 잘못된 bin에 배치 | 97 |
| Waste | 물체를 들지 않고 place | 26 |
| Waste | 실행 불가능한 fallback 선택 | 15 |
| Waste | prediction set에 fallback만 존재 | 1 |

초기 10회와 달리 전체 400회에는 관측 오류 관련 실패 외에 잘못된 행동과 fallback 종료도 포함된다.
모든 실패가 같은 원인이라고 단정하지 않았다. `complete=400`은 정상 프로세스 종료이며 실제 성공은 174/400이다.

## 비교 범위

- KnowNo/IntroPlan은 action oracle, Ours/Query-Action POMCP는 Boolean fact oracle을 사용한다.
  질문 1회의 정보량과 사용자 부담이 동일하다는 뜻은 아니다.
- IntroPlan은 현재 BRL adaptation(수동 knowledge, 단어 기반 검색)이며 원본 논문 재현 성능으로 일반화하지 않는다.
- 그래프의 행동 수는 성공 episode 한정, 시간은 전체 episode 기준이다.
  LaTeX 표의 시간은 기존 표 형식을 유지해 성공 episode 한정이며 CSV에는 두 값 모두 있다.

## 산출물

- `query_baseline_comparison.png` / `.pdf`: 통합 4-panel 그림
- `query_baseline_paper_table.csv` / `.tex`: 비교 표
- `query_baseline_summary.csv`: 평균·표준편차 및 유효 표본 수
- `query_baseline_scene_summary.csv`: scene별 성공률
- `query_baseline_paired_success_contrasts.csv`: 대응 비교
- `query_baseline_paired_runs.csv`: 실제 비교한 1,600개 레코드와 원본 경로
- `input_manifest.json`: IntroPlan 실행 로그 출처와 SHA-256

기존 `analysis_experiment.py`에 IntroPlan 파싱을 추가하고 `plot_query_baseline_figure.py`의
방법 목록·대응 검정·그림을 확장했다. 과거 그림 출력 폴더는 보존했다.
