# BRL 재보정 결과 — 2026-09-15

gpt-4o 실제 API 호출 완료. temperature=5.0, quantile_method=finite_sample.
동일한 고정 후보 프롬프트로 KnowNo와 IntroPlan을 점수화했다. 실행 기본값에는 95% 결과를 반영했다.

| 방법 | 도메인 | n | qhat 80% | qhat 85% | qhat 95% (적용) |
| --- | --- | --- | --- | --- | --- |
| knowno | tomato | 18 | 0.51357271 | 0.78396675 | 1.00000000 |
| introplan | tomato | 18 | 0.44891664 | 0.51397169 | 1.00000000 |
| knowno | wastesorting | 23 | 0.35930975 | 0.38439065 | 0.46632191 |
| introplan | wastesorting | 23 | 0.51501520 | 0.58186361 | 0.73006613 |

## 해석과 범위

- tomato는 n=18이므로 95% 순위 ceil(19×0.95)=19가 표본 수를 넘는다. 따라서 qhat=1을 사용한다. 반환된 모든 후보가 포함되는 보수적 설정이다.
- 기존 자료의 고유 context는 tomato 11개, waste 19개다. 새로운 독립 calibration/evaluation rollout을 수집한 결과가 아니다.
- 기존 데이터에는 반복 context, 고정 후보 및 과거 상태 표기가 포함된다. 현재 배포 프롬프트·후보 생성 분포에 대한 보장이나 최종 과업 성공률로 해석하지 않는다.
- IntroPlan: 수동 knowledge, lexical cosine 검색 top-k=3, 설명 후 점수화.
- Knowno scoring top-logprobs=5, IntroPlan=20으로 실제 BRL 설정을 따랐다.
- 재보정 기본값은 KnowNo 단일/반복 runner 및 전용 GUI, 도메인 CLI와 IntroPlan runner에 반영했다. `QHAT`으로 덮어쓸 수 있다.
- 이전 KnowNo runner 기본값은 tomato 0.8404, waste 0.8704. 이전 IntroPlan 기본값은 미보정 0.9.

## 파일

- `results.json`: 결과 요약 및 모델·데이터 해시
- `*_result.json`: 결과와 전체 점수
- `*_scored.json`: 레코드별 재개 checkpoint
- `*_prompts.json`: 사용한 프롬프트와 정답 라벨
- `*_trace.jsonl`: IntroPlan 검색·추론 내역
- `recalibrate.py`: 재현 스크립트. 설정이 바뀌면 새 출력 폴더를 사용하여 기존 checkpoint와 혼합하지 않는다.
- `run.log`: 실행 로그

## 보관 및 적용 상태 정정

이 폴더는 `scripts/baseline/calibration/20260915_brl/`로 이동했다.
위의 “적용” 표는 재보정 직후 상태의 기록이다. 이후 KnowNo 실행값은 기존 100개 보정의 tomato=0.8404, waste=0.8704로 복원했다.
이번 KnowNo 결과는 참고 기록으로 보관하고 IntroPlan의 실행값만 이번 결과를 사용한다.
`recalibrate.py`는 이동한 위치에 맞게 프로젝트 루트 경로만 수정했다.

## 후속 보정

IntroPlan도 이후 [복구한 각 100개 자료](../introplan_recovered100_20260915/README.md)로 다시 보정했다.
이 폴더의 18/23개 결과는 현재 실행 기본값에 사용하지 않는다.
