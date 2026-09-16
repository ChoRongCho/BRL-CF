# IntroPlan — 복구한 KnowNo 100개 자료로 보정

2026-09-15 실제 gpt-4o 호출 완료. 각 도메인 100개, 총 200개.
과거 KnowNo의 레코드 순서·관측/후보·정답·추론 전 점수화 프롬프트를 그대로 사용하고 IntroPlan 설명 후 새 점수를 계산했다.
KnowNo의 기존 실행 qhat 0.8404/0.8704는 변경하지 않았다.

## 결과

| Domain | qhat 75% | qhat 80% | qhat 85% | qhat 95% (적용) |
| --- | --- | --- | --- | --- |
| tomato | 0.96090371 | 0.96875988 | 0.97182393 | 0.98094750 |
| wastesorting | 0.83086378 | 0.87804552 | 0.92130061 | 0.96153422 |

## 조건 및 해석

- 모델 gpt-4o, score temperature=5.0, 목표 coverage=95%, quantile_method=legacy_higher. 기존 KnowNo 보정과 분위수 방식을 맞췄다.
- knowledge snapshot은 기존 BRL 수동 예시, lexical cosine 검색 top-k=3. 추론 512 tokens 제한, scoring top-logprobs=20.
- finite_sample 방식의 qhat도 결과 JSON에 함께 저장했다.
- 이번 95% qhat은 tomato=0.9809474992495626, wastesorting=0.9615342162270937이며 IntroPlan 기본 실행에 적용했다.
- KnowNo보다 qhat이 높다는 것은 이 보정 점수에서 정답 포함을 위해 더 낮은 점수 문턱이 필요했다는 의미다. 성능 우위를 뜻하지 않는다.
- 이는 과거 고정 후보 자료에 대한 calibration이며, 현재 BRL rollout의 최종 작업 성공률을 검증한 결과가 아니다.
- 과거 KnowNo와 API 호출 시점은 다르며, 원본 IntroPlan의 SBERT/LLM 생성 knowledge 대신 BRL adaptation을 사용한다.

## 보관 파일

- `*_inputs.json`: 복구한 100개 입력과 과거 점수화 프롬프트. 정답은 로컬 보정용이며 LLM 설명 생성에 전달하지 않는다.
- `*_manifest.json`: 모델·온도·출처·SHA-256.
- `knowledge_snapshot.json`: 실제 사용한 검색 예시.
- `*_checkpoint.json`: 입력 인덱스별 완료 점수.
- `*_traces/000.jsonl` 등: 샘플별 검색 ID, 생성 설명, scoring prompt, usage.
- `*_result.json`: 전체 점수, 두 분위수 방식 및 목표별 qhat.
- `*_run.log`: 실제 실행 로그.
- `calibrate.py`: 도메인별 재개 가능한 실행 코드. checkpoint와 입력·조건이 다른 경우 새 실행 폴더를 사용한다.

실행 기본값 반영 후 dry-run으로 qhat과 온도를 확인했다.
