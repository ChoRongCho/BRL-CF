# KnowNo 데이터 복구 — 2026-09-15

- 원본 출처: `knowno_legacy/tomato/gpt4/20260624_151751/` 및 `knowno_legacy/wastesorting/gpt4/20260624_152507/`의 `calibration_summary.json`.
- 각 도메인의 100개 원문·선택지·정답을 `knowno/data/`에 JSON과 TXT로 복구했다.
- 이 폴더의 `tomato-mc-gen-prompt.txt`와 `waste-mc-gen-prompt.txt`는 복구 전 18개/23개 자료를 그대로 보존한 파일이다.
- `manifest.json`: 복구 출처, 원본 SHA-256, 복구 및 백업 경로.
- `knowno_*_verified.json`: 과거 저장 점수로 재계산한 결과. API 호출 없음.

| 도메인 | n | T | 목표 coverage | 재확인 qhat | 실행값 |
| --- | --- | --- | --- | --- | --- |
| tomato | 100 | 5 | 95% | 0.8403607106115039 | 0.8404 |
| wastesorting | 100 | 5 | 95% | 0.8703607098481373 | 0.8704 |

당시 사용한 `legacy_higher` 분위수 방식으로 정확히 재현했다.
기존 100개 데이터셋과 초기 소규모 데이터셋은 구분해 보관한다.
Git/폴더 정리가 유실 원인인지는 확인되지 않았다.
