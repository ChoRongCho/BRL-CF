# Baseline calibration archive

보정 산출물과 로그를 원 실행별로 보관한다. 새 보정은 새 날짜·실행 폴더에 저장한다.

| 폴더 | 내용 | 현재 적용 여부 |
| --- | --- | --- |
| [knowno_legacy](knowno_legacy/summary.md) | 기존 KnowNo 보정 JSON·CSV, 모델별 실패 로그, 원본 요약 | gpt-4o 100개 보정의 95% 값 사용 |
| [20260915_brl](20260915_brl/README.md) | 초기 18/23개 보정 기록 | 이전 결과, 현재 기본값에 미적용 |
| [recovery_20260915](recovery_20260915/README.md) | 과거 100개 데이터 복구 출처·이전 소규모 파일·저장 점수 검증 | KnowNo 기본 입력을 복구한 100개로 연결 |

## 현재 실행 qhat

| 방법 | tomato | wastesorting | 근거 |
| --- | --- | --- | --- |
| KnowNo | 0.8404 | 0.8704 | 기존 gpt-4o, n=100, T=5, 목표 95% |
| IntroPlan | 0.9809474992495626 | 0.9615342162270937 | 복구된 각 100개, T=5, 목표 95%, legacy_higher |

KnowNo 원본 결과:

- [Tomato JSON](knowno_legacy/tomato/gpt4/20260624_151751/calibration_summary.json), [CSV](knowno_legacy/tomato/gpt4/20260624_151751/calibration_scores.csv)
- [Waste JSON](knowno_legacy/wastesorting/gpt4/20260624_152507/calibration_summary.json), [CSV](knowno_legacy/wastesorting/gpt4/20260624_152507/calibration_scores.csv)

이번 보정은 기존 고정 후보 자료를 사용하며, 과거 KnowNo의 100개 보정과 표본 수·분위수 방식이 다르다.
이를 같은 조건의 성능 비교로 해석하지 않는다. 초기 18/23개 결과는 과거 이력으로만 보관한다.

## 보관 원칙과 이동 기록

- 이전 위치: `experiments_logs/calibration_log/`. 그 위치에는 새 경로 안내만 남겼다.
- 과거 JSON·CSV·로그와 요약 안의 원래 절대 경로는 실행 당시 출처이므로 그대로 보존했다.
- [migration_manifest.json](migration_manifest.json)은 이동 전 경로, 보관 경로와 SHA-256을 기록한다.
  53개 파일의 이동 직후 바이트 일치를 확인했다.
- 이동 후 `20260915_brl/recalibrate.py`의 프로젝트 루트 경로와 해당 README의 적용 상태 설명만 수정했다.
  따라서 이 두 파일의 manifest 해시는 수정 전 원본 기준이다. 점수·프롬프트·추론·로그 파일은 변경하지 않았다.
- 기존 checkpoint가 있는 폴더에서 설정을 바꾸어 보정을 재실행하지 않는다. 다른 조건은 새 실행 폴더로 보관한다.

최신 IntroPlan 결과: [introplan_recovered100_20260915](introplan_recovered100_20260915/README.md). 복구한 KnowNo의 각 100개 자료로 재보정했다.
