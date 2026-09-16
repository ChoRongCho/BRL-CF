# KnowNo calibration data

과거 보정 JSON에 저장된 원문·선택지·정답을 이용해 도메인별 100개 데이터셋을 복구했다.
여러 과거 실행의 100개 레코드를 비교했으며 입력 내용과 순서가 동일함을 확인했다.

| 파일 | 내용 |
| --- | --- |
| `tomato-calibration.json` | tomato 100개, 보정 CLI 기본 입력 |
| `waste-calibration.json` | wastesorting 100개, 보정 CLI 기본 입력 |
| `tomato-mc-gen-prompt.txt` / `waste-mc-gen-prompt.txt` | 같은 100개를 기존 텍스트 형식으로 복구 |
| `*-tasks-info.txt` | 기존 시나리오 정보. 복구한 보정 레코드 수와 별개 |
| `metabot-*` | 원본 Mobile Manipulation 자료. BRL tomato/waste 자료와 별개 |

추출한 필드는 `context`, `mc_gen_prompt`, `true_actions`, `options`, `true_options`이다.
점수와 모델 응답은 원본 archive에 보존하며, 복구 데이터에 정답을 새로 작성하지 않았다.
파일 내용은 복구했지만 원래 파일의 바이트나 Git 유실 경위를 복원한 것은 아니다.

출처·이전 소규모 파일·검증 결과: [recovery_20260915](../../calibration/recovery_20260915/README.md).
기존 KnowNo gpt-4o 보정값은 저장된 점수만으로 재계산하여 일치를 확인했다.
실행값은 기존 반올림 값인 tomato `0.8404`, waste `0.8704`를 유지한다.

```bash
# 프로젝트 루트에서. 실제 재점수화가 필요할 때만 실행 (API 호출 발생).
python scripts/baseline/knowno/compute_qhat.py --domain tomato --score-with-llm \
  --target-success 0.95 --output-json /tmp/knowno_tomato.json
```

보정 CLI의 기본 표본 수는 100개다. 초기 18/23개와 합친 117/120개는 서로 다른
버전의 합집합이므로 기존 qhat을 재현할 때 혼합하지 않는다.
