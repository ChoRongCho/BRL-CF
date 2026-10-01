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
Tomato는 복구된 라벨을 유지한다. Waste는 2026-09-29에 단일 rollout 행동만
정답으로 둔 오류를 수정했다. 빈손 상태에서는 현재 관측된 남은 쓰레기에 대한
모든 `pick`을 복수 정답으로 두며, `detect`와 `place` 상태의 라벨은 유지한다.

출처·이전 소규모 파일·검증 결과: [recovery_20260915](../../calibration/recovery_20260915/README.md).
Waste의 기존 단일 정답 보정값은 복수 정답 실험 정의의 최종 보정값으로 사용하지
않는다. 기존 100개 정적 레코드는 실제 rollout보다 place 상태가 지나치게 적으므로,
저장된 실제 실행 로그를 이용한 별도 calibration audit 결과를 함께 확인해야 한다.

```bash
# 프로젝트 루트에서. 실제 재점수화가 필요할 때만 실행 (API 호출 발생).
python scripts/baseline/knowno/compute_qhat.py --domain tomato --score-with-llm \
  --target-success 0.95 --output-json /tmp/knowno_tomato.json
```

보정 CLI의 기본 표본 수는 100개다. 초기 18/23개와 합친 117/120개는 서로 다른
버전의 합집합이므로 기존 qhat을 재현할 때 혼합하지 않는다.
