# IntroPlan 초기 10회 분석

대상: `iterate_introplan_20260915_203137_137802.csv`의 도메인별 scene 01–05 각 1회.
모델 gpt-4o, oracle=exact action oracle, T=5, top-k=3, max steps=50.
qhat: tomato 0.9809474992495626, waste 0.9615342162270937.
새 배치가 시작되어 첫 배치 로그는 `archive/baselines_20260915_204042_983169/`로 이동했다.
이 분석은 해당 보관본의 Summary JSON을 사용한다.

## 요약

| 도메인 | 성공 | 평균 질문 | 평균 행동 수 | 평균 시간 |
| --- | --- | --- | --- | --- |
| tomato | 4/5 (80%) | 3.6 | 12.0 | 56.7초 |
| wastesorting | 3/5 (60%) | 4.4 | 9.8 | 40.5초 |
| 전체 | 7/10 (70%) | 4.0 | 10.9 | 48.6초 |

모든 평균은 실패 episode를 포함한다. 총 109회 의사결정 중 40회 질문(36.7%),
질문 시 prediction set 크기는 가중 평균 3.2개다. 총 토큰 315,098개(설명 생성 포함).
배치 출력의 `complete=10`은 프로세스 정상 종료이며 과업 성공 10회를 의미하지 않는다.

## Scene별

| 도메인 | Scene | 성공 | 질문 | 행동 |
| --- | --- | --- | --- | --- |
| tomato | 01 | 성공 | 5 | 14 |
| tomato | 02 | 성공 | 3 | 13 |
| tomato | 03 | 실패 | 2 | 5 |
| tomato | 04 | 성공 | 4 | 15 |
| tomato | 05 | 성공 | 4 | 13 |
| wastesorting | 01 | 실패 | 3 | 7 |
| wastesorting | 02 | 성공 | 4 | 10 |
| wastesorting | 03 | 성공 | 3 | 11 |
| wastesorting | 04 | 실패 | 6 | 11 |
| wastesorting | 05 | 성공 | 6 | 10 |

## 실패 3회의 공통 양상

1. Tomato 03, step 5: 실제 fresh인 tomato1을 scan에서 rotten으로 관측한 뒤 discard.
   prediction set은 discard 하나, 점수 약 0.9739. Oracle 질문 없이 실행.
   정답 행동 place도 후보에 있었지만 점수 0.0117로 문턱 0.0191 미만이었다.
2. Waste 01, step 7: 실제 can인 waste4를 plastic으로 관측하고 plastic bin에 배치.
   prediction set은 해당 배치 하나, 점수 약 0.9511. Oracle 질문 없이 실행.
   올바른 can bin 배치는 후보에 없었다.
3. Waste 04, step 11: 실제 can인 waste3를 plastic으로 관측하고 plastic bin에 배치.
   prediction set은 해당 배치 하나, 점수 약 0.9618. Oracle 질문 없이 실행.
   올바른 can bin 배치는 후보에 없었다.

각 실패 직전 IntroPlan 설명은 잘못된 관측을 사실로 받아들이고 해당 행동을 정당화했다.
즉, 이 3회는 oracle 오답이 아니라 **잘못된 관측에 대해 확신한 자율 실행**이었다.
설명 생성이 관측 오류를 자동으로 찾아내는 것은 아니며, 행동 후보의 확신과 세계 상태의 정확성이
분리될 수 있다는 초기 사례다. 원인에 관한 이 해석은 해당 로그에 한정한다.

## 다음 분석 기준

- 현재 각 scene 1회이므로 방법의 우열이나 통계적 차이는 판단하지 않는다.
- KnowNo/Ours와 동일 domain·scene·seed의 성공률·질문 수를 짝지어 비교한다.
- 오류 관측 뒤 질문 여부, 실패 행동의 자율/질문 선택, 올바른 행동의 후보 포함 여부를 집계한다.
- 이 10개 결과를 보고 qhat을 조정한 뒤 동일 자료를 최종 평가로 사용하는 것은 피한다.
- 이번 결과는 BRL adaptation에 대한 결과이며 원본 IntroPlan 전체에 대한 결론이 아니다.

원자료 경로와 추출 지표는 `episodes.csv`, 전체 Summary는 `episodes.json`에 저장했다.
