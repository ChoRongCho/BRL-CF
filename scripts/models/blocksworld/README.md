# Blocksworld 모델 의미 정리

이 디렉터리는 `blocksworld` 도메인의 실행 모델을 구현한다.

- `trans.py`: block manipulation action이 runtime state를 어떻게 바꾸는지 정의한다.
- `obs.py`: action 실행 후 관찰되는 symbolic fact를 정의한다.
- `rw.py`: goal tower 구성 진행도와 최종 goal reward를 계산한다.
- `answer.py`: `true_init` 기준으로 feedback 질문에 답한다.

`blocksworld`는 여러 block을 table 위나 다른 block 위에 쌓아 목표 tower 구조를 만드는 고전적인 symbolic planning 도메인이다.

## Transition Model

`TransitionBlocksworld`는 네 가지 조작 action을 stochastic action으로 모델링한다.

| Action | Transition 의미 |
| --- | --- |
| `pickup(B,T)` | 확률 `0.95`로 table `T` 위의 clear block `B`를 집는다. 성공하면 `holding(B)`가 추가되고 `on_table(B,T)`, `handempty`, `clear(B)`가 제거된다. |
| `putdown(B,T)` | 확률 `0.95`로 들고 있는 block `B`를 table `T`에 내려놓는다. 성공하면 `on_table(B,T)`, `clear(B)`, `handempty`가 추가되고 `holding(B)`, `clear(T)`가 제거된다. |
| `unstack(B1,B2)` | 확률 `0.95`로 `B2` 위의 clear block `B1`을 들어 올린다. 성공하면 `holding(B1)`, `clear(B2)`가 추가되고 `on(B1,B2)`, `handempty`, `clear(B1)`가 제거된다. |
| `stack(B1,B2)` | 확률 `0.95`로 들고 있는 block `B1`을 clear block `B2` 위에 올린다. 성공하면 `on(B1,B2)`, `clear(B1)`, `handempty`가 추가되고 `holding(B1)`, `clear(B2)`가 제거된다. |

실패 outcome은 no-op이다. 즉 현재 state를 그대로 유지한다.

## Observation Model

`ObservationBlocksworld`는 action 이후 runtime state에서 참인 observation candidate를 반환한다.

예를 들어 `stack(b1,b2)` 성공 후 state에 `on(b1,b2)`, `clear(b1)`, `handempty`가 있으면 이 fact들이 확률 `0.95`로 관찰된다. 실패하거나 후보 fact 중 참인 것이 없으면 empty observation이 반환된다.

현재 blocksworld scene은 `facts`와 `true_init`이 거의 동일한 완전 관측 benchmark에 가깝다. 따라서 observation은 hidden world sensing보다는 action 결과 확인의 의미가 크다.

## Reward Model

`RewardBlocksworld`는 두 종류의 reward를 준다.

- 새로 만족한 goal fact 하나당 `1.0`
- 전체 goal fact가 처음으로 모두 만족되면 추가 `10.0`

Blocksworld는 tower를 한 단계씩 쌓아가는 도메인이므로, 최종 goal reward만 주는 것보다 goal fact 진행 보상이 있으면 planner가 중간 성과를 더 잘 구분할 수 있다.

## PO 요소

현재 `scene_01.yaml`에서는 대부분의 block 위치와 clear 상태가 `facts`와 `true_init`에 동시에 들어 있으므로 강한 PO 요소는 없다.

다만 구조적으로는 일부 `on`, `on_table`, `clear` fact를 `facts`에서 빼고 `true_init`에만 넣으면 부분 관측 blocksworld로 확장할 수 있다. 이 경우 observation field와 belief generation이 hidden stack relation을 다루게 된다.
