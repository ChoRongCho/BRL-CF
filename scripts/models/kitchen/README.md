# Kitchen 모델 의미 정리

이 디렉터리는 `kitchen` 도메인의 실행 모델을 구현한다.

- `trans.py`: 재료/도구 조작, 세척, 조리, serving action의 state 변화를 정의한다.
- `obs.py`: 냉장고 확인과 청결 검사, task 실행 결과 observation을 정의한다.
- `rw.py`: 세척, 조리, serving 진행도와 최종 goal reward를 계산한다.
- `answer.py`: `true_init` 기준으로 feedback 질문에 답한다.

`kitchen` 도메인은 고정된 robot manipulator가 ingredient를 찾아 pot에 넣고, soup를 만들어 dish에 serve하는 symbolic cooking task다.

## Transition Model

`TransitionKitchen`은 action별 성공/실패 outcome을 만든다.

| Action Group | Transition 의미 |
| --- | --- |
| `open_fridge` | `true_init`에 ingredient가 해당 fridge에 있으면 확률 `0.90`으로 `in_fridge(I,F)`를 state에 추가한다. 이미 들고 있는 ingredient는 다시 fridge에서 발견되지 않는다. |
| `inspect_ingredient`, `inspect_pot`, `inspect_dish` | 현재 state에 clean/dirty 정보가 있으면 그 정보를 우선하고, 없으면 `true_init`의 청결 상태를 확률 `0.90`으로 state에 추가한다. |
| `pick_*` | 확률 `0.90`으로 object를 든다. `pick_ingredient` 성공 시 해당 `in_fridge(I,F)`도 제거한다. |
| `wash_*` | 확률 `0.95`로 dirty object를 clean하게 만든다. |
| `place_*`, `discard_ingredient` | 확률 `0.95`로 들고 있는 object를 내려놓거나 ingredient를 pot에 넣는다. |
| `boil_*_soup` | 확률 `0.95`로 pot 안 재료를 cooked soup fact로 바꾼다. |
| `serve_*_soup` | 확률 `0.95`로 cooked soup를 clean dish에 serve한다. |

실패 outcome은 제거될 예정이던 fact를 보존한다. 예를 들어 `wash_ingredient`가 실패하면 `dirty(I)`가 유지되고, `place_ingredient_in_pot`이 실패하면 `holding(R,I)`가 유지된다.

## Observation Model

`ObservationKitchen`은 실제 환경처럼 sensing action과 task action을 구분한다.

- `open_fridge`: hidden `true_init`을 보고 ingredient가 어느 refrigerator에 있는지 관찰한다.
- `inspect_*`: 현재 state에 clean/dirty 정보가 있으면 현재 state를 우선하고, 아직 모르면 hidden `true_init`을 관찰한다.
- 나머지 task action: transition 이후 runtime state에서 참인 observation candidate를 관찰한다.

이 설계는 wash 이후 다시 inspect했을 때 `true_init`의 예전 dirty 상태가 되살아나지 않도록 한다. 한 번 runtime state에서 `clean(O)`가 참이 되면 inspect는 현재 state의 clean 정보를 우선한다.

## Reward Model

`RewardKitchen`은 다음 reward를 준다.

- object가 새로 `clean(O)`이 되면 `1.0`
- soup가 새로 cooked 상태가 되면 `3.0`
- soup가 새로 served 상태가 되면 `5.0`
- 전체 goal이 처음 만족되면 추가 `10.0`

## PO 요소

부분 관측 요소는 ingredient 위치와 object 청결 상태다.

- `in_fridge(I,F)`는 `open_fridge`로 확인한다.
- `clean(O)` / `dirty(O)`는 `inspect_*`로 확인한다.

초기 `facts`에는 object type과 `handempty(robot)` 정도만 있고, 실제 refrigerator assignment와 cleanliness는 `true_init`에 들어 있다.
