# Watering 모델 의미 정리

이 디렉터리는 `watering` 도메인의 실행 모델을 구현한다.

- `trans.py`: action을 실행했을 때 state가 어떻게 바뀌는지 정의한다.
- `obs.py`: action 실행 후 robot이 어떤 observation을 받는지 정의한다.
- `rw.py`: goal 기반 reward를 계산한다.
- `answer.py`: `true_init` 기준으로 feedback 질문에 답한다.

`watering` 도메인은 부분 관측 실내 물주기 task다. Robot은 basket을 찾고, basket을 집고, tap에서 물을 채운 뒤, plant 위치를 찾아 물을 줘야 한다.

## State 의미

중요한 predicate는 다음과 같다.

- `at(I,R)`: item 또는 robot `I`가 room `R`에 있다.
- `holding(A,C)`: agent `A`가 container `C`를 들고 있다.
- `free(A)`: agent `A`가 아무것도 들고 있지 않다.
- `water_loaded(C)`: container `C`에 물이 들어 있다.
- `watered(P)`: plant `P`에 물을 줬다.
- `connected(R1,R2)`: room `R1`과 `R2`가 연결되어 있다.

`scene_*.yaml`에는 두 종류의 초기 정보가 있다.

- `facts`: planner/runtime state가 처음부터 알고 있는 사실
- `true_init`: 실제 환경의 hidden ground truth

예를 들어 basket과 plant의 위치는 `facts`에는 없을 수 있지만, 실제 환경을 나타내는 `true_init`에는 들어 있다.

## Transition Model

`TransitionWatering`은 action 실행 후 가능한 state 변화의 확률 분포를 만든다. 각 outcome은 다음 정보를 가진다.

- `add_facts`: state에 추가할 fact
- `del_facts`: state에서 제거할 fact
- `probability`: 해당 outcome의 확률

Transition model은 `scripts/domain/watering/robot_skill.yaml`의 action schema를 따른다.

| Action | Transition 의미 |
| --- | --- |
| `move(A,R1,R2)` | 확률 `0.95`로 robot을 `R1`에서 `R2`로 이동시킨다. 실패하면 state 변화가 없다. |
| `find_basket(A,C,R)` | `true_init`에 `at(C,R)`가 있으면 확률 `0.90`으로 해당 위치 fact를 state에 추가한다. 실제 위치가 아니면 변화가 없다. Basket을 이미 들고 있으면 위치를 발견하지 않는다. |
| `find_plant(A,P,R)` | `true_init`에 `at(P,R)`가 있으면 확률 `0.90`으로 해당 위치 fact를 state에 추가한다. 실제 위치가 아니면 변화가 없다. |
| `pick_basket(A,C,R)` | 확률 `0.90`으로 `holding(A,C)`를 추가하고, `free(A)`와 `at(C,R)`를 제거한다. 실패하면 `free(A)`를 유지한다. |
| `load_water(A,C,T)` | 확률 `0.95`로 `water_loaded(C)`를 추가한다. 실패하면 state 변화가 없다. |
| `pour_water(A,C,P,R)` | 확률 `0.90`으로 `watered(P)`를 추가하고 `water_loaded(C)`를 제거한다. 실패하면 `water_loaded(C)`를 유지한다. |

`find_*` action은 물리적으로 물체를 움직이는 action이 아니라 active sensing action으로 모델링되어 있다. 즉, 실제 환경의 hidden 위치 정보를 `true_init`에서 확인하고, 발견에 성공하면 planner가 쓰는 runtime/belief state에 위치 fact를 추가한다.

## Observation Model

`ObservationWatering`은 action 실행 후 robot이 센서 피드백으로 무엇을 받는지 시뮬레이션한다.

핵심 설계는 다음과 같다.

- `find_basket`, `find_plant`는 hidden truth인 `true_init`을 관찰한다.
- 물리 action인 `move`, `pick_basket`, `load_water`, `pour_water`는 transition 이후 runtime state를 관찰한다.

즉 `obs.py`는 실제 환경처럼 동작한다. Sensing action은 실제 세계를 보고, task action은 실행 결과가 현재 state에 반영됐는지를 관찰한다.

| Action | Observation 의미 |
| --- | --- |
| `move(A,R1,R2)` | runtime state에서 참인 후보 fact를 관찰한다. 보통 `at(A,R2)`를 확률 `0.95`로 관찰한다. |
| `find_basket(A,C,R)` | 실제 hidden world에 `at(C,R)`가 있으면 확률 `0.90`으로 `at(C,R)`를 관찰한다. 실제 위치가 아니면 empty observation을 반환한다. Basket을 이미 들고 있으면 empty observation을 반환한다. |
| `find_plant(A,P,R)` | 실제 hidden world에 `at(P,R)`가 있으면 확률 `0.90`으로 `at(P,R)`를 관찰한다. 실제 위치가 아니면 empty observation을 반환한다. |
| `pick_basket(A,C,R)` | runtime state에서 참인 후보 fact를 관찰한다. 예를 들어 성공하면 `holding(A,C)`, 실패하면 `free(A)`를 확률 `0.95`로 관찰할 수 있다. |
| `load_water(A,C,T)` | runtime state에서 참인 후보 fact를 관찰한다. 예를 들어 `water_loaded(C)`와 `holding(A,C)`를 확률 `0.95`로 관찰할 수 있다. |
| `pour_water(A,C,P,R)` | runtime state에서 참인 후보 fact를 관찰한다. 예를 들어 성공하면 `watered(P)`, 실패하면 `water_loaded(C)`를 확률 `0.95`로 관찰할 수 있다. |

관찰 후보 중 참인 fact가 하나도 없으면 확률 `1.0`으로 empty observation을 반환한다.

## Likelihood Mode

Belief update에서는 observation을 각 candidate belief state에 대해 평가해야 한다.

그래서 `get_observation_distribution_for_likelihood()`는 `find_basket`, `find_plant`에 대해 hidden `true_init`을 쓰지 않고, belief updater가 넘겨준 candidate state를 기준으로 observation probability를 계산한다.

이 구조는 tomato와 wastesorting 도메인의 패턴과 같다.

- observation sampling: hidden truth를 사용한다.
- likelihood scoring: candidate state를 사용한다.

이 둘을 분리하지 않으면 belief update가 후보 world들을 제대로 구분하지 못한다.

## 예시

`true_init`에 다음 사실이 있다고 하자.

```text
at(basket1,kitchen)
at(plant1,office)
```

Robot이 다음 action을 실행하면:

```text
find_basket(robot,basket1,kitchen)
```

동작은 다음과 같다.

- transition은 확률 `0.90`으로 `at(basket1,kitchen)`을 runtime/belief state에 추가한다.
- observation은 확률 `0.90`으로 `at(basket1,kitchen)`을 반환한다.
- 반대로 `find_basket(robot,basket1,office)`를 실행하면, `true_init` 기준으로 false이므로 transition과 observation 모두 위치 정보를 반환하지 않는다.

이후 robot이 다음 action을 실행하면:

```text
pick_basket(robot,basket1,kitchen)
```

성공 시 `holding(robot,basket1)`가 추가되고, `free(robot)`와 `at(basket1,kitchen)`이 제거된다. Observation model은 transition 이후 runtime state에서 참인 fact를 기준으로 관찰을 반환한다.
