# Rover 모델 의미 정리

이 디렉터리는 `rover` 도메인의 실행 모델을 구현한다.

- `trans.py`: rover action이 runtime state를 어떻게 바꾸는지 정의한다.
- `obs.py`: action 실행 후 rover가 어떤 observation을 받는지 정의한다.
- `rw.py`: soil/image/communication 진행도와 goal 달성 reward를 계산한다.
- `answer.py`: `true_init` 기준으로 feedback 질문에 답한다.

`rover` 도메인은 waypoint graph 위에서 이동하며 soil sample과 objective image를 수집하고, lander로 데이터를 전송하는 task다.

## Transition Model

`TransitionRover`는 action별 state 변화 확률 분포를 만든다.

| Action | Transition 의미 |
| --- | --- |
| `navigate(R,W1,W2)` | 확률 `0.95`로 rover 위치를 `W1`에서 `W2`로 이동한다. 실패하면 위치 변화가 없다. |
| `detect_road(R,W1,W2)` | `true_init`에 `visible(W1,W2)` 또는 `can_traverse(W1,W2)`가 있으면 확률 `0.90`으로 해당 road fact를 state에 추가한다. |
| `sample_soil(R,W)` | 확률 `0.90`으로 `have_soil_analysis(R,W)`를 추가하고 `at_soil_sample(W)`를 제거한다. 실패하면 soil sample fact를 유지한다. |
| `take_image(R,O,W)` | 확률 `0.90`으로 `have_image(R,O)`를 추가한다. 실패하면 state 변화가 없다. |
| `communicate_soil_data(...)` | 확률 `0.95`로 `communicated_soil_data(W)`를 추가한다. |
| `communicate_image_data(...)` | 확률 `0.95`로 `communicated_image_data(O)`를 추가한다. |

`detect_road`는 watering의 `find_*`와 같은 active sensing action이다. 물리적 이동 없이 hidden road 정보를 `true_init`에서 확인하고, 성공하면 planner가 쓰는 state에 추가한다.

## Observation Model

`ObservationRover`는 action 실행 후 rover가 받는 sensor feedback을 만든다.

- `detect_road`는 hidden truth인 `true_init`을 관찰한다.
- `navigate`, `sample_soil`, `take_image`, `communicate_*`는 transition 이후 runtime state를 관찰한다.

| Action | Observation 의미 |
| --- | --- |
| `detect_road(R,W1,W2)` | 실제 hidden world에 road fact가 있으면 `visible(W1,W2)`, `can_traverse(W1,W2)`를 확률 `0.90`으로 관찰한다. 없으면 empty observation을 반환한다. |
| `navigate(R,W1,W2)` | runtime state에서 참인 `in(R,W1)` 또는 `in(R,W2)`를 확률 `0.95`로 관찰한다. |
| `sample_soil(R,W)` | runtime state에서 참인 `have_soil_analysis(R,W)` 또는 `at_soil_sample(W)`를 확률 `0.95`로 관찰한다. |
| `take_image(R,O,W)` | runtime state에서 참인 `have_image(R,O)`를 확률 `0.95`로 관찰한다. |
| `communicate_*` | runtime state에서 참인 communicated fact를 확률 `0.95`로 관찰한다. |

Belief update의 likelihood 계산에서는 `detect_road`도 hidden `true_init`이 아니라 candidate state를 기준으로 평가한다. Sampling은 실제 환경을 보고, likelihood는 후보 world를 본다는 tomato/watering 패턴과 같다.

## PO 요소

부분 관측 요소는 다음과 같다.

- 실제 traversable/visible road edge
- soil sample 위치
- objective가 어느 waypoint에서 visible한지

현재 `robot_skill.yaml`에는 road 탐지 action인 `detect_road`만 명시되어 있다. Soil sample과 objective visibility는 action precondition으로 쓰이므로, scene이나 belief generation 쪽에서 candidate world로 포함되어야 한다.
