# Watering Model

This directory implements the execution model for the `watering` domain.

- `trans.py`: transition outcomes after executing an action.
- `obs.py`: observations after action execution.
- `rw.py`: goal reward and empty find penalty.
- `answer.py`: feedback answers from `true_init`.

The robot always carries its water container. There is no search or pickup
step for a separate container. The container state is represented only by:

- `water_empty(A)`: agent `A`'s carried container is empty.
- `water_loaded(A)`: agent `A`'s carried container is loaded.

Initial scenes should start with `water_empty(robot)`.

## State

Important predicates:

- `at(I,R)`: item or robot `I` is in room `R`.
- `water_empty(A)`: agent `A` has an empty carried container.
- `water_loaded(A)`: agent `A` has a loaded carried container.
- `watered(P)`: plant `P` has been watered.
- `connected(R1,R2)`: rooms `R1` and `R2` are connected.

`scene_*.yaml` uses:

- `facts`: initially known runtime facts.
- `true_init`: hidden ground truth, such as plant locations.

## Actions

| Action | Transition meaning |
| --- | --- |
| `move(A,R1,R2)` | Moves the robot from `R1` to `R2` with probability `0.95`. |
| `find_plant(A,R)` | Reveals plant locations in room `R` probabilistically. |
| `load_water(A,T)` | Replaces `water_empty(A)` with `water_loaded(A)` at tap `T`. |
| `pour_water(A,P,R)` | Waters plant `P`, replacing `water_loaded(A)` with `water_empty(A)`. |

`find_plant` is active sensing: it reads hidden plant locations from
`true_init` during observation sampling. Physical actions observe the runtime
state after transition.
