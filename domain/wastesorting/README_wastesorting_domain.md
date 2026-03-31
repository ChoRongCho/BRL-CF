# WasteSorting Domain README

## 1. Overview

`WasteSorting`은 **부분 관측(partial observability)** 환경에서 고정형 매니퓰레이터가 폐기물을 파지하고 촉각 기반으로 식별한 뒤 분류하는 작업을 모델링한 도메인입니다.  
이 도메인은 다음 두 가지를 결합합니다.

1. **ASP 스타일 world generation**  
   관측되지 않은 폐기물의 재질과 hardness를 가능한 여러 world로 생성합니다.
2. **행동 모델(action model)**  
   매니퓰레이터가 폐기물을 탐지하고, 파지하고, tactile sensing으로 재질과 hardness를 확인하고, 그 결과에 따라 압축 또는 직접 배치를 수행합니다.

이 구조는 belief-space planning, contingent planning, 또는 관측 기반 task planning 실험에 적합합니다.

---

## 2. Domain Objective

이 도메인의 목적은 다음과 같습니다.

- sorting table 위의 여러 폐기물을 관측하고
- 파지 후 tactile sensing으로 재질을 분류하고
- 더 강한 파지로 hardness를 확인한 뒤
- 플라스틱과 캔은 hardness 기준상 압축 가능하면 압축해서 각 bin으로 보내고
- 유리와 일반 폐기물은 압축하지 않고 각 bin으로 보내고
- 위험 폐기물은 hazardous bin으로 보내는 것입니다.

현재 문제 인스턴스에서는 최종적으로:

- `w1`을 plastic bin에 넣고
- `w2`를 can bin에 넣고
- `w3`를 glass bin에 넣고
- `w4`를 general bin에 넣고
- `w5`를 hazardous bin에 넣도록

설계되어 있습니다.

---

## 3. File Structure

이 도메인은 보통 다음 세 파일로 구성됩니다.

### `domain_rule.yaml`
도메인의 정적 의미론과 belief state 생성 규칙을 정의합니다.

포함 내용:
- predicate 정의
- 파생 규칙(rule)
- choice rule
- consistency constraint
- show directive

### `initial_state.yaml`
특정 문제 인스턴스를 정의합니다.

포함 내용:
- object/type instance
- planner가 아는 초기 사실(`facts`)
- 실제 정답 상태(`true_init`)
- 목표 상태(`goal`)

### `robot_skill.yaml`
매니퓰레이터가 수행할 수 있는 action / skill 모델을 정의합니다.

포함 내용:
- action 이름
- 파라미터
- 비용(cost)
- preconditions
- add/delete effects
- observation

---

## 4. Domain Semantics

### 4.1 Core Assumptions

이 도메인은 다음 가정을 사용합니다.

- 각 폐기물은 정확히 하나의 재질 카테고리를 가진다.
- 카테고리는 다음 다섯 가지 중 하나이다.
  - `plastic`
  - `can`
  - `glass`
  - `general`
  - `hazardous`
- 각 폐기물은 정확히 하나의 hardness 상태를 가진다.
  - `soft`
  - `hard`
- 각 폐기물은 정확히 하나의 물리 상태를 가진다.
- 물리 상태는 sorting table 위에 있거나, 로봇에게 들려 있거나, 어떤 bin 안에 있는 상태 중 하나이다.
- bin의 종류와 bin 위치는 planning 시작 시점에 모두 알려져 있다.
- 미관측 폐기물은 초기 belief에서 sorting table 위에만 존재할 수 있다.
- 폐기물의 존재는 `detect`를 통해 확인되며, `detect`는 table 위 실제 대상을 관측하는 동작이다.
- 재질 분류는 파지 후 `scan_material`을 통해 확인된다.
- hardness는 더 강한 파지 후 `probe_hardness`를 통해 확인된다.
- `plastic`, `can`은 material과 hardness 결과에 따라 압축 후 배치될 수 있다.
- `glass`, `general`, `hazardous`는 압축 없이 배치된다.
- 매니퓰레이터는 이동하지 않으며, 조작 대상은 sorting table에서만 집을 수 있다.

즉, planner는 처음부터 모든 폐기물의 실제 재질과 hardness를 아는 것이 아니라, **탐지와 촉각 관측을 통해 belief를 줄여나가야** 합니다.

---

## 5. Predicates

### 5.1 Object Predicates

- `robot(R)` : 매니퓰레이터 로봇
- `waste(W)` : 폐기물 객체
- `location(L)` : 작업 공간 위치
- `bin(B)` : 폐기물 수거 bin

### 5.2 Bin Predicates

- `plastic_bin(B)` : plastic 전용 bin
- `can_bin(B)` : can 전용 bin
- `glass_bin(B)` : glass 전용 bin
- `general_bin(B)` : general 전용 bin
- `hazardous_bin(B)` : hazardous 전용 bin
- `bin_at(B, L)` : bin `B`가 위치 `L`에 있음

### 5.3 Waste Property Predicates

- `observed(W)` : 폐기물 `W`가 탐지됨
- `plastic(W)` : plastic 폐기물
- `can(W)` : can 폐기물
- `glass(W)` : glass 폐기물
- `general(W)` : general 폐기물
- `hazardous(W)` : hazardous 폐기물
- `soft(W)` : 압축 가능 hardness를 가진 폐기물
- `hard(W)` : 비압축 hardness를 가진 폐기물
- `compression_target(W)` : 압축 후 배치해야 하는 폐기물
- `direct_placeable(W)` : 비압축 상태로 바로 배치해야 하는 폐기물

### 5.4 Physical State Predicates

- `at(W, L)` : 폐기물 `W`가 위치 `L`에 있음
- `holding(R, W)` : 로봇 `R`이 폐기물 `W`를 들고 있음
- `handempty(R)` : 로봇 손이 비어 있음
- `in_bin(W, B)` : 폐기물 `W`가 bin `B` 안에 있음
- `compressed(W)` : 폐기물 `W`가 압축된 상태임

---

## 6. Derived Rules

### 6.1 Bin Typing

각 bin 타입 predicate는 `bin(B)`를 유도합니다.

- `plastic_bin(B)`이면 `bin(B)`
- `can_bin(B)`이면 `bin(B)`
- `glass_bin(B)`이면 `bin(B)`
- `general_bin(B)`이면 `bin(B)`
- `hazardous_bin(B)`이면 `bin(B)`

### 6.2 Location Typing

- `bin_at(B, L)`이면 `location(L)`입니다.

즉, bin이 놓인 영역은 workspace location으로 간주됩니다.

### 6.3 Manipulation Branch Typing

- `plastic(W)` 또는 `can(W)`이고 `soft(W)`이면 `compression_target(W)`입니다.
- `plastic(W)` 또는 `can(W)`이고 `hard(W)`이면 `direct_placeable(W)`입니다.
- `glass(W)`, `general(W)`, `hazardous(W)`이면 `direct_placeable(W)`입니다.

즉, 재질과 hardness가 함께 확인되면 후속 액션 경로가 자동으로 나뉩니다.

---

## 7. Choice Rules

이 도메인의 핵심은 폐기물 재질, hardness, 물리 상태를 정하는 choice rule입니다.

### 7.1 Waste Category

각 폐기물 `W`는 정확히 하나의 카테고리를 가져야 합니다.

- `plastic(W)`
- `can(W)`
- `glass(W)`
- `general(W)`
- `hazardous(W)`

### 7.2 Hardness State

각 폐기물 `W`는 정확히 하나의 hardness 상태를 가져야 합니다.

- `soft(W)`
- `hard(W)`

### 7.3 Waste Physical State

미관측 폐기물 `W`는 sorting table 위에만 존재할 수 있습니다.

- `at(W, sorting_table)`

관측된 폐기물 `W`는 정확히 하나의 물리 상태를 가져야 합니다.

- `at(W, L)` where `location(L)`
- `holding(R, W)` where `robot(R)`
- `in_bin(W, B)` where `bin(B)`

### 7.4 Bin Location

각 bin `B`는 정확히 하나의 위치를 가집니다.

- `bin_at(B, L)` where `location(L)`

---

## 8. Constraints

### 8.1 Hand Consistency

- `holding(R, W)`와 `handempty(R)`는 동시에 참일 수 없습니다.

### 8.2 Single Object in Hand

- 한 로봇은 동시에 두 개 이상의 폐기물을 들 수 없습니다.

### 8.3 Physical State Exclusivity

- 폐기물은 동시에 위치에 놓여 있으면서 들려 있을 수 없습니다.
- 폐기물은 동시에 들려 있으면서 bin 안에 있을 수 없습니다.
- 폐기물은 동시에 위치에 놓여 있으면서 bin 안에 있을 수 없습니다.

### 8.4 Single Bin Location

- 각 bin은 정확히 하나의 위치에 있어야 합니다.

### 8.5 Bin Compatibility

- `plastic` 폐기물은 `plastic_bin`에만 들어갈 수 있습니다.
- `can` 폐기물은 `can_bin`에만 들어갈 수 있습니다.
- `glass` 폐기물은 `glass_bin`에만 들어갈 수 있습니다.
- `general` 폐기물은 `general_bin`에만 들어갈 수 있습니다.
- `hazardous` 폐기물은 `hazardous_bin`에만 들어갈 수 있습니다.
- 이 호환성은 state constraint뿐 아니라 place action의 precondition에서도 직접 반영됩니다.

### 8.6 Compression Constraints

- `compression_target(W)`인 폐기물은 압축된 뒤에만 bin에 들어갈 수 있습니다.
- `compression_target(W)`가 아닌 폐기물은 압축되면 안 됩니다.

이 제약들은 촉각 기반 분류와 후속 조작의 일관성을 유지합니다.

---

## 8.7 Action-Level Execution Flow

현재 action model은 다음 순서를 기준으로 설계되어 있습니다.

1. `detect`
   - sorting table 위에 있는 폐기물을 관측하여 `observed(W)`를 얻습니다.
   - 이 action은 폐기물 위치를 새로 생성하지 않습니다.

2. `pick`
   - `observed(W)`이면서 `at(W, sorting_table)`인 대상만 집을 수 있습니다.

3. `scan_material` / `probe_hardness`
   - 파지한 뒤 재질과 hardness를 관측합니다.

4. `compress_item`
   - `compression_target(W)`인 경우에만 수행합니다.

5. `place_*`
   - 각 배치 action은 폐기물 종류와 bin 타입이 일치할 때만 실행 가능합니다.
   - 예를 들어 plastic은 plastic bin, hazardous는 hazardous bin으로만 보낼 수 있습니다.

---

## 9. Initial Problem Instance

현재 인스턴스에는 다음 객체들이 존재합니다.

### 9.1 Waste Items

- `w1`
- `w2`
- `w3`
- `w4`
- `w5`

### 9.2 Robot

- `r1`

### 9.3 Locations

- `sorting_table`
- `plastic_zone`
- `can_zone`
- `glass_zone`
- `general_zone`
- `hazardous_zone`

### 9.4 Bins

- `b_plastic`
- `b_can`
- `b_glass`
- `b_general`
- `b_hazardous`

초기 `facts` 기준으로 planner가 아는 것은 다음과 같습니다.

- 로봇 `r1`이 존재하고 손이 비어 있음
- 폐기물 `w1`~`w5`가 존재함
- 각 bin의 종류가 정해져 있음
- 각 bin의 위치가 정해져 있음

반면 각 폐기물의 실제 재질, hardness, 실제 위치는 `true_init`에만 들어 있고 planner에는 숨겨져 있습니다.

---

## 10. Ground-Truth Initial State

`true_init` 기준 실제 정답 상태는 다음과 같습니다.

- `w1` : plastic, soft, `sorting_table`
- `w2` : can, soft, `sorting_table`
- `w3` : glass, hard, `sorting_table`
- `w4` : general, hard, `sorting_table`
- `w5` : hazardous, hard, `sorting_table`

로봇 상태:
- `r1`의 손은 비어 있음

중요한 점은, 이 상태가 **실제 world state**이지 planner가 처음부터 전부 아는 정보라는 뜻은 아니라는 점입니다.

---

## 11. Goal Specification

현재 goal의 의미는 다음과 같습니다.

- `in_bin(w1, b_plastic)`
- `in_bin(w2, b_can)`
- `in_bin(w3, b_glass)`
- `in_bin(w4, b_general)`
- `in_bin(w5, b_hazardous)`

### Goal Interpretation

즉, 모든 폐기물은 촉각 기반 분류 결과에 따라 올바른 bin으로 이동되어야 합니다.  
특히 플라스틱과 캔은 hardness에 따라 압축 후 배치될 수 있고, 유리, 일반, 위험 폐기물은 비압축 상태로 전용 bin에 배치됩니다.

---

## 12. Action Model Summary

`robot_skill.yaml`에는 다음 행동들이 정의되어 있습니다.

### 12.1 `detect(R, W)`

- 폐기물의 존재와 위치를 관측합니다.
- 이 도메인에서는 `sorting_table` 위의 폐기물만 탐지할 수 있습니다.
- 결과적으로 `observed(W)`와 `at(W, sorting_table)`를 알게 됩니다.

### 12.2 `pick(R, W)`

- `sorting_table` 위에서 관측된 폐기물을 집습니다.
- 결과적으로 `holding(R, W)`가 되고 `handempty(R)`는 해제됩니다.

### 12.3 `scan_material(R, W)`

- 파지한 폐기물의 재질을 tactile sensing으로 분류합니다.
- 관측 결과는 `plastic/can/glass/general/hazardous` 중 하나입니다.

### 12.4 `probe_hardness(R, W)`

- 더 강한 파지로 폐기물의 hardness를 확인합니다.
- 관측 결과는 `soft` 또는 `hard`입니다.

### 12.5 `compress_item(R, W)`

- `compression_target(W)`인 폐기물을 압축합니다.
- 결과적으로 `compressed(W)`가 됩니다.

### 12.6 `place_compressed_in_bin(R, W, B)`

- 압축된 폐기물을 bin에 넣습니다.
- 주로 `plastic`, `can` 중 `soft`인 경우의 경로에 사용됩니다.

### 12.7 `place_direct_in_bin(R, W, B)`

- 손에 들고 있는 폐기물을 압축 없이 바로 bin에 넣습니다.
- `glass`, `general`, `hazardous`와, `hard`로 판정된 `plastic`, `can`에 사용됩니다.

---

## 13. Intended Planning Flow

이 도메인에서 planner가 따를 전형적인 흐름은 다음과 같습니다.

1. `detect`로 sorting table 위 폐기물을 확인
2. `pick`으로 폐기물을 집기
3. `scan_material`로 재질 확인
4. `probe_hardness`로 hardness 확인
5. `plastic` 또는 `can`이고 `soft`이면 `compress_item -> place_compressed_in_bin`
6. `glass`, `general`, `hazardous`이거나 `plastic/can`이 `hard`이면 `place_direct_in_bin`

즉, 이 도메인은 **탐지 -> 파지 -> 재질 판별 -> hardness 판별 -> 카테고리별 조작 분기 -> bin 배치**의 작업 흐름을 명시적으로 실험할 수 있도록 설계되어 있습니다.
