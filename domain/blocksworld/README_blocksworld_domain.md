# Blocksworld Domain README

## 1. Overview

`Blocksworld`는 **부분 관측(partial observability)** 이 있는 블록 쌓기 환경을 모델링한 도메인입니다.  
이 도메인은 다음 두 가지를 결합합니다.

1. **ASP 스타일 world generation**  
   관측되지 않은 블록의 위치를 가능한 여러 world로 생성합니다.
2. **행동 모델(action model)**  
   조작기(manipulator)가 블록을 관측하고, 집고, 분리하고, 테이블에 내려놓고, 다른 블록 위에 쌓는 행동을 수행합니다.

이 구조는 hidden initial state가 있는 block rearrangement, belief-space planning, 또는 관측 기반 task planning 실험에 적합합니다.

---

## 2. Domain Objective

이 도메인의 목적은 다음과 같습니다.

- 일부만 관측된 초기 블록 배치를 바탕으로
- 보이지 않는 블록의 가능한 위치를 고려하고
- 필요한 관측과 조작을 수행하여
- 최종 목표 적재 구조를 만족하는 것입니다.

현재 문제 인스턴스에서는 최종적으로:

- `b2`를 `table` 위에 두고
- `b1`을 `b2` 위에 두고
- `b4`를 `b1` 위에 두고
- `b5`를 `table` 위에 두고
- `b3`를 `b5` 위에 두도록

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
- object instance
- planner가 아는 초기 사실(`facts`)
- 실제 정답 상태(`true_init`)
- 목표 상태(`goal`)

### `robot_skill.yaml`
조작기가 수행할 수 있는 action / skill 모델을 정의합니다.

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

- 각 블록은 정확히 하나의 상태를 가진다.
- 그 상태는 다음 둘 중 하나이다.
  - 어떤 support 위에 놓여 있음
  - 조작기에 의해 들려 있음
- support는 `table`이거나 다른 블록이다.
- `visible(X)`는 블록이나 support가 직접 관측되었음을 뜻한다.
- 블록 적재 그래프에는 cycle이 없어야 한다.
- 한 블록 위에는 동시에 하나의 블록만 올라갈 수 있다.
- 조작기는 동시에 하나의 블록만 들 수 있다.

즉, planner는 처음부터 모든 블록의 실제 배치를 아는 것이 아니라, **부분적으로 보이는 구조와 제약을 이용해 가능한 world를 유지하고, 필요하면 관측을 통해 정보를 늘려가야** 합니다.

---

## 5. Predicates

### 5.1 Object Predicates

- `block(B)` : 블록 객체
- `support(S)` : 합법적인 지지체

### 5.2 Observation Predicate

- `visible(X)` : 객체 또는 support `X`가 직접 관측됨

### 5.3 Physical State Predicates

- `on(B, S)` : 블록 `B`가 support `S` 위에 있음
- `holding(B)` : 조작기가 블록 `B`를 들고 있음
- `handempty` : 조작기 손이 비어 있음
- `clear(X)` : `X` 위에 아무 블록도 없음

### 5.4 Relational / Aggregate Predicates

- `above(B1, B2)` : `B1`이 `B2`의 위쪽에 있음
- `block_count(N)` : 전체 블록 수

---

## 6. Derived Rules

### 6.1 Support Typing

`block(B)`이면 `support(B)`로도 간주됩니다.

즉, 블록은 객체이면서 동시에 다른 블록을 지지할 수 있는 support이기도 합니다.

### 6.2 Above Relation

- `on(B1, B2)`이면 `above(B1, B2)`
- `on(B1, B2)`이고 `above(B2, B3)`이면 `above(B1, B3)`

이 규칙은 직접 적재 관계와 전이적 위계 관계를 함께 표현합니다.

### 6.3 Clear Rule

- `support(S)`
- `visible(S)`
- `not on(_, S)`

를 만족하면 `clear(S)`가 됩니다.

즉, 현재 모델에서는 **관측된 support에 대해서만** clear 여부가 파생됩니다.

### 6.4 Counting Rule

- `block_count(N)` : 전체 블록 수를 count

이 규칙은 문제 크기 확인이나 상태 통계에 활용할 수 있습니다.

---

## 7. Choice Rules

이 도메인의 핵심은 각 블록의 물리 상태를 정하는 choice rule입니다.

### 7.1 Block Physical State

각 블록 `B`는 정확히 하나의 상태를 가져야 합니다.

- `on(B, S)` where `support(S)`
- `holding(B)`

즉, 각 블록은:

- `table` 위에 있거나
- 다른 블록 위에 있거나
- 현재 조작기에 들려 있어야 하며
- 이 상태들을 동시에 둘 이상 만족할 수 없습니다.

이 규칙은 partially observed initial state와 action execution 상태를 하나의 표현 안에 통합합니다.

---

## 8. Constraints

### 8.1 No Self-Support

- `on(B, B)`는 허용되지 않습니다.

즉, 블록이 자기 자신 위에 놓일 수 없습니다.

### 8.2 Single Block on a Block

- 같은 블록 `S` 위에 두 블록 `B1`, `B2`가 동시에 올라갈 수 없습니다.

단, 이 제약은 `S`가 블록일 때만 적용됩니다. 따라서 `table`은 여러 블록을 동시에 지지할 수 있습니다.

### 8.3 No Cycles

- `above(B, B)`는 허용되지 않습니다.

즉, support graph는 순환이 없는 stack 구조여야 합니다.

### 8.4 Hand Consistency

- `holding(B)`와 `handempty`는 동시에 참일 수 없습니다.

### 8.5 Single Held Block

- 동시에 두 개 이상의 블록을 들 수 없습니다.

이 제약들은 classical block manipulation planning의 물리적 일관성을 유지합니다.

---

## 9. Initial Problem Instance

현재 인스턴스에는 다음 객체들이 존재합니다.

### 9.1 Blocks

- `b1`
- `b2`
- `b3`
- `b4`
- `b5`

### 9.2 Supports

- `table`
- 각 블록 `b1`~`b5`도 support로 사용 가능

### 9.3 Initially Visible Blocks

- `b1`
- `b2`
- `b3`

즉, 초기 facts 기준으로 `b4`, `b5`는 planner가 직접 관측하지 못한 블록입니다.

### 9.4 Manipulator State

- 초기에는 `handempty`

---

## 10. Ground-Truth Initial State

`true_init` 기준 실제 정답 상태는 다음과 같습니다.

- `b1` : `table` 위
- `b2` : `b1` 위
- `b3` : `table` 위
- `b4` : `table` 위
- `b5` : `b4` 위
- 조작기 손은 비어 있음

중요한 점은, 이 상태가 **실제 world state**이지 planner가 처음부터 전부 아는 정보라는 뜻은 아니라는 점입니다.

초기 `facts`에서 planner가 아는 것은 다음뿐입니다.

- `b1`~`b5`가 존재함
- `table`이 support임
- `handempty`
- `b1`, `b2`, `b3`가 visible임
- `on(b1, table)`
- `on(b2, b1)`
- `on(b3, table)`

따라서 `b4`, `b5`의 실제 배치는 숨겨진 정보로 남아 있습니다.

---

## 11. Goal Specification

현재 goal의 의미는 다음과 같습니다.

- `on(b2, table)`
- `on(b1, b2)`
- `on(b4, b1)`
- `on(b5, table)`
- `on(b3, b5)`

### Goal Interpretation

- 현재 `b2`는 `b1` 위에 있으므로 이를 분리해 `table`로 옮겨야 함
- `b1`은 `b2` 위로 재배치되어야 함
- 숨겨져 있던 `b4`는 최종적으로 `b1` 위에 있어야 함
- `b5`는 `table` 위에 놓여야 함
- `b3`는 `b5` 위로 옮겨져야 함

즉, 목표는 두 개의 stack:

- `b4` on `b1` on `b2` on `table`
- `b3` on `b5` on `table`

형태를 만드는 것으로 해석할 수 있습니다.

---

## 12. Action Model

현재 action model은 다섯 개 행동으로 구성됩니다.

### 12.1 `observe_block(B)`
블록을 관측하여 visible 상태로 만듭니다.

**Preconditions**
- `block(B)`

**Effects**
- `visible(B)` 추가

**Observation**
- `visible(B)`
- `on(B, S)`
- `clear(B)`
- `holding(B)`

**의미**
- hidden block의 위치나 clear 여부를 확인하기 위한 관측 action입니다.
- `b4`, `b5`처럼 초기에는 보이지 않는 블록을 계획 도중 드러낼 수 있습니다.

---

### 12.2 `pickup(B)`
테이블 위에 있는 블록을 집습니다.

**Preconditions**
- `block(B)`
- `visible(B)`
- `on(B, table)`
- `clear(B)`
- `handempty`

**Effects**
- `holding(B)` 추가
- `on(B, table)` 삭제
- `handempty` 삭제

**의미**
- 테이블 위의 clear block을 들어 올리는 기본 조작입니다.

---

### 12.3 `unstack(B, S)`
다른 블록 `S` 위에 있는 블록 `B`를 분리해 집습니다.

**Preconditions**
- `block(B)`
- `block(S)`
- `visible(B)`
- `visible(S)`
- `on(B, S)`
- `clear(B)`
- `handempty`

**Effects**
- `holding(B)` 추가
- `clear(S)` 추가
- `on(B, S)` 삭제
- `handempty` 삭제

**의미**
- stacked block을 떼어내는 action입니다.
- 분리 후 아래 블록 `S`는 clear가 됩니다.

---

### 12.4 `putdown(B)`
들고 있는 블록을 테이블 위에 내려놓습니다.

**Preconditions**
- `block(B)`
- `holding(B)`

**Effects**
- `on(B, table)` 추가
- `handempty` 추가
- `visible(B)` 추가
- `holding(B)` 삭제

**의미**
- 손에 든 블록을 독립 stack의 바닥 블록으로 배치합니다.

---

### 12.5 `stack(B, S)`
들고 있는 블록 `B`를 clear한 블록 `S` 위에 쌓습니다.

**Preconditions**
- `block(B)`
- `block(S)`
- `holding(B)`
- `visible(S)`
- `clear(S)`

**Effects**
- `on(B, S)` 추가
- `handempty` 추가
- `visible(B)` 추가
- `holding(B)` 삭제
- `clear(S)` 삭제

**의미**
- 원하는 tower 구조를 만드는 핵심 action입니다.
- support가 되는 블록은 미리 관측되어 있어야 하며, 위가 비어 있어야 합니다.

---

## 13. Policy Branching by Block State

현재 도메인에서 블록 상태에 따른 일반적인 행동 분기는 다음과 같습니다.

### Hidden Block

```text
observe_block
```

보이지 않는 블록은 먼저 관측해야 조작 전제조건을 만족시킬 수 있습니다.

### Clear Block on Table

```text
pickup -> putdown / stack
```

테이블 위의 clear block은 바로 집어서 다른 위치로 옮길 수 있습니다.

### Clear Block on Another Block

```text
unstack -> putdown / stack
```

다른 블록 위에 놓인 clear block은 먼저 분리한 뒤 재배치합니다.

### Non-Clear Block

```text
first remove the block(s) above it
```

위에 다른 블록이 있으면 직접 집을 수 없으므로, 상단 블록부터 순차적으로 치워야 합니다.

---

## 14. Planning Interpretation

이 도메인은 단순 fully observable blocksworld보다 다음 문제에 더 가깝습니다.

- contingent planning
- belief-space planning
- partially observed rearrangement planning

핵심 구조는 다음과 같습니다.

```text
Possible world generation
        ->
Observation through observe_block
        ->
Manipulation through pickup / unstack / putdown / stack
```

즉, planner는 처음부터 모든 truth를 아는 것이 아니라, 행동 중에 얻은 관측을 바탕으로 후속 조작을 결정해야 합니다.

---

## 15. Known Modeling Notes

### 15.1 `clear`는 visible support에 대해서만 파생
현재 `clear(S)`는 `visible(S)`를 요구합니다.
즉, planner는 보지 못한 블록을 clear하다고 가정하지 않습니다.

### 15.2 `table`은 무한 용량처럼 사용
현재 제약은 block support에 대해서만 단일 상단 블록을 강제합니다.
따라서 `table`은 여러 블록을 동시에 지지할 수 있습니다.

### 15.3 Held Block은 support graph 밖으로 나감
블록을 들고 있는 동안 그 블록은 어떤 `on(B, S)` 관계도 갖지 않습니다.
이 점은 choice rule에서 `on(B, S)`와 `holding(B)`를 함께 다루는 방식으로 표현됩니다.

### 15.4 Observation Model은 단순화되어 있음
`observe_block(B)`는 특정 블록을 직접 관측하는 추상 action입니다.
현실적인 센서 위치나 시야 제약은 현재 모델에 포함되어 있지 않습니다.

---

## 16. Recommended Execution Flow

현재 인스턴스에서 자연스러운 고수준 실행 흐름은 다음과 같습니다.

```text
1. visible한 stack을 정리해 필요한 clear block 확보
2. hidden block(b4, b5)이 필요해지는 시점에 observe_block 수행
3. pickup / unstack으로 블록 분리
4. putdown으로 임시 staging stack 구성 가능
5. stack으로 목표 tower 재구성
6. 모든 goal on-relation 만족 여부 확인
```

---
