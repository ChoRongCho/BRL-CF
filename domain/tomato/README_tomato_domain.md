# TomatoHarvest Domain README

## 1. Overview

`TomatoHarvest`는 **부분 관측(partial observability)** 환경에서 토마토 수확 작업을 모델링한 도메인입니다.  
이 도메인은 다음 두 가지를 결합합니다.

1. **ASP 스타일 world generation**  
   관측되지 않은 토마토의 상태를 가능한 여러 world로 생성합니다.
2. **행동 모델(action model)**  
   로봇이 이동, 탐지, 파지, 스캔, 적재, 폐기 등의 행동을 수행하면서 상태를 바꾸고 관측을 획득합니다.

이 구조는 belief-space planning, contingent planning, 또는 관측 기반 task planning 실험에 적합합니다.

---

## 2. Domain Objective

이 도메인의 목적은 다음과 같습니다.

- 여러 개의 토마토 중 **익은 토마토는 수확**하고,
- **썩은 토마토는 폐기**하고,
- **안 익은 토마토는 남겨두는 것**입니다.

현재 문제 인스턴스에서는 최종적으로:

- `t1`, `t2`, `t4`를 수확하여 `loaded`
- `t5`를 `discarded`
- `t3`는 줄기(stem)에 그대로 유지
- 로봇은 마지막에 `dockstation`으로 복귀

하도록 설계되어 있습니다.

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
로봇이 수행할 수 있는 action / skill 모델을 정의합니다.

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

- 각 토마토는 정확히 하나의 상태를 가진다.
- 토마토의 품질 상태는 다음 중 하나이다.
  - `ripe`
  - `unripe`
  - `rotten`
- 관측된 토마토 상태는 hard evidence이다.
- 관측되지 않은 토마토는 제약을 만족하는 범위에서 여러 가능한 상태로 분기될 수 있다.
- 토마토 개수와 상태 공간은 사전에 고정되어 있다.

즉, planner는 처음부터 모든 토마토의 실제 상태를 아는 것이 아니라, **관측을 통해 belief를 줄여나가야** 합니다.

---

## 5. Predicates

### 5.1 Object Predicates

- `tomato(T)` : 토마토 객체
- `robot(R)` : 로봇 에이전트
- `location(L)` : 위치
- `stem(S)` : 토마토가 매달릴 수 있는 줄기 위치

### 5.2 Tomato Property Predicates

- `observed(T)` : 토마토가 관측됨
- `ripe(T)` : 익은 토마토
- `unripe(T)` : 안 익은 토마토
- `rotten(T)` : 썩은 토마토

### 5.3 Tomato Physical State Predicates

- `at(T, S)` : 토마토가 줄기 `S`에 있음
- `holded(T, R)` : 토마토가 로봇에게 들려 있음
- `loaded(T, R)` : 토마토가 수확/적재됨
- `discarded(T)` : 토마토가 폐기됨

### 5.4 Robot State Predicates

- `handempty(R)` : 로봇 손이 비어 있음
- `holding(R, T)` : 로봇이 토마토를 들고 있음
- `located(R, L)` : 로봇이 위치 `L`에 있음
- `navprepared(R)` : 로봇이 이동 준비 상태임

### 5.5 Aggregate Predicates

- `tomato_count(N)` : 전체 토마토 수
- `tomato_harvested(N)` : 적재된 토마토 수

---

## 6. Derived Rules

### 6.1 Location Typing

`stem(S)`이면 `location(S)`로 간주됩니다.

즉, 줄기는 별도 타입이지만 동시에 위치로도 해석됩니다.

### 6.2 Counting Rules

- `tomato_count(N)` : 전체 토마토 수를 count
- `tomato_harvested(N)` : `loaded(T, R)`의 수를 count

이 규칙은 상태 통계나 goal monitoring에 활용할 수 있습니다.

---

## 7. Choice Rules

이 도메인의 핵심은 choice rule입니다.

### 7.1 관측된 토마토의 물리 상태

관측된 토마토는 정확히 하나의 물리 상태를 가집니다.

- `at(T,S)`
- `holded(T,R)`
- `discarded(T)`
- `loaded(T,R)`

즉, 한 토마토가 동시에 줄기에 있으면서 적재된 상태일 수는 없습니다.

### 7.2 토마토 품질 상태

모든 토마토는 정확히 하나의 품질 상태를 가집니다.

- `ripe(T)`
- `unripe(T)`
- `rotten(T)`

### 7.3 관측되지 않은 토마토의 위치

관측되지 않은 토마토는 반드시 어떤 줄기 `S` 위에 존재해야 합니다.

즉, 아직 보지 못한 토마토는 planner 입장에서:

- 어디 stem에 있는지는 후보가 될 수 있지만,
- 손에 들려 있거나 폐기되거나 적재된 것으로는 가정되지 않습니다.

이 규칙은 **unobserved object의 물리적 가능 world를 제한**합니다.

---

## 8. Constraints

### 8.1 Hold Consistency

- `holded(T,R)`이면 반드시 `holding(R,T)`여야 합니다.

즉, 토마토 측 표현과 로봇 측 표현이 서로 일치해야 합니다.

### 8.2 Hand Consistency

- `holding(R,T)`와 `handempty(R)`는 동시에 참일 수 없습니다.

### 8.3 Single Object in Hand

- 한 로봇은 동시에 두 개 이상의 토마토를 들 수 없습니다.

### 8.4 Single Robot Location

- 각 로봇은 정확히 하나의 위치에 있어야 합니다.

이 제약들은 classical manipulation planning의 물리적 일관성을 유지합니다.

---

## 9. Initial Problem Instance

현재 인스턴스에는 다음 객체들이 존재합니다.

### 9.1 Tomatoes

- `t1`
- `t2`
- `t3`
- `t4`
- `t5`

### 9.2 Robot

- `changmin`

### 9.3 Locations

- `dockstation`
- `stem1`
- `stem2`
- `stem3`

---

## 10. Ground-Truth Initial State

`true_init` 기준 실제 정답 상태는 다음과 같습니다.

- `t1` : ripe, `stem1`
- `t2` : ripe, `stem1`
- `t3` : unripe, `stem2`
- `t4` : ripe, `stem3`
- `t5` : rotten, `stem3`

로봇 상태:
- `changmin`은 `dockstation`에 위치
- 손은 비어 있음
- `navprepared` 상태

중요한 점은, 이 상태가 **실제 world state**이지 planner가 처음부터 전부 아는 정보라는 뜻은 아니라는 점입니다.

---

## 11. Goal Specification

현재 goal의 의미는 다음과 같습니다.

- `loaded(t1, changmin)`
- `loaded(t2, changmin)`
- `at(t3, stem2)`
- `loaded(t4, changmin)`
- `discarded(t5)`
- `located(changmin, dockstation)`

### Goal Interpretation

- 익은 토마토(`t1`, `t2`, `t4`)는 수확
- 안 익은 토마토(`t3`)는 유지
- 썩은 토마토(`t5`)는 폐기
- 로봇은 작업 종료 후 출발 지점으로 복귀

---

## 12. Action Model

현재 action model은 다음 일곱 개 행동으로 구성됩니다.

### 12.1 `prepare_nav`
이동 준비를 수행합니다.

**Preconditions**
- `robot(R)`
- `handempty(R)`

**Effects**
- `navprepared(R)` 추가

**의미**
- 손이 비어 있을 때만 이동 준비 가능
- 이후 `navigate` 수행을 가능하게 함

---

### 12.2 `navigate(R, L1, L2)`
로봇을 한 위치에서 다른 위치로 이동시킵니다.

**Preconditions**
- `robot(R)`
- `location(L1)`
- `location(L2)`
- `located(R, L1)`
- `navprepared(R)`
- `handempty(R)`

**Effects**
- `located(R, L2)` 추가
- `located(R, L1)` 삭제
- `navprepared(R)` 삭제

**의미**
- 이동 전 반드시 준비되어 있어야 함
- 이동 후에는 다시 준비 상태가 사라짐
- 손에 토마토를 든 상태에서는 이동할 수 없음

---

### 12.3 `detect(R, S)`
현재 위치의 줄기에서 토마토를 탐지합니다.

**Preconditions**
- `handempty(R)`
- `robot(R)`
- `stem(S)`
- `located(R, S)`

**Effects / Observation**
- `observed(T)`
- `at(T, S)`
- 품질 정보 일부 관측

**의미**
- 탐지를 통해 토마토가 관측 가능한 객체로 전환됨
- 이후 `pick`의 전제조건인 `observed(T)`를 만족시킬 수 있음

---

### 12.4 `pick(R, T, S)`
토마토를 집습니다.

**Preconditions**
- `robot(R)`
- `tomato(T)`
- `observed(T)`
- `stem(S)`
- `located(R, S)`
- `at(T, S)`
- `handempty(R)`

**Effects**
- `holding(R, T)` 추가
- `holded(T, R)` 추가
- `at(T, S)` 삭제
- `handempty(R)` 삭제

**의미**
- 관측된 토마토만 파지 가능
- 파지 후 손은 더 이상 비어 있지 않음

---

### 12.5 `scan(R, T)`
들고 있는 토마토의 품질 상태를 정밀 판별합니다.

**Preconditions**
- `robot(R)`
- `tomato(T)`
- `observed(T)`
- `holding(R, T)`

**Observation**
- `ripe(T)`
- `unripe(T)`
- `rotten(T)`

**의미**
- 탐지보다 더 정밀한 품질 확인 단계
- 이후 `place` 또는 `discard`의 선택 기준이 됨

---

### 12.6 `place(R, T)`
익은 토마토를 적재합니다.

**Preconditions**
- `robot(R)`
- `tomato(T)`
- `observed(T)`
- `ripe(T)`
- `holding(R, T)`
- `holded(T, R)`

**Effects**
- `loaded(T, R)` 추가
- `handempty(R)` 추가
- `holding(R, T)` 삭제
- `holded(T, R)` 삭제

**의미**
- `ripe(T)`일 때만 실행 가능
- 성공하면 수확 완료 상태가 됨

---

### 12.7 `discard(R, T)`
토마토를 폐기합니다.

**Preconditions**
- `robot(R)`
- `tomato(T)`
- `observed(T)`
- `holding(R, T)`
- `holded(T, R)`

**Effects**
- `discarded(T)` 추가
- `handempty(R)` 추가
- `holding(R, T)` 삭제
- `holded(T, R)` 삭제

**의미**
- 들고 있는 토마토를 버리는 행동
- 특히 `rotten(T)`일 때 자연스러운 종료 action

---

## 13. Policy Branching by Tomato State

현재 도메인에서 토마토 상태에 따른 일반적인 행동 분기는 다음과 같습니다.

### Unknown / Unobserved

```text
prepare_nav -> navigate -> detect
```

관측되지 않은 토마토는 먼저 탐지해야 합니다.

### Ripe

```text
detect -> pick -> scan -> place
```

익은 토마토는 적재 대상으로 처리됩니다.

### Unripe

```text
detect -> (do not harvest)
```

안 익은 토마토는 줄기에 남겨두는 것이 목표와 일치합니다.
현재 도메인에는 unripe 토마토를 다시 줄기에 놓는 별도 action이 없으므로, 애초에 파지하지 않는 정책이 가장 자연스럽습니다.

### Rotten

```text
detect -> pick -> scan -> discard
```

썩은 토마토는 폐기 대상으로 처리됩니다.

---

## 14. Planning Interpretation

이 도메인은 단순한 fully observable planning보다는 다음 문제에 가깝습니다.

- contingent planning
- belief-space planning
- POMDP-lite task planning

핵심 구조는 다음과 같습니다.

```text
Possible world generation
        ->
Observation through detect / scan
        ->
Action branching (place / discard / skip)
```

즉, planner는 처음부터 모든 truth를 아는 것이 아니라, 행동 중에 얻은 관측을 바탕으로 후속 action을 결정해야 합니다.

---

## 15. Known Modeling Notes

### 15.1 `holded`와 `holding`의 중복
현재 도메인은 토마토 측 표현(`holded`)과 로봇 측 표현(`holding`)을 둘 다 사용합니다.
이는 일관성 제약으로 연결되어 있지만, 표현 중복으로 볼 수도 있습니다.

### 15.2 Navigation 제약
`navigate`는 `handempty(R)`를 요구합니다.
즉, 물체를 들고 이동하는 것을 허용하지 않습니다.
이 제약은 현실 로봇에 비해 다소 강할 수 있으나, 현재 모델에서는 navigation과 manipulation을 분리하는 역할을 합니다.

### 15.3 Detect Observation Coverage
현재 `detect`와 `scan`의 observation 설계는 토마토 품질 정보를 단계적으로 드러내는 구조입니다.
이 부분은 실험 목적에 따라 더 coarse/fine하게 조정할 수 있습니다.

---

## 16. Recommended Execution Flow

현재 인스턴스에서 자연스러운 고수준 실행 흐름은 다음과 같습니다.

```text
1. dockstation에서 시작
2. stem 위치로 이동
3. detect로 토마토 관측
4. 필요한 토마토를 pick
5. scan으로 상태 판별
6. ripe면 place
7. rotten이면 discard
8. unripe는 남김
9. 모든 목표 달성 후 dockstation 복귀
```

---

## 17. Summary

`TomatoHarvest` 도메인은 다음 특징을 갖는 benchmark입니다.

- 부분 관측 환경
- ASP 기반 possible-world generation
- 관측 기반 action branching
- 수확 / 유지 / 폐기 정책 분리
- 이동, 파지, 스캔, 적재, 폐기를 포함한 로봇 작업 모델

따라서 이 도메인은 단순 state-transition planning을 넘어서,  
**“센싱 결과에 따라 후속 행동이 달라지는 로봇 계획 문제”**를 표현하는 데 적합합니다.
