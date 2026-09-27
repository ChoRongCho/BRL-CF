# 질의 방법별 실제 실행 흐름

이 문서는 현재 저장소 코드에 구현된 다음 다섯 방법을 설명한다.

1. KnowNo
2. Query-as-Action POMCP
3. `01_cp_when.py`: Action-CP When + Ours EIG What
4. `02_value_when.py`: Query-value When + Ours EIG What
5. `03_value_what.py`: Ours confidence When + Query-value What

설명에서 **질의 시점(When)**은 질문을 시작할지 결정하는 부분이고,
**질의 내용(What)**은 어떤 사실 또는 어떤 행동을 물을지 결정하는 부분이다.

## 한눈에 보는 차이

| 방법 | 기본 계획 표현 | When | What | 사람이 답하는 내용 | 질문과 물리 행동의 관계 |
|---|---|---|---|---|---|
| KnowNo | LLM이 생성한 행동 선택지와 CP prediction set | 행동 prediction set이 singleton이 아니거나 fallback이 포함될 때 | prediction set 안의 행동 선택 | 다음에 실행할 행동 option | 질문 결과로 그 step의 물리 행동을 고름 |
| Query-as-Action POMCP | 물리 행동과 Boolean QueryAction을 합친 POMCP action space | 별도 trigger 없음. POMCP가 query의 Q-value를 가장 높게 평가할 때 | 가장 높은 Q-value를 가진 QueryAction의 target fact | Boolean state fact의 참/거짓 | 질문 자체가 물리 행동과 경쟁하는 한 decision |
| `01_cp_when.py` | BRL physical POMCP + KnowNo action CP | action prediction set이 singleton이 아니거나 fallback이 포함될 때 | 전체 belief에서 EIG가 가장 큰 Boolean state fact | 선택된 fact의 참/거짓 | Action CP는 trigger로만 쓰고 물리 행동 실행 후 별도 질의 |
| `02_value_when.py` | BRL physical POMCP + 보조 QaA value evaluator | 최고 query Q가 최선의 physical Q보다 클 때 | 별도로 계산한 Ours EIG fact 한 개 | 선택된 fact의 참/거짓 | QaA는 trigger 계산에만 쓰고 실제 action은 실행하지 않음 |
| `03_value_what.py` | BRL physical POMCP + 보조 QaA value evaluator | Ours의 entropy confidence가 threshold보다 낮을 때 | 모든 ambiguous fact 중 query Q가 가장 큰 fact | 선택된 fact의 참/거짓 | QaA는 질문 선택에만 쓰고 실제 action은 실행하지 않음 |

세 policy ablation의 공통 실행 순서는 다음과 같다.

```text
physical POMCP가 행동 선택
        ↓
물리 행동 실행
        ↓
observation으로 belief update
        ↓
각 조건의 When 평가
        ↓
When이 true이면 각 조건의 What으로 Boolean fact 하나 선택
        ↓
oracle 답변으로 belief particle filtering
        ↓
갱신된 belief에서 When을 다시 평가
        ↓
질문 종료 후 MAP state를 symbolic knowledge로 확정
```

따라서 세 ablation에서 질문은 물리 행동을 대체하지 않는다. 한 physical
step이 실행된 뒤에 질문 phase가 시작된다. 같은 physical step 안에서 fact
하나를 묻고 When을 다시 평가하므로, 조건이 계속 참이면 서로 다른 fact를
여러 번 물을 수 있다.

---

## 1. KnowNo

### 핵심 정의

KnowNo는 현재 상황에서 실행할 **행동 후보**를 LLM으로 생성하고, 각 행동
option의 확률에 conformal threshold를 적용한다. prediction set이 행동 하나로
충분히 좁혀지지 않으면 사람 또는 자동 oracle에게 어떤 행동을 실행할지 묻는다.

이 방법의 질문은 `ripe(tomato1)`과 같은 state fact의 참·거짓 질문이 아니다.
`A`, `B`, `C`, `D`, `E`와 같은 **행동 option 선택 질문**이다.

### 사전 calibration

각 calibration sample에서 정답 행동 option의 temperature-scaled probability를
`p_true`라고 하면 nonconformity score는 다음과 같다.

```text
score = 1 - p_true
```

calibration score들의 상위 quantile에서 domain별 `qhat`을 구한다. 현재 baseline
실험 설정은 Tomato `0.8404`, Waste Sorting `0.8704`를 사용한다. 실행 시 각
행동 option의 probability가 다음 조건을 만족하면 prediction set에 포함된다.

```text
p(option) >= 1 - qhat
```

`qhat`은 매 planning step마다 다시 학습되는 값이 아니다. calibration data로
미리 계산하여 실행 전체에서 고정해서 사용한다.

### 단일 episode 실행 흐름

```text
1. domain과 scene 설정을 읽고 작업 상태를 초기화한다.
   Tomato:
     - hidden ripeness, freshness, location
     - robot location, observed/scanned properties
     - held, loaded, discarded tomatoes
   Waste Sorting:
     - hidden waste category, occlusion
     - observed attributes
     - held, remaining, placed objects

2. 각 planning step을 시작하기 전에 성공 및 dead-end를 검사한다.

3. 현재까지 로봇이 관찰한 상태와 action history를 prompt로 만든다.
   hidden ground truth는 LLM prompt에 직접 제공하지 않는다.

4. LLM generation call로 다음 행동의 multiple-choice 후보를 생성한다.
   - option token은 A/B/C/D/E이다.
   - 일반 행동 후보와 "an option not listed here" fallback option을 포함한다.
   - 후보는 현재 관찰 상태, held object, 처리 완료 object, history를 반영한다.

5. 별도의 LLM scoring call을 수행한다.
   - 응답 token의 log probability를 얻는다.
   - score temperature로 temperature scaling한다.
   - p(option) >= 1-qhat인 option으로 prediction set을 만든다.

6. help_needed를 계산한다.

   help_needed =
       prediction set 크기가 1이 아니거나
       fallback option이 prediction set에 포함됨

7. help_needed가 false이면 prediction set의 유일한 action을 자율 선택한다.

8. help_needed가 true이면 외부에 행동 선택을 요청한다.
   - 사람 모드: prediction set에 포함된 A/B/C/D/E 중 하나를 입력받는다.
   - auto-answer 모드: exact task-state oracle이 적절한 option을 선택한다.
   - fallback을 선택한 경우 oracle은 기존 option에 없는 실행 action을 직접
     제공할 수 있다.
   - 사람 모드에서 prediction set 밖의 option은 다시 입력하게 한다.
   - 빈 prediction set은 사람 모드에서 PLAN FAILURE가 된다.

9. 선택된 문자열을 domain action으로 parse하고 실행 가능성을 검사한다.
   fallback recovery action도 별도로 validation한다.

10. hand-written task simulator에서 action을 실행한다.
    - detect/scan에는 성공 확률과 label error가 적용된다.
    - navigate/pick/place/discard에도 설정된 failure probability가 적용된다.
    - 실행 불가능한 action은 episode failure 원인이 될 수 있다.

11. observed state와 action history를 갱신한다.

12. success, dead-end, max_steps 중 하나가 발생할 때까지 2번부터 반복한다.

13. success, stop reason, action history, help count, autonomous count,
    prediction-set size, fallback recovery, action failure, LLM token usage와
    elapsed time을 저장한다.
```

### 질문 수와 plan length

- 한 planning step에서 help 요청은 최대 한 번 발생하고, 그 답으로 같은 step의
  물리 행동을 선택한다.
- 질문은 별도의 robot action step으로 추가되지 않는다.
- `planning_length` 또는 완료 iteration은 LLM planning/action cycle 수다.
- `max_steps`도 이 cycle을 제한한다.
- POMCP reward나 belief particle distribution은 사용하지 않는다.

### When과 What의 의미

```text
When: action prediction set이 singleton인지 여부
What: prediction set 중 어느 action을 실행할지
```

fallback option이 singleton으로 남은 경우에도 구체적인 실행 action이 없으므로
help를 요청한다.

---

## 2. Query-as-Action POMCP

### 핵심 정의

이 baseline은 로봇의 물리 행동과 Boolean state 질문을 하나의 POMCP action
space에 넣는다. 별도의 When module이나 What module이 없다. POMCP가 누적
보상 Q-value를 계산한 뒤, 물리 행동과 QueryAction 전체에서 가장 높은 action을
직접 선택한다.

### QueryAction 생성

```text
1. domain별 Boolean query schema를 코드에 미리 정의한다.

   Tomato 예:
     ripe(T), fresh(T), located(R,L), at(T,S), observed(T),
     holding(R,T), handempty(R), scanned(T), loaded(T,R), discarded(T)

   Waste Sorting 예:
     plastic(W), can(W), paper(W), general(W), detected(W),
     holding(R,W), handempty(R), in_bin(W,B)

2. 현재 scene에 선언된 object를 각 schema parameter에 대입한다.

3. 모든 grounded Boolean QueryAction을 생성한다.
   각 QueryAction은 다음 정보를 가진다.
     - action name
     - target_fact
     - query schema
     - object-type preconditions
     - query cost

4. root에서는 다음 action만 후보로 둔다.
   - 현재 MAP symbolic knowledge에서 applicable한 physical action
   - type precondition이 맞고 belief particle 사이에서 target fact가
     True/False로 갈리는 QueryAction
```

### POMCP 내부 transition과 reward

POMCP simulation에서 QueryAction을 선택하면 물리 state는 변하지 않고 Boolean
answer가 observation으로 돌아온다.

```text
query simulation:
    correct = sampled_state.has_fact(target_fact)
    answer = correct                     with answer_accuracy
             not correct                 otherwise
    next_state = current_state
    reward = -query_cost
```

물리 action은 sampled particle에서 precondition이 성립하면 transition과
observation을 simulation한다. root에서 선택 가능한 action이더라도 특정 sampled
particle에서 실행 불가능할 수 있다. 이 경우 해당 simulation branch는 다음과
같이 처리한다.

```text
observation = ("invalid_action", action_name)
reward = -failure_penalty
terminal = True
```

물리 행동 reward는 다음과 같다.

- Tomato
  - 같은 detect/scan/navigate를 연속 두 번째 수행: `-5`
  - fresh tomato를 성공적으로 place: `+10`
  - rotten tomato를 성공적으로 discard: `+10`
  - 그 외: `0`
- Waste Sorting
  - detect를 연속 세 번째 이상 수행: `-10`
  - 올바른 bin에 성공적으로 place: `+5`
  - 처음 goal state에 도달: 추가 `+10`
  - 그 외: `0`

POMCP는 위 immediate reward와 `gamma`가 적용된 future reward로 Q-value를
추정한다. search가 끝나면 root에서 visit된 action 중 Q-value가 가장 큰 action을
선택한다.

### 단일 episode 실행 흐름

```text
1. query_cost, failure_penalty, answer_accuracy와 공통 POMCP 설정을 읽는다.

2. Environment, BeliefManager, QueryAsActionPOMCPPlanner를 만든다.
   초기 state로 symbolic belief를 초기화한다.

3. 매 decision에서 최신 posterior로 POMCP tree를 새로 만든다.

4. root candidate를 구성한다.
   candidates = applicable physical actions
                + currently ambiguous grounded QueryActions

5. belief particle을 weight에 따라 sample하며 POMCP simulation을 실행한다.
   search tree 안에서 physical action과 QueryAction은 동일한 UCB/Q backup
   과정으로 평가된다.

6. root Q-value가 가장 높은 action을 고른다.

7. 선택 결과가 QueryAction이면:
   a. QueryAction에 연결된 target_fact를 oracle에게 묻는다.
   b. 가장 최근 physical action과 observation을 함께 넘겨 detect/scan 같은
      epistemic fact의 의미를 일관되게 판정한다.
   c. answer_accuracy 확률로 oracle answer를 유지하고, 나머지는 반전한다.
   d. 답과 일치하는 belief particle만 남기고 weight를 정규화한다.
   e. posterior MAP state를 symbolic knowledge로 갱신한다.
   f. 실제 cumulative reward에서도 query_cost를 차감한다.
   g. physical action을 실행하지 않고 다음 decision으로 돌아간다.

8. 선택 결과가 physical action이면:
   a. env.step(action)으로 실제 transition과 observation을 얻는다.
   b. 실제 physical reward를 cumulative reward에 더한다.
   c. action과 observation으로 belief를 갱신한다.
   d. posterior MAP state를 symbolic knowledge로 갱신한다.
   e. physical step을 하나 증가시킨다.

9. GOAL DONE, PLAN FAILURE 또는 전체 decision max_step에서 종료한다.

10. success, end reason, physical steps, total actions, question count,
    cumulative reward, 전체 decision sequence, timing과 final knowledge를 저장한다.
```

### 질문 수와 plan length

- QueryAction 하나는 독립적인 POMCP decision 하나다.
- `total_actions = physical actions + QueryActions`다.
- `max_step`은 `total_actions`에 적용되어 무한 질문을 막는다.
- 논문의 plan length와 로그의 `steps`/`physical_steps`는 physical action만 센다.
- 질문 수는 `total_questions`에 별도로 기록한다.

### When과 What의 의미

명시적인 분리는 없지만 분석을 위해 다음과 같이 해석할 수 있다.

```text
When: max_a Q(b,a)의 action이 QueryAction일 때
What: 그중 실제 argmax로 선택된 QueryAction의 target_fact
```

즉, 질문 여부와 질문 내용이 동일한 joint POMCP optimization에서 동시에
결정된다.

---

## 세 When–What policy ablation의 공통 구조

`01_cp_when.py`, `02_value_when.py`, `03_value_what.py`는 같은 physical execution
runner를 공유한다. 서로 다른 부분은 When과 What policy뿐이다.

### 공통 physical loop

```text
1. BRL Environment, BeliefManager, physical POMCPPlanner를 만든다.

2. physical_planner.search(belief)로 물리 행동 하나를 선택한다.
   이 planner의 action space에는 QueryAction이 없다.

3. env.step(action)을 실행한다.
   physical reward만 cumulative reward에 더한다.

4. observation과 action으로 posterior belief를 계산한다.
   posterior distribution을 유지한 채 MAP particle을 symbolic knowledge로 쓴다.

5. 해당 조건의 query episode를 실행한다.

6. query episode가 끝나면 최종 MAP state를 symbolic knowledge로 확정하고
   belief를 그 knowledge 하나로 reset한다.

7. env.check_done과 physical POMCP tree pruning을 수행한다.

8. GOAL DONE, MAX STEP 또는 PLAN FAILURE까지 반복한다.
```

### 공통 Boolean 질문 처리

```text
asked_facts = empty set

while True:
    현재 posterior에서 When을 평가한다.
    if When == false:
        질문 phase 종료

    이미 물은 fact를 제외하고 What으로 fact 하나를 고른다.
    if fact가 없으면:
        질문 phase 종료

    exact auto oracle에 fact의 True/False를 묻는다.
    답과 일치하는 belief particle만 유지하고 weight를 정규화한다.
    질문 전후 confidence와 frontier 크기를 기록한다.

    if 질문이 frontier를 줄이지 못했으면:
        질문 phase 종료

    갱신된 posterior에서 When을 다시 평가한다.
```

따라서 "한 번 질문"은 한 번의 trigger 평가에서 fact 하나만 묻는다는 뜻이다.
답변 뒤에도 When이 계속 참이면 같은 physical step에서 다른 fact를 추가로 묻는다.
같은 fact는 한 query episode 안에서 두 번 묻지 않는다.

### 공통 belief confidence와 ambiguous fact

belief particle weight를 `w_i`, particle 수를 `N`이라고 하면 confidence는
normalized entropy로 계산한다.

```text
H(b) = -sum_i w_i log2(w_i)
confidence(b) = 1 - H(b) / log2(N)
```

particle이 0개 또는 1개이면 confidence는 `1.0`이다. ambiguous fact 후보는
frontier 일부 particle에는 있고 일부에는 없는 symbolic fact다.

### Ours EIG What

`01_cp_when.py`와 `02_value_when.py`가 사용하는 정보이득 기반 What이다.
`01_cp_when.py`는 CP set으로 제한한 belief에서 계산하고, `02_value_when.py`는
전체 현재 belief에서 계산한다.

```text
current_entropy = H(b)

for each ambiguous and not-yet-asked fact f:
    true branch와 false branch로 particle을 나눈다.
    expected_entropy(f)
        = P(f=True)  * H(b | f=True)
        + P(f=False) * H(b | f=False)
    EIG(f) = current_entropy - expected_entropy(f)

가장 큰 EIG를 가진 fact를 선택한다.
동률이면 fact 문자열의 사전순으로 선택한다.
```

### Value evaluator의 역할

`02_value_when.py`와 `03_value_what.py`는 Query-as-Action POMCP 코드를 보조
evaluator로 재사용한다. 이 evaluator는 root Q-value만 계산하며 action을 실제
환경에 실행하지 않는다.

- 질문 후보는 고정 query schema 전체가 아니라 현재 ablation이 넘긴
  ambiguous fact로 동적으로 만든다.
- 각 query action은 `query_fact(f)`이며 target은 `f`다.
- query simulation reward, physical task reward와 invalid-action penalty는
  Query-as-Action baseline과 같은 모델을 사용한다.
- root 후보가 모두 최소 한 번 방문되도록 simulation budget을
  `max(configured simulations, root candidate 수 + 1)`로 올린다.
- `(episode seed, physical step, query index, root mode, candidate facts)`에서
  별도 seed를 만들어 evaluator의 난수가 실제 episode RNG를 바꾸지 않게 한다.
- evaluator의 `query_cost`, `failure_penalty`, `answer_accuracy`는 Q-value에만
  영향을 준다.
- 실제 질문 답변은 현재 runner의 exact auto oracle을 사용한다.
- 실제 episode cumulative reward에는 ablation 질문 비용을 차감하지 않는다.

---

## 3. `01_cp_when.py`: Action-CP When + Ours EIG What

### 목적

What을 Ours EIG로 고정하고, 질문을 시작하는 시점만 KnowNo의 action
prediction-set ambiguity로 바꾼다. 이 조건은 KnowNo가 제안한 행동을 실행하거나
사람에게 행동을 고르게 하지 않는다. Action CP 결과는 Boolean state-fact 질문을
시작할지 정하는 trigger로만 사용한다.

### Action-CP calibration과 scoring

KnowNo baseline과 같은 domain별 calibration과 scoring을 사용한다.

```text
Tomato:       qhat = 0.8404
WasteSorting: qhat = 0.8704
Temperature:  5.0
cutoff:       1 - qhat
```

각 physical action과 observation으로 belief를 갱신한 뒤, 현재 MAP symbolic
knowledge, 남은 객체, 관측 속성, 보유 객체, 행동 이력을 KnowNo prompt에 넣는다.
LLM은 다음 로봇 행동 후보와 `an option not listed here`를 만들고, 별도 scoring
prompt가 각 option의 확률을 계산한다. 확률이 cutoff 이상인 option들이 action CP
prediction set을 이룬다. 이때 CP class는 belief state가 아니라 **다음 행동
option**이다.

### runtime When

```text
if action_prediction_set is empty:
    질문 시작
else if fallback option is in action_prediction_set:
    질문 시작
else if action_prediction_set size != 1:
    질문 시작
else:
    질문하지 않음
```

이는 KnowNo의 help trigger와 같다. 빈 set과 fallback singleton도 로봇이 실행할
명확한 행동을 얻지 못한 경우이므로 질문한다.

### runtime What

When이 true이면 action option을 묻지 않는다. 현재 전체 belief의 ambiguous
Boolean state fact 각각에 대해 기대 정보 이득(EIG)을 계산하고, EIG가 가장 큰
fact 하나를 선택한다. oracle의 Boolean 답으로 belief particle을 filter한 뒤
Action CP When을 다시 평가한다. 따라서 비교에서 바뀌는 요소는 When이고,
What은 Ours와 동일하다.

### 단일 episode pseudo code

```text
initialize physical POMCP and belief
load KnowNo action-CP qhat and score temperature

while task is active:
    physical_action = physical_planner.search(belief)
    execute physical_action
    belief = update(belief, action, observation)
    sync symbolic knowledge to posterior MAP state

    asked = empty set
    while True:
        options = LLM_generate_action_options(current context)
        scores = LLM_score_action_options(options)
        cp_set = {option if score(option) >= 1-qhat}

        if cp_set is one non-fallback action:
            break

        fact = argmax over unasked ambiguous facts of EIG(fact | belief)
        if no fact:
            break

        answer = oracle(fact)
        belief = filter_and_normalize(belief, fact, answer)
        asked.add(fact)

        if belief frontier did not shrink:
            break

    commit posterior MAP state
    check task termination
```

### reward, step과 로그

- Physical POMCP와 실제 episode reward에는 physical action만 들어간다.
- CP 질문에는 별도 query reward/cost를 적용하지 않는다.
- `steps`와 `max_step`은 physical action 기준이다.
- 질문은 `total_questions`와 step별 `query_count`로 따로 기록한다.
- 생성 option, prediction set, fallback, option score, qhat 판단 근거를
  `policy_trace.json`에 기록한다.

---

### 목적

What을 Ours EIG로 고정하고, 질문을 시작하는 시점만 Query-as-Action의 value
comparison으로 바꾼다. Query-as-Action planner가 최종 행동을 직접 실행하지
않으며, query Q-value가 높은지를 **trigger로만 사용**한다.

### When 계산

```text
1. 현재 ambiguous fact 전체를 query 후보로 만든다.

2. 보조 QaA evaluator의 root에 다음 후보를 넣는다.
   - MAP symbolic knowledge에서 applicable한 모든 physical actions
   - 각 ambiguous fact에 대응하는 QueryAction(f) 전체

3. POMCP simulation으로 root Q-values를 계산한다.

4. 다음 값을 비교한다.
   best_query = max_f Q(b, QueryAction(f))
   best_physical = max_a Q(b, physical_action_a)

5. best_query > best_physical이면 질문 trigger를 켠다.
   같으면 질문하지 않는다.
```

QaA가 최고로 평가한 query fact는 trigger 근거로만 기록한다. 그 fact 자체를
실제 질문으로 실행하지 않는다.

### What 계산

trigger가 켜지면 전체 현재 belief에서 Ours EIG를 별도로 계산하여 가장 큰 fact
하나를 묻는다. QaA의 최고 query fact와 실제 EIG fact는 달라도 된다. 답변으로
belief를 갱신한 뒤 같은 physical step에서는 추가 질문이나 value trigger 재평가를
하지 않는다.

### 단일 episode pseudo code

```text
initialize physical POMCP, belief, and auxiliary QaA value evaluator

while task is active:
    physical_action = physical_planner.search(belief)
    execute physical_action
    belief = update(belief, action, observation)

    candidates = all ambiguous facts
    Q_query, Q_physical = auxiliary_QaA_search(
        root_queries=candidates,
        root_physical=applicable physical actions
    )

    if max(Q_query) > max(Q_physical):
        f_EIG = highest-EIG ambiguous fact on the current belief
        answer = oracle(f_EIG)
        belief = filter_and_normalize(belief, f_EIG, answer)
        # 이 physical step에서는 여기서 질문을 끝낸다.

    commit posterior MAP state
    check task termination
```

### reward, step과 로그

- 보조 evaluator 안에서는 query reward `-query_cost`, invalid physical action
  `-failure_penalty`, domain task reward와 gamma가 Q-value에 반영된다.
- evaluator는 action을 실제 환경에 실행하지 않는다.
- 실제 episode cumulative reward에는 physical `env.step()` reward만 더한다.
- 실제 Boolean 질문 비용은 cumulative reward에서 차감하지 않는다.
- `steps`와 `max_step`은 physical action 기준이다.
- `policy_trace.json`에 trigger가 된 최고 query fact/Q, 실제 EIG fact/score,
  모든 query/physical Q, `best query Q - best physical Q`, visit 수와 simulation
  budget을 기록한다.

---

## 5. `03_value_what.py`: Ours confidence When + Query-value What

### 목적

질문 시점은 Ours의 belief confidence threshold로 고정하고, 무엇을 물을지만
Query-as-Action value로 바꾼다. 따라서 이 조건은 When을 통제하고 What 선택
방법만 비교한다.

### When 계산

```text
confidence = 1 - H(b) / log2(number of particles)

if confidence < threshold:
    질문 시작
else:
    질문하지 않음
```

코드는 strict `<`를 사용하므로 confidence가 threshold와 같으면 질문하지 않는다.

### What 계산

```text
1. 현재 ambiguous하고 아직 묻지 않은 fact 전체를 candidates로 만든다.

2. 각 fact f를 QueryAction(f)로 만든다.

3. 보조 QaA evaluator root에는 query action만 넣는다.
   root_mode = query_only

4. 각 QueryAction의 root Q-value를 계산한다.
   root에서 physical action과 비교하지 않는다.
   다만 query 이후 search depth에서는 inherited QaA dynamics에 따라
   future physical action과 future query가 value에 반영될 수 있다.

5. Q-value가 가장 큰 fact를 선택한다.
   동률이면 fact 문자열의 사전순으로 선택한다.
```

### 단일 episode pseudo code

```text
initialize physical POMCP, belief, and auxiliary QaA value evaluator

while task is active:
    physical_action = physical_planner.search(belief)
    execute physical_action
    belief = update(belief, action, observation)

    asked = empty set
    while confidence(belief) < threshold:
        candidates = unasked ambiguous facts
        if candidates is empty:
            break

        query_values = auxiliary_QaA_search(
            root_queries=candidates,
            root_physical=[]
        )
        fact = argmax_f query_values[f]

        answer = oracle(fact)
        belief = filter_and_normalize(belief, fact, answer)
        asked.add(fact)

        if belief frontier did not shrink:
            break

        # confidence를 다시 계산한다.

    commit posterior MAP state
    check task termination
```

### reward, step과 로그

- `query_cost`, `failure_penalty`, `answer_accuracy`, domain reward와 gamma는
  보조 evaluator의 query Q-value에만 영향을 준다.
- 실제 질문은 exact auto oracle을 사용한다.
- 실제 cumulative reward에는 physical action reward만 기록한다.
- `steps`와 `max_step`은 physical action 기준이다.
- `policy_trace.json`에 confidence/threshold, 각 fact의 query Q, visit 수,
  선택한 fact와 simulation budget을 기록한다.

---

## 해석할 때 지켜야 할 핵심 구분

### KnowNo와 `01_cp_when.py`는 같은 Action CP를 서로 다르게 사용한다

```text
KnowNo:
    class = LLM이 생성한 다음 행동 option
    set size가 모호하면 행동 선택을 요청

01_cp_when.py:
    class = KnowNo와 같은 LLM 다음 행동 option
    action CP set이 모호하면 질문 phase를 시작
    질문 내용 = 전체 belief에서 Ours EIG가 선택한 Boolean state fact
```

두 방법은 같은 action ambiguity trigger를 공유한다. KnowNo는 사람에게 행동
option을 고르게 하지만, CP-When ablation은 action set을 When에만 쓰고 실제
질문 내용은 Ours EIG가 고른 state fact로 유지한다.

### Query-as-Action baseline과 Value ablation의 QaA 사용 범위는 다르다

```text
Query-as-Action baseline:
    QaA POMCP의 argmax action을 실제로 실행
    QueryAction과 physical action이 동일 decision budget을 소비
    실제 reward에도 query cost를 반영

02_value_when.py / 03_value_what.py:
    QaA POMCP는 root Q-value를 계산하는 보조 evaluator
    evaluator가 선택한 action을 실제로 실행하지 않음
    질문은 physical action 실행 후 별도 phase에서 수행
    실제 episode reward와 physical max_step에 질문을 포함하지 않음
```

### `02_value_when.py`와 `03_value_what.py`가 통제하는 요소

```text
02_value_when.py:
    When = query value vs physical value
    What = Ours EIG

03_value_what.py:
    When = Ours entropy-confidence threshold
    What = query-action value
```

이 구조 때문에 두 조건을 통해 질문 시점과 질문 내용의 효과를 분리해서 볼 수
있다.

## 구현 위치

- KnowNo episode:
  - `scripts/baseline/knowno/scripts/knowno_multistep_tomato.py`
  - `scripts/baseline/knowno/scripts/knowno_multistep_wastesorting.py`
- KnowNo CP scoring/calibration:
  - `scripts/baseline/knowno/scripts/prompt.py`
  - `scripts/baseline/knowno/compute_qhat.py`
- Query-as-Action episode:
  - `scripts/baseline/targeted_query_pomdp/run_experiment.py`
- Query-as-Action action schemas and planner:
  - `scripts/baseline/targeted_query_pomdp/query_actions.py`
  - `scripts/baseline/targeted_query_pomdp/planner.py`
- Query-as-Action reward:
  - `scripts/baseline/targeted_query_pomdp/rw.py`
- policy ablation entry points:
  - `scripts/ablation/when_what_policy_ablation/01_cp_when.py`
  - `scripts/ablation/when_what_policy_ablation/02_value_when.py`
  - `scripts/ablation/when_what_policy_ablation/03_value_what.py`
- shared When/What policies:
  - `scripts/ablation/when_what_policy_ablation/script/policies.py`
- Action-CP trigger implementation:
  - `scripts/ablation/when_what_policy_ablation/script/cp_when.py`
- Archived belief-state CP experiment:
  - `scripts/ablation/when_what_policy_ablation/belief_state_cp_archive/`
- auxiliary query-value evaluator:
  - `scripts/ablation/when_what_policy_ablation/script/value_evaluator.py`
- shared query loop and experiment runner:
  - `scripts/ablation/when_what_policy_ablation/script/query_episode.py`
  - `scripts/ablation/when_what_policy_ablation/script/runner.py`
