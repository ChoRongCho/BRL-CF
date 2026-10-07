# When–What policy ablation 실험 설정

작성일: 2026-10-05. 재실험 전 조건별 구조, 할인율, 질문 비용 및 보상 정의를 점검하기 위한 문서이다.

기존 실험의 설정값은 이 폴더의 `00_raw/`에 보관된 trace와 [README.md](README.md)를 기준으로 확인했다. 동작과 보상 계산 설명은 현재 작업 트리의 소스 코드를 기준으로 한다. 과거 실행 당시의 소스 전체를 복원한 검증은 아니므로, 기록된 설정과 현재 구현을 구분해서 읽어야 한다.

## 1. 비교 목적과 실험 구성

실험의 의도는 Ours의 질문 시점(When)과 질문 내용(What)을 기준으로, 시점만 다른 정책으로 바꾸거나 내용만 다른 정책으로 바꾸는 것이다. 물리 행동은 공통으로 기본 POMCP가 계획한다. **그러나 현재 구현에서는 Value-When의 1개 질문 제한과 CP-When의 반복 gate 평가 때문에 질문 반복·종료 규칙도 달라진다. 아래 표는 현재 구현이며, 의도한 통제 설계와의 차이는 2절 후반에 명시한다.**

| 조건 | When: 언제 질문하는가 | What: 무엇을 질문하는가 | 한 물리 행동 이후 질문 방식 |
|---|---|---|---|
| Ours (`ours`) | belief confidence가 0.8 미만 | 기대 정보 이득(EIG)이 가장 큰 사실 | 답변으로 belief를 갱신하고 confidence 기준에 따라 추가 질문 |
| Action-CP When (`cp_when`) | LLM의 action prediction set이 단일 유효 행동으로 결정되지 않음 | Ours와 같은 EIG | 답변마다 CP When을 다시 평가 |
| Value-When (`value_when`) | 최고 질문 Q가 최고 물리 행동 Q보다 큼 | Ours와 같은 EIG | 조건이 충족돼도 물리 행동당 최대 1개 질문 |
| Value-What (`value_what`) | Ours와 같은 confidence < 0.8 | 질문 후보 중 Q가 가장 큼 | 답변마다 confidence를 재평가하여 추가 질문 |

실험은 Tomato와 Waste sorting 두 도메인, 각 scene 1–5, scene·조건별 40회로 구성된다. 조건당 400회, 총 1,600개 실행 슬롯이다. 기존 실행일은 2026-09-21~22이며, CP-When의 형식 파싱 오류 4건을 제외한 정상 결과는 1,596개이다. 기존 분석은 이 오류 4건도 성공률 분모에 포함하고 실패로 집계한다. 조건 비교에는 같은 domain·scene·seed를 대응시킨다.

## 2. 실행 구조와 조건별 의미

공통 흐름은 다음과 같다.

```text
기본 POMCP로 물리 행동 선택·실행
  → 관측으로 belief 갱신
  → When으로 질문 여부 판단
  → 필요하면 What으로 Boolean 사실 질문 선택
  → Oracle 답변으로 belief 갱신
  → 조건별 추가 질문 또는 질문 종료
  → MAP 상태를 반영하고 다음 물리 행동 계획
```

따라서 이 실험의 When은 물리 행동 실행·관측 이후의 질문 여부이다. Value 조건에서도 가치 평가기가 선택한 물리 행동을 직접 실행하지 않는다. 실제 행동 실행용 POMCP와 질문 가치 평가용 POMCP는 별도이다.

### Ours

belief 후보의 확률 분포에 대해 정규화 엔트로피 기반 confidence를 계산한다.

`confidence ≈ 1 − H(b) / log₂(N)`

여기서 N은 belief 후보 수이며, N이 0 또는 1이면 confidence는 1이다. 구현에는 로그 계산용 작은 epsilon이 포함된다. confidence가 0.8 미만이면, `H(b) − E[H(b | answer)]`가 최대인 사실을 질문한다. 질문 후 confidence가 기준에 도달하거나 질문 후보가 없어지는 등의 종료 조건까지 진행한다.

### Action-CP When

LLM의 행동 선택 점수에 temperature scaling을 적용하고, `score ≥ 1 − qhat`인 선택지로 prediction set을 구성한다. 이 집합이 유효 행동 하나만 포함하면 질문하지 않는다. 집합이 비어 있거나, 여러 선택지를 포함하거나, fallback인 NoOpT를 포함하면 질문한다.

의도한 설계에서 CP는 질문 episode의 시작 여부만 결정하고 실제 질문 내용은 EIG가 고른다. 현재 구현은 답변 후 CP를 다시 평가하므로 CP가 추가 질문 여부도 결정한다. 이 패키지는 **action prediction set 기반 CP**이며, 별도로 보관된 belief-state CP 방식과 구분된다.

### Value-When

모호한 사실들에 대한 질문 액션과 현재 symbolic knowledge에서 적용 가능한 물리 행동을 가치 평가용 POMCP의 root 후보로 함께 넣는다(`root_mode="compare"`).

`질문 시작 ⇔ max_q Q(b, q) > max_a Q(b, a)`

동점이면 질문하지 않는다. 질문 시작을 결정한 최고 Q 질문과 실제로 묻는 질문은 다를 수 있다. **What은 여전히 EIG**이기 때문이다. 현재 구현은 질문을 시작해도 물리 행동당 최대 1개만 묻는다. 따라서 Ours와 비교할 때 When 판단뿐 아니라 질문 반복 제한의 차이도 고려해야 한다.

### Value-What

질문 시작 여부는 Ours와 같은 confidence 기준이다. 질문이 필요할 때 모호한 사실들의 질문 액션만 root 후보로 넣고(`root_mode="query_only"`), 최고 Q 질문을 선택한다.

`선택 질문 = argmax_q Q(b, q)`

root에서 물리 행동과 비교하여 질문을 취소하는 구조가 아니다. 모든 질문 Q가 음수여도 When이 참이고 후보가 있으면 가장 높은 Q 질문을 선택한다. root 이후의 가치 평가는 Query-as-Action POMCP로 진행하며, rollout은 적용 가능한 물리 행동을 사용한다.

### 질의 시점을 시간 순서로 명시

`a_t`는 이미 실행한 물리 행동, `b_{t+1}^{obs}`는 그 행동과 관측으로 갱신한 posterior belief이다. 이 갱신은 detect에만 한정되지 않는다. 모든 실행된 물리 행동에 대해 runner가 `update_belief(belief, observation, action)`을 호출한다.

여기서 belief가 “확정됐다”는 것은 **관측을 반영한 분포가 계산됐다**는 뜻이다. 실제 상태를 확실히 알게 됐다는 뜻은 아니며, 질문 답변으로 그 분포를 더 갱신할 수 있다.

```python
# 시간 t의 물리 행동
a_t = physical_pomcp.search(b_t)
o_next = env.step(a_t).observation
b_obs = update_belief(b_t, o_next, a_t)  # b_{t+1}^{obs}

# 다음 물리 행동 a_{t+1}을 실행하기 전에 질문할지 판단
query_values, physical_values = query_value_pomcp.evaluate(b_obs)
start_query = max(query_values) > max(physical_values)
# physical_values: 이미 실행한 a_t가 아니라, b_obs에서 다음에 할 행동들의 Q
# query_values: 지금 질문하고 후속 행동을 이어가는 경우의 Q

if start_query:
    b_after_query = question_episode(b_obs)
else:
    b_after_query = b_obs

b_next = commit_map_state(b_after_query)
a_next = physical_pomcp.search(b_next)  # a_{t+1}을 별도로 계획
```

위 코드는 순서를 보여주는 의사코드이다. 가치 평가에서 최고 Q를 받은 물리 행동을 그대로 실행하지 않는다. 질문 episode가 끝난 후 기본 POMCP가 다음 물리 행동을 계획한다. 또한 `Q(query) > Q(physical)`은 불확실성 크기 자체의 판정이 아니라 **비용과 할인된 후속 보상을 고려했을 때 질문이 더 유리하다는 추정**이다. confidence threshold와 같은 조건은 아니다.

### 의도한 설계: When은 진입, threshold는 반복·종료

2026-10-05 논의에서 확인한 설계는 다음과 같다. When만 비교하는 조건은 질문 episode 진입 기준만 바꾼다. 일단 진입하면 네 조건 모두 답변 후 confidence를 확인하여 추가 질문 여부를 결정한다. 이 원칙은 기존 `review/when_what_ablation_report.md`의 17절에도 명시돼 있다.

```python
# 의도한 설계의 의사코드 — 현재 실행 코드가 아님
def question_episode_intended(belief, condition):
    # When은 episode 진입 시 한 번 판단
    if condition in ("ours", "value_what"):
        start = confidence(belief) < 0.8
    elif condition == "cp_when":
        start = cp_when(belief)
    elif condition == "value_when":
        start = best_query_q(belief) > best_next_physical_q(belief)

    if not start:
        return belief

    asked = set()
    while True:
        candidates = ambiguous_facts(belief) - asked
        if not candidates:
            break

        # value_what에서만 What을 변경
        if condition == "value_what":
            fact = select_highest_query_q(belief, candidates)
        else:
            fact = select_highest_eig(belief, candidates)

        before_count = frontier_size(belief)
        answer = oracle(fact)
        belief = update_with_answer(belief, fact, answer)
        asked.add(fact)

        # 네 조건에 공통인 반복·종료 기준
        if confidence(belief) >= 0.8:
            break
        if frontier_size(belief) >= before_count:
            break
        # confidence < 0.8이고 후보가 남으면 다음 질문
        # Value-When의 1개 제한이나 CP/Value gate 재평가는 없음

    return belief
```

CP/Value When이 true이면 질문 후보가 있는 한 최소 한 번 묻는다. 진입 당시 confidence가 이미 0.8 이상이더라도 첫 질문은 허용하고, 그 답변 이후 공통 threshold로 종료 여부를 판단한다. “여러 번 질문”은 고정 횟수가 아니라 confidence, 후보 존재 여부, belief 감소 여부에 따라 필요한 만큼 반복한다는 뜻이다.

### 현재 구현: 공통 반복 규칙을 대체한 지점

아래는 `script/query_episode.py`와 `script/policies.py`의 현재 동작을 축약한 의사코드이다.

```python
def question_episode_current(belief, condition):
    asked = set()
    limit = 1 if condition == "value_when" else None  # 차이 A

    while True:
        # 차이 B: 진입 시뿐 아니라 답변 이후에도 조건별 When을 재평가
        if not when_policy[condition].should_start(belief):
            break

        fact = what_policy[condition].select(belief, exclude=asked)
        if fact is None:
            break

        before_count = frontier_size(belief)
        belief = update_with_answer(belief, fact, oracle(fact))
        asked.add(fact)

        if frontier_size(belief) >= before_count:
            break
        if limit is not None and len(asked) >= limit:
            break  # Value-When: confidence가 낮아도 여기서 종료

        # 모든 조건에 공통인 confidence >= 0.8 종료 판정은 없음
        # 다음 반복에서 각 조건의 When으로 돌아감

    return belief
```

| 지점 | 의도한 설계 | 현재 구현 | 비교에 미치는 영향 |
|---|---|---|---|
| Value-When 첫 답변 이후 | confidence < 0.8이면 후보·belief 감소 조건에 따라 추가 질문 | `max_questions_per_step=1`로 종료 | When뿐 아니라 질문 횟수 제한도 달라짐 |
| CP-When 첫 답변 이후 | 공통 confidence threshold로 추가 질문 판단 | CP prediction set으로 When을 다시 판단 | When뿐 아니라 episode 종료 기준도 달라짐 |
| Ours / Value-What 반복 | confidence threshold 사용 | confidence threshold 사용 | 이 반복 기준에 대해서는 의도와 일치 |

구체적인 코드 위치는 `script/policies.py`의 Value-When bundle에 있는 `max_questions_per_step=1`, `script/query_episode.py`의 `while True` 내부 `bundle.when.should_start(context)`, 그리고 답변 처리 뒤 `question_limit_reached`로 종료하는 분기이다.

예를 들어 Value-When이 질문을 시작하고 첫 답변 이후 confidence가 0.55가 됐으며 belief가 줄고 후보도 남아 있다면, **의도한 설계는 다음 EIG 질문을 계속하지만 현재 구현은 종료하고 다음 물리 행동 계획으로 넘어간다.** CP-When도 같은 상황에서 CP가 singleton action을 반환하면 현재 구현은 종료하지만, 의도한 설계는 confidence 0.8에 못 미쳤으므로 계속 질문한다.

**판정:** 현재 구현은 “When만 교체하고 질문 반복 구조는 공통으로 유지한다”는 통제 설계와 불일치한다. 특히 Value-When의 1개 제한은 이 설계 목적에 비추어 구현 오류로 봐야 한다. 다만 누가 언제 어떤 의도로 변경했는지는 여기서 확인하지 않았으며, 변경 주체나 의도를 단정하지 않는다. 기존 성능 차이를 When 기준만의 효과로 해석해서도 안 된다. 이번 작업은 이 차이를 문서화한 것이며 실행 코드는 변경하지 않았다.

### Value-What의 질의 시점과 반복 구조

Value-What도 `a_t` 실행과 관측으로 `b_{t+1}^{obs}`를 계산한 뒤, 다음 물리 행동 `a_{t+1}`을 실행하기 전에 질문한다. 여기서는 **질문 여부를 Q로 비교하지 않는다.** Ours와 동일한 confidence threshold가 질문 여부를 결정하고, 질문이 필요할 때만 Q가 질문 내용을 선택한다.

```python
# Value-What의 의도한 흐름과 현재 구현을 요약한 의사코드
a_t = physical_pomcp.search(b_t)
o_next = env.step(a_t).observation
belief = update_belief(b_t, o_next, a_t)  # b_{t+1}^{obs}
asked = set()

while confidence(belief) < 0.8:          # When: Ours와 동일
    candidates = ambiguous_facts(belief) - asked
    if not candidates:
        break

    # What: 현재 belief에서 질문 후보들의 Q를 새로 계산
    values = query_value_pomcp.evaluate(
        belief,
        query_facts=candidates,
        root_mode="query_only",        # root에는 질문 후보만 배치
    )
    fact = argmax_query_q(values)        # EIG 대신 Q로 질문 내용 선택
    # max Q(query) > max Q(physical) 비교는 하지 않음
    # 모든 Q(query)가 음수여도 가장 높은 Q의 질문을 선택

    before_count = frontier_size(belief)
    answer = oracle(fact)
    belief = update_with_answer(belief, fact, answer)
    asked.add(fact)

    if frontier_size(belief) >= before_count:
        break
    # 1개 제한 없음: 다음 반복에서 갱신된 confidence를 검사
    # 0.8 미만이고 후보가 남으면 갱신된 belief로 질문 Q를 재계산

b_next = commit_map_state(belief)
a_next = physical_pomcp.search(b_next)   # 질문 종료 후 다음 물리 행동 계획
```

실제 공통 runner는 `while True` 안에서 `BeliefThresholdWhen.should_start()`를 호출한다. Value-What에서는 그 함수가 `confidence < 0.8`을 검사하므로 위 의사코드와 같은 반복 기준이 된다. `PolicyBundle`에 별도의 질문 수 제한을 지정하지 않아 `max_questions_per_step`은 기본값 `None`이다.

| 점검 항목 | 의도한 Value-What | 현재 구현 | 판정 |
|---|---|---|---|
| 질문 시점 | 관측 belief 갱신 후, 다음 물리 행동 전 | 동일 | 일치 |
| 질문 시작 | confidence < 0.8 | `BeliefThresholdWhen` | 일치 |
| 질문 후보 | 현재 모호한 사실 중 이번 episode에서 아직 묻지 않은 사실 | `context.ambiguous_facts()` | 일치 |
| 질문 선택 | 후보별 질문 Q 중 최댓값 | `QueryValueWhat`, `root_mode="query_only"` | 일치 |
| 첫 답변 이후 | confidence가 낮으면 추가 질문 | 갱신된 belief로 confidence와 질문 Q를 다시 계산 | 일치 |
| 질문 수 제한 | 고정 1개 제한 없음 | `max_questions_per_step=None` | 일치 |
| 종료 | confidence 기준 도달, 후보 없음, belief가 줄지 않음 | 동일 | 일치 |

예를 들어 첫 답변 이후 confidence가 0.55이고 belief가 줄고 후보도 남아 있다면, Value-What은 **다음 후보들의 Q를 다시 계산해 두 번째 질문을 한다.** 그 답변 이후 confidence가 0.85가 되면 질문을 종료하고 다음 물리 행동을 계획한다. 첫 번째 평가에서 만들어 둔 질문 순서를 그대로 따라가는 방식이 아니다.

**Value-What 판정:** 확인한 질문 진입·선택·반복·종료 구조에는 Value-When의 1개 제한 같은 불일치가 없다. Ours의 EIG 질문 선택을 Q 기반 질문 선택으로 바꾸고, threshold 기반 질문 반복은 유지한다. 앞의 Value-When·CP-When에 대한 설계 불일치 판정을 Value-What까지 확대하지 않는다.

비용 1.0은 Value-What의 각 질문 Q에도 즉시 보상 −1.0으로 포함된다. 다만 비용을 포함한 Q는 **어떤 질문을 할지** 고르는 데 사용하며, 질문을 중단할지는 confidence가 결정한다. `query_only`는 root 후보 제한이지, 후속 물리 행동 보상을 계산하지 않는다는 뜻이 아니다. 확인 근거는 `script/policies.py`의 `QueryValueWhat` 및 Value-What bundle, `script/value_evaluator.py`의 `evaluate()`, `script/query_episode.py`의 반복과 종료 분기이다.

## 3. 실험 파라미터

아래 공통 값은 Ours 400개, CP-When 396개, Value-When 400개, Value-What 400개의 정상 trace에서 확인했다. 오류 4건에는 정상 trace가 없다.

| 항목 | 값 | 적용 의미 |
|---|---:|---|
| 할인율 gamma | 0.2 | 기본 POMCP 및 Value 조건의 질문 가치 평가 |
| query cost | 1.0 | 질문 가치 평가에서 질문당 즉시 보상 −1.0 |
| failure penalty | 10.0 | 질문 가치 평가 중 적용 불가능한 물리 행동에 −10.0 |
| answer accuracy | 1.0 | 가치 평가기의 질문 응답 정확도; 실제 질문은 auto Oracle 사용 |
| confidence threshold | 0.8 | Ours·Value-What의 When에 사용 |
| n_simulations | 100 | 기본 탐색 예산; 가치 평가기는 아래 보정 적용 |
| max_depth | 20 | 탐색 깊이 상한 |
| epsilon | 0.005 | 할인 계수에 따른 조기 탐색 종료 |
| UCB c | 1.0 | 탐색 시 exploration 계수 |
| max_step | 50 | 물리 행동 실행 횟수 상한 |

query cost와 failure penalty가 모든 조건의 메타데이터에 기록돼 있어도, Ours와 CP-When의 EIG/CP 판단이 이 비용으로 Q를 계산한다는 뜻은 아니다. Value-When과 Value-What의 별도 가치 평가기에 적용되는 값이다. 마찬가지로 threshold 0.8은 Value-When의 질문 시작 조건이 아니다.

CP-When의 정상 trace 396개에서 확인한 추가 설정은 Tomato `qhat=0.8404`, Waste sorting `qhat=0.8704`, `score_temperature=5.0`이다.

가치 평가기의 실제 simulation 예산은 `max(100, root 후보 수 + 1)`이다. 첫 simulation이 root 초기화 rollout에 사용되므로 각 root 후보를 최소 한 번 방문할 수 있도록 보정한다.

### gamma와 유효 탐색 깊이

현재 POMCP는 `gamma^depth < epsilon`이면 해당 깊이에서 0을 반환한다. 따라서 gamma 0.2, epsilon 0.005에서는 다음과 같다.

| depth | gamma^depth | 할인 조건에 의한 종료 |
|---:|---:|---|
| 0 | 1 | 아니오 |
| 1 | 0.2 | 아니오 |
| 2 | 0.04 | 아니오 |
| 3 | 0.008 | 아니오 |
| 4 | 0.0016 | 예 |

즉 max_depth가 20이어도 현재 설정에서는 depth 0–3의 보상까지만 포함할 수 있다. 질문도 탐색 깊이 한 단계를 소비하므로, root 질문 뒤의 물리 행동 보상은 한 번 할인된다. 예를 들어 질문 직후 다음 행동에서 +10을 확실히 받고 이후 보상이 없다면 root 질문 가치는 `−1 + 0.2 × 10 = 1`이다. 이는 계산 예시이며 실제 추정 Q는 전이·관측·후속 정책에 따라 달라진다.

## 4. 비용과 보상 정의

### 물리 행동의 도메인 보상

기본 POMCP와 실제 환경은 도메인별 `RewardModel`을 사용한다. 질문 가치 평가기는 별도의 로컬 task reward 구현을 사용하며, 현재 코드의 보상 규칙은 아래와 같다.

| 도메인 | 조건 | 보상 |
|---|---|---:|
| Tomato | fresh 토마토를 place하여 loaded 사실이 새로 생김 | +10 |
| Tomato | rotten 토마토를 discard하여 discarded 사실이 새로 생김 | +10 |
| Tomato | detect / scan / navigate의 해당 streak가 연속 2회째 이상 | −5 |
| Tomato | 그 밖의 행동 및 별도 goal state 보상 | 0 |
| Waste sorting | 올바른 분류의 bin에 넣어 in_bin 사실이 새로 생김 | +5 |
| Waste sorting | 전체 goal을 미달성 상태에서 새로 달성 | +10 추가 |
| Waste sorting | detect_waste 연속 3회째 이상 | −10 |
| Waste sorting | 그 밖의 행동 | 0 |

Tomato의 detect/scan streak는 각각 대상 인자에 대응하는 fluent, navigate는 navigate streak로 판정한다. 단순히 전체 실행에서 해당 행동이 두 번 나왔다는 의미가 아니다. Waste sorting의 마지막 올바른 배치가 전체 goal을 새로 달성하면 보상은 +15가 될 수 있다. 두 도메인 모두 모든 물리 행동에 일괄 적용하는 step cost는 현재 비활성화돼 있다(`STEP_REWARD` 관련 코드가 주석).

### 가치 평가용 POMCP의 추가 보상

| 평가 대상 | 즉시 보상 | 후속 처리 |
|---|---:|---|
| 질문 액션 | `−action.cost = −1.0` | 상태를 복사하고 답변 관측을 생성한 뒤 탐색 계속 |
| sampled state에서 precondition이 성립하지 않는 물리 행동 | `−failure_penalty = −10.0` | 해당 simulation 경로 종료 |
| 적용 가능한 물리 행동 | 위 도메인 task reward | 전이·관측 후 탐색 계속 |

여기서 −10은 모든 과제 실패나 확률적 행동 실패에 일괄 부여하는 값이 아니다. 가치 평가 중 샘플링된 상태에서 물리 행동의 precondition이 성립하지 않을 때 적용된다.

비용 전달 경로는 다음과 같다.

```text
run_when_what_policy_ablation.sh: QUERY_COST="1.0"
  → --query-cost
  → args.policy.query_cost
  → QueryValueEvaluator / RestrictedRootQueryPlanner
  → QueryAction.cost
  → QueryAsActionRewardModel.query_reward(): −action.cost
  → simulate(): reward + gamma × 후속 return
  → root action Q 갱신
```

현재 코드에서 두 Value 조건을 실행 설정으로 초기화하고 `_generate()`를 호출하여, 각각 `query_cost=1.0`, `QueryAction.cost=1.0`, 질문 즉시 보상 `−1.0`임을 확인했다. 보관된 두 Value 조건의 trace 800개도 모두 query cost 1.0이다.

### 로그의 누적 reward와 계획 내부 Q의 차이

실행 로그의 `reward.cumulated`는 환경의 물리 행동 보상을 단순 합산한다. 질문할 때 −1을 별도로 차감하지 않으며, 실행 누적값에 gamma 할인도 적용하지 않는다. 따라서 이 값은 **질문 비용을 차감한 discounted return이 아니다**. 질문 수는 별도 지표로 기록된다. 가치 평가 내부의 invalid-action penalty 역시 실제 실행 누적 reward에 자동 합산되는 구조가 아니다.

## 5. 재실험 전에 유지·변경 여부를 정할 사항

- 현재 단일 실행 shell은 `QUERY_COST="1.0"`을 고정하고 `--query-cost` 변경 옵션을 받지 않는다. Python 진입점은 이 옵션을 지원한다. 배치에서 비용을 바꾸려면 전달 경로도 함께 수정해야 한다.
- 재실험의 의도한 설계는 When을 episode 진입에만 사용하고, 이후 네 조건의 질문 반복·종료를 공통 confidence threshold로 맞추는 것이다. 현재 Value-When의 1개 제한과 CP-When의 반복 gate 평가는 이 설계와 불일치한다. 2절의 의사코드는 수정 기준이며 아직 실행 코드에 반영되지 않았다.
- gamma를 바꾸면 미래 보상의 가중치뿐 아니라 epsilon에 의한 유효 탐색 깊이도 바뀐다. gamma 비교에서는 두 효과가 함께 발생한다.
- Value-What은 비용이 포함된 Q를 쓰지만 질문 자체를 할지 결정하는 기준은 confidence이다. 비용 증가가 직접적인 질문 취소 기준으로 작동하지 않는다.
- Ours는 별도 `when_what_random/run_experiment.py` 실행 경로의 결과를 이 패키지에 합친다. policy ablation 배치 자체의 기본 조건은 `cp_when value_when value_what`이므로, 전체 네 조건 재실험 시 Ours 실행도 포함해야 한다.

## 6. 확인에 사용한 파일

아래 소스 경로는 `02_BRL_POMDP_CODE/` 기준이다.

| 파일 | 확인 내용 |
|---|---|
| `run/run_when_what_policy_ablation.sh` | 단일 실행 기본값과 Python 인자 전달 |
| `run/iterate_when_what_policy_ablation.sh` | 조건·도메인·scene·반복 수 |
| `scripts/ablation/when_what_policy_ablation/script/batch.py` | paired seed 및 배치 실행 |
| `scripts/ablation/when_what_policy_ablation/script/runner.py` | 기본 POMCP와 가치 평가기 분리, 실제 reward 집계 |
| `scripts/ablation/when_what_policy_ablation/script/policies.py` | 조건별 When·What 및 질문 제한 |
| `scripts/ablation/when_what_policy_ablation/script/query_episode.py` | 질문·답변·belief 갱신 반복 |
| `scripts/ablation/when_what_policy_ablation/script/cp_when.py` | action CP prediction set 구성 |
| `scripts/ablation/when_what_policy_ablation/script/value_evaluator.py` | root 후보 제한, 비용 전달, simulation 예산 |
| `scripts/baseline/targeted_query_pomdp/planner.py` 및 `rw.py` | 질문 Q 탐색, query cost, invalid-action penalty |
| `scripts/planners/pomcp.py` | 기본 POMCP 할인과 탐색 종료 |
| `scripts/models/tomato/rw.py`, `scripts/models/wastesorting/rw.py` | 실행 환경의 도메인 보상 |
| `scripts/models/feedback_manager.py` | confidence와 EIG 계산 |
| `scripts/ablation/when_what_random/run_experiment.py` | Ours 실행과 누적 reward |

기존 결과 수치는 [report.md](report.md), 실행별 자료는 [01_processed/episodes.csv](01_processed/episodes.csv), 원본 파일 목록은 [00_raw_sources.csv](00_raw_sources.csv)를 참고한다. 이 문서를 작성하면서 실험 설정이나 기존 결과는 변경하지 않았다.
