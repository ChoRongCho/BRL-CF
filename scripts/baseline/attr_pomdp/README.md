# Active Search baseline (grounded-fact Adapted Attr-POMDP)

Yang et al., *Interactive Robotic Grasping with Attribute-Guided
Disambiguation* (ICRA 2022)의 의사결정 구조를 symbolic domain에 맞게 옮긴
baseline이다.

원 논문의 color/location 질문을 grounded Boolean fact 질문으로 바꾼다.

```text
Original:
  AskAttr(color), AskAttr(location), AskPoint(object), Grasp(object)

This adaptation:
  QueryFact(symbolic fact), TaskAction(domain action)
```

따라서 이 구현은 논문의 vision-language pipeline 재현이 아니라
`Adapted Attr-POMDP with grounded fact queries`다. 본 프로젝트의 실험과 결과
경로에서는 이 방법의 canonical 이름을 `active-search`로 사용한다.

## 구현된 기능

- Tomato 및 WasteSorting의 불확실한 attribute fact로 초기 particle belief 생성
- task action과 query action을 같은 finite-horizon Bellman search에서 평가
- query의 no-op world transition
- oracle `yes/no` observation branch
- answer-conditioned Bayesian belief update
- query cost와 task reward의 expected-return 비교
- 동일 fact 반복 질문 방지
- action, answer, belief 변화 및 Q-value 기록

질문 후보는 현재 belief에서 참일 확률이 0과 1 사이인 fact만 사용한다.

```text
Tomato:
  ripe(T), unripe(T), rotten(T), at(T,S)

WasteSorting:
  plastic(W), can(W), paper(W), general(W)
```

## Feedback provider

| Provider | 상태 | 동작 |
|---|---|---|
| `oracle` | 구현 | `models.<domain>.answer.answer_question("oracle", ...)` 직접 호출 |
| `vlm` | placeholder | `NotImplementedError` |
| `human` | placeholder | `NotImplementedError` |

VLM과 Human을 Oracle로 대체하지 않는다.

Oracle의 truth rule은 이 폴더에 복사하지 않았다. runtime answer와 planner의
hypothetical particle answer가 모두 기존 domain `answer.py`를 호출한다.

## 구조

```text
attr_pomdp/
├── main.py
├── main_clean.py
├── domains/
│   ├── tomato/robot_skill.yaml
│   └── wastesorting/robot_skill.yaml
├── scripts/
│   ├── belief.py
│   ├── initial_belief.py
│   ├── planner.py
│   ├── providers.py
│   ├── query.py
│   └── rewards.py
└── tests/
```

`paper/`와 `paper_review.md`는 기존 위치 구조를 유지한 채 함께 이동했다.

## 실행

저장 로그를 포함한 실행:

```bash
python3 scripts/baseline/attr_pomdp/main.py \
  --domain wastesorting \
  --scene 01 \
  --feedback-source oracle \
  --query-cost 1.0 \
  --max-depth 2 \
  --max-step 25
```

Tomato:

```bash
python3 scripts/baseline/attr_pomdp/main.py \
  --domain tomato \
  --scene 01 \
  --feedback-source oracle \
  --query-cost 1.0
```

E3 전체 실행:

```bash
METHODS="ours attr-pomdp knowno" ./run/e3_baselines.sh
```

## Reward

Task action에는 프로젝트의 공통 domain reward를 그대로 사용한다. 따라서 baseline만
별도의 terminal reward를 사용하지 않으며, 다른 방법과 동일한 task objective에서
비교한다. Query action만 설정된 `query_cost`만큼의 음수 reward를 받는다.

```text
Q(b, query)
  = -query_cost
  + gamma * sum_answer P(answer | b, query) V(b_query,answer)
```

별도 entropy bonus는 기본 planner reward에 넣지 않는다. 질문이 이후 task 결정을
개선할 때만 가치가 생기도록 하기 위해서다.

## 현재 제한 및 placeholder

- 원 논문의 MAttNet/UOIS-Net, color/location matrix, pointing gesture는 구현하지 않음
- VLM provider 미구현
- Human provider 미구현
- Oracle은 기존 domain 구현의 의미론을 그대로 사용
- 현재 질문 vocabulary는 domain별 정적 attribute predicate로 제한
- 기본 planning depth는 계산량 때문에 2이며, horizon 밖의 장기 질문 가치는
  평가되지 않음
- 초기 particle은 domain의 attribute 조합을 균등분포로 둠

향후 VLM/Human 구현 시 provider interface에 availability, answer accuracy,
response cost를 추가해야 한다.
