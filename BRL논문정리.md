# BRL 논문 방향성 정리

## 1. 현재 상황

- 기존 유저 스터디는 N=12로 진행했다.
- 각 피험자는 도메인별로 5가지 조건을 경험했고, 총 10개 조건을 수행했다.
- 한 조건은 약 3~4분짜리 로봇 동영상을 보면서, 로봇이 맞게 행동했는지 확인하는 방식이었다.
- 동영상이 끝난 뒤 설문 조사를 진행했다.

문제는 다음과 같다.

- 조건별 피로도나 인지된 작업부하 차이가 유의미하지 않았다.
- Our1, Our2 조건 차이가 오히려 해석을 어렵게 만들었다.
- 시스템의 원래 목적은 사용자가 계속 로봇을 감시하는 것이 아니라, 필요할 때만 시스템이 사용자에게 묻는 것이다.
- 따라서 "동영상을 보면서 계속 맞는지 확인하는 과제"는 시스템 취지와 잘 맞지 않는다.
- 일부 데이터는 오염 가능성이 있었고, N=6으로 다시 봐도 명확한 결과가 나오지 않았다.

현재의 핵심 문제는 단순히 N이 부족한 것이 아니라, 논문에서 주장하려는 내용과 실험 설계가 잘 맞지 않았다는 점이다.

## 2. 현재 고민

가능한 방향은 크게 두 가지였다.

1. 시스템 논문으로 먼저 끊고, 이후 더 큰 N의 유저 스터디 논문을 따로 쓰기
2. 유저 스터디를 N=30으로 다시 설계해서 원래 논문 안에 포함하기

하지만 "시스템 논문 1개 + 후속 N=30 논문" 전략은 위험하다.

이유는 다음과 같다.

- 시스템 논문에서 이미 시스템을 제안한 뒤, 후속으로 단순히 더 큰 유저 스터디만 하면 독립적인 연구 질문이 약해질 수 있다.
- 후속 논문이 "시스템을 더 많은 사람으로 검증했다" 수준이면 반쪽짜리 논문이 될 가능성이 있다.
- 반대로 시스템 논문도 유저 스터디를 빼면 "사람을 위해 만든 시스템인데 사람 평가가 약하다"는 비판을 받을 수 있다.

따라서 가장 중요한 판단은 다음이다.

- N=30을 한다면, 그것은 후속 논문이 아니라 현재 논문을 완성하기 위한 핵심 평가로 넣는 것이 자연스럽다.
- 시스템 논문으로 먼저 끊는다면, 후속 N=30은 단순 검증이 아니라 별도의 HRI 연구 질문으로 완전히 재설계되어야 한다.

## 3. 시스템의 핵심 정체성

우리 시스템은 단순히 "사람에게 언제 물어볼까"를 다루는 시스템이 아니다. POMDP는 부분관측 stochastic domain에서 belief를 유지하며 행동을 선택하는 표준적 정식화이고, POMCP는 큰 POMDP에서 online Monte-Carlo planning을 수행하는 대표적 방법이다 [@kaelbling1998planning; @silver2010monte; @ross2008online].

더 정확히는 다음과 같다.

로봇은 내부적으로 task 수행을 위한 상태, 지식, 전이 모델을 가지고 있다. 실행 중 실제 관측이 들어왔을 때, 이 관측이 로봇의 전이 모델이 예상한 분포와 맞지 않으면 로봇은 자신의 지식이나 상태 추정이 잘못되었을 가능성을 감지한다. 이때 사람에게 feedback을 요청한다.

즉, 사람에게 묻는 이유는 단순히 불확실성이 높아서가 아니라, "로봇이 예상한 세계"와 "실제로 관측된 세계" 사이에 불일치가 있기 때문이다.

정리하면:

- POMDP는 로봇의 task planning을 담당한다 [@kaelbling1998planning].
- 질문 자체를 POMDP action space에 꼭 넣을 필요는 없다.
- 질문은 POMDP의 belief, uncertainty, observation-transition mismatch를 감시하는 별도 feedback trigger layer에서 발생할 수 있다.
- 일정 threshold를 넘으면 사람에게 feedback을 요청한다.
- feedback은 belief나 knowledge를 업데이트하고, 이후 planning에 다시 반영된다.

전체 loop는 다음과 같다.

```text
POMDP planning
-> action execution
-> observation
-> belief / uncertainty update
-> mismatch or uncertainty threshold check
-> human feedback request
-> knowledge / belief update
-> replanning
```

## 4. KnowNo와의 비교

KnowNo와의 비교는 논문에서 빠지기 어렵다 [@ren2023robots].

중요한 차이는 다음과 같다.

- KnowNo는 주로 action-level uncertainty를 다룬다.
- 즉, "지금 어떤 action을 해야 하는가?"가 불확실할 때 사람에게 도움을 요청한다.
- 우리 시스템은 state-level 또는 knowledge-level uncertainty를 다룬다.
- 즉, "내가 가진 상태나 지식 중 무엇이 실제 세계와 맞지 않는가?"를 확인하기 위해 사람에게 묻는다.

이 차이는 중요하다.

KnowNo식 질문은 다음 action을 정하는 데 직접 도움을 준다. 하지만 그 답변은 주로 현재 시점의 action 선택 문제를 해결한다.

우리 시스템의 질문은 내부 belief나 knowledge를 수정한다. 따라서 한 번의 feedback이 이후 여러 planning step에 영향을 줄 수 있다.

논문에서 사용할 수 있는 핵심 대비는 다음과 같다.

```text
KnowNo: 다음 action의 불확실성을 해결한다.
Ours: planning에 쓰이는 상태/지식의 불일치를 해결한다.
```

또는:

```text
KnowNo는 "무엇을 할까?"를 묻는다.
우리 시스템은 "내가 무엇을 잘못 알고 있나?"를 묻는다.
```

## 5. 가능한 contribution

현재 contribution은 다음처럼 정리하는 것이 가장 자연스럽다.

### 5.1 POMDP planning과 feedback trigger의 결합

우리는 POMDP 기반 planning 과정에서 발생하는 belief uncertainty와 observation-transition mismatch를 이용해 사람에게 feedback을 요청하는 구조를 제안한다 [@kaelbling1998planning; @silver2010monte].

중요한 점은 질문을 POMDP action으로 넣는 것이 아니다. 질문을 action으로 넣으면 search space가 커지고, 실제 구현과도 맞지 않을 수 있다.

따라서 contribution은 다음에 가깝다.

- POMDP는 planning을 담당한다.
- feedback trigger는 planning 중 생성되는 belief와 관측 불일치를 감시한다.
- threshold를 넘으면 사람에게 묻는다.
- feedback은 다시 belief나 knowledge update로 들어가고, replanning에 반영된다.

### 5.2 Observation-transition mismatch 기반 질문

기존 방식처럼 fixed interval이나 단순 반복 질문을 하지 않는다. HRI에서는 반복적이거나 부적절한 도움 요청이 annoyance와 interruption cost를 만들 수 있으므로, 요청 타이밍 자체가 중요한 설계 변수다 [@banerjee2018effects; @dahiya2023impact; @bajones2016help].

우리 시스템은 실제 관측이 전이 모델의 예측과 맞지 않을 때 질문한다.

즉, feedback request는 다음 상황에서 발생한다.

- 로봇의 예측과 실제 관측이 다르다.
- 특정 knowledge나 state에 대한 uncertainty가 threshold를 넘는다.
- 이 불일치가 이후 planning에 영향을 줄 가능성이 있다.

### 5.3 State/knowledge-level feedback

우리 시스템은 다음 action을 바로 물어보는 것이 아니라, planning에 쓰이는 지식이나 상태를 확인한다. 이는 다음 action 후보 집합이 불확실할 때 사람에게 선택을 요청하는 KnowNo식 action-level ask-for-help와 구분된다 [@ren2023robots].

이 점에서 KnowNo류의 action-level ask-for-help와 구분된다.

### 5.4 Closed-loop replanning

사람의 feedback은 단순히 로그로 저장되는 것이 아니라, 시스템 내부의 belief나 knowledge를 바꾼다. 이 점은 human input을 단순 post-hoc 평가가 아니라 closed-loop decision process의 일부로 다루는 HRI/robot planning 관점과 연결된다 [@goodrich2008human; @chen2014human].

이 업데이트는 이후 plan에 다시 영향을 준다.

따라서 contribution은 "로봇 시스템에 통합했다"가 아니라, "feedback이 planning loop 안에서 실제로 닫힌 고리를 만든다"로 주장해야 한다.

### 5.5 사람 응답의 noise 모델

현재 시스템은 사람의 답변을 oracle처럼 볼 수 있다. 하지만 실제 사람은 항상 정확하지 않다. 따라서 실제 HRI 평가에서는 human factor, workload, attention, situation awareness를 함께 고려해야 한다 [@goodrich2008human; @chen2014human; @endsley2018automation].

따라서 사람의 답변은 별도의 feedback observation model로 다루는 것이 좋다.

가능한 조건:

- oracle feedback
- noisy feedback
- biased feedback
- 모른다고 답하는 경우
- 특정 질문 유형에서만 오류가 커지는 경우

이것은 유저 스터디 없이도 simulation이나 ablation에서 중요한 평가축이 될 수 있다.

## 6. 피해야 할 주장

다음 표현은 조심해야 한다.

```text
Human feedback is modeled as a costly sensing action in a POMDP.
```

질문을 실제로 POMDP action space에 넣지 않는다면, 이 표현은 과장이다.

더 정확한 표현은 다음과 같다.

```text
POMDP는 로봇 planning에 사용하고, feedback request는 POMDP의 belief uncertainty와 observation-transition mismatch를 감시하는 별도 layer에서 threshold 기반으로 발생한다.
```

또한 "로봇 시스템에 통합했다"는 말 자체는 contribution으로 약하다.

통합을 contribution으로 쓰려면 다음을 보여야 한다.

- 어떤 요소들이 어떻게 상호작용하는가
- 그 결합 때문에 기존 방식보다 무엇이 좋아지는가
- ablation이나 baseline 비교에서 그 효과가 드러나는가

## 7. 기존 유저 스터디의 문제

기존 유저 스터디는 다음 점에서 시스템 취지와 어긋났다.

- 사용자가 계속 동영상을 보며 로봇이 맞는지 감시했다.
- 이는 "필요할 때만 묻는 시스템"이라는 목표와 맞지 않는다.
- 실제 사용 시나리오에서는 사용자가 원래 하던 일이 있고, 로봇의 질문은 그 중간에 interruption으로 들어와야 한다.
- 따라서 단순히 동영상을 보는 방식은 workload 차이를 잘 드러내지 못했을 가능성이 크다.

또한 5개 조건은 너무 많았다.

- All
- No
- Ours1
- Ours2
- KnowNo

특히 Ours1과 Ours2의 차이는 사용자 입장에서 명확히 느끼기 어렵고, 해석을 복잡하게 만들었다.

## 8. HRI 연구원이 제안한 새 실험

제안은 다음과 같다.

1. 조건을 5개에서 3개로 줄인다.
   - All
   - Ours1
   - KnowNo
2. Main task를 수행하게 하면서, 보조 task로 태블릿 feedback을 하게 한다.
3. N=30으로 늘린다.

이 방향은 대체로 타당하다.

이유는 다음과 같다.

- 시스템의 핵심은 사용자의 주 작업을 방해하지 않으면서 필요한 feedback을 얻는 것이다.
- 따라서 main task 중간에 feedback 요청이 들어오는 구조가 더 현실적이다.
- 기존처럼 사용자가 계속 로봇을 감시하는 방식보다 시스템 목적에 잘 맞는다.
- 조건을 3개로 줄이면 피로도와 순서 효과가 줄고, 해석이 명확해진다.

하지만 새 실험은 단순 workload study가 아니라, interruption과 feedback efficiency를 보는 실험으로 정의해야 한다 [@banerjee2018effects; @dahiya2023impact].

## 9. 새 유저 스터디의 핵심 claim

기존 claim:

```text
우리 시스템은 workload를 줄인다.
```

이 표현은 너무 넓고 약하다.

새 claim은 다음처럼 바꾸는 것이 좋다.

```text
우리 시스템은 사용자의 주 작업을 덜 방해하면서 필요한 feedback을 얻는다.
```

또는:

```text
우리 시스템은 불필요한 질문을 줄이면서, planning에 필요한 state/knowledge-level feedback을 얻는다.
```

따라서 측정해야 할 것은 단순 설문만이 아니다.

가능한 측정값:

- 주 작업 성능
- 주 작업 수행 시간
- feedback 정확도
- feedback 응답 시간
- 질문 개수
- 질문 후 주 작업으로 복귀하는 데 걸린 시간
- perceived workload
- interruption burden
- 질문 타이밍이 적절하다고 느꼈는지
- trust 또는 perceived usefulness

NASA-TLX는 주관적 workload 측정에 널리 쓰이고, SAGAT/SART는 situation awareness를 각각 probe 기반/주관적 rating 기반으로 평가하는 대표 도구다 [@hart1988development; @endsley1988design; @taylor1990situational].

## 10. 조건 선택에 대한 판단

3개 조건으로 줄이는 것은 좋다.

가장 자연스러운 조합은 다음이다.

```text
All vs Ours vs KnowNo
```

이 조합은 "feedback이 필요한가?"보다 "어떤 방식으로 물어보는 것이 좋은가?"에 집중한다. 특히 KnowNo는 action-level clarification baseline으로 두고, Ours는 state/knowledge-level correction baseline으로 두면 비교축이 명확해진다 [@ren2023robots].

각 조건의 의미:

- All: 많이 묻는 baseline
- Ours: uncertainty/mismatch 기반 state-level feedback
- KnowNo: action-level ask-for-help baseline

No 조건을 빼는 것은 아쉽지만, 현재 논문의 핵심이 feedback 유무가 아니라 feedback 방식이라면 받아들일 수 있다.

Ours1과 Ours2는 유저 스터디에서 동시에 비교하지 않는 것이 좋다.

- 사용자 입장에서 차이가 모호하다.
- 해석이 복잡해진다.
- Ours1/Ours2 차이는 simulation이나 system ablation에서 다루는 것이 더 적절하다.

## 11. 가능한 hypothesis

새 실험의 가설은 다음처럼 잡는 것이 좋다.

```text
H1. Ours는 All보다 사용자의 주 작업 방해와 인지된 작업부하를 줄인다.

H2. Ours는 All보다 적은 질문으로 비슷한 수준의 feedback 품질을 유지한다.

H3. Ours는 KnowNo보다 이후 planning에 재사용 가능한 state/knowledge-level feedback을 더 효율적으로 얻는다.

H4. 사용자는 Ours의 질문 타이밍을 All이나 KnowNo보다 더 적절하다고 인식한다.
```

이렇게 해야 N=30 실험이 단순히 "피험자를 늘린 실험"이 아니라, 시스템의 핵심 주장과 직접 연결된다.

## 12. 현재 가장 현실적인 논문 구조

현재 가장 자연스러운 구조는 논문 하나로 통합하는 것이다.

즉:

```text
시스템 제안
+ simulation / ablation
+ N=30 user study
```

이 구조가 가장 탄탄하다.

시스템 파트에서는 다음을 보인다.

- POMDP planning
- uncertainty / mismatch 기반 feedback trigger
- state/knowledge-level feedback
- KnowNo식 action-level 질문과의 차이
- oracle/noisy feedback 조건에서의 robustness

유저 스터디에서는 다음을 보인다.

- 사용자가 주 작업을 수행하는 중에 feedback 요청이 들어올 때의 burden
- All, Ours, KnowNo 비교
- Ours가 질문 수를 줄이면서도 필요한 feedback을 얻는지
- Ours가 주 작업 방해를 줄이는지

## 13. 교수님께 말할 수 있는 요지

교수님께는 "N=30을 하기 싫다"가 아니라, 다음처럼 말하는 것이 좋다.

```text
기존 N=12 실험은 단순히 표본 수가 부족한 문제가 아니라, 실험 과제가 시스템 취지와 맞지 않는 문제가 있었습니다.

우리 시스템은 사용자가 로봇을 계속 감시하게 하는 것이 아니라, 사용자의 주 작업 중 필요한 순간에만 feedback을 요청하는 구조입니다.

따라서 N=30을 한다면 기존 설계를 확장하는 것이 아니라, main task 중간에 system-initiated query가 들어오는 방식으로 실험을 재설계해야 합니다.

또한 조건은 5개에서 3개로 줄이고, Ours1/Ours2 차이는 유저 스터디가 아니라 simulation/ablation에서 다루는 것이 더 적절합니다.
```

그리고 논문 방향은 다음처럼 정리할 수 있다.

```text
이 논문은 POMDP 기반 로봇 planning 과정에서 발생하는 belief uncertainty와 observation-transition mismatch를 이용해, 사람에게 필요한 순간에만 state/knowledge-level feedback을 요청하는 closed-loop 시스템을 제안한다.

평가는 simulation/ablation으로 시스템 특성을 보이고, N=30 유저 스터디로 주 작업 중 feedback 요청이 사용자에게 주는 방해와 효율성을 검증한다.
```

## 14. 최종 판단

현재 상태에서 "시스템 논문 하나 + 후속 N=30 논문"으로 나누는 것은 위험하다.

이유:

- 시스템 논문은 human evidence가 약하다는 비판을 받을 수 있다.
- 후속 N=30 논문은 단순 validation처럼 보일 수 있다.
- 둘 다 반쪽짜리가 될 가능성이 있다.

더 나은 선택지는 다음 중 하나다.

1. 하나의 논문으로 통합한다.
   - 시스템
   - simulation / ablation
   - N=30 user study

2. 시스템 논문으로 끊되, 후속 N=30은 기대하지 않는다.
   - 이 경우 유저 스터디는 별도의 HRI 연구 질문으로 완전히 다시 잡아야 한다.

현재 논문을 가장 탄탄하게 만드는 방향은 1번이다.

다만 N=30을 한다면 반드시 기존 실험을 그대로 확장하면 안 된다. 실험의 핵심을 "동영상 감시"에서 "주 작업 중 system-initiated feedback의 방해와 효율"로 바꾸어야 한다.

## 15. 관련 선행연구: main task 중 interruption / sub task를 넣는 방식

새 실험 설계의 핵심은 다음이다.

```text
사용자는 main task를 수행한다.
로봇 또는 시스템은 중간에 feedback / help / correction 요청을 보낸다.
사용자는 잠시 sub task를 처리한 뒤 main task로 돌아온다.
조건별로 main task 성능, sub task 처리, 주관적 workload를 비교한다.
```

이 방식은 HRI와 human factors 쪽에서 이미 사용된 구조다 [@banerjee2018effects; @dahiya2023impact; @bajones2016help].

### 15.1 Banerjee et al. 2018: interruptibility-aware robot behavior [@banerjee2018effects]

논문:

- Siddhartha Banerjee, Andrew Silva, Karen Feigh, Sonia Chernova.
- "Effects of Interruptibility-Aware Robot Behavior"
- https://arxiv.org/abs/1804.06383

핵심:

- 로봇이 사람에게 도움을 요청해야 할 때, 아무 때나 interrupt하지 않고 사람이 interruptible한 순간을 예측해서 접근한다.
- 실험은 mock manufacturing assembly 환경에서 수행했다.
- 참가자는 조립 작업을 수행하고, 로봇은 자기 작업을 위해 참가자에게 도움을 요청한다.

task 구조:

- Main task: 참가자가 나무 블록 구조물을 조립한다.
- Robot interruption / sub task: 로봇이 자기 조립 과제를 들고 와서 참가자에게 도움을 요청한다.
- 참가자는 로봇의 요청을 받아들이거나 무시할 수 있다.
- 로봇 요청은 random, wizard, model-based 조건으로 나뉜다.

측정 metric:

- human task performance
  - 참가자가 idle 상태로 보낸 시간
  - 완료한 task 수
- robot task performance
  - interrupt된 build 비율
  - participant가 build 중일 때와 idle일 때 로봇이 기다린 시간
  - interruption 수
  - ignored interruption 수
  - interruption lag: 로봇이 도움을 요청한 시점부터 사람이 로봇 작업을 시작할 때까지의 시간
  - interruption duration: 로봇이 요청 뒤 기다린 전체 시간
- subjective measure
  - 질문 타이밍의 적절성
  - 로봇이 workload를 고려한다고 느꼈는지
  - social aptitude / considerateness

중요한 결과:

- interruptibility-aware robot은 사람의 main task 성능을 크게 개선하지는 않았다.
- 하지만 로봇 task performance와 사회적 인식은 좋아졌다.
- 즉, 주 작업 성능 차이가 안 나와도 "더 적절한 타이밍", "무시되는 요청 감소", "로봇 효율 증가", "로봇이 더 배려 깊게 보임"으로 논문이 성립할 수 있다.

우리에게 주는 의미:

- main task 성능만 보면 효과가 안 나올 수 있다.
- 따라서 main task success만이 아니라 query 처리 효율과 subjective appropriateness를 반드시 같이 봐야 한다.
- "질문 타이밍이 적절했는가", "시스템이 내 작업 부담을 고려한다고 느꼈는가"는 우리 실험에도 직접 넣을 만하다.

### 15.2 Dahiya et al. 2023: multi-robot supervision에서 intrinsic / extrinsic interruption 비교 [@dahiya2023impact]

논문:

- Abhinav Dahiya, Yifan Cai, Oliver Schneider, Stephen L. Smith.
- "On the Impact of Interruptions During Multi-Robot Supervision Tasks"
- https://arxiv.org/abs/2306.16501

핵심:

- 사람은 여러 로봇을 감시하는 main task를 수행한다.
- 중간에 secondary task가 interruption으로 들어온다.
- interruption은 main task와 관련 있는 intrinsic interruption과 관련 없는 extrinsic interruption으로 나뉜다.

task 구조:

- Main task: 여러 로봇의 비디오 피드를 보면서 fault를 감지하고 report한다.
- Intrinsic sub task: 특정 로봇 fault correction task를 수행한다. main task와 관련이 있다.
- Extrinsic sub task: 메시지를 다시 타이핑해서 보내는 task를 수행한다. main task와 무관하다.
- No interruption 조건도 포함했다.

측정 metric:

- main task performance
  - fault reported percentage: 발생한 fault 중 사용자가 report한 비율
  - response time: fault 발생 후 report까지 걸린 평균 시간
- subjective workload
  - NASA-TLX
  - 조건 간 NASA-TLX 차이를 개인별 변화량으로 비교
- post-questionnaire
  - interruption이 disruptive했는지
  - monitoring task에서 sub task로 전환하기 어려웠는지
  - sub task 이후 monitoring task로 돌아오기 어려웠는지

중요한 결과:

- interruption 종류는 main task performance에 유의미한 영향을 주지 않았다.
- 하지만 interruption은 perceived workload를 증가시켰다.
- extrinsic interruption은 intrinsic interruption보다 workload와 task switching difficulty를 더 크게 만들었다.

우리에게 주는 의미:

- 우리 feedback 요청은 extrinsic이 아니라 intrinsic에 가깝다. 로봇 task와 직접 관련된 correction / feedback이기 때문이다.
- main task success가 조건별로 크게 안 달라도 workload, switching difficulty, resumption difficulty에서는 차이가 날 수 있다.
- 새 실험에서는 "질문이 방해됐는가"뿐 아니라 "질문에서 main task로 돌아오기 어려웠는가"를 물어야 한다.

### 15.3 KnowNo: action-level ask-for-help baseline [@ren2023robots]

논문:

- Allen Z. Ren et al.
- "Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners"
- https://arxiv.org/abs/2307.01928

핵심:

- LLM planner가 다음 action 후보들 사이에서 확신이 부족하면 사람에게 도움을 요청한다.
- conformal prediction으로 prediction set을 만들고, set이 하나가 아니면 사람에게 물어본다.
- 목표는 task success를 보장하면서 human help를 최소화하는 것이다.

task 구조:

- Main robot task: language instruction을 따라 manipulation task를 수행한다.
- Human help: 다음 action 후보가 여러 개일 때 사람이 후보 중 하나를 선택한다.
- 사람에게 묻는 질문은 state/knowledge 수정이라기보다 action selection에 가깝다.

측정 metric:

- task completion rate
- help request 수 또는 prediction set size
- desired success level과 실제 success rate 차이
- baseline 대비 help 감소율

우리에게 주는 의미:

- KnowNo는 유저 스터디의 main-sub task 설계 근거라기보다, algorithmic baseline에 가깝다.
- 비교 포인트는 action-level question vs state/knowledge-level feedback이다.
- 유저 스터디에서는 KnowNo 조건을 "다음 action을 고르게 하는 질문"으로 구현하고, 우리 조건은 "로봇의 state/knowledge mismatch를 수정하는 질문"으로 구현하는 것이 좋다.

### 15.4 Bajones et al. 2016: 로봇 malfunction 후 사용자 도움 요청 [@bajones2016help]

논문:

- Markus Bajones, Astrid Weiss, Markus Vincze.
- "Help, Anyone? A User Study For Modeling Robotic Behavior To Mitigate Malfunctions With The Help Of The User"
- https://arxiv.org/abs/1606.02547

핵심:

- 서비스 로봇이 navigation malfunction을 겪을 때 사용자에게 도움을 요청한다.
- 사용자가 반복적으로 로봇을 도와주는지, 누가 도와주는지, 로봇 인식이 어떻게 바뀌는지 본다.

task 구조:

- Main task: 두 명의 사용자가 Lego 모델을 만든다.
- Robot role: Lego 블록이나 instruction을 운반한다.
- Help request: 로봇이 길을 잃거나, 목표 위치에 도달하지 못하거나, 장애물에 걸렸을 때 사용자에게 도와달라고 요청한다.

측정 metric:

- behavioral data
  - 사용자가 도와준 횟수
  - 누가 도와줬는지
  - 로봇이 도움을 요청한 뒤 도움을 받을 때까지 걸린 시간
  - 로봇과 사용자 사이 거리
- self-report
  - perceived intelligence
  - likability
  - task contribution
  - open-ended responses

중요한 결과:

- 사용자는 malfunction 상황에서도 로봇을 도와줬다.
- 사용자가 직접 해결 가능한 recovery strategy를 제공하면 로봇에 대한 부정적 인식이 크게 악화되지 않을 수 있다.
- 하지만 반복적인 도움 요청은 점점 귀찮게 느껴질 수 있다.

우리에게 주는 의미:

- "로봇이 실패하거나 불확실할 때 사람에게 묻는 것" 자체는 HRI에서 자연스러운 시나리오다.
- 다만 반복 질문은 annoyance를 만들 수 있으므로, query count와 perceived annoyance를 측정해야 한다.

## 16. 선행연구에서 보이는 공통 실험 패턴

관련 연구를 보면 성공률을 한 가지만 보지 않는다.

대체로 다음 세 층위를 나눠 본다.

### 16.1 Main task 성능

사용자가 원래 해야 하는 작업이 얼마나 유지되는지 본다.

예시 metric:

- task completion rate
- completed tasks 수
- task completion time
- fault detection rate
- response time
- error rate
- idle time

우리 실험에 대응시키면:

- main task 정답률
- main task 완료 시간
- main task 중단 횟수
- feedback 요청 후 main task 복귀 시간
- main task error 증가량

### 16.2 Sub task / feedback 처리 성능

시스템이 요청한 도움이나 feedback이 얼마나 잘 처리됐는지 본다.

예시 metric:

- help request accepted rate
- ignored request 수
- response latency
- feedback accuracy
- correction success
- robot task completion rate
- query당 얻은 정보량 또는 correction 수

우리 실험에 대응시키면:

- feedback 응답률
- feedback 정답률
- feedback 응답 시간
- query count
- query당 uncertainty 감소량
- query당 plan correction 효과
- 잘못된 feedback 비율

### 16.3 주관적 부담과 interruption 경험

성능 차이가 안 나와도 workload와 perceived interruption은 차이가 날 수 있다.
NASA-TLX는 workload 측정 도구로, task switching difficulty와 resumption difficulty는 interruption 연구에서 조건 차이를 설명하는 보조 지표로 쓸 수 있다 [@hart1988development; @dahiya2023impact].

예시 metric:

- NASA-TLX
- perceived timing appropriateness
- perceived disruptiveness
- task switching difficulty
- resumption difficulty
- annoyance
- trust
- perceived usefulness
- robot considerateness

우리 실험에 대응시키면:

- NASA-TLX
- "질문 타이밍이 적절했다"
- "질문이 내 작업을 방해했다"
- "질문 후 원래 작업으로 돌아오기 어려웠다"
- "시스템이 내 부담을 고려한다고 느꼈다"
- "질문이 필요한 내용이라고 느꼈다"

## 17. 우리 N=30 실험에 대한 구체적 설계 제안

선행연구를 보면, 새 실험은 다음처럼 잡는 것이 가장 타당하다.

### 17.1 실험 조건

```text
All vs Ours vs KnowNo
```

- All: 모든 중요한 시점에 사용자에게 묻는다.
- Ours: uncertainty / mismatch threshold를 넘을 때 state/knowledge-level feedback을 요청한다.
- KnowNo: 다음 action이 불확실할 때 action-level help를 요청한다.

Ours1/Ours2는 유저 스터디에서 빼는 것이 좋다.

- 사용자가 체감하기 어렵다.
- 해석이 복잡해진다.
- 내부 알고리즘 차이는 simulation / ablation에서 다루는 것이 맞다.

### 17.2 Main task

main task는 사용자가 지속적으로 수행해야 하고, 중간에 feedback 요청으로 방해받을 수 있어야 한다.

좋은 main task 조건:

- 일정 시간 계속 집중해야 한다.
- 정답률이나 수행 시간을 측정할 수 있다.
- 너무 어렵지 않아야 한다.
- feedback 요청 때문에 interruption cost가 생길 여지가 있어야 한다.

가능한 형태:

- 화면에서 다른 작업을 수행하면서 로봇 상태를 가끔 확인한다.
- 간단한 분류/탐색/기억 과제를 수행하면서 로봇 질문에 답한다.
- 로봇 task와 약하게 연결된 monitoring task를 수행한다.

중요한 점:

- 기존처럼 "로봇 동영상을 계속 감시하는 것"이 main task가 되면 안 된다.
- 사용자는 원래 자기 일을 하고 있고, 로봇 질문은 그 일에 끼어드는 형태여야 한다.

### 17.3 Sub task

sub task는 로봇의 feedback request이다.

조건별로 sub task의 성격이 달라진다.

- All: 자주 묻는다.
- Ours: 필요할 때만 state/knowledge-level로 묻는다.
- KnowNo: 다음 action 선택을 묻는다.

sub task에서 반드시 기록할 것:

- 질문 발생 시점
- 질문 종류
- 응답 여부
- 응답 시간
- 응답 정답 여부
- 질문 후 main task 복귀 시간
- 해당 feedback이 실제 plan 변경으로 이어졌는지

### 17.4 핵심 비교 metric

최소한 다음 metric은 필요하다.

```text
Main task:
- accuracy
- completion time
- error rate
- resumption time

Feedback task:
- query count
- response rate
- response time
- feedback accuracy
- ignored query count
- plan correction success

Subjective:
- NASA-TLX
- timing appropriateness
- disruptiveness
- resumption difficulty
- annoyance
- trust / usefulness
```

### 17.5 성공률 비교 방식

성공률은 하나로 합치지 말고 분리해서 봐야 한다.

```text
Main task success:
사용자가 원래 작업을 얼마나 잘 수행했는가?

Feedback success:
사용자가 로봇 질문에 얼마나 정확히 답했는가?

Robot task success:
그 feedback 덕분에 로봇 planning이 얼마나 잘 되었는가?

Interaction success:
적은 질문으로 필요한 feedback을 얻었는가?
```

가능한 aggregate metric:

```text
feedback efficiency = robot task success / query count

interruption cost = main task performance drop + resumption time + subjective disruptiveness

query usefulness = plan correction으로 이어진 query 수 / 전체 query 수
```

## 18. 지금 설계에 대한 판단

HRI 연구원이 제안한 방식은 관련 연구와 잘 맞는다.

특히 다음 점에서 타당하다.

- main task를 두고 feedback을 sub task로 넣는 구조는 선행연구에서 이미 쓰인다.
- 성능 차이가 안 나와도 workload, timing appropriateness, switching difficulty에서 효과를 볼 수 있다.
- 조건을 3개로 줄이는 것은 해석 가능성을 높인다.
- All vs Ours vs KnowNo는 비교축이 명확하다.

하지만 주의할 점도 있다.

- main task 성능만 보면 또 차이가 안 날 수 있다.
- 따라서 feedback efficiency와 interruption experience를 반드시 같이 측정해야 한다.
- KnowNo는 유저 스터디에서 "action-level question"으로 명확히 구현되어야 한다.
- Ours는 "state/knowledge-level feedback"으로 명확히 구현되어야 한다.
- 질문 수가 줄어드는 것만으로는 부족하고, 줄어든 질문으로도 필요한 correction이 유지되는지를 보여야 한다.

따라서 새 실험의 핵심 문장은 다음처럼 잡는 것이 좋다.

```text
우리는 system-initiated feedback이 사용자의 주 작업을 얼마나 방해하는지, 그리고 적은 질문으로 로봇 planning에 필요한 correction을 얼마나 효율적으로 얻는지를 평가한다.
```

이렇게 잡으면 N=30 유저 스터디는 기존 실험의 단순 반복이 아니라, 시스템의 핵심 목적을 직접 평가하는 실험이 된다.

## 19. 선행연구에서 실제로 사용한 main task

관련 연구들을 보면 main task는 대체로 다음 네 가지 유형으로 나뉜다.

1. 조립 / 제작 과제
2. 로봇 monitoring 과제
3. 협업 과제
4. language instruction 기반 robot task

우리 실험에 가장 직접적으로 쓸 수 있는 것은 1번과 2번이다.

### 19.1 조립 / 제작 과제

대표 연구:

- Banerjee et al. 2018, "Effects of Interruptibility-Aware Robot Behavior" [@banerjee2018effects]
- Bajones et al. 2016, "Help, Anyone?" [@bajones2016help]

실제 main task:

- 참가자가 블록이나 Lego를 이용해 정해진 구조물을 만든다.
- 화면이나 태블릿에 조립 지시가 주어진다.
- 참가자는 제한 시간 안에 구조물을 완성해야 한다.
- 실험 중 로봇이 중간에 다가와 도움을 요청한다.

Banerjee et al. 2018의 경우:

- 참가자는 mock manufacturing 환경에서 나무 블록 구조물을 조립했다.
- 각 build session에는 제한 시간이 있었다.
- tablet에 build instruction이 표시됐다.
- 중간중간 로봇이 자기 build를 위해 참가자에게 도움을 요청했다.
- 로봇 요청은 random, wizard, model-based 조건에 따라 다른 타이밍에 발생했다.

측정한 main task metric:

- 참가자가 완료한 task 수
- 참가자가 idle로 보낸 시간
- build 중 interrupt됐는지
- build 중일 때 로봇이 얼마나 기다렸는지

특징:

- embodied task라서 사용자가 실제로 손을 움직인다.
- interruption이 들어와도 참가자가 자기 main task 상태를 눈으로 계속 볼 수 있다.
- 그래서 main task 성능 차이는 크게 안 날 수 있다.
- 대신 robot task performance, ignored interruption, timing appropriateness, social perception에서 차이가 났다.

우리에게 주는 의미:

- 노인 대상 assistive robot 시나리오와 잘 맞는다.
- 사용자가 "원래 하던 일"을 하고 있고, 로봇 질문은 중간에 들어오는 구조를 만들기 쉽다.
- 하지만 조립 task는 준비물이 필요하고, 실험 통제가 조금 복잡하다.

우리 실험으로 바꾼다면:

```text
사용자는 간단한 조립/분류/정리 과제를 수행한다.
로봇은 별도의 task를 수행하다가 불확실성이 생기면 tablet으로 질문한다.
사용자는 하던 일을 멈추고 답한 뒤 다시 main task로 돌아간다.
```

가능한 main task 예시:

- 블록을 색/모양 규칙에 따라 정렬하기
- 카드나 물체를 카테고리별로 분류하기
- 간단한 조립 설명을 보고 구조물 만들기
- 물건 목록을 보고 바구니에 맞게 담기
- 퍼즐 또는 순서 맞추기

장점:

- 실제 assistive robot 맥락과 자연스럽다.
- interruption cost를 측정하기 쉽다.
- main task completion time과 error를 측정할 수 있다.

단점:

- 사람마다 손재주 차이가 생길 수 있다.
- 물리 실험 세팅이 필요하다.
- 노인 대상이면 난이도 조절이 중요하다.

### 19.2 로봇 monitoring 과제

대표 연구:

- Dahiya et al. 2023, "On the Impact of Interruptions During Multi-Robot Supervision Tasks" [@dahiya2023impact]

실제 main task:

- 참가자는 웹 기반 interface에서 여러 로봇의 비디오 feed를 본다.
- 각 로봇은 실내를 이동한다.
- 일부 로봇에는 fault가 발생한다.
- 참가자는 fault를 발견하면 해당 로봇을 선택하고 "Report Fault" 버튼을 누른다.

fault 종류:

- 로봇이 멈춤
- 제자리에서 원을 그림
- 앞으로 가지 않고 좌우로 흔들림

측정한 main task metric:

- fault reported percentage
  - 발생한 fault 중 참가자가 report한 비율
- response time
  - fault 발생부터 report까지 걸린 시간
- 조건별 NASA-TLX
- condition 간 NASA-TLX 변화량

sub task:

- intrinsic interruption: fault correction task
  - 로봇 fault와 관련된 질문에 답한다.
- extrinsic interruption: messaging task
  - 주어진 메시지를 다시 타이핑한다.
- no interruption 조건도 있었다.

특징:

- main task가 정량화하기 쉽다.
- 화면 기반이라 실험 구현이 쉽다.
- fault detection rate와 response time이 명확하다.
- 하지만 사용자가 결국 "감시자" 역할을 하므로, 우리 시스템의 원래 시나리오와 완전히 같지는 않다.

우리에게 주는 의미:

- 현재 기존 실험의 "동영상 보기"와 가장 가까운 형태다.
- 다만 기존 실험처럼 로봇 행동을 계속 평가하는 것이 아니라, 사용자가 별도 monitoring 업무를 하다가 가끔 feedback 요청을 받는 구조로 바꿔야 한다.

우리 실험으로 바꾼다면:

```text
사용자는 화면에서 여러 간단한 상태 변화를 감지하는 main task를 수행한다.
중간에 로봇이 tablet feedback 요청을 보낸다.
사용자는 요청에 답한 뒤 다시 monitoring task로 돌아간다.
```

가능한 main task 예시:

- 여러 비디오/이미지 feed 중 이상 상태 찾기
- 화면에 나타나는 target event 감지하기
- 특정 색/기호/상태 변화가 나타나면 버튼 누르기
- 간단한 robot monitoring dashboard에서 오류 상태 report하기

장점:

- 구현이 쉽다.
- 정답률과 반응 시간을 자동 기록하기 좋다.
- N=30 실험을 빠르게 돌리기 좋다.

단점:

- 사용자가 여전히 "감시자" 역할에 가까워질 수 있다.
- 노인 assistive robot의 자연스러운 사용 맥락과 거리가 생길 수 있다.
- main task가 너무 단순하면 조건 차이가 안 나올 수 있다.

### 19.3 협업형 Lego task

대표 연구:

- Bajones et al. 2016, "Help, Anyone?" [@bajones2016help]

실제 main task:

- 두 명의 참가자가 함께 Lego 모델을 만든다.
- Same Task 조건에서는 두 참가자가 둘 다 builder 역할을 한다.
- Different Task 조건에서는 한 명은 director, 한 명은 builder 역할을 한다.
- 로봇은 instruction이나 Lego block을 운반한다.
- 중간에 로봇이 navigation malfunction을 겪고, 사용자에게 도움을 요청한다.

로봇 malfunction 종류:

- localization uncertainty
- goal pose unreachable
- collision with human or obstacle

측정 metric:

- 누가 로봇을 도와줬는지
- 도움 요청부터 실제 도움까지 걸린 시간
- 로봇과 사용자 사이 거리
- 사용자가 반복적으로 도와주는지
- perceived intelligence
- likability
- task contribution

특징:

- main task와 robot help request가 같은 상황 안에 자연스럽게 묶인다.
- 사용자가 로봇을 도와야 전체 task flow가 이어진다.
- 반복적인 도움 요청이 annoyance를 만들 수 있음을 보여준다.

우리에게 주는 의미:

- "로봇이 불확실하거나 실패했을 때 사람에게 도움을 요청한다"는 framing을 뒷받침한다.
- 다만 dyad 실험은 복잡하므로, 현재 논문에는 과할 수 있다.

우리 실험으로 바꾼다면:

```text
한 명의 사용자가 간단한 household task를 수행한다.
로봇은 보조적으로 움직이거나 정보를 제공한다.
로봇이 불확실한 상태가 되면 사용자에게 확인 질문을 한다.
```

가능한 main task 예시:

- 물건 정리 task
- 약 복용/식사 준비 순서 확인 task
- 카테고리별 물건 배치 task
- 로봇이 가져온 물건이 맞는지 확인하는 task

장점:

- assistive robot 맥락과 잘 맞는다.
- 사용자가 로봇에게 feedback을 주는 이유가 자연스럽다.

단점:

- 실험 시간이 길어질 수 있다.
- task script와 WoZ 통제가 필요하다.
- 사람마다 협업 방식 차이가 커질 수 있다.

### 19.4 Language instruction 기반 robot manipulation task

대표 연구:

- KnowNo, "Robots That Ask For Help" [@ren2023robots]

실제 main task:

- 로봇이 자연어 instruction을 따라 table-top rearrangement 또는 mobile manipulation task를 수행한다.
- 예를 들어 특정 bowl, block, object를 옮긴다.
- instruction에 ambiguity가 있으면 LLM planner가 여러 action 후보를 만든다.
- 후보 set이 하나로 좁혀지지 않으면 사람에게 도움을 요청한다.

실제 질문 형태:

- "어떤 bowl을 microwave에 넣어야 하는가?"
- "어느 위치에 물체를 둬야 하는가?"
- "다음 action 후보 중 무엇이 맞는가?"

측정 metric:

- task completion rate
- human help request 수
- prediction set size
- 원하는 success level과 실제 success rate 차이
- baseline 대비 help 감소율

특징:

- 사용자의 main task라기보다 robot의 main task가 중심이다.
- 사람은 robot planner가 애매할 때 action-level clarification을 제공한다.
- human workload나 interruption cost를 본 연구라기보다는 algorithmic uncertainty alignment 연구다.

우리에게 주는 의미:

- KnowNo는 main task 설계의 직접 참고보다는 baseline 구현 참고에 가깝다.
- 유저 스터디에서 KnowNo 조건을 넣는다면, 사람에게 "다음 action 후보 중 하나를 고르는 질문"으로 구현하는 것이 적절하다.

우리 실험으로 바꾼다면:

```text
KnowNo 조건:
로봇이 다음 action 후보를 제시하고, 사용자가 맞는 action을 고른다.

Ours 조건:
로봇이 내부 state/knowledge에 대한 질문을 하고, 사용자가 그 지식을 수정한다.
```

비교 포인트:

- action-level 질문은 당장 다음 행동을 정하는 데 좋다.
- state/knowledge-level 질문은 이후 planning에도 재사용될 수 있다.
- 따라서 query당 장기 효과를 비교해야 한다.

## 20. 우리 실험에 가장 적합한 main task 후보

선행연구를 기준으로 보면, 우리에게 가능한 main task는 크게 세 가지다.

### 20.1 후보 A: 간단한 조립/정리 task

형태:

```text
사용자가 물체를 정리하거나 조립한다.
중간에 로봇이 tablet으로 feedback 요청을 보낸다.
사용자는 답변 후 원래 task로 돌아간다.
```

측정:

- main task 완료 시간
- main task 오류 수
- interruption 후 복귀 시간
- feedback 응답 시간
- feedback 정확도
- NASA-TLX
- annoyance / timing appropriateness

장점:

- assistive robot 시나리오와 가장 자연스럽다.
- Banerjee et al. 2018과 Bajones et al. 2016의 설계와 잘 연결된다.

위험:

- 실험 구현과 통제가 복잡하다.
- 물리 task 난이도 차이가 클 수 있다.

### 20.2 후보 B: 화면 기반 monitoring / detection task

형태:

```text
사용자가 화면에서 계속 target event를 찾는다.
중간에 로봇 feedback 요청이 들어온다.
사용자는 답변 후 detection task로 돌아간다.
```

측정:

- target detection accuracy
- missed event 수
- false alarm 수
- reaction time
- feedback 응답 시간
- resumption time
- NASA-TLX

장점:

- 구현이 쉽다.
- 정량 metric이 깔끔하다.
- Dahiya et al. 2023과 직접 연결된다.

위험:

- 기존 비디오 감시 실험과 비슷해질 수 있다.
- assistive robot의 실제 사용 맥락이 약해질 수 있다.

### 20.3 후보 C: 생활 보조 시나리오 기반 tablet task

형태:

```text
사용자가 tablet에서 생활 보조 관련 main task를 수행한다.
예: 일정 확인, 물건 분류, 약 복용 순서 확인, 간단한 기억 과제.
중간에 로봇이 feedback 요청을 보낸다.
```

측정:

- tablet main task accuracy
- task completion time
- interruption 후 첫 클릭까지 시간
- feedback accuracy
- feedback response time
- subjective workload
- perceived usefulness
- timing appropriateness

장점:

- 노인 대상 assistive robot 맥락과 연결하기 쉽다.
- 물리 task보다 통제가 쉽다.
- 기존 tablet feedback 시스템을 재활용하기 쉽다.

위험:

- 너무 인위적인 task가 될 수 있다.
- "로봇"의 존재감이 약해질 수 있다.

## 21. 현재 판단: 어떤 main task가 제일 나은가

가장 논문적으로 안정적인 선택은 다음이다.

```text
생활 보조 시나리오 기반 tablet/main task
+ 로봇 feedback request sub task
```

이유:

- 기존 시스템이 tablet feedback을 포함하고 있다.
- 노인 assistive robot 시나리오와 연결된다.
- 물리 조립 task보다 실험 통제가 쉽다.
- monitoring task보다 "사용자가 자기 일을 하고 있다"는 구도가 자연스럽다.
- main task accuracy, time, resumption time을 자동 기록할 수 있다.

다만 HRI 쪽 설득력을 더 높이고 싶으면 간단한 물리 정리 task도 가능하다.

추천 우선순위:

1. 생활 보조 시나리오 기반 tablet task
2. 간단한 물체 분류/정리 task
3. 화면 기반 monitoring task

피해야 할 것:

- 사용자가 로봇 동영상을 계속 감시하는 task
- 정답률이 너무 쉬워 ceiling effect가 나는 task
- 조건별 interruption이 들어와도 main task에 영향이 거의 없는 task
- sub task가 main task와 너무 무관해서 우리 feedback 요청이 extrinsic interruption처럼 보이는 task

## 22. 2026-06-30 발표자료 기준 앞으로 할 일

발표자료 기준으로 보면, 앞으로의 논문 방향은 단순히 "사용자 부담을 줄이는 HRI 시스템"이 아니라 **부분 관측 환경에서 action frontier의 불확실성을 이용해 query timing과 query content를 함께 결정하는 planning system**으로 잡는 것이 좋다.

핵심 framing은 다음이다.

```text
로봇은 partial observability 때문에 현재 상태를 완전히 알 수 없다.
작은 상태 오판은 잘못된 action 선택으로 이어지고, 누적되면 task failure가 된다.
사람은 로봇이 직접 알기 어려운 정보를 줄 수 있지만, 계속 묻는 것은 비용이 크다.
따라서 언제 물을지와 무엇을 물을지를 planning과 연결해서 결정해야 한다.
```

이 방향에서 논문은 다음 질문에 답해야 한다.

```text
When: 언제 사람에게 물어볼 것인가?
What: 무엇을 물어볼 것인가?
How: 어떤 형식의 질문이 planning에 더 효과적인가?
```

### 22.1 기존 유저 스터디는 파일럿으로 정리

기존 N=12 유저 스터디는 그대로 밀기 어렵다. 피험자와 담당자 사이의 상호작용으로 일부 데이터가 오염되었고, 오염 가능성이 있는 피험자를 제외하면 N=6 수준으로 줄어든다. 이 경우 모든 지표에서 통계적 유의성이 사라진다.

핵심 문제는 표본 수만이 아니다.

- 참가자가 3~4분짜리 영상을 보며 계속 로봇 행동을 확인했다.
- 실험 시간이 짧고 터치 기반 인터페이스라 fatigue 차이가 잘 나기 어렵다.
- 모든 조건에서 참가자가 계속 집중하고 있으므로, "필요할 때만 묻는 시스템"의 장점이 잘 드러나지 않는다.
- 조건이 많아 해석이 복잡하고, Ours 계열의 차이가 사용자에게 명확하게 전달되지 않는다.

따라서 기존 유저 스터디는 본 실험이라기보다 파일럿으로 정리하는 것이 맞다. N을 단순히 늘리는 것이 아니라, claim, method, task, condition을 다시 맞춰야 한다.

### 22.2 논문 contribution 재정리

발표자료에서 가장 중요한 contribution 구조는 When, What, How이다. 이를 논문 contribution으로 정리하면 다음과 같다.

#### When: query timing

우리는 부분 관측 환경에서 현재 action 선택과 직접 관련된 불확실성을 정량화하고, 로봇이 사람의 도움이 필요한 시점을 스스로 판단하는 메커니즘을 제안한다.

중요한 점은 전체 state space의 불확실성을 보는 것이 아니라, **현재 행동으로 도달 가능한 action frontier** 위에서 uncertainty를 평가한다는 것이다. 이렇게 해야 query timing이 planning과 직접 연결된다.

평가할 것:

- Threshold query vs All vs No vs Random
- query 횟수
- query 비율
- task success rate
- operation time 또는 task completion time

#### What: query content

우리는 로봇이 자신의 행동 선택에 필요한 상태 정보를 식별하고, 그 상태 정보를 질의 대상으로 선택하는 방법을 제안한다.

즉, "불확실하니까 묻는다"가 아니라, 현재 frontier에서 어떤 predicate가 action 선택을 가장 많이 흔드는지를 고르는 것이 핵심이다.

평가할 것:

- Ours vs active search / active sensing baseline
- query 횟수
- query 비율
- task success rate
- 탐색 시간 또는 sensing/search cost

추가 ablation:

- search를 한 개씩 수행하는 방식
- 여러 후보를 한 번에 탐색하는 방식
- heuristic search 방식
- entropy 기반 predicate selection 방식

#### How: state-level query format

우리는 불확실성을 action 단위가 아니라 state/predicate 단위로 질의한다. 사용자 피드백이 특정 상태 변수에 대응되기 때문에, 한 번의 답변이 다음 action 하나만이 아니라 이후 planning에도 재사용될 수 있다.

비교 대상은 KnowNo-style action-level query이다.

평가할 것:

- Ours vs KnowNo-Conformal Prediction
- task success rate
- query count
- query당 planning 개선 효과
- 사용자가 다음 action을 추론해야 하는 부담이 줄어드는지

핵심 대비는 다음이다.

```text
KnowNo: 다음 action 후보가 여러 개일 때, 사람에게 맞는 action을 고르게 한다.
Ours: action 선택을 흔드는 state/predicate uncertainty를 사람에게 확인한다.
```

### 22.3 Method 설명에서 강조할 점

Method에서는 시스템이 세계를 어떻게 바라보는지를 더 분명히 설명해야 한다.

현재 정리해야 할 loop는 다음이다.

```text
state / knowledge / transition model
-> action frontier generation
-> frontier belief / entropy 계산
-> query timing 결정
-> query content 선택
-> human feedback
-> belief / knowledge update
-> replanning
```

중요한 문장:

```text
우리는 현재 행동으로 도달 가능한 action frontier에서만 uncertainty를 평가하여,
planning과 직접 관련된 uncertainty만을 대상으로 query timing과 query content를 동시에 결정한다.
```

이 문장은 논문의 system/method identity에 가깝다. 기존처럼 단순히 "사용자에게 필요한 순간에만 묻는다"라고 쓰면 HRI 시스템처럼 보이지만, action frontier를 강조하면 planning contribution이 살아난다.

### 22.4 Related Work에서 잡아야 할 비교축

관련 연구는 크게 두 축으로 정리하면 된다.

#### Active sensing / active information gathering

이 계열은 불확실성이 높을 때 robot sensing action을 수행하거나, human-in-the-loop setting에서는 사람에게 질문한다.

하지만 차이는 다음이다.

- sensing/query action이 planning action space 안에 포함되면 search space가 커진다.
- 질문 시점이 reward 또는 sensing action의 기대 보상에 의해 결정되는 경우가 많다.
- 무엇을 물어볼지가 사전에 action으로 정의되는 경우가 많다.
- 우리 방법은 현재 action frontier posterior에서 query timing과 query content를 함께 도출한다.

#### LLM-based query / KnowNo

KnowNo류 방법은 다음 action prediction이 여러 개일 때 사용자에게 질문한다.

차이는 다음이다.

- 현재 상태에 대한 불확실성을 명시적으로 다루지 않는다.
- 질문이 action-level clarification에 가깝다.
- 우리 방법은 state-level uncertainty를 predicate 단위로 해소하고, 그 결과를 belief update와 replanning에 반영한다.

### 22.5 System evaluation 계획

시스템 실험은 contribution별로 대응되게 설계해야 한다.

#### 실험 1: query timing

목적:

```text
필요할 때만 묻는 방식이 All처럼 성공률을 유지하면서 query 수를 줄이는가?
```

조건:

- No query
- All query
- Random query
- Threshold query
- Ours

metric:

- task success rate
- query count
- query probability per step
- operation time
- task completion time

#### 실험 2: query content

목적:

```text
현재 action frontier에서 planning에 필요한 predicate를 고르는 것이
active search/sensing보다 효율적인가?
```

조건:

- active search
- active sensing
- random predicate query
- entropy-based predicate query
- Ours

metric:

- success rate
- query count
- search/sensing cost
- operation time
- 탐색 시간

#### 실험 3: query format

목적:

```text
state-level query가 action-level query보다 planning에 더 효과적인가?
```

조건:

- Ours
- KnowNo-Conformal Prediction
- 가능하면 action-level oracle query

metric:

- task success rate
- query count
- operation time
- response time
- failure case: 잘못된 상태 인지 때문에 잘못된 action을 고르는 경우

### 22.6 Oracle 실험을 먼저 해야 하는 이유

유저 스터디로 바로 가기 전에 oracle 실험을 먼저 해야 한다. 이유는 system contribution이 먼저 검증되어야 하기 때문이다.

먼저 확인할 것:

- 완벽한 oracle에서 Ours가 성공률과 수행 시간을 개선하는가?
- query count를 줄이면서 All과 유사한 성공률을 유지하는가?
- active search/sensing보다 적은 cost로 필요한 정보를 얻는가?
- KnowNo-style action query보다 state-level query가 failure를 더 잘 줄이는가?

다만 oracle은 완벽한 정답기 하나만 두면 부족하다. 이후 유저 스터디와 연결하려면 human-like oracle도 같이 준비해야 한다.

oracle 조건:

- perfect oracle
- noisy oracle
- biased oracle
- unknown-capable oracle
- delayed-response oracle

human-like oracle에서 반영할 것:

- 사람 응답 정확도
- response time
- "모름" 응답
- 질문 유형별 오류율
- action-level 질문과 state-level 질문의 난이도 차이

### 22.7 도메인 확장 계획

Tomato 하나만으로는 domain-specific system처럼 보일 수 있다. 발표자료 기준으로도 필요하면 domain을 확장해야 한다.

우선 검토할 도메인:

- tomato harvesting
- waste sorting
- Blocksworld
- pick-and-delivery
- assembly
- household manipulation
- RoboCasa 계열 household benchmark

도메인 선정 기준:

- partial observability가 명확해야 한다.
- hidden state나 hidden predicate가 있어야 한다.
- action frontier가 자연스럽게 생겨야 한다.
- state-level query와 action-level query를 모두 정의할 수 있어야 한다.
- active search/sensing baseline을 만들 수 있어야 한다.
- success rate와 task completion time을 자동 측정할 수 있어야 한다.
- 가능하면 기존 benchmark를 사용하고, 새 환경 구현은 최소화한다.

현실적인 순서:

1. tomato와 waste sorting을 먼저 정리한다.
2. Blocksworld 또는 pick-and-delivery처럼 symbolic 확장이 쉬운 도메인을 추가한다.
3. assembly 또는 household manipulation을 검토한다.
4. RoboCasa는 embodied benchmark로서 가능성은 크지만, 구현 비용을 보고 결정한다.

### 22.8 새 N>=30 유저 스터디 방향

사람에게 묻는 시스템이므로 유저 스터디는 필요하다. 다만 현재 논문에서 유저 스터디의 역할은 HRI factor를 깊게 분석하는 것이 아니라, system-initiated query가 실제 사용자에게 어떤 부담과 응답 특성을 만드는지 확인하는 것이다.

조건은 5개가 아니라 3개로 줄인다.

```text
All vs Ours vs KnowNo
```

새 실험 구조:

```text
사용자는 main task를 수행한다.
로봇 질문은 알람처럼 sub task로 들어온다.
사용자는 질문에 답한 뒤 main task로 돌아간다.
조건별로 main task performance와 feedback response를 비교한다.
```

보고 싶은 것:

- 사용자의 집중력을 덜 방해하면서 필요한 feedback을 얻는가?
- Ours가 All보다 적은 질문으로 성공률을 유지하는가?
- Ours가 KnowNo보다 사용자가 느끼는 작업부하를 줄이는가?
- state-level 질문이 action-level 질문보다 응답하기 쉬운가?

측정 metric:

- success rate
- response time
- task completion time
- NASA-RTLX
- fatigue
- situation awareness
- SAGAT
- SART

다만 fatigue는 기존 실험에서 차이가 잘 안 났으므로 핵심 metric으로 두기 어렵다. 더 중요한 것은 task performance, response time, workload, situation awareness이다.

### 22.9 Main task 후보

발표자료에서 나온 main task 후보는 다음이다.

- 문서 교정
- 틀린 그림 찾기
- 타이핑 과제
- 다른 로봇 관리 또는 관제
- N-back 기억력 테스트
- Rapid Serial Visual Presentation, RSVP attention test

선행연구 구조와 맞추면 main task와 sub task의 역할은 다음처럼 잡는다.

```text
Main task:
사용자가 지속적으로 수행해야 하는 기본 작업

Sub task:
로봇이 평가하고 싶은 query interaction
```

선택 기준:

- main task 성능과 response time을 자동 기록할 수 있어야 한다.
- interruption 이후 복귀가 측정 가능해야 한다.
- 너무 쉽거나 너무 짧아서 ceiling effect가 나면 안 된다.
- 로봇 질문이 완전히 무관한 extrinsic interruption처럼 보이면 안 된다.

현재 가장 현실적인 후보:

1. RSVP 또는 N-back
   - 집중력과 반응 시간 측정이 좋다.
   - 구현이 쉽고 자동 로그가 가능하다.
2. 문서 교정 또는 틀린 그림 찾기
   - main task로 이해하기 쉽다.
   - 정확도와 수행 시간을 측정하기 쉽다.
3. 다른 로봇 관리/관제
   - HRI 맥락은 좋지만 기존 동영상 감시 실험과 비슷해질 위험이 있다.

### 22.10 논문 claim 재정리

이전 claim이 다음에 가까웠다면,

```text
우리 시스템은 사용자의 workload를 줄인다.
```

이제는 다음처럼 바꾸는 것이 좋다.

```text
우리 시스템은 partial observability 환경에서
action frontier의 belief uncertainty를 이용해
query timing과 query content를 동시에 결정하고,
state-level feedback을 belief update와 replanning에 반영한다.
```

짧게 쓰면 다음이다.

```text
Action-frontier belief entropy를 이용한
state-level human query mechanism for closed-loop task planning.
```

논문에서 직접 주장할 수 있는 효과:

- 필요한 순간에만 query하여 query 수를 줄인다.
- All query 대비 성공률을 유지한다.
- active search/sensing 대비 정보 획득 cost를 줄인다.
- KnowNo-style action query 대비 state misrecognition에 의한 실패를 줄인다.
- human feedback이 belief update와 replanning에 즉시 반영되는 closed-loop pipeline을 보인다.

### 22.11 당장 실행 순서

앞으로 할 일은 다음 순서가 가장 자연스럽다.

1. 기존 tomato / waste sorting system evaluation 코드를 contribution별로 다시 분류한다.
   - When: Threshold, All, No, Random
   - What: Active search/sensing, predicate selection ablation
   - How: Ours vs KnowNo-Conformal Prediction
2. action frontier, belief entropy, query timing, query content가 한 loop로 설명되도록 Method section을 다시 정리한다.
3. oracle 실험을 먼저 구성한다.
   - perfect oracle
   - noisy oracle
   - unknown-capable oracle
   - delayed-response oracle
4. 공통 metric을 통일한다.
   - success rate
   - query count
   - query probability per step
   - operation time
   - task completion time
   - response time
5. active search/sensing baseline을 구현 또는 정리한다.
6. KnowNo-Conformal Prediction baseline을 action-level query baseline으로 명확히 둔다.
7. tomato와 waste sorting에서 먼저 결과를 만든다.
8. 필요하면 Blocksworld, pick-and-delivery, assembly 중 하나를 추가한다.
9. N>=30 유저 스터디는 All vs Ours vs KnowNo, main task + query sub task 구조로 재설계한다.
10. 유저 스터디는 HRI factor 중심이 아니라 response time, task performance, workload를 보조적으로 확인하는 역할로 둔다.

### 22.12 최종 정리

이번 발표자료 기준으로 가장 중요한 결론은 다음이다.

```text
논문의 중심은 "사람 부담을 줄이는 인터페이스"가 아니라,
부분 관측 planning에서 현재 action frontier의 불확실성을 이용해
언제 무엇을 물어볼지 결정하고,
state-level feedback을 closed-loop planning에 반영하는 방법이다.
```

따라서 앞으로의 작업은 유저 인터페이스를 키우는 것이 아니라, contribution별 실험을 명확히 만드는 데 집중해야 한다.

```text
When -> Threshold/All/No/Random 비교
What -> Active search/sensing 및 search ablation 비교
How -> KnowNo-Conformal Prediction과 state-level query 비교
User study -> All/Ours/KnowNo, main task + sub task, N>=30
```

## 참고문헌

[@kaelbling1998planning] Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1-2), 99-134.

[@ross2008online] Ross, S., Pineau, J., Paquet, S., & Chaib-Draa, B. (2008). Online planning algorithms for POMDPs. *Journal of Artificial Intelligence Research*, 32, 663-704.

[@silver2010monte] Silver, D., & Veness, J. (2010). Monte-Carlo planning in large POMDPs. *Advances in Neural Information Processing Systems*, 23.

[@goodrich2008human] Goodrich, M. A., & Schultz, A. C. (2008). Human-robot interaction: A survey. *Foundations and Trends in Human-Computer Interaction*, 1(3), 203-275.

[@chen2014human] Chen, J. Y. C., & Barnes, M. J. (2014). Human-agent teaming for multirobot control: A review of human factors issues. *IEEE Transactions on Human-Machine Systems*, 44(1), 13-29.

[@ren2023robots] Ren, A. Z., Dixit, A., Bodrova, A., Singh, S., Tu, S., Brown, N., Xu, P., Takayama, L., Xia, F., Varley, J., Xu, Z., Sadigh, D., Zeng, A., & Majumdar, A. (2023). Robots that ask for help: Uncertainty alignment for large language model planners. *Conference on Robot Learning (CoRL)*. https://arxiv.org/abs/2307.01928

[@banerjee2018effects] Banerjee, S., Silva, A., Feigh, K., & Chernova, S. (2018). Effects of interruptibility-aware robot behavior. https://arxiv.org/abs/1804.06383

[@dahiya2023impact] Dahiya, A., Cai, Y., Schneider, O., & Smith, S. L. (2023). On the impact of interruptions during multi-robot supervision tasks. https://arxiv.org/abs/2306.16501

[@bajones2016help] Bajones, M., Weiss, A., & Vincze, M. (2016). Help, anyone? A user study for modeling robotic behavior to mitigate malfunctions with the help of the user. https://arxiv.org/abs/1606.02547

[@hart1988development] Hart, S. G., & Staveland, L. E. (1988). Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research. In P. A. Hancock & N. Meshkati (Eds.), *Human Mental Workload* (Advances in Psychology, Vol. 52, pp. 139-183). North-Holland.

[@endsley1988design] Endsley, M. R. (1988). Design and evaluation for situation awareness enhancement. *Proceedings of the Human Factors Society Annual Meeting*, 32(2), 97-101.

[@endsley2018automation] Endsley, M. R. (2018). Automation and situation awareness. In *Automation and Human Performance* (pp. 163-181). CRC Press.

[@taylor1990situational] Taylor, R. M. (1990). Situational awareness rating technique (SART): The development of a tool for aircrew systems design. In *Situational Awareness in Aerospace Operations* (AGARD-CP-478). NATO AGARD.
