# Related Work 보강 정리

Created: 2026-06-23

Source: `01_stanford_review_notes.md`, `00_stanford_review_notes.md`, current `section/02_related.tex`

## 1. 핵심 판단

두 Stanford review가 공통으로 지적한 Related Work 리스크는 크게 두 가지다.

1. **VOI/VOC/VPI 및 reward-aware information gathering과의 관계가 약하다.**
   현재 논문은 expected entropy reduction으로 predicate-level query를 고르지만, reviewer는 이것이 downstream task reward를 직접 최적화하는 VOI가 아니라는 점을 지적한다.

2. **Contingent/conditional symbolic planning과 sensing action literature가 부족하다.**
   현재 `Symbolic Planning under Uncertainty`는 ASP/POMDP hybrid와 symbolic uncertainty planning은 다루지만, classical symbolic planning 쪽에서 sensing action, conditional plan, knowledge-level plan을 다루는 계열을 거의 언급하지 않는다.

현재 `02_related.tex`는 이미 넓게 잘 써져 있다. 따라서 대규모 재작성보다는 다음 두 지점을 보강하면 된다.

- `Planning under Uncertainty` 안에 **VOI / belief-dependent reward / active information gathering** 문단 추가
- `Symbolic Planning under Uncertainty` 안에 **contingent planning / conditional planning / sensing action** 문단 추가

## 2. 현재 Related Work 구조와 삽입 위치

현재 구조:

```latex
\section{Related Work}
  \subsection{Knowledge-Based State and Action Models}
  \subsection{Planning under Uncertainty}
    \subsubsection{POMDP}
    \subsubsection{Belief Approximation}
    \subsubsection{Symbolic Planning under Uncertainty}
  \subsection{Human Corrective System}
```

추천 삽입 위치:

1. `\subsubsection{POMDP}` 또는 `\subsubsection{Belief Approximation}` 뒤에 새 문단:
   - 제목을 새 subsubsection으로 만들면 `\subsubsection{Information-Gathering Objectives}` 정도.
   - 너무 길어지면 subsection 없이 `POMDP` 말미에 한 문단으로 넣어도 된다.

2. `\subsubsection{Symbolic Planning under Uncertainty}` 초반, standard symbolic planning limitation 설명 직후:
   - contingent/conditional planning과 sensing action을 다루는 문단을 추가.

3. `\subsection{Human Corrective System}`의 KnowNo 문단 말미:
   - state-level query와 action-level query의 차이를 이미 잘 설명하고 있으므로, VOI와 query cost는 여기보다 Planning under Uncertainty 쪽에서 처리하는 게 낫다.

## 3. 추가해야 할 Related Work 주제

### 3.1 Contingent / Conditional Symbolic Planning

리뷰어 지적:

> Limited discussion of contingent/conditional planning and sensing within symbolic planning.

이 주제는 우리 논문과 직접 비교 대상이다. 왜냐하면 contingent planning도 uncertainty를 줄이기 위해 sensing action을 plan 안에 넣기 때문이다.

다만 차이는 명확하다.

- Contingent planning:
  - offline 또는 symbolic search에서 conditional branch를 포함한 plan을 생성
  - sensing action의 outcome에 따라 branch
  - 보통 human query가 아니라 robot sensing action을 다룸

- Our framework:
  - online POMCP execution 중 belief uncertainty를 평가
  - symbolic frontier에서 ambiguous predicate를 고름
  - sensing만으로 안 풀리는 uncertainty를 human-facing predicate query로 해결

따라서 관련 문헌을 넣되, "우리가 contingent planner를 이겼다"가 아니라 "비슷하게 uncertainty/action branching을 다루지만 query target과 execution setting이 다르다"로 쓰는 게 좋다.

필요한 citation 후보:

- `bonet2000planning` 또는 `bonet2001planning` 계열: planning with incomplete information as heuristic search
- `hoffmann2005conformant` 또는 conformant/contingent planning survey
- `brafman2012replanning` / `albore2011acting` 등 contingent planning with sensing/replanning
- `petrick2002knowledge` 또는 PKS 계열: knowledge-level planning
- `hoffmann2003contingent` 등 FF 확장 계열

현재 `ref.bib`에 이 계열 citation은 거의 없다. `ghallab2016automated`, `aeronautiques1998pddl`, `fikes1971strips`, `hoffmann2001ff` 정도만 있다. 제대로 쓰려면 bib 추가가 필요하다.

#### Draft paragraph

```latex
Another line of work addresses uncertainty within symbolic planning through contingent, conformant, or knowledge-level planning. These methods extend classical planning by introducing sensing actions, conditional effects, or belief/knowledge states so that a plan can branch depending on observations. Such formulations are closely related to our use of symbolic predicates under partial observability, because they also treat information gathering as part of task execution. However, they typically construct conditional plans or policies over symbolic belief states offline, and the information-gathering actions are usually robot sensing actions rather than human-facing predicate queries. In contrast, our framework performs online planning with POMCP and uses the current symbolic frontier to decide which uncertain predicate should be verified by the user during execution.
```

이 문단은 `Symbolic Planning under Uncertainty` 안에 넣는 것이 가장 자연스럽다.

### 3.2 VOI / VOC / VPI / Belief-Dependent Reward

리뷰어 지적:

> Prior work on value-of-information and VOC/VPI-based query selection in planning under uncertainty is not compared.

우리 논문에서 entropy query는 "uncertainty reduction"이지 "task utility improvement"를 직접 최적화하는 게 아니다. 이 차이를 Related Work에서 먼저 인정하고, Method/Discussion에서 scope를 명확히 하는 구조가 좋다.

관련 개념:

- VOI: Value of Information
- VPI: Value of Perfect Information
- VOC: Value of Computation
- belief-dependent reward / rho-POMDP
- active perception / information gathering reward

현재 `ref.bib`에 이미 있는 관련 citation:

- `somani2013despot`
- `silver2010monte`
- `sunberg2018pomcpow`
- `lauri2016exploration`
- `kim2021plgrim`
- `xiao2019objectsearch`
- `pajarinen2022composition`

현재 `ref.bib`에 없어서 추가가 필요한 후보:

- VOI/VPI 고전 문헌: Howard, Matheson 계열
- VOC: Russell and Wefald, Hay et al. 등
- rho-POMDP / belief-dependent reward: Araya-Lopez et al. 계열
- active perception / belief-space information gain: Spaan, Kurniawati, Bai/Hsu 계열 중 적절한 것

#### Draft paragraph

```latex
Information gathering can also be formulated through value-aware objectives. In decision-theoretic formulations, value of information (VOI), value of perfect information (VPI), and value of computation (VOC) select observations or computations according to their expected utility improvement rather than their pure uncertainty reduction. Related POMDP variants with belief-dependent rewards, such as rho-POMDP formulations, can encode information gain or query cost directly in the reward. These approaches provide a principled way to align information acquisition with downstream task value. Our query selection is different in scope: it uses expected entropy reduction as a lightweight predicate-level disambiguation criterion within the current symbolic frontier, while long-horizon task optimization remains handled by POMCP over task rewards. Thus, our method favors interpretability and online tractability, but does not claim to solve reward-optimal query selection.
```

이 문단은 `Planning under Uncertainty`의 POMDP 설명 뒤, 또는 새 `Information-Gathering Objectives` subsubsection으로 넣는 것이 좋다.

### 3.3 Active Perception / Interactive POMDP

리뷰어 지적:

> Prior interactive/active perception and human-in-the-loop POMDP works that reason explicitly about information value and query costs could be discussed.

현재 related work에는 navigation/object search 쪽 active information gathering이 일부 들어가 있다.

이미 있는 문장:

```latex
In navigation,~\cite{lauri2016exploration} combines POMCP with model predictive control to maximize information gain over occupancy-grid maps...
```

따라서 새로 길게 쓰기보다는 VOI 문단에서 다음 차이를 명확히 하면 된다.

- active perception: robot viewpoint/control/sensing action을 선택
- ours: human-facing predicate query를 선택
- 둘 다 information gathering이지만 query actuator가 다름

#### Draft sentence

```latex
This also distinguishes our setting from active perception and visual search, where the robot chooses sensing or viewpoint actions to improve its own observations; here, the information action is a user-facing predicate verification grounded in the shared knowledge base.
```

## 4. Related Work에서 유지해야 할 방어 논리

### 4.1 VOI와 entropy의 관계

너의 논리:

- 우리는 planning 성능 자체를 최적화하는 새 planner를 만드는 게 scope가 아니다.
- Long-horizon task planning은 POMCP reward가 담당한다.
- Query module은 현재 frontier에서 predicate ambiguity를 줄이는 역할이다.
- 따라서 VOI-optimal query selection은 다른 문제다.

Related Work에서는 이걸 너무 방어적으로 쓰지 말고, 다음처럼 균형 있게 쓰면 된다.

```text
VOI는 더 일반적이고 reward-aware한 formulation이다.
우리는 그 대신 predicate-level verification에 맞는 lightweight entropy criterion을 쓴다.
이 선택은 online tractability와 interpretability를 위한 design choice다.
```

이렇게 쓰면 reviewer에게 "VOI를 몰라서 안 한 게 아니라 scope가 다르다"가 전달된다.

### 4.2 Contingent planning과의 차이

핵심 차이:

- Contingent planning은 plan structure 자체가 observation branch를 포함한다.
- 우리 방법은 online execution 중 belief frontier를 갱신하고, 필요한 경우 human query를 발생시킨다.
- Contingent planning의 sensing action은 보통 robot의 observation action이고, 우리 query는 user-facing predicate verification이다.

Related Work에서는 이 차이를 명시하면 충분하다.

### 4.3 KnowNo-style action query와의 차이

현재 `Human Corrective System` 문단은 이미 충분히 좋다. 다만 리뷰어가 계속 지적하므로 한 문장 정도만 더 명확히 할 수 있다.

추가 가능 문장:

```latex
We therefore use KnowNo-style methods as an action-query reference point rather than as a direct algorithmic substitute for our state-query mechanism.
```

하지만 이 문장은 Related Work보다는 experiment/user-study discussion에 더 적합할 수 있다.

## 5. 구체적 수정 계획

### Step 1. `ref.bib` 보강

추가 후보:

```text
contingent planning / knowledge-level planning:
- Bonet and Geffner, Planning with Incomplete Information as Heuristic Search
- Petrick and Bacchus, A Knowledge-Based Approach to Planning with Incomplete Information and Sensing
- Hoffmann and Brafman, Contingent Planning via Heuristic Forward Search

VOI / VPI / VOC / belief reward:
- Howard, Information Value Theory
- Russell and Wefald, Principles of Metareasoning / Do the Right Thing
- Araya-Lopez et al., A POMDP Extension with Belief-Dependent Rewards

active perception / information gathering:
- 필요하면 기존 `lauri2016exploration`, `xiao2019objectsearch`, `pajarinen2022composition`으로 충분히 연결 가능
```

정확한 bib entry는 나중에 실제 논문 제목/venue 확인 후 넣는 게 좋다. 지금은 `03_revise.md`에서는 후보로만 둔다.

### Step 2. `02_related.tex`에 새 subsubsection 추가

추천 제목:

```latex
\subsubsection{Information-Gathering Objectives}~\label{rel:pu_info_gain}
```

위치:

- `\subsubsection{Belief Approximation}` 뒤
- `\subsubsection{Symbolic Planning under Uncertainty}` 앞

내용:

- VOI/VPI/VOC
- belief-dependent reward / rho-POMDP
- active perception과 ours의 차이
- expected entropy criterion의 위치

### Step 3. `Symbolic Planning under Uncertainty`에 contingent planning 문단 추가

위치:

```latex
As described in Section~\ref{rel:kb}, symbolic planning formalisms ...
However, standard predicate-based representations ...
```

이 다음에 contingent planning 문단 삽입.

### Step 4. Table caption 또는 table category는 굳이 안 바꿔도 됨

현재 Table~\ref{tab:comparison}은 이미 충분히 큰 table이다. Contingent planning과 VOI를 table row로 추가하면 너무 복잡해질 수 있다.

추천:

- Table은 그대로 둔다.
- 본문에서 "not shown separately in the table" 같은 표현은 쓰지 않는다.
- Related Work 문단으로만 보강한다.

## 6. 바로 넣을 수 있는 최소 수정안

추가 실험 없이 reviewer risk를 줄이는 최소 수정은 아래 두 문단이다.

### 문단 A: VOI / reward-aware information gathering

```latex
Information gathering can also be formulated through value-aware objectives. In decision-theoretic formulations, value of information (VOI), value of perfect information (VPI), and value of computation (VOC) select observations or computations according to their expected utility improvement rather than pure uncertainty reduction. Related POMDP variants with belief-dependent rewards can encode information gain or query cost directly in the reward. These approaches provide a principled way to align information acquisition with downstream task value. Our query selection is different in scope: it uses expected entropy reduction as a lightweight predicate-level disambiguation criterion within the current symbolic frontier, while long-horizon task optimization remains handled by POMCP over task rewards. Thus, our method favors interpretability and online tractability, but does not claim to solve reward-optimal query selection.
```

### 문단 B: Contingent / conditional symbolic planning

```latex
Another line of work addresses uncertainty within symbolic planning through contingent, conformant, or knowledge-level planning. These methods extend classical planning by introducing sensing actions, conditional effects, or belief/knowledge states so that a plan can branch depending on observations. Such formulations are closely related to our use of symbolic predicates under partial observability, because they also treat information gathering as part of task execution. However, they typically construct conditional plans or policies over symbolic belief states, and the information-gathering actions are usually robot sensing actions rather than human-facing predicate queries. In contrast, our framework performs online planning with POMCP and uses the current symbolic frontier to decide which uncertain predicate should be verified by the user during execution.
```

## 7. 주의할 점

- Related Work에서 "VOI보다 entropy가 낫다"라고 쓰면 안 된다.
- "VOI는 scope 밖"이라고만 쓰면 reviewer가 회피로 읽을 수 있다.
- 가장 좋은 표현은 "VOI is more general and reward-aware; our entropy objective is a lightweight predicate-disambiguation criterion integrated with symbolic frontier planning."
- Contingent planning은 적으로 만들 필요 없다. "유사한 문제를 다루지만, our information action is human-facing predicate verification"로 구분하면 된다.
- Citation 없이 일반명사처럼 VOI/VPI/VOC를 많이 쓰면 약해 보인다. `ref.bib`에 최소 3-5개 정도는 추가하는 게 좋다.

## 8. 우선순위 결론

Related Work에서 가장 먼저 할 일은 다음 두 가지다.

1. **VOI/VPI/VOC, belief-dependent reward, active perception과 entropy query의 관계 정리**
2. **Contingent/conditional symbolic planning과 human-facing predicate query의 차이 정리**

이 두 문단만 추가해도 `01_stanford_review_notes.md`의 missing related work 지적은 상당히 방어된다.
