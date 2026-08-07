# 02 수정 현황: Method 중심 점검

작성일: 2026-06-23

기준 파일:

- `latex_zips/latex_tro/01_revise.md`
- `latex_zips/latex_tro/section/04_method.tex`

## 1. Method 섹션 현재 상태

Method 쪽은 리뷰어가 지적한 핵심 중 상당 부분이 반영되었다. 특히 belief/frontier 설명, notation consistency, query가 predicate-level verification이라는 점은 이전보다 명확해졌다.

### 완료된 수정

#### 1.1 Frontier 정의 명확화

현재 frontier는 symbolic transition model의 support로 설명된다.

```latex
the frontier is the finite support of $T(\cdot \mid s,a)$
```

따라서 frontier가 임의 particle set이 아니라, 현재 state와 action에서 transition model이 유도하는 distinct symbolic successor set이라는 점이 명확해졌다.

현재 설명의 핵심:

- `Frontier(s,a)=\{s_1^\star,\dots,s_j^\star\}`
- 각 `s_k^\star`는 distinct reachable successor state
- `w_k = T(s_k^\star \mid s,a)` 또는 `T(s_k^\star \mid s^K,a)`로 prior frontier weight 정의
- `pick(tomato_1)` 예시로 success/failure successor와 prior weight 설명

#### 1.2 Local frontier belief 설명 추가

현재 문장:

```latex
The belief is therefore maintained over the currently reachable frontier rather than over all symbolic worlds.
```

이 문장은 limitation이라기보다 방법 정의에 가깝다. 즉, full posterior 전체를 유지하는 것이 아니라 current reachable frontier 위에서 belief를 유지한다는 점을 명확히 한다.

#### 1.3 Observation likelihood notation 정리

기존:

```latex
O(o \mid s_k^\star)
```

수정 후:

```latex
O(o \mid s_k^\star, a)
```

Preliminaries의 POMDP notation과 맞도록 action-conditioned observation likelihood로 통일되었다.

#### 1.4 Weight notation 정리

현재 구조:

- `w_k`: observation 전 frontier prior weight
- `w_k^\prime`: observation likelihood를 곱한 unnormalized posterior weight
- `\tilde{w}_k^\prime`: normalized posterior weight

`P(c)`와 `P(\neg c)`는 normalized weight 기준으로 수정되었다.

```latex
P(c)=\sum_{s_k^{\star}: c \in s_k^{\star}} \tilde{w}_k^\prime
```

MAP state selection도 normalized weight 기준으로 정리되었다.

#### 1.5 Confidence notation 충돌 제거

Algorithm에서 confidence scalar가 기존 `c`에서 `\rho`로 바뀌었다.

```latex
\rho \gets \ComputeConfidence{\mathcal{B}}
\While{$\rho < \tau$}
```

이로써 candidate hypothesis `c \in \mathcal{C}`와 confidence scalar `c`의 충돌이 사라졌다.

#### 1.6 Candidate set notation 충돌 제거

Hypothesis candidate set은 `\mathcal{C}`로 유지하고, POMCP action candidate set은 `\mathcal{A}_{cand}`로 변경했다.

```latex
\mathcal{A}_{cand} = \{a \mid a\llbracket s \rrbracket \in \mathcal{A},\ ha \in T\}
```

#### 1.7 Predicate-level query 예시 추가

Entropy query가 action-level decision이 아니라 predicate-level hypothesis verification이라는 점을 예시로 명확히 했다.

```latex
For example, the selected hypothesis may be a fact such as
$c^{*}=\texttt{ripe}(\text{tomato}_1)$
or
$c^{*}=\texttt{plastic}(\text{waste}_2)$.
```

#### 1.8 POMCP scope 방어 추가

Method에서 POMCP를 새로 제안하거나 최적화하는 것이 논문 scope가 아님을 밝혔다.

현재 취지:

- POMCP는 off-the-shelf online planner
- 본 논문의 초점은 symbolic belief representation과 human-verifiable uncertainty resolution을 통합하는 것
- POMCP simulation은 generative model을 사용하되, symbolic model은 applicable actions와 reachable symbolic successors로 sampling/search를 제한한다

## 2. Method에서 아직 확인할 부분

### 2.1 `reachable successors states` 오타

현재 문장에 오타가 있다.

```latex
reachable successors states
```

수정 권장:

```latex
reachable successor states
```

또는 더 안전하게:

```latex
reachable symbolic successor states
```

### 2.2 Algorithm argmax 표기 확인

현재:

```latex
$s^{K} \gets \arg\max_{(s_k^{\star}, \tilde{w}_k^\prime) \in \mathcal{B}} \tilde{w}_k^\prime$
```

의미는 맞다. 다만 LaTeX 독자가 조금 무겁게 느낄 수 있으므로 간결하게 하려면 아래도 가능하다.

```latex
$s^{K} \gets \arg\max_{s_k^{\star}} \tilde{w}_k^\prime$
```

현재 표기도 틀리지는 않으므로 필수 수정은 아니다.

### 2.3 Filtering 후 renormalization 문장

현재 filtering equation은 normalized weight `\tilde{w}_k^\prime`를 사용한다. 이후 문장:

```latex
The remaining weights are renormalized
```

은 retained subset 위에서 다시 정규화한다는 뜻으로 읽히므로 괜찮다. 더 정확히 쓰려면:

```latex
The retained weights are renormalized over the remaining particles
```

정도지만 필수는 아니다.

### 2.4 Method에 더 넣지 않는 것이 좋은 내용

다음 내용은 Method보다 Discussion/Limitation에 두는 것이 적절하다.

- MAP commit이 alternative global hypotheses를 prune할 수 있다는 점
- early mis-commitment 가능성
- observation/transition probability misspecification
- POMCP scalability 개선 방향
- VOI/rho-POMDP와의 차이

Method에서는 현재처럼 방법 정의와 알고리즘 설명 중심으로 충분하다.

## 3. Method 밖에서 아직 남은 주요 작업

### 3.1 System Discussion

`section/05_exp_system.tex`에서 더 보강할 부분:

- frontier/local belief approximation의 실용적 trade-off
- fixed transition/observation model probability의 calibration limitation
- scalability bottleneck: frontier size, POMCP tree expansion, observation-likelihood evaluation
- progressive widening, DESPOT-style scenario sampling, tree reuse/caching 등 future direction

### 3.2 Entropy/VOI scope 정리

Method에서는 predicate-level query 예시로 충분히 완화했다. 하지만 Related Work 또는 Discussion에서는 다음 차이를 명확히 할 필요가 있다.

- 본 논문의 query는 POMDP action으로 query cost/future value를 최적화하는 VOI problem이 아니다.
- Query는 current frontier의 symbolic state ambiguity를 줄이는 verification mechanism이다.
- Long-horizon task action selection은 POMCP와 reward model이 담당한다.
- VOI/rho-POMDP는 related/future work로 언급하면 충분하다.

### 3.3 Real-robot section

`section/05_real_robot.tex`에서 다음을 명확히 해야 한다.

- closed-loop quantitative hardware evaluation인지
- 아니면 integrated hardware feasibility demonstration인지

정량 hardware 결과가 없다면, main quantitative claims는 controlled evaluation과 WoZ study 기반이라고 분리해야 한다.

### 3.4 User study claim 완화

Abstract, user-study summary, conclusion에서 다음 표현을 낮추는 것이 좋다.

- `confirm reduced cognitive workload`
- `effectively lowers user workload`
- `real-world settings`

대신:

- `provide preliminary evidence`
- `suggest`
- `supplementary usability check`

같은 표현이 안전하다.

### 3.5 KnowNo baseline 차이 설명

남은 핵심:

- system-level KnowNo와 WoZ KnowNo의 차이
- tomato WoZ에서 KnowNo가 높은 이유
- waste에서 action-level query가 어려운 이유
- action-level baseline이 context/interface design에 민감하다는 점

## 4. Method 섹션 결론

현재 Method는 리뷰어가 지적한 reproducibility/clarity 문제 중 상당 부분을 해결했다.

특히 다음 항목은 충분히 개선됨:

- frontier 생성 방식
- duplicate symbolic successor의 의미
- prior/posterior weight 흐름
- observation likelihood notation
- confidence/hypothesis notation conflict
- predicate-level query의 구체적 의미
- POMCP가 새 solver contribution이 아니라 integration component라는 scope

따라서 Method에서 남은 것은 큰 구조 수정이 아니라 소규모 문장/오타 정리 수준이다. 다음 단계는 `05_exp_system.tex`, `05_real_robot.tex`, `05_exp_user.tex`, `06_conclusion.tex`, `02_related.tex` 쪽에서 scope와 limitation을 맞추는 것이다.
