# Query Convergence와 Frontier Coverage의 분리

이 문서는 symbolic belief refinement에서 발생하는 두 가지 문제를 논문 관점에서 분리해서 정리한다.

핵심은 다음과 같다.

1. **Query convergence**: true state가 이미 frontier 안에 들어 있다고 가정할 때, 질의 정책이 true state로 수렴할 수 있는가?
2. **Frontier coverage**: 제한된 크기의 frontier가 true state를 포함할 확률을 어떻게 확보할 것인가?

이 두 문제는 서로 다르다. 첫 번째는 질의 정책의 이론적 수렴성 문제이고, 두 번째는 large symbolic state space에서 sampling 또는 frontier generation의 품질 문제이다.

---

## 1. 전체 가능한 world 집합

전체 가능한 symbolic world 집합을 다음과 같이 둔다.

$$
\mathcal{S} = \{s_1, s_2, \dots, s_N\}.
$$

토마토 도메인에서 각 토마토가 세 가지 hidden state 중 하나를 가질 수 있다고 하자.

예를 들어 각 토마토의 hidden state가 다음 세 가지라면,

$$
\{\text{ripe}, \text{unripe}, \text{rotten}\},
$$

토마토가 12개일 때 가능한 조합 수는 다음과 같다.

$$
N = 3^{12} = 531{,}441.
$$

이는 단순한 예시일 뿐이고, 실제 시스템에서는 위치, 관측 여부, holding 여부, loaded/discarded 여부, action outcome까지 포함되면 가능한 symbolic world 수는 더 커질 수 있다.

따라서 전체 공간 $\mathcal{S}$를 매 step마다 명시적으로 전부 다루는 것은 계산적으로 어렵다.

---

## 2. Frontier 구성

전체 possible world 집합 $\mathcal{S}$를 모두 유지하는 대신, 시스템은 일부 상태만 뽑아 frontier를 구성한다.

$$
\mathcal{F} \subseteq \mathcal{S},
\qquad
|\mathcal{F}| = M.
$$

여기서 $M$은 실제로 유지하는 frontier 크기이다.

예를 들어 전체 가능한 world 수가 $N=531{,}441$이더라도, 계산 비용 때문에 실제 시스템은 $M=250$, $M=8000$ 등 일부 상태만 유지할 수 있다.

원래 belief distribution을 $p(s)$라고 하면, frontier 위에서는 이를 다시 정규화한다.

$$
p_{\mathcal{F}}(s)
=
\frac{p(s)}
{\sum_{s' \in \mathcal{F}} p(s')},
\qquad s \in \mathcal{F}.
$$

즉 원래 belief를 완전히 버리는 것이 아니라, frontier에 남은 상태들만 대상으로 posterior를 재정규화하는 것이다.

하지만 이 과정에서 중요한 문제가 생긴다.

만약 true state $s^*$가 frontier 안에 포함되지 않으면,

$$
s^* \notin \mathcal{F},
$$

아무리 질의를 잘 수행하더라도 true state로 수렴할 수 없다. 이 경우 질의 정책은 frontier 내부에서 가장 그럴듯한 state 또는 equivalence class로 수렴할 수 있을 뿐이다.

따라서 query refinement의 이론적 분석은 두 문제로 나누어야 한다.

---

## 3. 두 개의 서로 다른 문제

### Problem 1: Query Convergence

첫 번째 문제는 다음과 같다.

> True state가 frontier 안에 들어 있다고 가정했을 때, query policy는 true state를 찾아낼 수 있는가?

수식으로는 다음 조건을 가정한다.

$$
s^* \in \mathcal{F}.
$$

이 문제는 sampling 품질과 별개이다. 이미 올바른 상태가 후보 안에 존재한다고 가정하고, 질의 정책이 잘못된 후보들을 제거해 나갈 수 있는지를 묻는다.

### Problem 2: Frontier Coverage

두 번째 문제는 다음과 같다.

> 제한된 frontier $\mathcal{F}$가 true state를 포함할 확률을 어떻게 크게 만들 것인가?

즉 다음 확률을 어떻게 확보할 것인가의 문제이다.

$$
P(s^* \in \mathcal{F}).
$$

이 문제는 importance sampling, particle filtering, likelihood weighting, observation-consistent generation 등이 다루는 영역이다.

---

## 4. Query Convergence를 위한 가정

query convergence를 보이기 위해 다음 세 가지 가정을 둔다.

### Assumption 1: True State Inclusion

True state는 frontier 안에 포함되어 있다.

$$
s^* \in \mathcal{F}.
$$

### Assumption 2: Correct Oracle

사람 또는 oracle은 모든 query에 대해 true state 기준으로 정확한 답을 준다.

어떤 query predicate $q$에 대해 oracle answer는 항상 다음과 같다.

$$
a(q) = q(s^*).
$$

여기서 $q(s)$는 state $s$에서 predicate $q$의 truth value이다.

### Assumption 3: Query Separability

Frontier 안의 임의의 두 서로 다른 state는 적어도 하나의 query predicate로 구별 가능하다.

$$
\forall s_i, s_j \in \mathcal{F}, \; s_i \neq s_j,
\quad
\exists q \in \mathcal{Q}
\quad
\text{s.t.}
\quad
q(s_i) \neq q(s_j).
$$

여기서 $\mathcal{Q}$는 시스템이 질문할 수 있는 predicate 집합이다.

이 가정은 중요하다. 만약 두 state가 모든 query predicate에 대해 같은 truth value를 가진다면, 어떤 질의를 해도 둘을 구분할 수 없다. 이 경우 단일 true state로의 수렴은 불가능하고, query-indistinguishable equivalence class로만 수렴할 수 있다.

---

## 5. Candidate Set Update

초기 후보 집합을 frontier로 둔다.

$$
C_0 = \mathcal{F}.
$$

시점 $t$에서 query $q_t$를 수행하고, oracle answer를 받았다고 하자.

oracle은 정확하므로 답은 다음과 같다.

$$
q_t(s^*).
$$

그러면 다음 후보 집합은 oracle answer와 일치하는 state만 남긴다.

$$
C_{t+1}
=
\{s \in C_t
\mid
q_t(s) = q_t(s^*)\}.
$$

즉 현재 후보 중 true state와 같은 답을 내는 state만 살아남는다.

---

## 6. Lemma 1: True State는 제거되지 않는다

### Statement

모든 시점 $t$에 대해,

$$
s^* \in C_t
$$

가 성립한다.

### Proof

초기 조건에서 $C_0=\mathcal{F}$이고, Assumption 1에 의해 $s^*\in\mathcal{F}$이므로,

$$
s^* \in C_0.
$$

이제 $s^* \in C_t$라고 가정하자.

후보 집합 update는 다음과 같다.

$$
C_{t+1}
=
\{s \in C_t
\mid
q_t(s) = q_t(s^*)\}.
$$

$s=s^*$를 대입하면,

$$
q_t(s^*) = q_t(s^*)
$$

는 항상 참이다.

따라서 $s^*$는 $C_{t+1}$에도 포함된다.

귀납법에 의해 모든 $t$에 대해 $s^*\in C_t$이다.

---

## 7. Lemma 2: 후보가 둘 이상이면 제거 가능한 잘못된 state가 존재한다

### Statement

만약 $|C_t|>1$이면, 적어도 하나의 잘못된 state를 제거할 수 있는 query가 존재한다.

### Proof

$|C_t|>1$이고 Lemma 1에 의해 $s^*\in C_t$이다.

따라서 어떤 state $s \in C_t$가 존재하여,

$$
s \neq s^*.
$$

Assumption 3, 즉 query separability에 의해 $s$와 $s^*$를 구별하는 query $q$가 존재한다.

$$
q(s) \neq q(s^*).
$$

이 query를 수행하면 후보 update는 다음과 같다.

$$
C_{t+1}
=
\{s' \in C_t
\mid
q(s') = q(s^*)\}.
$$

그런데 선택한 잘못된 state $s$는 $q(s)\neq q(s^*)$이므로 $C_{t+1}$에서 제거된다.

반면 Lemma 1에 의해 $s^*$는 제거되지 않는다.

따라서 적어도 하나의 잘못된 state를 제거하는 query가 존재한다.

---

## 8. Theorem 1: Query Convergence

### Statement

다음 조건이 성립한다고 하자.

1. $s^* \in \mathcal{F}$.
2. Oracle answer는 항상 정확하다.
3. Frontier 안의 모든 state pair는 query predicate로 구별 가능하다.
4. Query policy는 후보 집합을 실제로 줄이는 positive-gain query가 존재할 때, 그런 query를 선택한다.

그러면 query policy는 유한 단계 내에 true state로 수렴한다.

즉 어떤 $T \le M-1$이 존재하여,

$$
C_T = \{s^*\}
$$

가 성립한다.

### Proof

Lemma 1에 의해 true state $s^*$는 모든 시점에서 후보 집합에 남아 있다.

만약 $|C_t|=1$이면, $s^*\in C_t$이므로 곧바로

$$
C_t=\{s^*\}
$$

이다.

반대로 $|C_t|>1$이면 Lemma 2에 의해 적어도 하나의 잘못된 state를 제거할 수 있는 query가 존재한다.

Assumption 4에 의해 query policy는 후보를 줄이는 query를 선택하므로,

$$
|C_{t+1}| < |C_t|.
$$

즉 후보 집합의 크기는 단조 감소하며, 감소할 때마다 적어도 1개 이상의 잘못된 state가 제거된다.

초기 후보 수는

$$
|C_0| = |\mathcal{F}| = M
$$

이다.

따라서 최대 $M-1$번의 query 이후에는 후보 집합의 크기가 1이 된다.

그리고 Lemma 1에 의해 true state는 제거되지 않았으므로,

$$
C_T = \{s^*\}.
$$

따라서 query policy는 유한 단계 내에 true state로 수렴한다.

---

## 9. Entropy 또는 Information Gain 기반 Query Policy와의 관계

현재 시스템의 query policy는 후보 fact 중 posterior entropy를 가장 크게 줄이는 query를 선택한다.

현재 후보 집합 $C_t$와 weight distribution $w_t(s)$가 있을 때, query $q$를 수행하면 후보는 true branch와 false branch로 나뉜다.

$$
C_t^+(q) = \{s \in C_t \mid q(s)=\text{True}\},
$$

$$
C_t^-(q) = \{s \in C_t \mid q(s)=\text{False}\}.
$$

query가 모든 후보에 대해 같은 값을 가지면 후보 집합을 나누지 못한다. 이 경우 information gain은 0이다.

반대로 어떤 query가 후보 집합을 둘로 나눈다면, 즉

$$
C_t^+(q) \neq \emptyset
\quad \text{and} \quad
C_t^-(q) \neq \emptyset,
$$

그 query는 posterior uncertainty를 줄일 수 있다.

따라서 positive information gain을 갖는 query가 존재하고, query policy가 최대 information gain query를 선택한다면, 후보 집합을 줄이는 query가 선택된다.

이 관점에서 entropy 기반 query policy는 Theorem 1의 Assumption 4를 만족하는 한 수렴성 증명에 포함될 수 있다.

다만 주의할 점이 있다.

Entropy 기반 policy는 "사람이 보기에 중요한 질문"을 먼저 고르는 것이 아니라, 현재 후보 집합을 가장 잘 나누는 질문을 고른다. 따라서 detect 이후 위치 질문이 자연스럽게 보이더라도, posterior frontier를 더 균형 있게 나누는 ripeness 질문이 먼저 선택될 수 있다.

이것은 query convergence의 실패가 아니라, query objective의 성격이다.

---

## 10. N = M인 경우

먼저 전체 possible world를 모두 frontier에 포함하는 이상적인 경우를 생각한다.

$$
\mathcal{F} = \mathcal{S},
\qquad
M=N.
$$

이 경우 true state는 당연히 frontier 안에 있다.

$$
s^* \in \mathcal{F}.
$$

따라서 Assumption 2와 Assumption 3이 성립하면 Theorem 1에 의해 query policy는 true state로 수렴한다.

즉 full enumeration setting에서는 query convergence 자체는 어렵지 않다.

문제는 $N$이 매우 크다는 점이다.

토마토 12개만 보아도,

$$
N = 3^{12}=531{,}441
$$

이다.

따라서 $N=M$인 full frontier 방식은 이론적으로는 깔끔하지만 계산적으로는 비현실적이다.

---

## 11. M < N이고 true state가 frontier 안에 있는 경우

실제 시스템에서는 $M \ll N$인 frontier를 사용한다.

이때도 만약

$$
s^* \in \mathcal{F}
$$

가 성립한다면 Theorem 1은 그대로 적용된다.

즉 중요한 점은 다음이다.

> Query convergence theorem은 전체 possible world를 다 포함할 필요가 없다. True state가 sampled frontier 안에 포함되어 있고, frontier 내부 state들이 query로 구별 가능하면 충분하다.

따라서 $M<N$이라도 다음 조건이 성립하면 수렴한다.

$$
s^* \in \mathcal{F}
$$

and

$$
\forall s_i, s_j \in \mathcal{F}, \; s_i \neq s_j,
\quad
\exists q \in \mathcal{Q}
\quad
\text{s.t.}
\quad
q(s_i) \neq q(s_j).
$$

이것이 논문에서 주장할 수 있는 핵심 이론적 결과이다.

---

## 12. Frontier Coverage 문제

이제 남는 문제는 frontier가 true state를 포함할 확률이다.

$$
P(s^* \in \mathcal{F}).
$$

이 문제는 query convergence와 별개의 문제이다.

Query convergence theorem은 조건부 명제이다.

$$
s^* \in \mathcal{F}
\quad \Rightarrow \quad
\text{query convergence}.
$$

하지만 실제 시스템에서는 먼저 $s^* \in \mathcal{F}$를 만족하도록 frontier를 생성해야 한다.

이것이 frontier coverage 문제이다.

---

## 13. Uniform Sampling의 한계

전체 possible world가 $N$개이고, 그중 true state가 하나라고 하자.

균등하게 $M$개를 replacement 없이 sampling하면 true state가 포함될 확률은 다음과 같다.

$$
P(s^* \in \mathcal{F})
=
\frac{M}{N}.
$$

95% coverage를 원하면,

$$
\frac{M}{N} \ge 0.95.
$$

즉

$$
M \ge 0.95N.
$$

토마토 12개 예시에서 $N=531{,}441$이면,

$$
M \ge 504{,}869.
$$

이는 사실상 full enumeration과 거의 같다.

따라서 단순 uniform sampling으로는 large symbolic state space에서 true state coverage를 확보하기 어렵다.

---

## 14. Importance Sampling 관점

Importance sampling 또는 proposal-based sampling을 사용하면 상황이 달라진다.

sampling distribution을 $\pi(s)$라고 하자.

Replacement sampling으로 $M$번 뽑는다면 true state가 적어도 한 번 포함될 확률은 다음과 같다.

$$
P(s^* \in \mathcal{F})
=
1 - (1-\pi(s^*))^M.
$$

95% coverage를 원하면,

$$
1 - (1-\pi(s^*))^M \ge 0.95.
$$

따라서

$$
M
\ge
\frac{\log(0.05)}
{\log(1-\pi(s^*))}.
$$

예를 들어,

| $\pi(s^*)$ | 95% coverage에 필요한 $M$ |
|---:|---:|
| 0.01 | 약 299 |
| 0.001 | 약 2,995 |
| 0.0001 | 약 29,956 |

따라서 true state에 높은 proposal probability를 부여할 수 있다면 필요한 particle 수를 크게 줄일 수 있다.

하지만 핵심 질문은 다음이다.

> 어떻게 $\pi(s^*)$를 크게 만들 것인가?

이것이 observation-consistent frontier generation, likelihood weighting, particle filtering이 필요한 이유이다.

---

## 15. Observation-Consistent Frontier Generation

현재 시스템은 observation과 transition model을 이용해 frontier를 구성한다.

즉 action $a$와 observation $o$가 주어졌을 때, 다음과 같은 posterior에 가까운 상태들을 유지하려고 한다.

$$
p(s' \mid o, a)
\propto
P(o \mid s', a)
P(s' \mid a).
$$

이 관점에서 frontier generation은 단순 uniform sampling이 아니라 observation-consistent proposal을 만드는 문제로 볼 수 있다.

좋은 frontier generator는 다음 조건을 만족해야 한다.

1. Observation likelihood가 높은 state를 우선 포함한다.
2. True state와 observationally consistent한 state를 높은 확률로 포함한다.
3. 너무 작은 particle cap으로 posterior support를 잘라내지 않는다.
4. 낮은 probability state를 제거하더라도 true state coverage를 크게 해치지 않는다.

이번 실험에서 확인된 중요한 현상은 다음이다.

작은 particle cap, 예를 들어 $M=250$, 은 계산을 빠르게 만들지만 posterior support를 심하게 왜곡할 수 있다.

반면 $M=8000$처럼 충분히 큰 frontier를 유지하면 true state 또는 true state와 일관된 후보가 frontier 안에 남을 가능성이 커지고, query refinement가 정상적으로 작동한다.

즉 particle cap은 단순한 runtime parameter가 아니라 posterior fidelity를 결정하는 핵심 hyperparameter이다.

---

## 16. Query Convergence와 Frontier Coverage의 논문 내 역할

논문에서는 두 문제를 명확히 분리해서 쓰는 것이 좋다.

### Theoretical Claim

Query refinement 자체는 다음 조건부 수렴성을 가진다.

> If the true state is included in the sampled frontier and the query predicates separate frontier states, then the proposed query policy converges to the true state in finite queries under a correct oracle.

한국어로는 다음과 같이 쓸 수 있다.

> True state가 sampled frontier 안에 포함되어 있고, query predicate들이 frontier state들을 구별할 수 있으며, oracle answer가 정확하다면, 제안한 query policy는 유한 번의 질의 후 true state로 수렴한다.

### Practical Challenge

실제 어려움은 다음이다.

> How can the system construct a compact frontier that contains the true state with high probability?

한국어로는 다음과 같이 쓸 수 있다.

> 실제 어려움은 제한된 계산 예산 안에서 true state를 높은 확률로 포함하는 compact frontier를 생성하는 것이다.

---

## 17. 한계로서의 Intractability

현재 시스템의 근본적 한계는 flat symbolic frontier enumeration이다.

토마토가 $N_T$개이고 각 토마토가 3개 hidden state를 가진다면 hidden assignment 수는 다음과 같이 증가한다.

$$
3^{N_T}.
$$

토마토 8개는 다음과 같다.

$$
3^8 = 6{,}561.
$$

토마토 12개는 다음과 같다.

$$
3^{12} = 531{,}441.
$$

이 차이는 매우 크다.

8개 토마토에서는 전체 frontier를 유지하거나 큰 particle cap으로 근사하는 것이 가능할 수 있다. 하지만 12개 토마토에서는 같은 방식이 빠르게 intractable해진다.

따라서 large scene에서는 다음 trade-off가 발생한다.

1. 작은 $M$: 빠르지만 true state coverage가 낮고 posterior가 왜곡될 수 있다.
2. 큰 $M$: posterior fidelity는 높지만 update/query 계산 비용이 커진다.

이 trade-off는 방법론의 중요한 한계로 명시할 수 있다.

---

## 18. 후속 방법론 방향

이 한계를 해결하기 위한 방향은 크게 세 가지이다.

### 18.1 Applicable Action Pruning

POMCP search time을 줄이기 위한 방법이다.

현재 POMCP search cost는 대략 다음과 같이 증가한다.

$$
\text{search cost}
\approx
n_{\text{sim}}
\times
d_{\text{rollout}}
\times
|\mathcal{A}_{\text{app}}|
\times
\text{transition/observation cost}.
$$

따라서 applicable action 수를 줄이면 search time을 줄일 수 있다.

가능한 전략은 다음과 같다.

- 현재 위치와 관련 없는 pick/place action 제거
- 이미 충분히 관측된 stem에 대한 detect action 제거
- goal 달성에 기여하지 않는 action 제거
- domain heuristic으로 top-k action만 rollout에 사용
- symbolic precondition과 task progress를 이용한 action masking

이 방향은 belief frontier 크기를 직접 줄이지는 않지만, planning search branch factor를 줄인다.

### 18.2 Symbolic Frontier Reduction

Belief update의 frontier 폭발을 줄이는 방법이다.

핵심은 $3^N$개의 world를 명시적으로 만들지 않는 것이다.

가능한 전략은 다음과 같다.

- observation-consistent constraint를 강하게 적용
- observed object와 unobserved object를 분리해서 factorized belief 유지
- 독립적인 object-level belief factor를 유지하고, 필요한 경우에만 joint state 생성
- domain invariant로 불가능한 조합 사전 제거
- query와 planning에 필요한 marginal만 계산

이 방향은 flat possible-world enumeration을 structured belief representation으로 바꾸는 것이다.

### 18.3 Probability-Aware Frontier Pruning

Transition 또는 observation probability가 매우 낮은 branch를 줄이는 방법이다.

가능한 전략은 다음과 같다.

- top-p cumulative probability support 유지
- likelihood가 낮은 observation branch 제거
- beam search 기반 frontier 유지
- importance sampling 또는 likelihood weighting 기반 frontier 생성
- posterior weight가 낮은 particles resampling

다만 이 방식은 조심해야 한다.

이번 실험에서 보았듯이 particle cap을 너무 작게 두면 true state 또는 중요한 posterior support가 사라져 query refinement가 잘못된 state로 수렴할 수 있다.

따라서 probability-aware pruning은 단순 truncation이 아니라 coverage guarantee 또는 empirical calibration과 함께 설계되어야 한다.

---

## 19. 논문에 넣을 수 있는 문장 초안

### 한계 문장

본 방법은 sampled frontier 안에서의 query refinement에 대해서는 조건부 수렴성을 갖지만, large symbolic state space에서는 true state가 frontier 안에 포함된다는 보장이 별도로 필요하다. 특히 object 수가 증가하면 가능한 symbolic world 수가 조합적으로 증가하며, 작은 particle cap은 posterior support를 왜곡하여 잘못된 knowledge refinement를 유발할 수 있다.

### 수렴성 문장

True state가 sampled frontier에 포함되어 있고, query predicate set이 frontier 내 state들을 구별할 수 있으며, oracle answer가 정확하다고 가정하면, 제안한 query policy는 유한 번의 질의 후 true state로 수렴한다. 이는 query 과정에서 true state가 제거되지 않고, 각 informative query가 적어도 하나의 잘못된 state를 제거하기 때문이다.

### Frontier coverage 문장

반면 true state가 sampled frontier에 포함될 확률은 별도의 frontier coverage 문제이다. Uniform sampling에서는 true state가 단일 world일 때 coverage probability가 $M/N$에 불과하므로, $M \ll N$인 large scene에서는 충분한 coverage를 기대하기 어렵다. 따라서 observation-consistent proposal, importance sampling, likelihood weighting, factorized symbolic belief와 같은 구조적 frontier generation이 필요하다.

### Future work 문장

향후 연구에서는 action-space pruning, structured symbolic belief representation, probability-aware frontier reduction을 결합하여 large scene에서도 posterior fidelity를 유지하면서 계산 가능한 query refinement를 수행하는 방법을 탐구할 필요가 있다.

---

## 20. 정리

논리 구조는 다음과 같이 정리할 수 있다.

1. 전체 possible world는 $\mathcal{S}$이고 크기는 $N$이다.
2. 계산 가능성을 위해 sampled frontier $\mathcal{F}$를 유지하며 크기는 $M$이다.
3. Query convergence는 $s^*\in\mathcal{F}$를 조건으로 하는 문제이다.
4. 이 조건이 만족되고 query predicate가 state들을 구별할 수 있으면, query policy는 유한 단계 내에 $s^*$로 수렴한다.
5. 하지만 $s^*\in\mathcal{F}$ 자체는 보장되지 않는다.
6. 따라서 large scene에서 핵심 난점은 frontier coverage이다.
7. Uniform sampling은 $M/N$ coverage만 제공하므로 large $N$에서 비효율적이다.
8. Importance sampling, likelihood weighting, observation-consistent generation, factorized belief가 필요하다.
9. 따라서 논문에서는 query convergence theorem과 frontier coverage limitation을 분리해서 제시하는 것이 가장 논리적이다.

