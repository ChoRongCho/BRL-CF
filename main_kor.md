# 부분 관측 환경에서 로봇 작업 계획 중 언제 무엇을 질의할 것인가

*When and What to Query in Robot Task Planning under Partial Observability*

> 현재 `latex_zips/BRL_achitecture_changmin/main.tex`에서 불러오는 원고의 한국어 번역본이다. 주석과 비활성 문안은 제외했다. 수정 제안이나 해석을 추가하지 않고 원문의 서술과 주장 강도를 유지했다. 인용은 원문의 참고문헌 키로 표기하며, 그림은 원본 PDF 링크와 번역한 설명을 제공한다. 표의 밑줄은 원문과 같이 최상 평균값을 나타낸다.

**저자:** Changmin Park¹, Seongah Park¹, Nayoung Kim¹, Sungjin Park¹, Gayoung Ryu¹, Geonyeong Ko¹, Junho Lee¹*, Changjoo Nam¹˒²*

Changmin Park와 Seongah Park는 본 연구에 동등하게 기여하였다. 별표는 교신저자를 나타낸다.

**소속:** ¹ 서강대학교, 대한민국 서울. ² Vertical Labs, 대한민국 서울.

**이메일:** Changmin Park: cmp9877@gmail.com; Seongah Park: ppseongah@gmail.com; Nayoung Kim: nayoungkim0826@gmail.com; Sungjin Park: sungjin.park@sogang.ac.kr; Gayoung Ryu: gyryu.sg@gmail.com; Geonyeong Ko: 2567bear@naver.com; Junho Lee: junho.lee@sogang.ac.kr; Changjoo Nam: cjnam@sogang.ac.kr.

## 초록

본 논문에서는 작업 계획 중 로봇이 언제 외부 정보를 요청하고 무엇을 질문해야 하는지 결정하는 방법을 제시한다. 불완전한 정보를 바탕으로 동작하는 로봇은 현재 환경의 상태를 판단하지 못할 수 있으며, 이는 작업 실패로 이어질 수 있다. 그러나 이러한 모호성을 모두 외부 질의로 해결하면 불필요한 비용이 발생한다. 따라서 로봇은 언제 질의할지와 어떤 정보를 요청할지를 모두 결정해야 한다. 이 문제를 해결하기 위해 가능한 현재 상태들에 대한 가중 belief를 유지한다. Belief의 불확실성을 이용하여 질의 시점을 결정하고, 현재 상태에 관한 질문 중 불확실성을 가장 크게 줄일 것으로 기대되는 질문을 선택한다. 외부 에이전트의 응답을 받은 후 belief를 갱신하고 작업 계획을 계속한다. 두 로봇 도메인에서 작업 계획 중 외부 피드백을 요청하는 다른 시스템들과 프레임워크를 비교 평가한다. 정확한 응답을 사용하는 시뮬레이션에서 제안 방법은 두 도메인 모두에서 가장 성능이 좋은 baseline보다 높은 성공률을 달성하며, 각각 100.00% 대 76.00%, 97.50% 대 56.00%를 보인다. 이러한 성공률 우위는 실제 환경에서 비전 언어 모델과 사람의 피드백을 사용한 실험에서도 유지된다.

In this paper, we present a method for deciding when a robot should request external information and what it should ask during task planning. A robot operating with incomplete information may fail to determine the current state of the world, which can lead to task failure. However, resolving every such ambiguity through external queries would incur unnecessary costs. Therefore, a robot must determine both when to query and what information to request. To solve this problem, we maintain a weighted belief over possible current states. We use belief uncertainty to decide when to query and select the question about the current state expected to reduce uncertainty the most. After receiving an answer from an external agent, the belief is updated and task planning continues. We evaluate the framework on two robotic domains against other systems that request external feedback during task planning. In simulation with correct responses, our method achieves higher success rates than the best-performing baseline across the two domains, with 100.00% versus 76.00% and 97.50% versus 56.00%. The success advantage also holds with vision-language model and human feedback in real-world experiments.


수정: 
우리는 작업 계획 중 로봇이 언제 외부 정보를 요청하고 무엇을 질의할지 결정하는 방법을 제시한다. 불완전한 상태 정보는 잘못된 행동과 작업 실패로 이어질 수 있지만, 모든 불확실성을 외부 질의로 해소하면 불필요한 비용이 발생한다. 따라서 작업 성공을 높이면서 질의 수를 줄이려면 적절한 질의 시점과 질의 선택이 필요하다. 우리는 가능한 현재 상태들에 대한 가중 belief를 구성한다. Belief의 신뢰도로 질의 시점을 결정하고, 불확실성을 가장 크게 줄일 것으로 기대되는 질의를 선택한다. 외부 에이전트의 응답으로 belief를 갱신한 뒤 작업 계획을 계속한다. 분리수거와 토마토 수확에서 기존 선제적 질의 시스템들과 전체 시스템 성능을 비교한다. 정확한 응답을 사용하는 시뮬레이션에서 제안 시스템의 성공률은 각각 100.00%와 97.50%로, 가장 성능이 좋은 baseline의 76.00%와 56.00%보다 높다. 이러한 성공률 우위는 비전 언어 모델과 사람의 피드백을 사용하는 실제 로봇 실험에서도 유지된다. 또한 공통 행동 플래너를 사용하는 ablation 실험으로 질의 시점과 선택의 기여를 분석한다. 제안한 시점 기준은 질의 선택 방법이 달라져도 높은 성공률을 유지하며, 제안한 선택 기준은 동일한 시점 기준에서 질의 수를 줄인다.

In this paper, we present a method for deciding when a robot should request external information and what it should ask during task planning. A robot with incomplete information may fail to determine the current state of the world, which can lead to task failure. However, resolving all uncertainty through external queries would incur unnecessary costs. Therefore, a robot must determine both when to query and what information to request. To solve this problem, we construct a weighted belief over possible current states. We use belief uncertainty to decide when to query and select the \cp{query} expected to reduce uncertainty the most. After receiving an answer from an external agent, \cp{we update the belief and continue task planning}. We evaluate the framework on two robotic domains against existing proactive querying systems during task planning. In simulation with correct responses, our method \cp{achieves success rates of $100.00\%$ and $97.50\%$ versus $76.00\%$ and $56.00\%$ for the best baseline}. The success advantage \cp{holds with vision language model and human feedback on physical robots}. \cp{Ablation experiments show that our query timing maintains high success, while our query selection reduces query counts under the same timing criterion.}

**핵심어:** 불확실성하의 계획, 온라인 기호적 계획, 로봇 시스템

## 1. 서론

로봇은 인식 오류 [kaipa2016addressing]와 행동 실패 [kaelbling1998planning]로 인해 작업 계획 및 실행 중 불확실성을 자주 마주한다. 예를 들어 토마토가 익었는지, 집기 행동이 성공했는지 확신하지 못할 수 있다. 로봇은 외부 에이전트가 계속 실행을 감시해야 할 필요를 줄이기 위해, 부족한 정보를 제공하거나 다음 행동을 선택해 달라고 선제적으로 요청할 수 있다. 그러나 질의가 너무 적으면 불확실성이 해소되지 않아 작업이 실패할 수 있고, 질의가 잦으면 응답 지연, 비전 언어 모델(VLM)의 토큰 사용량, 사람의 작업 부담과 같은 상호작용 비용이 증가한다. 또 다른 어려움은 질의 수를 늘리는 것만으로 작업 성공이 반드시 개선되지는 않는다는 점이다. 질의가 너무 늦거나 불확실성을 해소하지 못하면 로봇은 여전히 실패할 수 있다. 따라서 로봇은 질의 수를 제한하면서 작업을 성공시키기 위해 **언제** 도움을 요청하고 **어떤** 정보를 요청할지 결정해야 한다.

기존 방법들은 주로 예측된 행동의 신뢰도나 기대 작업 수익을 이용하여 언제 무엇을 질의할지 결정한다. Conformal prediction 기반 방법은 여러 후보 행동이나 계획이 남아 있으면 외부 에이전트에게 질의한다 [ren2023robots, liang2024introspective]. 그러나 후보가 하나라는 사실이 올바른 상태 추정을 보장하지는 않으며, 예측 집합에서 올바른 행동이나 계획이 제외될 수도 있다. 부분 관측 마르코프 의사결정 과정(POMDP) 기반 접근법은 질의를 계획 행동으로 모델링하고 [burks2021collaborative, burks2023harps], 기대 누적 보상에 따라 선택한다. 그러나 두 접근법 모두 현재 상태에 대해 요구되는 신뢰도 수준에 도달하기 위해 언제 무엇을 질의할지 명시적으로 결정하지는 않는다. 그 결과 로봇은 상태 정보가 부족한 채 행동하거나 충분한 신뢰도에 도달하기 위해 추가 질의를 필요로 할 수 있다.

본 논문에서는 현재 기호적 상태의 불확실성을 이용하여 언제 질의하고 무엇을 질문할지 결정하는 방법을 제시한다. 작업 계획을 POMDP로 정식화하고 [kaelbling1998planning], 기호적 사실과 행동 모델 [davis1993knowledge, fikes1971strips]을 사용하여 도달 가능한 상태들에 대한 가중 belief를 구성한다. 질의 시점을 결정하기 위해 로봇은 정규화된 엔트로피로부터 belief confidence를 계산한다. 엔트로피는 확률이 하나의 상태로 집중될수록 감소한다. Belief confidence가 임계값보다 낮으면 외부 에이전트에게 질의한다. 무엇을 질의할지 결정하기 위해서는 confidence 임계값에 도달하는 데 필요한 질의 수를 줄이도록 belief 엔트로피의 기대 감소량이 가장 큰 상태 사실을 선택한다. 각 응답은 belief 갱신을 위한 증거를 제공한다. Confidence가 임계값에 도달하면 최대 사후 확률(MAP) 상태 추정치로 지식 베이스를 갱신하고 계획과 실행을 재개한다.

35번줄 수정: 
본 논문에서는 현재 기호적 상태의 불확실성을 이용하여 언제 질의하고 무엇을 질문할지 결정하는 방법을 제시한다. 작업 계획을 POMDP로 정식화하고 [kaelbling1998planning], 기호적 사실과 행동 모델 [davis1993knowledge, fikes1971strips]을 사용하여 도달 가능한 상태들에 대한 가중 belief를 구성한다. 질의 시점을 결정하기 위해 로봇은 정규화된 엔트로피로부터 belief confidence를 계산한다. 엔트로피는 확률이 하나의 상태로 집중될수록 감소한다. Belief confidence가 임계값보다 낮으면 외부 에이전트에게 질의한다. 무엇을 질의할지 결정하기 위해서는 confidence 임계값에 도달하는 데 필요한 질의 수를 줄이도록 belief 엔트로피의 기대 감소량이 가장 큰 상태 사실을 선택한다. 각 응답은 belief 갱신을 위한 증거를 제공한다. Confidence가 임계값에 도달하면, 우리는 최대 사후 확률(MAP) 상태로 지식 베이스를 갱신하고 계획과 실행을 재개한다. 계산 비용을 제한하기 위해 전체 belief를 전파하는 대신, 이 MAP 상태에서 다음 belief를 구성한다.

In this paper, we present a method that uses uncertainty about the current symbolic state to determine when to query and what to ask. We formulate task planning as a POMDP~\citep{kaelbling1998planning} and use symbolic facts and action models~\citep{davis1993knowledge,fikes1971strips} to construct a weighted belief over reachable states. To determine when to query, the robot computes belief confidence from normalized entropy, which decreases as probability concentrates on a single state. The robot queries an external agent if belief confidence falls below a threshold. To determine what to query, the robot selects the state fact with the greatest expected reduction in belief entropy to reduce the number of queries needed to reach the confidence threshold. Each response provides evidence for updating the belief. Once confidence reaches the threshold, \cp{we update} the knowledge base with the maximum a posteriori \cp{(MAP)} state estimate and \cp{resume} planning and execution. \cp{To limit computational cost, we construct the next belief from the MAP state rather than propagate the full belief.}


센싱 오류가 있는 부분 관측 환경의 두 로봇 작업에서 제안 방법을 평가한다. Oracle 에이전트를 사용하는 시뮬레이션에서 conformal prediction과 기대 작업 보상에 각각 기반한 두 질의 전략과 제안 방법을 비교한다. 이후 VLM과 사람 참가자를 외부 에이전트로 사용하는 실제 로봇 실험을 수행하여 conformal prediction 기반 baseline과 비교한다. 제안 방법은 시뮬레이션과 VLM 및 사람 피드백을 사용하는 실제 로봇 실험 모두에서 가장 높은 성공률을 달성한다. Ablation 실험은 질의 시점이 주로 작업 성공에 영향을 주며, 제안 질의 선택 방법은 동일한 시점 기준에서 질의 수를 줄인다는 것을 보여준다. 또한 confidence 임계값을 변화시켜 작업 성공과 질의 수 사이의 절충 관계를 살펴본다. 마지막으로 참가자 다섯 명 모두와의 인터뷰를 통해 두 방법 사이의 작업 성능 차이를 설명한다.

수정 (37번 라인만):

센싱 오류가 있는 부분 관측 환경의 두 로봇 작업에서 제안 방법을 평가한다. 먼저 oracle 응답을 사용하는 시뮬레이션에서 conformal prediction과 기대 작업 보상에 기반한 기존 선제적 질의 시스템들과 전체 시스템 성능을 비교한다. VLM과 사람의 피드백을 사용하는 실제 로봇에서도 conformal prediction 기반 baseline과 비교한다. 제안 시스템은 모든 평가 환경에서 비교한 시스템들 중 가장 높은 성공률을 달성한다. 다음으로 질의 시점과 선택이 작업 성공률과 질의 수에 미치는 영향을 구분하기 위해, 같은 POMCP 행동 플래너를 사용하는 ablation 실험을 수행한다. Ablation 실험은 confidence 기반 시점이 질의 선택 방법이 달라져도 높은 성공률을 유지하고, 엔트로피 기반 선택이 동일한 시점 기준에서 질의 수를 줄임을 보였다. 또한 confidence 임계값을 변화시켜 작업 성공과 질의 수 사이의 절충 관계를 살펴본다. 마지막으로 참가자 다섯 명 모두와의 인터뷰를 통해 두 방법 사이의 작업 성능 차이를 설명한다.

We evaluate our method on two robotic tasks in partially observable environments with sensing errors. In simulations with an oracle agent, we \cp{first compare our system against existing proactive querying systems} based on conformal prediction and expected task reward. We then conduct physical robot experiments \cp{with VLM and human feedback} to compare our method with a baseline based on conformal prediction. Our method achieves the highest success rates \cp{among the evaluated systems} in both simulation and physical robot experiments with VLM and human feedback. \cp{We then conduct ablation experiments with the same POMCP action planner to distinguish the effects of query timing and selection on task success and query count.} Ablation experiments show that \cp{query timing based on confidence maintains high success across selection strategies}, while \cp{query selection based on expected entropy reduction} reduces the number of queries \cp{under} the same timing criterion. We also vary the confidence threshold to examine the \cp{tradeoff} between task success and query count. Finally, interviews with all five participants help explain the differences in task performance between the two methods.



본 연구의 기여는 다음과 같다.

- Belief 불확실성으로부터 질의 시점을 결정하여 작업 성공을 개선한다.
- 기대 엔트로피 감소량으로 무엇을 질의할지 선택하여, 고정된 시점 기준에서 질의 수를 줄인다.
- 시뮬레이션과 VLM 및 사람 피드백을 사용하는 실제 로봇 실험에서 평가한 baseline들보다 높은 작업 성공률을 보인다.

## 2. 관련 연구

기호적 계획에서 상태 불확실성을 표현하는 연구와 불확실성하에서 언제 무엇을 질의할지 결정하는 연구를 검토한다.

### 2.1. 기호적 계획에서의 상태 불확실성

본 연구는 현재 작업 상태에서 어떤 사실이 성립하는지에 관한 불확실성을 고려한다. 인식 오류와 불확실한 행동 결과 때문에 여러 상태가 주어진 관측과 일치할 수 있다. 기호적 계획은 각 상태를 사실의 집합으로 표현하고, 전제조건과 효과로 행동을 기술한다 [fikes1971strips, mcdermott1998pddl]. 고전적 계획은 현재 상태가 알려져 있고 행동 결과가 결정적이라고 가정한다. 확률적 PDDL(PPDDL)은 확률적 행동 효과와 보상으로 이 표현을 확장한다 [younes2004ppddl1]. Relational Dynamic Influence Diagram Language(RDDL)는 부분 관측 도메인에서 확률적 상태 전이와 관측 모델을 기술한다 [sanner2010rddl]. POMDP는 가능한 현재 상태들의 불확실성을 belief로 표현하고, 행동 선택 시 예상되는 관측을 활용한다 [kaelbling1998planning]. 이러한 연구 방향에서 Zhang 등의 연구 [zhang2015corpp]는 상식 추론으로 POMDP 계획의 사전 belief를 구성하고, Sanner 등의 연구 [sanner2010symbolic]는 기호적 동적 계획법에 일차 논리 구조를 활용한다. 본 연구는 실행 중 기호적 사실과 행동 모델로 도달 가능한 후속 상태들에 대한 가중 belief를 구성한다.

큰 상태 공간에서 모든 가능한 상태에 대한 정확한 belief를 유지하는 것은 계산적으로 다루기 어려운 경우가 많다. Particle 표현은 유한한 수의 샘플링된 상태로 belief를 근사한다 [thrun2005probabilistic]. 온라인 POMDP 방법은 전체 belief 공간의 정책을 계산하지 않고 현재 belief에서 계획함으로써 계산량을 줄인다 [ross2008online]. Partially Observable Monte-Carlo Planning(POMCP)은 Monte Carlo belief 갱신과 Monte Carlo tree search를 결합하여 온라인으로 행동을 선택한다 [silver2010monte]. Xiao 등의 연구 [xiao2019objectsearch]는 인식 및 조작 불확실성이 있는 복잡한 환경에서 목표 물체를 탐색하도록 POMCP를 확장한다. 그러나 기존 온라인 POMDP 계획 연구는 외부 에이전트에게 명시적으로 질의하지 않은 채 상태 불확실성하에서 행동을 선택하는 데 초점을 맞춘다. 본 연구는 행동 선택과 질의를 통합하는 방법을 제안한다.

### 2.2. 불확실성하의 질의 시점과 선택

기존 방법들은 불확실성 추정치, 사람의 신뢰, 후보 행동 간 모호성을 바탕으로 질의 시점을 결정한다. 일부 방법은 작업 수익의 불확실성 [singi2023hula]이나 시각 기반 주행 정책에서 학습한 암묵적 불확실성 추정치 [igbinedion2024learning]를 사용한다. Mangalindan 등의 연구 [mangalindan2025seek]는 로봇에 대한 사람의 신뢰를 잠재 POMDP 상태로 모델링하고, 팀 성능을 위한 도움 요청을 선택할 때 신뢰 모델을 사용한다. 온라인 conformal prediction은 상호작용 모방 학습에서 전문가 피드백 요청을 안내한다 [zhao2025conformalized]. KnowNo는 conformal prediction 집합에 여러 후보가 있으면 행동 선택을 요청하고 [ren2023robots], IntroPlan은 conformal prediction과 검색된 자기성찰적 추론 예시를 결합한다 [liang2024introspective]. KnowNo와 IntroPlan의 질의 시점은 기호적 작업 상태에 대한 명시적 belief 없이 후보 행동 사이의 모호성에 의존한다.

POMDP 기반 방법은 정보 요청을 계획 행동으로 모델링하고, 기대 누적 보상을 통해 질의 시점과 선택을 결정한다 [armstrongcrews2007oracular, burks2021collaborative, burks2023harps]. Oracular POMDP는 고정 비용으로 정확한 상태를 알려주는 oracle을 고려한다 [armstrongcrews2007oracular]. 사람 보조 의미적 센싱은 사람의 관측을 belief에 통합하고 계획 목적에 따라 질의를 선택한다 [burks2021collaborative, burks2023harps]. 따라서 질의 비용과 작업 보상이 플래너의 정보 요청 여부와 사용할 질의를 결정한다. 제안 방법은 행동 계획의 목적과 독립적으로 belief confidence 임계값을 사용하여 확인이 필요한 시점을 결정한다.

질의 선택은 작업에 필요한 정보에 따라 달라진다. Kaipa 등의 연구 [kaipa2016addressing]는 bin-picking에서 사람의 입력으로 인식 불확실성을 해소한다. Deits 등의 연구 [deits2013clarifying]는 자연어 명령과 환경 내 개체 사이의 대응 관계에 대한 엔트로피를 줄이는 명확화 질문을 선택한다. Anand 등의 연구 [anand2025adaptive]는 피드백 비용과 가용성을 고려하면서 페널티 함수 학습의 중요한 상태에서 정보 이득으로 피드백 형식을 선택한다. 제안 방법은 확인했을 때 belief 엔트로피의 기대 감소량이 가장 큰 상태 사실을 선택한다. 외부 에이전트는 상태 사실을 확인하고, 플래너는 행동을 선택한다.

## 3. 사전 지식

기호적 계획 문제는 튜플 $\mathcal{P}=\langle\mathcal{F},\mathcal{A},s_0,\mathcal{G}\rangle$로 정의한다. $\mathcal F$는 유한한 사실 집합, $\mathcal A$는 행동 집합, $s_0\subseteq\mathcal F$는 초기 상태, $\mathcal G\subseteq\mathcal F$는 목표 조건이다. 각 상태는 $\mathcal F$의 부분집합이다. 행동 $a\in\mathcal A$는 전제조건 $\mathrm{pre}(a)$, 추가 효과 $\mathrm{add}(a)$, 삭제 효과 $\mathrm{del}(a)$를 갖는다. $\mathrm{pre}(a)\subseteq s$이면 상태 $s$에서 행동 $a$를 적용할 수 있으며 이를 $a\llbracket s\rrbracket$로 표기한다. 적용 결과는 $s'=(s\setminus\mathrm{del}(a))\cup\mathrm{add}(a)$이다. 계획 $\pi=(a_1,\ldots,a_T)$는 $s_0$에서 행동을 적용하여 $\mathcal G\subseteq s_T$를 만족하는 상태 $s_T$에 도달한다.

부분 관측 마르코프 의사결정 과정(POMDP)은 튜플 $\langle\mathcal S,\mathcal A,T,R,\Omega,O,\gamma\rangle$이다. $\mathcal S$, $\mathcal A$, $\Omega$는 각각 상태, 행동, 관측 공간이다. 모델 $T(s'\mid s,a)$와 $O(o\mid s',a)$는 전이 확률과 관측 확률을 제공하고, $R(s,a)$는 즉시 보상이며, $\gamma\in(0,1]$는 미래 보상의 할인율이다. Belief $b$는 상태에 대한 확률분포이다. 행동 $a$와 관측 $o$ 이후 Bayes filter는 다음과 같이 belief를 갱신한다.

$$
b'(s')=\eta\,O(o\mid s',a)\sum_{s\in\mathcal S}T(s'\mid s,a)b(s).
$$

여기서 $\eta$는 분포를 정규화한다. 정책은 기대 할인 보상 $\mathbb E[\sum_{t=0}^{\infty}\gamma^t r_t]$을 최대화하도록 belief를 행동에 대응시킨다.

POMCP는 Monte Carlo tree search와 각 이력의 particle belief $B(h)$를 이용하여 행동–관측 이력 $h=\langle a_1o_1\cdots a_to_t\rangle$를 탐색한다. 다음 upper confidence bound(UCB) 규칙으로 행동을 선택한다.

$$
a^*=\arg\max_a\left(V(ha)+c\sqrt{\frac{\log N(h)}{N(ha)}}\right). \tag{1}
$$

$V(ha)$는 추정 행동 가치, $N(h)$와 $N(ha)$는 방문 횟수이며, $c$는 탐색을 조절한다. 시뮬레이션에서는 생성 모델로부터 전이, 관측, 보상을 샘플링한다. 할인 수익 $R=r+\gamma R'$를 사용하여 $V(ha)\leftarrow V(ha)+(R-V(ha))/N(ha)$로 가치를 갱신한다. 각 이력에서 샘플링한 상태들은 $B(h)$를 구성하여, 탐색 트리 내부의 명시적 Bayes 갱신 없이 사후분포를 근사한다.

## 4. 방법

먼저 상태 불확실성으로 언제 무엇을 질의할지 결정하는 온라인 계획 및 질의 절차를 설명한다. 이어서 로봇 실행을 위한 시스템 구현을 제시한다.

**그림 1. 온라인 계획과 질의.** 행동 실행과 관측 후 로봇은 후보 상태들에 대한 frontier belief를 구성한다. Belief confidence가 $\tau$보다 낮으면 기대 사후 엔트로피를 최소화하는 사실 $q^*$를 질의한다. 불리언 응답 $y\in\{\texttt{true},\texttt{false}\}$로 일치하지 않는 상태를 걸러내고 나머지 가중치를 정규화한다. Confidence 임계값을 충족할 때까지 질의를 반복한다. 이후 로봇은 MAP 상태를 지식 베이스에 확정하여 반영하고 다음 계획 단계로 진행한다.

[그림 1 원본](latex_zips/BRL_achitecture_changmin/figures/fig_method.pdf)

### 4.1. 계획과 질의

기호적 계획과 실행에는 지식 상태 $s_t^K$를 사용한다. 온라인 행동 선택에는 POMCP [silver2010monte]를 사용하며, 트리 확장과 rollout을 적용 가능한 행동으로 제한한다. 각 단계에서 플래너는 $s_t^K$에 적용 가능한 행동 $a_t\in\mathcal A$를 선택하며 이를 $a_t\llbracket s_t^K\rrbracket$로 표기한다. $a_t$와 $o_{t+1}$ 이후에는 $(s_t^K,a_t)$로 정의되는 도달 가능 상태 frontier에서 샘플링하여 가중 belief $\mathcal B_{t+1}$를 구성한다. 이를 통해 행동 결과의 불확실성을 포착하면서 belief 구성을 기호적으로 유효한 후속 상태로 제한한다.

#### 4.1.1. 언제 질의할 것인가

질의 시점을 결정하기 위해 가중 belief의 엔트로피로 현재 상태의 불확실성을 평가한다. 행동 후에도 여러 후속 상태가 타당하게 남아 있고 어느 한 상태가 우세하지 않으면 로봇은 현재 상태를 확신 있게 식별할 수 없다. 이 경우 여러 가능한 상태에 가중치가 분산되므로 belief 엔트로피가 높다. 이러한 성질을 사용하여 정규화된 belief 엔트로피로부터 confidence를 계산한다 [qu2023stochastic]. Confidence가 낮을수록 상태 불확실성이 크다.

기호적 행동은 일반적으로 상태의 일부만 바꾼다는 점에 주목한다. 예를 들어 토마토를 집는 행동은 로봇이 어떤 물체를 잡고 있는지는 바꿀 수 있지만 로봇 위치는 유지한다. 이를 바탕으로 도달 가능한 후속 상태의 **frontier**를 다음과 같이 정의한다.

$$
\mathrm{Frontier}(s_t^K,a_t)=\{s_{t+1}^{1},\ldots,s_{t+1}^{j}\}.
$$

Frontier 내 불확실한 결과들은 상호 배타적이며, 동일한 후속 상태는 없다. 전체 기호적 상태 공간 대신 이 frontier로 particle 샘플링을 제한한다.

그림 1과 같이 $j$개 frontier 상태에 정규화된 가중치 $w_k$를 부여하여 가중 belief를 구성한다. 여기서 $\mathcal B_{t+1}=\{(s_{t+1}^k,w_k)\}_{k=1}^{j}$이고 $\sum_{k=1}^{j}w_k=1$이다. 구체적으로 $s_t^K$로부터의 전이 확률과 관측 likelihood의 곱을 정규화하여 가중치를 계산한다.

$$
\begin{aligned}
\bar w_k&=T(s_{t+1}^k\mid s_t^K,a_t)\,O(o_{t+1}\mid s_{t+1}^k,a_t),\\
w_k&=\bar w_k/\sum_{\ell=1}^{j}\bar w_\ell.
\end{aligned}\tag{2}
$$

모든 $\bar w_k$가 0이면 $w_k=1/j$로 설정한다. Confidence 계산과 질의 선택에는 이 가중 frontier belief를 사용한다. POMCP는 행동 가치를 추정하기 위해 탐색 트리 안에 샘플링된 상태들을 유지한다. 두 구성 요소는 동일한 기호적 전이 모델과 관측 모델을 사용한다.

Belief 엔트로피는 $\mathcal H(\mathcal B_{t+1})=-\sum_{k=1}^{j}w_k\log_2w_k$로 계산한다. 최대 엔트로피는 표현된 가능한 세계의 수에 따라 달라지므로 $\log_2j$로 정규화하여 다음과 같이 정의한다.

$$
\mathrm{Conf}(\mathcal B_{t+1})=1-\frac{\mathcal H(\mathcal B_{t+1})}{\log_2j}.\tag{3}
$$

$j=1$이면 $\mathrm{Conf}(\mathcal B_{t+1})=1$로 정의한다. 표현된 상태들의 가중치가 비슷하면 점수는 0에 가까워지고, belief 가중치 대부분이 하나의 상태에 있으면 1에 가까워진다.

미리 정의한 임계값 $\tau\in[0,1]$에 대해 $\mathrm{Conf}(\mathcal B_{t+1})<\tau$이면 로봇이 질의를 시작한다. $\tau$가 높을수록 더 집중된 belief를 요구하며 질의 수가 증가할 수 있다.

#### 4.1.2. 무엇을 질의할 것인가

Belief confidence가 $\tau$보다 낮으면 로봇은 질의를 시작하고 확인할 사실을 결정해야 한다. 확인은 응답과 일치하지 않는 상태 가설을 제거하므로, belief 엔트로피의 기대 감소량이 가장 큰 사실을 선택한다.

**알고리즘 1. 계획과 질의**

```text
입력: 초기 상태 s₀ᴷ, 목표 조건 G, confidence 임계값 τ, 단계 제한 T_max
출력: 실행한 행동 시퀀스 π

 1. B ← {(s₀ᴷ, 1)}, t ← 0, π ← ()
 2. while t < T_max 이고 G ⊈ s_tᴷ:
 3.     a_t ← SEARCH(B)
 4.     if a_t = ∅:
 5.         break
 6.     a_t를 실행하고 o_(t+1)을 수신
 7.     π ← π || a_t
 8.     B ← UPDATE(s_tᴷ, a_t, o_(t+1))                식 (2)
 9.     c ← Conf(B)                                    식 (3)
10.     while c < τ:
11.         q* ← SELECT-FACT(B)                         식 (4)
12.         if q* = ∅:
13.             break
14.         y ← QUERY(q*)
15.         B ← FILTER(B, q*, y)                        식 (5)
16.         c ← Conf(B)                                식 (3)
17.     s_(t+1)ᴷ ← argmax_(sᵏ ∈ B) w_k
18.     t ← t + 1
19. return π
```

구체적으로 belief에 남은 frontier 상태들 사이에서 참·거짓이 달라지는 원자적 사실을 추출하여 질의 후보 집합을 정의한다.

$$
\mathcal Q(\mathcal B)=\left(\bigcup_{(s,w)\in\mathcal B}s\right)\setminus\left(\bigcap_{(s,w)\in\mathcal B}s\right).
$$

각 후보 사실 $q\in\mathcal Q(\mathcal B)$에 대해 각 상태에서 $q$가 성립하는지에 따라 belief를 분할한다.

$$
\mathcal B^{q+}=\{(s^k,w_k)\mid q\in s^k\},\qquad
\mathcal B^{q-}=\{(s^k,w_k)\mid q\notin s^k\}.
$$

$P(q)=\sum_{k:q\in s^k}w_k$, $P(\neg q)=1-P(q)$로 둔다. 각 부분집합의 엔트로피는 가중치를 조건부로 정규화한 후 계산한다. $q$를 질문한 후의 기대 사후 엔트로피는 $\mathbb E[\mathcal H\mid q]=P(q)\mathcal H(\mathcal B^{q+})+P(\neg q)\mathcal H(\mathcal B^{q-})$이다. 선택하는 질의는 다음과 같다.

$$
q^*=\arg\min_{q\in\mathcal Q(\mathcal B)}\left[P(q)\mathcal H(\mathcal B^{q+})+P(\neg q)\mathcal H(\mathcal B^{q-})\right].\tag{4}
$$

식 (4)는 기대 엔트로피 감소량을 최대화하는 사실 $q^*$를 선택한다. 로봇은 이 사실을 현재 상태에 관한 이진 질문으로 변환한다. 예를 들어 $q^*=\texttt{ripe}(\text{tomato}_1)$은 “토마토 1은 익었나요?”라는 질문이 된다. 응답 $y\in\{\texttt{true},\texttt{false}\}$에 따라 보고된 진릿값과 일치하지 않는 상태를 제거하여 belief를 갱신한다.

$$
\mathcal B\leftarrow
\begin{cases}
\{(s^k,w_k)\mid q^*\in s^k\},&y=\texttt{true},\\
\{(s^k,w_k)\mid q^*\notin s^k\},&y=\texttt{false}.
\end{cases}\tag{5}
$$

로봇은 유지한 가중치를 재정규화하고 confidence를 다시 계산하며, 여전히 $\tau$보다 낮으면 다른 사실을 선택한다. Confidence가 $\tau$에 도달하거나 상태를 구별하는 사실이 남지 않으면 확인 루프를 종료한다. 이후 최대 사후 확률 상태(MAP) $s_{t+1}^K=\arg\max_{s^k\in\mathcal B_{t+1}}w_k$를 다음 기호적 계획에 사용할 상태로 확정한다. Belief 갱신의 계산 비용을 제한하기 위해 다음 frontier는 확정한 지식 상태와 다음 행동으로부터 구성한다.

알고리즘 1은 전체 온라인 계획과 질의 절차를 설명한다. 로봇은 먼저 belief, 단계 카운터, 행동 시퀀스를 초기화한다(1행). $t<T_{\max}$이고 목표 조건이 만족되지 않는 동안, 즉 $\mathcal G\nsubseteq s_t^K$인 동안 실행을 계속한다(2행). 각 단계에서 POMCP가 행동을 선택하고(3행), 사용할 행동이 없으면 실행을 중단한다(5행). 그렇지 않으면 행동을 실행하고 기록한 뒤(6–7행), 가중 frontier belief를 구성하고 confidence를 계산한다(8–9행). Confidence가 $\tau$보다 낮으면 후보 사실을 선택한다(11행). 후보가 없으면 질의 루프를 종료한다(13행). 그렇지 않으면 외부 에이전트에게 질의하여 응답을 얻고 belief를 필터링한 뒤(14–15행), confidence를 다시 계산한다(16행). 질의 루프 이후 가장 확률이 높은 잔여 상태가 다음 기호적 계획 상태가 된다(17행).

### 4.2. 시스템 구현

**그림 2. 제안한 계획 및 질의 방법의 로봇 실행 프레임워크.** 지식 베이스(KB)는 공유 기호적 사실과 행동 모델을 저장하고, 센싱 및 인식(SP)은 로봇 센서의 관측을 제공한다. 계획 관리자(PM)는 행동을 선택하고 질의를 결정한다. Belief confidence가 $\tau$보다 낮으면 피드백 관리자(FM)가 선택한 사실을 질문으로 변환하고, 사용자 인터페이스(UI)가 이를 사람에게 제시한다. FM은 응답을 불리언 증거로 변환하며 PM은 이를 사용하여 belief를 갱신하고 계획을 계속한다.

[그림 2 원본](latex_zips/BRL_achitecture_changmin/figures/fig_architecture.pdf)

그림 2와 같이 제안한 계획 및 질의 방법을 로봇에서 실행하기 위한 모듈형 프레임워크를 구현한다. KB는 기호적 사실과 행동 모델을 저장하고, SP는 센서 데이터에서 관측과 수치적 상태 추정치를 제공한다. PM은 이 정보로 로봇 행동을 선택하고 frontier belief를 구성하며 언제 무엇을 질의할지 결정한다. FM은 선택한 사실을 질문으로, 응답을 belief 갱신 증거로 변환한다. UI는 사람과 상호작용할 때 질문을 제시하고 응답을 수집한다.

각 기호적 행동은 로봇 스킬을 호출한다. $\texttt{navigate}(\text{robot},l_1,l_2)$와 $\texttt{pick}(\text{robot},\text{tomato}_1)$ 같은 행동은 작업 수준의 의도를 지정하고, MoveIt [chitta2012moveit]과 같은 기존 모션 플래너는 KB의 수치적 fluent를 이용하여 저수준 모션을 생성한다.

FM은 선택한 술어를 미리 정의된 질문 템플릿에 대응시키고 응답을 불리언 값으로 변환한다. 예를 들어 술어 $\mathrm{ripe}(t_1)$은 “토마토 $t_1$은 익었나요?”로 변환된다. 사람 참가자와 상호작용할 때 태블릿 UI는 카메라 화면을 표시하고, 해당 물체를 바운딩 박스로 강조하며, 간단한 응답 선택지를 제시한다. 참가자는 표시된 정보를 바탕으로 응답하고, 플래너는 이에 따라 belief를 갱신하고 다음 로봇 행동을 선택한다.

## 5. 실험

Baseline 비교, ablation 연구, confidence 임계값 변화 실험을 통해 분리수거(*Sort*)와 토마토 수확(*Harvest*)에서 제안 방법을 평가한다. Baseline 비교는 외부 에이전트에게 질의하는 완전한 계획 시스템들을 평가한다. Oracle 시뮬레이션에서는 Ours를 KnowNo, IntroPlan, POMCP-Q와 비교한다. 실제 로봇 실험에서는 VLM과 사람 피드백을 사용하여 Ours와 KnowNo를 비교하고, 피드백 제공자가 달라져도 성능 경향이 유지되는지 살펴본다. Ablation 연구에서는 시뮬레이션으로 질의 시점과 선택의 기여를 살펴본다. 임계값 변화 실험에서는 작업 성공과 질의 수의 절충 관계를 살펴본다.

Oracle 실험은 도메인 및 조건마다 200회 시행한다. VLM 실험은 도메인 및 방법마다 40회 시행한다. 사람 평가는 참가자 다섯 명으로 구성되며, 도메인 및 방법마다 10회, 총 40회 시행한다.

### 5.1. 실험 설정

Oracle 실험은 시뮬레이션 작업 실행과 정확한 응답을 사용한다. VLM 및 사람 실험은 실험실 환경의 실제 로봇을 사용한다. 그림 3은 실제 실험 구성을 보여준다. Sort는 고정형 UR5e 로봇 팔을 사용한다. Harvest는 UR5e 로봇 팔을 장착한 Summit 모바일 플랫폼을 사용한다. 모바일 플랫폼은 인식을 위해 RGB-D 카메라를, 위치 추정을 위해 LiDAR를 사용한다. 물체 탐지에는 Sort에서 SAM3 [carion2025sam3segmentconcepts], Harvest에서 YOLOv12 [tian2025yolov12]를 사용한다.

**그림 3. 두 작업 도메인의 실제 로봇 실험 구성.** (a) Sort에서는 고정형 UR5e 팔이 물체를 적절한 수거함으로 분류한다. (b) Harvest에서는 UR5e 팔을 장착한 Summit 모바일 플랫폼이 토마토 줄기 사이를 이동하며 토마토를 수확한다. 주석은 로봇 구성 요소, 센서, 작업 장비를 표시한다.

- [(a) Sort](latex_zips/BRL_achitecture_changmin/figures/03_SETUP/01_WasteSortingEnv.pdf)
- [(b) Harvest](latex_zips/BRL_achitecture_changmin/figures/03_SETUP/02_TomatoHarvestEnv.pdf)

#### 5.1.1. 작업 도메인

서로 다른 불확실성 원인을 가진 두 도메인을 평가한다(그림 3). Sort에서 로봇은 물체를 탐지하고 집어서 네 가지 폐기물 범주에 해당하는 수거함에 놓는다. 물체 범주는 처음에 알려져 있지 않으며, 가림과 잡음이 있는 관측으로 물체의 존재와 범주가 불확실해진다. Harvest에서 로봇은 두 줄기 사이를 이동하며 신선하고 익은 토마토를 수확하고 썩은 토마토를 버리며, 익지 않은 토마토는 그대로 둔다. 탐지는 익음 여부를 추정하고, 집은 뒤 수행하는 스캔은 신선한 토마토와 썩은 토마토를 구별한다. 잡음이 있는 관측으로 토마토 상태에 불확실성이 발생한다.

#### 5.1.2. 구현 세부 사항

제안 POMCP 플래너는 계획 단계마다 100회 시뮬레이션을 수행하고 최대 시뮬레이션 깊이는 20이다. 할인율은 $\gamma=0.2$, UCB 탐색 상수는 $c=1.0$이다. 기본 confidence 임계값은 $\tau=0.8$이다. VLM 실험의 외부 에이전트로는 NVFP4 양자화를 적용한 Gemma 4 31B IT [gemmateam2026gemma4]를 사용한다. 부록 A에 baseline 설정과 추가 실험 세부 사항을 제공한다.

#### 5.1.3. 평가 지표

성공률은 전체 시행 중 로봇이 작업 목표를 달성한 시행의 비율이다. Oracle 실험에서는 시행당 최대 20회의 물리적 행동을 허용한다. POMCP-Q에는 시행당 최대 30회의 질의도 허용한다. 질의 수는 시행당 질문 횟수이다. 질의 확률은 한 번 이상의 질의가 발생한 물리적 단계의 비율이다. 계획 길이는 시행당 실행한 물리적 행동 수이다. 계획 시간은 시행당 계획 탐색에 소요된 누적 시간이다. 상호작용 시간은 질의와 응답에 소요된 누적 시간이다. 총시간은 계획, 로봇 실행, 상호작용을 포함하여 실제 로봇의 각 시행에 소요된 경과 시간으로 측정한다. Oracle 시행에서는 계획 시간을 총시간으로 보고한다. 상호작용 시간은 VLM 및 사람 실험에서만 보고한다. 성공률을 제외한 모든 지표는 성공한 시행들의 평균과 표본 표준편차로 보고한다.

### 5.2. 시스템 평가

**표 1. 질의 시점(when)과 질의 선택(what) 전략.** KnowNo, IntroPlan, POMCP-Q는 baseline이다. R, S1–S2, T1–T3는 ablation 구성이다. 대시는 본 연구에서 평가하지 않은 조합이다.

| 선택(what) / 시점(when) | 무작위 | Conformal prediction | 보상 기반 | Confidence |
|---|---|---|---|---|
| 무작위 | R | — | — | S1 |
| Conformal 집합 | — | KnowNo, IntroPlan | — | — |
| 보상 기반 | — | — | POMCP-Q | S2 |
| 엔트로피 이득 | T1 | T2 | T3 | **Ours** |

표 1은 baseline과 ablation 구성을 요약한다. 5.2.1절의 baseline 비교는 Ours를 KnowNo, IntroPlan, POMCP-Q와 비교한다. 이 비교는 행동을 계획하고 실행하기 위해 외부 에이전트에게 질의하는 완전한 작업 계획 시스템들을 평가한다. Oracle, VLM, 사람 피드백을 사용하여 피드백 제공자에 걸쳐 관찰된 성능 경향이 유지되는지 살펴본다. 5.2.2절의 ablation 연구는 Ours를 R, S1–S2, T1–T3와 비교한다. T1–T3는 질의 시점(when)을, S1–S2는 질의 선택(what)을 살펴본다. 마지막으로 5.2.3절에서는 confidence 임계값이 작업 성공과 질의 수에 미치는 영향을 살펴본다. Ablation과 임계값 연구는 모두 시뮬레이션에서 oracle 응답을 사용한다.

#### 5.2.1. Baseline 비교

**Oracle.** Oracle 비교는 표 1에 요약한 질의 시점과 선택 전략을 갖는 세 baseline을 포함한다. KnowNo [ren2023robots]는 conformal prediction 집합이 하나의 일반 행동을 지정하지 않으면 행동 선택을 요청한다. IntroPlan [liang2024introspective]은 KnowNo를 확장하며, 행동 선택을 요청하기 전에 검색한 예시와 설명을 후보 평가에 통합한다. 또한 기대 누적 보상을 최대화하여 물리적 행동과 외부 에이전트에 대한 불리언 질의 중 하나를 선택하는 HARPS 기반 POMCP baseline(POMCP-Q)을 구현한다 [burks2023harps]. HARPS를 본 연구의 기호적 작업 도메인에 맞게 조정하고 물리적 행동과 질의를 별도로 실행한다. 표 2와 그림 4는 네 방법을 비교한다.

**그림 4. Oracle baseline 비교.** (a), (e) Ours는 두 도메인에서 가장 높은 성공률을 달성한다. (b)–(c), (f)–(g) Ours는 KnowNo와 IntroPlan보다 많은 질의를 사용하지만 POMCP-Q보다는 적게 사용하며, 질의 확률은 IntroPlan과 POMCP-Q보다 낮다. (d), (h) Ours는 두 도메인에서 가장 짧은 계획 시간을 보인다. 이러한 결과는 질의 수만으로 작업 성공이나 계획 시간을 설명할 수 없음을 나타낸다.

| 패널 | 원본 그림 |
|---|---|
| (a) Sort 성공률(%) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_waste_success.pdf) |
| (b) Sort 질의 수 | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_waste_queries.pdf) |
| (c) Sort 질의 확률(%) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_waste_query_probability.pdf) |
| (d) Sort 계획 시간(초) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_waste_plan_time.pdf) |
| (e) Harvest 성공률(%) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_tomato_success.pdf) |
| (f) Harvest 질의 수 | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_tomato_queries.pdf) |
| (g) Harvest 질의 확률(%) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_tomato_query_probability.pdf) |
| (h) Harvest 계획 시간(초) | [PDF](latex_zips/BRL_achitecture_changmin/figures/04_ORACLE_BASELINE/exp_oracle_tomato_plan_time.pdf) |

표 2와 그림 4와 같이 Ours의 성공률은 Sort에서 100.00%, Harvest에서 97.50%이며, POMCP-Q의 76.00%, 56.00%보다 높고 두 도메인 모두에서 더 적은 질의를 사용한다. POMCP-Q는 행동 공간에 질의를 포함하므로 고정된 시뮬레이션 예산 안에서 평가할 대안이 증가한다. 보상 기준도 현재 상태의 confidence를 명시적으로 평가하지 않는다. 이러한 차이는 낮은 성공률과 높은 질의 수에 기여할 수 있다. Ours는 관측 후 상태 사실을 확인하고 행동 계획을 물리적 행동으로 제한한다.

Ours는 belief confidence가 $\tau$에 도달할 때까지 확인을 계속하지만 KnowNo와 IntroPlan은 trigger마다 한 번의 행동 선택을 요청하므로, Ours의 질의 수가 더 많다. 이 차이는 Ours가 IntroPlan보다 낮은 질의 확률을 보이면서도 전체 질문 수는 더 많을 수 있는 이유를 설명한다. 비슷한 질의 수에서도 결과는 크게 달라진다. Harvest에서 IntroPlan과 Ours는 각각 8.96회와 9.16회 질의하지만 성공률은 40.00%와 97.50%이다. 따라서 질의 수만으로 작업 성공을 설명할 수 없다.

Ours의 계획 시간도 두 도메인 모두에서 가장 짧다. KnowNo와 IntroPlan은 후보 생성과 점수 계산을 위해 언어 모델을 호출하며, IntroPlan은 후보 평가를 위한 설명도 추가로 생성한다. Ours는 belief 상태로부터 질의 시점과 선택을 계산하고 물리적 행동만 탐색한다. 따라서 반복적인 언어 모델 추론을 피하고 POMCP-Q보다 탐색 공간을 제한한다.

**표 2. Baseline 결과.** 밑줄은 가장 좋은 평균을 나타낸다. `±` 뒤의 값은 표준편차이다. 표는 가독성을 위해 조건별로 나누어 표시한다.

**Oracle — Sort**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 48.50 | <u>2.48</u> ± 1.42 | <u>22.82</u> ± 12.61 | 10.78 ± 1.35 | 22.54 ± 3.04 | — | 22.54 ± 3.04 |
| IntroPlan | 51.50 | 6.83 ± 1.53 | 61.97 ± 9.35 | 10.99 ± 1.49 | 44.43 ± 8.05 | — | 44.43 ± 8.05 |
| POMCP-Q | 76.00 | 23.20 ± 4.39 | 86.90 ± 5.29 | 11.72 ± 1.49 | 10.16 ± 1.36 | — | 10.16 ± 1.36 |
| **Ours** | **<u>100.00</u>** | **7.83 ± 2.16** | **30.55 ± 10.41** | **<u>10.35</u> ± 0.88** | **<u>0.91</u> ± 0.09** | — | **<u>0.91</u> ± 0.09** |

**Oracle — Harvest**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 43.00 | <u>2.45</u> ± 1.06 | <u>17.27</u> ± 6.67 | 14.01 ± 1.07 | 20.72 ± 2.25 | — | 20.72 ± 2.25 |
| IntroPlan | 40.00 | 8.96 ± 2.04 | 61.13 ± 10.77 | 14.61 ± 1.46 | 73.80 ± 8.96 | — | 73.80 ± 8.96 |
| POMCP-Q | 56.00 | 11.09 ± 3.59 | 51.72 ± 10.50 | 14.77 ± 1.36 | 5.46 ± 1.11 | — | 5.46 ± 1.11 |
| **Ours** | **<u>97.50</u>** | **9.16 ± 1.91** | **40.14 ± 4.50** | **<u>13.93</u> ± 1.03** | **<u>1.10</u> ± 0.16** | — | **<u>1.10</u> ± 0.16** |

**VLM — Sort**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 50.00 | <u>3.15</u> ± 0.88 | 28.64 ± 7.96 | <u>11.00</u> ± 0.00 | <u>186.51</u> ± 10.90 | <u>15.46</u> ± 4.45 | 19.99 ± 4.76 |
| **Ours** | **<u>92.50</u>** | **6.70 ± 1.76** | **<u>19.81</u> ± 3.27** | **11.24 ± 0.49** | **193.69 ± 13.52** | **26.83 ± 7.76** | **<u>0.99</u> ± 0.05** |

**VLM — Harvest**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 25.00 | <u>7.20</u> ± 1.87 | 44.10 ± 7.51 | 16.10 ± 1.73 | 217.82 ± 17.78 | 44.99 ± 13.10 | 26.90 ± 3.07 |
| **Ours** | **<u>92.50</u>** | **8.51 ± 1.52** | **<u>37.63</u> ± 3.04** | **<u>13.27</u> ± 0.77** | **<u>180.39</u> ± 11.18** | **<u>34.12</u> ± 7.62** | **<u>1.38</u> ± 0.19** |

**사람 — Sort**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 20.00 | <u>3.00</u> ± 1.41 | 25.76 ± 10.71 | 11.50 ± 0.71 | <u>184.48</u> ± 24.08 | <u>11.59</u> ± 7.94 | 16.94 ± 0.39 |
| **Ours** | **<u>90.00</u>** | **7.11 ± 1.83** | **<u>18.18</u> ± 0.00** | **<u>11.00</u> ± 0.00** | **190.15 ± 50.72** | **36.49 ± 47.55** | **<u>0.94</u> ± 0.04** |

**사람 — Harvest**

| 방법 | 성공률(%) | 질의 수 | 질의 확률(%) | 계획 길이 | 총시간(초) | 상호작용 시간(초) | 계획 시간(초) |
|---|---|---|---|---|---|---|---|
| KnowNo | 30.00 | <u>3.67</u> ± 0.58 | <u>28.21</u> ± 4.44 | <u>13.00</u> ± 0.00 | 195.50 ± 8.26 | <u>19.04</u> ± 8.16 | 23.52 ± 1.21 |
| **Ours** | **<u>100.00</u>** | **8.50 ± 1.78** | **37.36 ± 3.74** | **13.10 ± 0.32** | **<u>180.02</u> ± 13.14** | **23.89 ± 13.47** | **<u>1.32</u> ± 0.08** |

**VLM.** 실제 로봇에서 VLM 피드백을 사용했을 때 oracle 실험의 경향이 유지되는지 평가한다. VLM 프롬프트는 부록 B에 제시한다. KnowNo는 oracle 실험에서 IntroPlan과 비슷한 성공률을 보이면서 IntroPlan이나 POMCP-Q보다 적은 질의를 사용하므로 비교 대상으로 선택한다. 실제 실험에서는 Ours와 KnowNo의 작업 성공과 상호작용 비용을 비교한다. 방법 및 도메인마다 40회 시행한다. 그림 5에 결과를 제시한다.

**그림 5. VLM 및 사람 응답을 사용한 실제 로봇 평가.** (a) VLM 피드백에서 Ours는 두 도메인 모두 KnowNo보다 높은 작업 성공률을 달성한다. (b) 사람 피드백에서도 같은 성공률 우위가 유지된다.

- [(a) VLM 응답](latex_zips/BRL_achitecture_changmin/figures/05_PHYSICAL_BASELINE/exp_physical_vlm_group.pdf)
- [(b) 사람 응답](latex_zips/BRL_achitecture_changmin/figures/05_PHYSICAL_BASELINE/exp_physical_human_group.pdf)
- [범례](latex_zips/BRL_achitecture_changmin/figures/05_PHYSICAL_BASELINE/exp_physical_legend.pdf)

Ours는 두 도메인 모두 92.50%의 성공률을 달성하며, KnowNo는 Sort에서 50.00%, Harvest에서 25.00%를 달성한다. Ours는 두 도메인 모두 더 짧은 탐색 시간을 필요로 한다. 성공한 Harvest 시행에서 KnowNo는 Ours의 8.51회보다 적은 7.20회 질의하지만, 더 긴 계획을 실행하고 총시간도 더 오래 걸린다. 평균 총시간은 KnowNo가 217.82초, Ours가 180.39초이다. Sort에서 Ours는 성공률이 더 높지만 성공한 시행에 걸리는 시간은 더 길다. 평균 총시간은 Ours가 193.69초, KnowNo가 186.51초이다. 평균 상호작용 시간도 Ours가 26.83초로 KnowNo의 15.46초보다 길다. 따라서 Ours는 두 도메인 모두 작업 성공을 개선하지만 총시간의 우위는 Harvest에 한정된다. 질의 수만으로 전체 작업 시간을 설명할 수 없다.

VLM 응답 오류와 플래너 실패를 구분한다. 각 도메인에서 Ours의 세 번의 실패는 모두 잘못된 VLM 피드백을 포함한다. Sort에서 KnowNo는 VLM 응답 실패가 3회, 플래너 실패가 17회이며, 후자는 전제조건이 충족되지 않았는데도 자율적으로 행동을 선택한 경우이다. Harvest에서 KnowNo는 VLM 응답 실패가 18회, 플래너 실패가 12회이다. VLM 실패 중 6회는 실행 가능한 대안이 있는데도 탐지를 반복 선택하여 단계 제한을 소진한 경우이며, 나머지 12회는 잘못된 행동 선택을 포함한다.

이러한 실패는 질의 시점과 질의 내용이 모두 중요함을 보여준다. 예측한 행동이 하나이더라도 실행 전에 상태 확인이 필요할 수 있다. 올바른 행동을 선택하려면 VLM은 로봇 위치, 물체 상태, 행동 전제조건을 평가해야 한다. 그러나 VLM은 여전히 실행 불가능하거나 작업을 진전시키지 않는 행동을 선택할 수 있다. Ours는 VLM에 현재 상태 사실을 확인하도록 요청하고, 갱신한 belief를 이후 계획에 사용하며, 행동 선택은 플래너에 맡긴다.

**사람.** 동일한 로봇 플랫폼과 작업 도메인을 사용하여 사람 피드백에서도 oracle 및 VLM 경향이 유지되는지 평가한다. 참가자는 다섯 명(남성 1명, 여성 4명)이며 평균 연령은 26.4세이다(표준편차 5.68, 범위 18–33세). 각 참가자는 방법 및 도메인마다 두 번 시행하여 조건당 10회를 구성한다. 참가자는 작업 지침을 받고 태블릿 인터페이스로 질의에 응답한다. 도메인 및 방법 순서는 참가자마다 다르다. 부록 C에 지침과 시행 일정을 제시한다. 결과는 그림 5에 보고한다.

Ours는 Sort에서 90.00%, Harvest에서 100.00%의 성공률을 달성하며, KnowNo는 각각 20.00%, 30.00%이다. Ours는 두 도메인 모두 계획 시간이 더 짧다. 성공 및 실패 시행 전체에서 총 상호작용 시간을 총 질의 수로 나누면, Sort의 질의당 시간은 Ours가 5.26초, KnowNo가 3.86초이며, Harvest에서는 각각 2.81초와 6.42초이다.

Sort에서 Ours의 높은 값은 아홉 번의 질의에 159.44초의 상호작용 시간이 걸린 P3의 한 시행에 영향을 받는다. 실험자는 연속적인 작업 실패 이후 망설임을 관찰했다. 두 방법과 두 도메인에 걸친 P3의 여덟 시행을 모두 제외하면, Sort의 질의당 시간은 Ours가 3.07초, KnowNo가 3.24초이며, Harvest에서는 각각 2.64초와 4.99초이다. 이 추가 분석은 Sort에서 상대적인 상호작용 시간이 한 참가자의 긴 응답 시간에 민감함을 보여준다.

실패 분석에서는 잘못된 사람 응답과 자율적인 플래너 결정을 모두 확인한다. Sort에서 Ours의 유일한 실패는 물체가 보인다는 잘못된 확인 이후 발생한다. KnowNo는 집기 전에 놓기를 선택한 사람 응답 실패가 4회, 탐지하지 않은 물체를 질의 없이 집은 플래너 실패가 4회이다. Harvest에서 Ours는 실패가 없다. KnowNo 실패 중 6회는 필요한 선행 행동을 건너뛴 사람의 선택 이후 발생하고, 1회는 상태 확인 질의 없이 썩은 토마토를 잘못 인식한 이후 발생한다. 이러한 사례는 올바른 행동을 선택하려면 참가자가 실행의 선행 요건을 해석해야 하며, 자율적인 결정은 잘못된 인식을 수정하지 않은 채 남길 수 있음을 보여준다.

#### 5.2.2. Ablation 연구

Baseline 비교는 서로 다른 행동 플래너와 질의 정책을 가진 완전한 계획 시스템을 평가한다. Ablation 연구는 POMCP 행동 플래너를 공유하는 일곱 구성을 사용하여 oracle 피드백하에서 질의 시점과 선택이 작업 성공과 질의 수에 미치는 영향을 살펴본다. 간결한 참조를 위해 시점 변형은 **T**, 선택 변형은 **S**로 표기한다. 엔트로피 이득 선택을 고정한 상태에서 무작위(T1), conformal prediction(T2), 보상 기반(T3) 시점을 비교한다. Confidence 시점을 고정한 상태에서 무작위(S1)와 보상 기반(S2) 선택을 비교한다. R은 질의 시점과 선택을 모두 무작위화하며, Ours는 confidence 시점과 엔트로피 이득 선택을 결합한다. 표 1은 구성을 요약한다. 각 구성은 도메인마다 200회 시행한다. 표 3과 그림 6에 결과를 제시한다. 부록 A.4에 질의 정책 설정을 제공한다.

**그림 6. Ablation 연구 결과.** (a)–(b) Confidence 기반 시점은 두 도메인에서 높은 성공률을 유지한다. (c)–(d) 엔트로피 기반 선택은 S1–S2보다 질의 수를 줄인다. 제안 방법은 질의 수를 줄이면서 높은 작업 성공률을 달성한다.

- [(a) Sort 성공률(%)](latex_zips/BRL_achitecture_changmin/figures/08_ABLATION/exp_ablation_waste_success.pdf)
- [(b) Harvest 성공률(%)](latex_zips/BRL_achitecture_changmin/figures/08_ABLATION/exp_ablation_tomato_success.pdf)
- [(c) Sort 질의 수](latex_zips/BRL_achitecture_changmin/figures/08_ABLATION/exp_ablation_waste_queries.pdf)
- [(d) Harvest 질의 수](latex_zips/BRL_achitecture_changmin/figures/08_ABLATION/exp_ablation_tomato_queries.pdf)

**표 3. 질의 정책 ablation.** T1–T3는 시점 기준을, S1–S2는 선택 기준을 비교한다. 동률을 포함하여 가장 높은 성공률과 가장 낮은 질의 수에 밑줄을 표시한다.

| 조건 | 시점 | 선택 | Sort 성공률(%) | Sort 질의 수 | Harvest 성공률(%) | Harvest 질의 수 |
|---|---|---|---|---|---|---|
| R | 무작위 | 무작위 | 73.50 | 10.25 ± 6.11 | 72.00 | 9.08 ± 3.94 |
| T1 | 무작위 | 엔트로피 이득 | 73.50 | <u>6.65</u> ± 3.29 | 72.00 | 7.44 ± 2.73 |
| T2 | Conformal prediction | 엔트로피 이득 | 74.00 | 7.09 ± 2.58 | 60.50 | <u>3.91</u> ± 1.71 |
| T3 | 보상 | 엔트로피 이득 | 99.00 | 17.65 ± 2.89 | 66.50 | 12.02 ± 2.88 |
| S1 | Confidence | 무작위 | <u>100.00</u> | 15.21 ± 3.53 | 96.50 | 12.70 ± 2.53 |
| S2 | Confidence | 보상 | <u>100.00</u> | 22.79 ± 6.26 | 97.00 | 15.27 ± 2.95 |
| **Ours** | Confidence | 엔트로피 이득 | **<u>100.00</u>** | **7.83 ± 2.16** | **<u>97.50</u>** | **9.16 ± 1.91** |

시점 비교는 R, T1–T3, Ours를 평가한다. Sort에서 R과 T1은 73.50%, T2는 74.00%의 성공률을 달성한다. T3의 성공률은 99.00%로 Ours의 100.00%와 비교되지만, 질의 수는 Ours의 7.83회에 비해 17.65회를 필요로 한다. R, T1, T2는 각각 10.25회, 6.65회, 7.09회 질의한다. Harvest에서 R과 T1은 72.00%, T2와 T3는 각각 60.50%, 66.50%로 Ours의 97.50%와 비교된다. 평균 질의 수는 R 9.08회, T1 7.44회, T2 3.91회, T3 12.02회, Ours 9.16회이다.

이 결과는 작업 성공이 질의 선택뿐 아니라 질의 trigger에도 의존함을 보여준다. T1은 R의 무작위 선택을 엔트로피 이득으로 대체하여 질의 수를 줄이지만 성공률은 개선하지 않는다. 무작위 시점은 trigger가 현재 상태 불확실성에 의존하지 않으므로 필요한 확인을 놓칠 수 있다. T2도 행동 예측 집합이 하나의 후보로 좁혀지면, 그 행동을 실행하는 데 필요한 상태 사실이 여전히 불확실하더라도 질의를 종료할 수 있다. T3는 기대 작업 보상으로 질의 시작 여부를 결정한 뒤 엔트로피 이득으로 질의를 선택하고, confidence가 $\tau$에 도달할 때까지 확인을 반복한다. Sort에서는 물체 범주의 확인이 올바른 수거함 선택과 작업 보상에 직접 연결되므로, 반복 확인이 99.00%의 높은 성공률을 뒷받침할 수 있다. 그러나 T3는 Ours보다 125.42% 많은 질의를 필요로 한다. Harvest에서 T3는 Ours보다 31.22% 많은 질의를 사용하지만 성공률은 31.00%p 낮다. 따라서 보상 기반 시점은 전체적으로 더 많은 질의를 요청하면서도 적절한 시점에 필요한 상태 정보를 확인하지 못할 수 있다. Ours는 현재 상태의 confidence를 질의 trigger로 사용하여 두 도메인 모두에서 가장 높은 성공률을 달성하면서 보상 기반 시점보다 적은 질의를 사용한다.

Confidence 기반 질의 시점을 고정하면 서로 다른 질의 선택 전략은 높은 성공률을 유지하지만 필요한 질의 수는 달라진다. 선택 비교는 R, S1, S2, Ours를 평가한다. R은 Sort에서 73.50%, Harvest에서 72.00%의 성공률을 달성하고 각각 10.25회, 9.08회 질의한다. S1은 무작위 선택을 유지하면서 confidence 시점을 사용하여 성공률이 각각 100.00%, 96.50%로 증가한다. Confidence 시점을 고정하면 S1, S2, Ours 모두 Sort에서 100.00%의 성공률을 달성한다. 평균 질의 수는 S1 15.21회, S2 22.79회, Ours 7.83회이다. Harvest에서 S1과 S2의 성공률은 96.50%, 97.00%이며 Ours는 97.50%이다. 평균 질의 수는 S1 12.70회, S2 15.27회, Ours 9.16회이다.

Confidence 시점은 세 선택 전략 모두에서 Sort 성공률 100.00%, Harvest 성공률 최소 96.50%를 유지한다. Confidence 시점하에서 질의 선택은 주로 요구 confidence에 도달하는 데 필요한 질의 수에 영향을 미친다. 무작위 선택은 상태 불확실성의 기대 감소량에 따라 사실의 우선순위를 정하지 않는다. 보상 기반 선택은 응답이 제공하는 정보를 직접 측정하지 않고 기대 작업 보상을 평가한다. 엔트로피 이득은 상태 불확실성의 기대 감소량에 따라 사실을 선택하므로, 선택 기준이 질의 에피소드를 종료하는 confidence 요건과 부합한다. S1 및 S2와 비교하여 Ours는 Sort에서 각각 48.52%, 65.64%, Harvest에서 27.87%, 40.01% 질의 수를 줄인다. 따라서 제안 질의 선택 방법은 작업 성공을 낮추지 않으면서 confidence 시점하에서 필요한 질의 수를 줄인다.

이 결과는 질의 시점과 선택의 구별되는 역할을 통해 제안 방법의 주요 기여를 뒷받침한다. 질의 시점은 로봇이 상태 불확실성하에서 행동을 실행하기 전에 추가 정보를 요청하는지 결정한다. Confidence 시점은 무작위, 보상 기반, 엔트로피 이득 선택에서 높은 성공률을 달성하는 반면, 무작위 시점하의 엔트로피 이득은 R보다 성공률을 높이지 않고 질의 수만 줄인다. Confidence 시점이 높은 성공 수준을 확보하면 엔트로피 이득은 요구 confidence에 도달하는 데 필요한 질의 수를 줄인다. 따라서 제안 방법은 작업 성공을 뒷받침하는 시점 기준과 그 성공 수준에서 질의 수를 줄이는 선택 기준을 결합한다.

#### 5.2.3. 임계값 실험

Ablation 비교 이후 confidence 임계값이 작업 성공과 질의 수를 어떻게 제어하는지 살펴본다. 엔트로피 이득 선택과 시뮬레이션의 정확한 oracle 응답을 고정하고 $\tau$를 0.0부터 1.0까지 0.1 간격으로 변화시킨다. 각 임계값은 도메인마다 200회 시행한다. $\tau=0.8$에서는 표 2와 표 3에 보고한 Ours 시행을 재사용한다. 그림 7에 결과를 제시한다.

**그림 7. Confidence 임계값 $\tau$가 작업 성공과 질의 수에 미치는 영향.** (a) Sort는 $\tau=0.7$, Harvest는 $\tau=1.0$에서 성공률 100.00%에 도달한다. (b) 평균 질의 수는 두 도메인 모두 $\tau=1.0$에서 급증한다. 기본값 $\tau=0.8$(점선)과 비교하면 $\tau=1.0$은 질의 수가 약 두 배이지만, Sort의 성공률은 변하지 않고 Harvest는 2.50%p 증가한다.

- [(a) 작업 성공](latex_zips/BRL_achitecture_changmin/figures/09_THRESHOLD/exp_threshold_success.pdf)
- [(b) 평균 질의 수](latex_zips/BRL_achitecture_changmin/figures/09_THRESHOLD/exp_threshold_queries.pdf)

그림 7은 두 도메인이 confidence 임계값에 서로 다르게 반응함을 보여준다. Sort 성공률은 $\tau=0.6$의 53.50%에서 $\tau=0.7$의 100.00%로 증가하며 더 높은 임계값에서도 유지된다. 평균 질의 수는 $\tau=0.7$–$0.9$에서 7.74–7.83회이며 $\tau=1.0$에서 14.97회로 증가한다. Harvest 성공률은 $\tau=0.3$의 51.50%에서 $\tau=0.4$의 86.00%로 증가하고, $\tau=0.5$–$0.7$에서는 90.00%를 유지한다. 임계값을 더 높이면 $\tau=0.8$에서 97.50%, $\tau=0.9$에서 99.50%, $\tau=1.0$에서 100.00%에 도달한다. 평균 질의 수는 $\tau=0.4$의 5.41회에서 $\tau=0.8$의 9.16회, $\tau=1.0$의 19.00회로 증가한다.

시스템 주도 질의는 동일한 POMCP 행동 플래너에서 작업 성공을 개선한다. 질의가 없을 때($\tau=0.0$) Ours의 성공률은 Sort 55.00%, Harvest 51.00%이다. 두 값 모두 oracle 피드백을 사용하는 KnowNo와 IntroPlan보다 높다. $\tau=0.8$에서는 성공률이 각각 45.00%p와 46.50%p 증가한다. 성공한 시행은 Sort에서 평균 7.83회, Harvest에서 평균 9.16회 질의를 필요로 한다.

작업 성공률이 최대에 가까워지면 추가 질의의 이득은 제한된다. $\tau$를 0.8에서 1.0으로 높이면 평균 질의 수가 Sort에서 91.19%, Harvest에서 107.42% 증가하지만, 성공률 증가는 각각 0.00%p와 2.50%p이다. Harvest에서 마지막으로 $\tau=0.9$에서 1.0으로 높이는 것만으로도 성공률 0.50%p 증가를 위해 7.98회의 질의가 추가된다. 두 도메인에 걸쳐 작업 성공과 질의 수의 균형을 맞추기 위해 공통 임계값으로 $\tau=0.8$을 사용한다.

#### 5.2.4. 인터뷰

두 도메인에서 Ours와 KnowNo를 사용한 후 다섯 참가자 모두를 인터뷰한다. P1과 P2는 평가한 작업을 완료하는 데 Ours가 KnowNo보다 신뢰할 만하다고 설명한다. P1은 Ours에서 오류가 발생할 가능성이 더 낮다고 생각한다. P2는 상태 확인을 제공한 후 Ours가 작업을 완료할 것으로 기대하지만, KnowNo는 카메라 화면에서 실행 가능해 보이는 행동을 선택한 후에도 가끔 실패한다고 설명한다.

올바른 행동을 선택하려면 참가자가 각 행동의 범위와 실행에 필요한 선행 단계를 모두 해석해야 한다. P1은 토마토 품질 확인과 같은 후속 행동의 일부로 KnowNo가 집기를 수행할 것으로 기대한다. P3는 집기 전에 품질 확인을 선택한 뒤 실패한 경험을 회상하고, P4는 토마토 작업 실패의 원인을 필요한 단계를 건너뛴 것으로 설명한다. 보고된 실패는 필요한 선행 행동을 생략한 선택 이후 KnowNo가 실패했다는 실패 분석과 일치한다. Ours는 참가자에게 상태 사실의 확인을 요청하고 행동 선택과 순서 결정은 플래너에 맡긴다. 따라서 참가자는 선택한 행동에 필요한 선행 단계가 포함되어 있는지 판단할 필요가 없다. 이러한 역할 분담은 사람 평가에서 Ours의 더 높은 작업 성공률을 설명하는 데 도움이 된다.

#### 5.2.5. 논의 및 한계

질의 시점은 주로 작업 성공에, 질의 선택은 주로 필요한 질의 수에 영향을 미친다는 것을 확인한다. Ablation 연구는 confidence 기반 시점이 여러 선택 전략에서 높은 성공률을 유지하고, 시점이 고정되면 엔트로피 기반 선택이 질의 수를 줄인다는 것을 보여준다. 제안 방법은 confidence 기반 시점과 엔트로피 기반 선택을 결합하여 작업 성공과 질의 효율의 균형을 맞춘다. Sort와 Harvest에서 제안 방법은 시뮬레이션과 VLM 및 사람 피드백을 사용하는 실제 로봇 모두에서 평가한 방법 중 가장 높은 작업 성공률을 달성한다.

효율적인 질의 선택은 주어진 confidence 수준에 도달하는 데 필요한 질의 수를 줄인다. Confidence 임계값은 불확실성을 받아들이는 것과 추가 확인을 요청하는 것 사이의 절충을 제어한다. $\tau<1$이면 다른 상태에 belief 가중치가 남아 있어도 로봇은 하나의 상태를 확정할 수 있다. 불확실성을 받아들이면 질의 수를 줄일 수 있지만, 잘못된 상태 추정이 이후 행동에 영향을 줄 수 있다. $\tau=1$이면 belief 엔트로피가 0이 된 후에만 실행을 재개한다. 임계값 실험은 질의 수를 늘리는 것이 작업 성공률을 반드시 높이지는 않음을 보여준다. 추가 질의의 이득은 남은 불확실성이 작업 성공에 영향을 주는지에 따라 달라진다. 두 도메인은 높은 작업 성공률을 달성하는 데 필요한 confidence 수준도 다르다.

제안 방법의 한계는 belief 갱신이 외부 에이전트 응답의 오류를 고려하지 않는다는 점이다. 응답 신뢰도와 관계없이 응답과 일치하지 않는 상태 가설을 제거한다. 따라서 잘못된 피드백은 실제 상태를 belief에서 제외하여 로봇이 잘못된 상태를 확신하게 만들 수 있다. 잘못된 VLM 및 사람 응답이 포함된 실패는 피드백 오류의 실제 영향을 보여준다. Confidence는 남아 있는 가설들에 대해 계산하므로 임계값을 높여도 올바른 상태 추정을 보장할 수 없다. 응답 신뢰도를 belief 갱신에 통합하면 일치하지 않는 가설을 완전히 제거하는 대신 가중치를 조정하여 타당한 대안들을 보존할 수 있을 것이다.

## 6. 결론 및 향후 연구

로봇 작업 계획 중 상태 불확실성을 이용하여 언제 무엇을 질의할지 결정하는 방법을 제시하였다. Belief confidence로 질의 시점을 결정하고, 기대 엔트로피 감소량으로 상태 사실의 선택을 안내한다. Sort와 Harvest 실험은 질의 시점이 작업 성공을 뒷받침하고 질의 선택이 필요한 질의 수를 줄인다는 것을 보여준다. 향후 연구에서는 외부 에이전트의 잘못된 응답에 대한 강건성을 높이기 위해 응답 신뢰도를 belief 갱신에 통합할 것이다.

## 감사의 글

본 연구는 대한민국 정부(과학기술정보통신부)의 재원으로 한국연구재단(NRF)의 지원을 일부 받아 수행되었다(과제번호 RS-2024-00461583 및 RS-2024-00411007).

## 윤리 승인 및 동의

본 연구는 서강대학교 기관생명윤리위원회의 승인을 받았다(승인번호 SGUIRB-A-2501-04-2). 연구는 해당 기관의 윤리 기준에 따라 수행되었다. 참여 전에 모든 참가자에게 충분한 설명을 제공하고 동의를 받았다.

## 참고문헌

인용 키에 대응하는 서지 정보는 원문의 [ref.bib](latex_zips/BRL_achitecture_changmin/ref.bib)에 수록되어 있다. 아래에는 현재 원고에서 인용하는 문헌의 서지 정보를 원문 표기로 제공한다.

- **[kaipa2016addressing]** Kaipa, Krishnanand N and Kankanhalli-Nagendra, Akshaya S and Kumbla, Nithyananda B and Shriyam, Shaurya and Thevendria-Karthic, Srudeep Somnaath and Marvel, Jeremy A and Gupta, Satyandra K (2016). Addressing perception uncertainty induced failure modes in robotic bin-picking. Robotics and Computer-Integrated Manufacturing (RCIM). 권 42. 쪽 17--38. DOI: 10.1016/j.rcim.2016.05.002.
- **[kaelbling1998planning]** Kaelbling, Leslie Pack and Littman, Michael L and Cassandra, Anthony R (1998). Planning and acting in partially observable stochastic domains. Artificial Intelligence (AIJ). 권 101. 호 1-2. 쪽 99--134. DOI: 10.1016/S0004-3702(98)00023-X.
- **[ren2023robots]** Ren, Allen Z and Dixit, Anushri and Bodrova, Alexandra and Singh, Sumeet and Tu, Stephen and Brown, Noah and Xu, Peng and Takayama, Leila and Xia, Fei and Varley, Jake and Xu, Zhenjia and Sadigh, Dorsa and Zeng, Andy and Majumdar, Anirudha (2023). Robots that ask for help: Uncertainty alignment for large language model planners. Proc. of the Conference on Robot Learning (CoRL).
- **[liang2024introspective]** Liang, Kaiqu and Zhang, Zixu and Fisac, Jaime F (2024). Introspective planning: Aligning robots' uncertainty with inherent task ambiguity. Proc. of Advances in Neural Information Processing Systems (NeurIPS). 권 37. 쪽 71998--72031.
- **[burks2021collaborative]** Burks, Luke and Ahmed, Nisar and Loefgren, Ian and Barbier, Luke and Muesing, Jeremy and McGinley, Jamison and Vunnam, Sousheel (2021). Collaborative human-autonomy semantic sensing through structured POMDP planning. Robotics and Autonomous Systems (RAS). 권 140. 쪽 103753. DOI: 10.1016/j.robot.2021.103753.
- **[burks2023harps]** Burks, Luke and Ray, Hunter M and McGinley, Jamison and Vunnam, Sousheel and Ahmed, Nisar (2023). HARPS: An online POMDP framework for human-assisted robotic planning and sensing. IEEE Transactions on Robotics (T-RO). 권 39. 호 4. 쪽 3024--3042. DOI: 10.1109/TRO.2023.3263460.
- **[davis1993knowledge]** Davis, Randall and Shrobe, Howard and Szolovits, Peter (1993). What is a knowledge representation?. AI Magazine. 권 14. 호 1. 쪽 17--33. DOI: 10.1609/aimag.v14i1.1029.
- **[fikes1971strips]** Fikes, Richard E and Nilsson, Nils J (1971). STRIPS: A new approach to the application of theorem proving to problem solving. Artificial Intelligence (AIJ). 권 2. 호 3-4. 쪽 189--208. DOI: 10.1016/0004-3702(71)90010-5.
- **[mcdermott1998pddl]** McDermott, Drew and Ghallab, Malik and Howe, Adele and Knoblock, Craig and Ram, Ashwin and Veloso, Manuela and Weld, Daniel and Wilkins, David (1998). PDDL -- The Planning Domain Definition Language. 호 CVC TR-98-003/DCS TR-1165.
- **[younes2004ppddl1]** Younes, H\aakan LS and Littman, Michael L (2004). PPDDL1. 0: The language for the probabilistic part of IPC-4. Proc. of the International Planning Competition (IPC). 쪽 70--74.
- **[sanner2010rddl]** Scott Sanner (2010). Relational Dynamic Influence Diagram Language (RDDL): Language Description. http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf
- **[zhang2015corpp]** Zhang, Shiqi and Stone, Peter (2015). CORPP: Commonsense reasoning and probabilistic planning, as applied to dialog with a mobile robot. Proc. of the Conference on Artificial Intelligence (AAAI). 권 29. 호 1. DOI: 10.1609/aaai.v29i1.9385.
- **[sanner2010symbolic]** Sanner, Scott and Kersting, Kristian (2010). Symbolic dynamic programming for first-order POMDPs. Proc. of the Conference on Artificial Intelligence (AAAI). 권 24. 호 1. 쪽 1140--1146. DOI: 10.1609/aaai.v24i1.7747.
- **[thrun2005probabilistic]** Thrun, Sebastian and Burgard, Wolfram and Fox, Dieter (2005). Probabilistic Robotics. MIT Press.
- **[ross2008online]** Ross, St\'ephane and Pineau, Joelle and Paquet, S\'ebastien and Chaib-Draa, Brahim (2008). Online planning algorithms for POMDPs. Journal of Artificial Intelligence Research (JAIR). 권 32. 쪽 663--704. DOI: 10.1613/jair.2567.
- **[silver2010monte]** Silver, David and Veness, Joel (2010). Monte-Carlo planning in large POMDPs. Proc. of Advances in Neural Information Processing Systems (NeurIPS). 권 23.
- **[xiao2019objectsearch]** Xiao, Yuchen and Katt, Sammie and ten Pas, Andreas and Chen, Shengjian and Amato, Christopher (2019). Online planning for target object search in clutter under partial observability. Proc. of the IEEE International Conference on Robotics and Automation (ICRA). 쪽 8241--8247. DOI: 10.1109/ICRA.2019.8793494.
- **[singi2023hula]** Singi, Siddharth and He, Zhanpeng and Pan, Alvin and Patel, Sandip and Sigurdsson, Gunnar A and Piramuthu, Robinson and Song, Shuran and Ciocarlie, Matei (2024). Decision making for human-in-the-loop robotic agents via uncertainty-aware reinforcement learning. Proc. of the IEEE International Conference on Robotics and Automation (ICRA). DOI: 10.1109/ICRA57147.2024.10611425.
- **[igbinedion2024learning]** Igbinedion, Ifueko and Karaman, Sertac (2024). Learning when to ask for help: Efficient interactive navigation via implicit uncertainty estimation. arXiv preprint arXiv:2305.16502. DOI: 10.48550/arXiv.2305.16502.
- **[mangalindan2025seek]** Mangalindan, Dong Hae and Rovira, Ericka and Srivastava, Vaibhav (2026). When to seek help: Trust-aware assistance-seeking in human-supervised autonomy. ACM Transactions on Human-Robot Interaction (THRI). 권 15. 호 2. 쪽 1--23. DOI: 10.1145/3773900.
- **[zhao2025conformalized]** Zhao, Michelle and Simmons, Reid and Admoni, Henny and Ramdas, Aaditya and Bajcsy, Andrea (2025). Conformalized interactive imitation learning: Handling expert shift and intermittent feedback. Proc. of the International Conference on Learning Representations (ICLR).
- **[armstrongcrews2007oracular]** Armstrong-Crews, Nicholas and Veloso, Manuela (2007). Oracular partially observable Markov decision processes: A very special case. Proc. of the IEEE International Conference on Robotics and Automation (ICRA). 쪽 2477--2482. DOI: 10.1109/ROBOT.2007.363691.
- **[deits2013clarifying]** Deits, Robin and Tellex, Stefanie and Thaker, Pratiksha and Simeonov, Dimitar and Kollar, Thomas and Roy, Nicholas (2013). Clarifying commands with information-theoretic human-robot dialog. Journal of Human-Robot Interaction (JHRI). 권 2. 호 2. 쪽 58--79. DOI: 10.5898/JHRI.2.2.Deits.
- **[anand2025adaptive]** Anand, Yashwanthi and Nwagwu, Nnamdi and Sabbe, Kevin and Fitter, Naomi T and Saisubramanian, Sandhya (2026). Adaptive querying for reward learning from human feedback. Frontiers in Robotics and AI. 권 12. 쪽 1734564. DOI: 10.3389/frobt.2025.1734564.
- **[qu2023stochastic]** Qu, Kaige and Zhuang, Weihua and Wu, Wen and Li, Mushu and Shen, Xuemin and Li, Xu and Shi, Weisen (2023). Stochastic Cumulative DNN Inference With RL-Aided Adaptive IoT Device-Edge Collaboration. IEEE Internet of Things Journal (IoT-J). 권 10. 호 20. 쪽 18000--18015. DOI: 10.1109/JIOT.2023.3280746.
- **[chitta2012moveit]** Chitta, Sachin and Sucan, Ioan and Cousins, Steve (2012). Moveit![ros topics]. IEEE Robotics and Automation Magazine (RAM). 권 19. 호 1. 쪽 18--19.
- **[carion2025sam3segmentconcepts]** Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and Lei, Jie and Ma, Tengyu and Guo, Baishan and Kalla, Arpit and Marks, Markus and Greer, Joseph and Wang, Meng and Sun, Peize and R\"adle, Roman and Afouras, Triantafyllos and Mavroudi, Effrosyni and Xu, Katherine and Wu, Tsung-Han and Zhou, Yu and Momeni, Liliane and Hazra, Rishi and Ding, Shuangrui and Vaze, Sagar and Porcher, Francois and Li, Feng and Li, Siyuan and Kamath, Aishwarya and Cheng, Ho Kei and Doll\'ar, Piotr and Ravi, Nikhila and Saenko, Kate and Zhang, Pengchuan and Feichtenhofer, Christoph (2025). SAM 3: Segment Anything with Concepts. https://arxiv.org/abs/2511.16719
- **[tian2025yolov12]** Tian, Yunjie and Ye, Qixiang and Doermann, David (2025). YOLOv12: Attention-Centric Real-Time Object Detectors. arXiv preprint arXiv:2502.12524.
- **[gemmateam2026gemma4]** Gemma Team (2026). Gemma 4 Technical Report. arXiv preprint arXiv:2607.02770. DOI: 10.48550/arXiv.2607.02770.

## 부록 A. 실험 매개변수

### A.1. 공통 설정

표 A1은 공통 설정을 제시한다. Oracle 시행은 도메인 및 조건마다 200회이다. 실제 로봇 평가는 방법 및 도메인마다 VLM 40회, 사람 10회 시행한다.

**표 A1. 공통 플래너 및 평가 매개변수.**

| 매개변수 | 값 | 매개변수 | 값 |
|---|---|---|---|
| 단계당 POMCP 시뮬레이션 | 100 | 최대 시뮬레이션 깊이 | 20 |
| 기본 할인율 $\gamma$ | 0.2 | UCB 탐색 상수 $c$ | 1.0 |
| Confidence 임계값 $\tau$ | 0.8 | 할인 중단 기준 $\epsilon$ | 0.005 |

$\gamma=0.2$, $\epsilon=0.005$이면 할인 중단 기준에 따라 깊이 4에서 시뮬레이션을 중단한다. Oracle 실험은 시행당 최대 20회의 물리적 행동을 허용한다. POMCP-Q에는 시행당 최대 30회의 질의도 허용한다.

### A.2. 보상 설정

표 A2는 행동 계획과 보상 기반 질의 평가에 사용하는 도메인 보상을 제시한다. 목표 보상과 행동 보상은 합산한다. 그 밖의 적용 가능한 물리적 행동은 보상이 0이다.

**표 A2. 작업 보상과 페널티.**

| 도메인 | 결과 또는 행동 | 보상 |
|---|---|---|
| Sort | 물체를 올바른 수거함에 놓기 | +5 |
| Sort | 작업 목표 달성 | +10 |
| Sort | 연속 탐지 중 세 번째 이후의 탐지 행동 | −10 |
| Harvest | 신선한 토마토를 놓거나 썩은 토마토를 버리기 | +10 |
| Harvest | 같은 줄기에서 연속 탐지 중 두 번째 이후의 행동 | −5 |
| Harvest | 같은 토마토의 연속 스캔 중 두 번째 이후의 행동 | −5 |
| Harvest | 연속 이동 중 두 번째 이후의 행동 | −5 |

POMCP-Q와 T3 및 S2의 보상 평가기는 동일한 도메인 보상을 사용한다. 가치 모델은 각 질의의 즉시 보상을 0, 적용 불가능한 물리적 행동의 페널티를 −10으로 설정한다.

### A.3. Baseline 매개변수

표 A3은 도메인별 설정을 제시한다. KnowNo와 IntroPlan은 GPT-4o를 사용하며, 행동 생성 temperature는 0.3, 점수 temperature는 5.0이다. KnowNo와 IntroPlan의 보정 임계값 $\hat q$는 예측 집합 coverage 95%를 목표로 한다. 교환가능성 가정하에서 conformal prediction은 예측 집합이 최소 하나의 올바른 행동을 포함할 확률이 최소 0.95임을 보장한다 [ren2023robots].

POMCP-Q는 의사결정당 100회 시뮬레이션, 질의 비용 0, 실패 페널티 10, 가정한 응답 정확도 1.0을 사용한다. 할인율 $\gamma$와 기타 POMCP-Q 설정은 평가한 구성 중 가장 높은 작업 성공률을 기준으로 도메인마다 별도로 선택한다. 대시는 적용되지 않는 설정이다.

**표 A3. Sort와 Harvest의 baseline 매개변수.**

| 도메인 | 매개변수 | Ours | KnowNo | IntroPlan | POMCP-Q |
|---|---|---|---|---|---|
| Sort | 보정 임계값 $\hat q$(oracle) | — | 0.845165 | 0.980493 | — |
| Sort | 행동 생성 temperature(oracle) | — | 0.3 | 0.3 | — |
| Sort | 할인율 $\gamma$ | 0.2 | — | — | 0.9 |
| Harvest | 보정 임계값 $\hat q$(oracle) | — | 0.901119 | 0.985552 | — |
| Harvest | 행동 생성 temperature(oracle) | — | 0.3 | 0.3 | — |
| Harvest | 할인율 $\gamma$ | 0.2 | — | — | 0.5 |

### A.4. Ablation 매개변수

표 A4는 질의 시점과 선택을 명시한다. CP는 conformal prediction, Entropy는 기대 엔트로피 이득이다. 모든 구성은 단계당 POMCP 시뮬레이션 100회와 $\tau=0.8$을 사용한다. T3와 S2는 질의 비용 0, 실패 페널티 10, 가정한 응답 정확도 1.0을 사용한다. T3와 S2의 할인율 $\gamma$는 평가한 구성 중 가장 높은 작업 성공률을 기준으로 도메인마다 별도로 선택한다.

**표 A4. Sort와 Harvest의 ablation 매개변수.**

| 도메인 | 매개변수 | R | T1 | T2 | T3 | S1 | S2 | Ours |
|---|---|---|---|---|---|---|---|---|
| Sort | 질의 시점 | 무작위 | 무작위 | CP | 보상 | Confidence | Confidence | Confidence |
| Sort | 질의 선택 | 무작위 | Entropy | Entropy | Entropy | 무작위 | 보상 | Entropy |
| Sort | 무작위 trigger 확률 | 0.4 | 0.4 | — | — | — | — | — |
| Sort | 할인율 $\gamma$ | 0.2 | 0.2 | 0.2 | 0.9 | 0.2 | 0.9 | 0.2 |
| Sort | 보정 임계값 $\hat q$ | — | — | 0.8704 | — | — | — | — |
| Sort | 후보 점수 temperature | — | — | 5.0 | — | — | — | — |
| Sort | 질의 평가 시뮬레이션 | — | — | — | ≥100 | — | ≥100 | — |
| Harvest | 질의 시점 | 무작위 | 무작위 | CP | 보상 | Confidence | Confidence | Confidence |
| Harvest | 질의 선택 | 무작위 | Entropy | Entropy | Entropy | 무작위 | 보상 | Entropy |
| Harvest | 무작위 trigger 확률 | 0.4 | 0.4 | — | — | — | — | — |
| Harvest | 할인율 $\gamma$ | 0.2 | 0.2 | 0.2 | 0.5 | 0.2 | 0.5 | 0.2 |
| Harvest | 보정 임계값 $\hat q$ | — | — | 0.8404 | — | — | — | — |
| Harvest | 후보 점수 temperature | — | — | 5.0 | — | — | — | — |
| Harvest | 질의 평가 시뮬레이션 | — | — | — | ≥100 | — | ≥100 | — |

T2는 conformal prediction 집합에 여러 행동이 포함되어 있는 동안 확인을 시작하고 계속한다. T3는 가장 높은 질의 가치가 가장 높은 물리적 행동 가치보다 클 때 확인을 시작한다. S2는 물리적 행동 가치와 비교하지 않고 가치가 가장 높은 사실을 선택한다. Trigger 이후에는 T2를 제외한 모든 구성이 confidence가 $\tau$에 도달하거나 구별 가능한 사실이 남지 않을 때까지 확인을 계속한다. 파싱할 수 없는 T2 출력 네 건은 실패로 처리한다.

### A.5. 시행 재사용 및 결과 보고

도메인별 동일한 Ours 200회 시행을 baseline, ablation, $\tau=0.8$의 임계값 비교에서 재사용한다. 다른 임계값에서는 별도 시행 묶음을 사용한다. Oracle 조건은 각 초기 구성과 반복에 대해 seed를 공유하지만, 정책별로 난수를 소비하는 방식은 다를 수 있다.

성공률은 모든 시행을 사용하고, 다른 시행 수준 지표는 성공한 시행의 평균과 표본 표준편차를 사용한다. 계획 시간은 기록된 누적 탐색 시간이며, KnowNo와 IntroPlan의 후보 생성 및 점수 계산을 포함한다. Oracle 총시간은 계획 시간과 같다. 사람의 질의당 상호작용 시간은 성공 및 실패 시행 전체의 상호작용 시간과 질의 수를 합산하여 계산한다.

## 부록 B. VLM 프롬프트

두 도메인 모두 샘플링 temperature 0.3의 `nvidia/Gemma-4-31B-IT-NVFP4`를 사용한다. 각 요청은 현재 RGB 카메라 이미지 두 장, 현재 로봇 행동, 도메인 행동 모델을 포함한다. Ours는 한국어 상태 확인 질문과 질의한 술어의 판단 기준을 제공한다. KnowNo는 후보 행동, 응답 토큰, 행동 선택 기준을 제공한다. Harvest는 고정된 물체 식별자와 줄기 배정 정보도 제공한다. 원문의 상자들은 프롬프트 지침을 영어로 요약하며, 보충 설정 파일에는 원래 영어 및 한국어 텍스트를 보존한다.

### 공통 요청 지침

상태 확인은 첫째 줄에 Yes 또는 No, 둘째 줄에 간단한 이유를 요구한다. 클라이언트는 응답을 belief 갱신에 사용할 불리언 값으로 변환한다. 행동 선택은 후보 토큰 뒤에 간단한 한국어 이유를 요구한다. VLM은 이미지에서 물체 식별, 로봇 상태, 행동 전제조건을 확인한다. 행동을 시도했다는 사실만으로 실행 성공이나 전제조건 충족이 성립하지 않는다. 실행 조건을 만족하는 후보가 없으면 수동 행동 입력 선택지를 선택한다. 클라이언트는 반환한 토큰이 후보 집합에 속하는지 확인한다.

### Sort 도메인 지침

VLM은 번호로 질의 대상 물체를 식별하고 가시성, 재질, 형태를 확인한다. 가려졌다는 사실이 부재를 의미하지 않으며, 색상이나 후보 수거함 이름만으로 범주가 결정되지 않는다. 그리퍼와 수거함 내부는 잡고 있음과 놓기 사실의 증거를 제공한다. 새로 드러난 물체를 포함하여 집기 전에는 탐지가 선행되어야 한다. 집기는 빈 그리퍼를 요구하고, 놓기는 물체를 잡고 있으며 물체 범주가 수거함과 일치할 것을 요구한다. 물체와 바운딩 박스가 보인다는 사실만으로 탐지 완료가 성립하지 않는다. 완료 조건은 빈 작업대, 올바르게 분류된 물체, 빈 그리퍼이다.

### Harvest 도메인 지침

VLM은 고정 토마토 식별자와 ArUco 마커로 위치를 판단한다. 우세한 빨강 또는 초록 색상으로 익음 여부를 판단한다. 큰 베이지색 부패 영역은 썩음을 나타내지만 작은 흠집, 반사, 변색은 그렇지 않다. VLM은 두 카메라 시점으로 잡고 있는 토마토를 검사하고 그리퍼와 바구니 내부를 직접 확인한다. 이동과 탐지는 빈 그리퍼를 요구한다. 집기는 탐지가 완료되어 있고 현재 줄기에 익은 토마토가 있을 것을 요구한다. 스캔은 집기 이후에 수행하며 토마토를 놓을지 버릴지 결정한다. 보이는 토마토와 시도한 행동만으로 탐지나 스캔의 완료가 성립하지 않는다. 완료 조건은 처리할 익은 토마토가 남지 않고 그리퍼가 비어 있는 것이다.

## 부록 C. 사람 평가 및 인터뷰

### C.1. 지침과 시행 순서

참가자는 작업 목표, 물체 범주, 사용 가능한 행동을 안내받고 카메라 화면을 보여주는 태블릿을 통해 질의에 응답한다(그림 C1). Ours는 불리언 상태 확인을 요청하며 한 에피소드에서 여러 번 요청할 수 있다. KnowNo는 trigger마다 하나의 행동 선택을 요청하며 수동 입력을 허용한다. 각 참가자는 방법 및 도메인마다 두 번 시행한다.

A는 Ours, B는 KnowNo를 뜻한다. P1과 P5는 Sort 이후 Harvest를 수행하고 방법 순서는 AABB이다. P2는 도메인을 반대 순서로 수행하며 방법 순서는 ABAB이다. P3는 Sort 이후 Harvest를 수행하고 BBAA 순서를 따른다. P4는 도메인을 반대 순서로 수행하고 BABA 순서를 따른다. 각 방법 시퀀스는 두 도메인 모두에 적용된다.

**그림 C1. 태블릿 인터페이스를 사용하는 사람 평가.** (a)–(b) 참가자는 각각 Sort와 Harvest에서 로봇 카메라 영상을 보며 질의에 응답한다.

- [(a) P3, Sort](latex_zips/BRL_achitecture_changmin/figures/10_APPX/p3_test.pdf)
- [(b) P2, Harvest](latex_zips/BRL_achitecture_changmin/figures/10_APPX/p2_test.pdf)

### C.2. 인터뷰 요약

표 C1은 5.2.4절에서 논의한 다섯 인터뷰를 요약한다.

**표 C1. 사람 응답을 사용한 baseline 비교에 관한 인터뷰 발췌.**

| 참가자 | 인터뷰 발췌 | 해석 |
|---|---|---|
| P1 | “[Ours]는 오류가 발생할 가능성이 훨씬 낮다고 생각해요.” KnowNo에 대해서는 “물체를 집고 품질을 확인해 줄 것으로 기대했어요.” | 인지된 신뢰성을 적은 오류와 연결하며, 선택한 행동에 선행 단계가 포함될 것이라는 기대를 확인한다. |
| P2 | “[Ours]는 제가 잘못 대답하지 않는 한 작업이 완료될 것이라고 생각해요. [KnowNo]는 화면에서 선택지가 실행 가능해 보였는데 선택하고 나면 작동하지 않는 경우가 있었어요.” | 작업 완료에 대한 기대를 상태 확인과 연결하고, 실행 가능해 보이는 선택에도 실행이 실패했음을 보고한다. |
| P3 | “목표에 집중하다 보니 품질 확인을 선택했는데, 집기 전에 선택해서 그런지 실패가 나왔어요.” | 전제조건이 충족되기 전에 행동을 선택한 Harvest 실패와 일치한다. |
| P4 | KnowNo에 대해서는 “돌이켜보니 제가 조급해서 한 단계를 건너뛴 것 같고, 그래서 실패한 것 같아요.” Ours에 대해서는 “저는 상태만 판단하고 로봇이 대부분의 행동을 처리해요.” | 생략한 단계를 실패와 연결하고 이후 행동을 로봇에 맡긴다고 설명하여, 본문에서 논의한 역할 분담을 뒷받침한다. |
| P5 | “한 로봇은 무엇이 맞는지 물었고, 다른 로봇은 다음 행동을 지정해 달라고 했어요.” | 사실 확인과 행동 결정을 구별하며, 두 방법에서 참가자에게 부여한 역할과 일치한다. |
