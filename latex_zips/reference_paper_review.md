# Reference Paper Review: Uncertainty-Driven Help/Query Strategies

이 문서는 `latex_zips/reference_paper` 폴더에 있는 9개 논문을 리뷰한 보고서이다. 정리 기준은 다음 다섯 가지이다.

1. Motivation
2. 풀고자 하는 문제
3. 방법론
4. 기여
5. 한계

## Overall Summary

검토한 논문들은 모두 넓은 의미에서 "로봇 또는 에이전트가 언제 혼자 행동하고, 언제 사람에게 묻거나 도움을 받아야 하는가"를 다룬다. 하지만 질문의 대상과 불확실성 표현 방식은 크게 다르다.

가장 직접적으로 우리 연구와 가까운 축은 KnowNo와 Introspective Planning이다. 이 계열은 LLM planner의 불확실성을 prediction set으로 표현하고, 가능한 다음 행동 후보가 하나로 좁혀지지 않을 때 사람에게 clarification/help를 요청한다. 장점은 conformal prediction을 통해 통계적 성공 보장을 제공한다는 점이고, 한계는 perception, low-level execution, human response error를 planner uncertainty와 통합하지 못한다는 점이다.

ELBA와 EACL 2024 clarification 논문은 "언제 어떤 질문을 생성할 것인가"를 다루지만, symbolic planning이나 robot belief update보다는 embodied vision-language task 또는 collaborative dialogue에 가깝다. HULA, interactive navigation, ConformalDAgger, trust-aware help seeking은 질문 내용보다는 "도움을 요청할 시점"을 결정하는 데 초점을 둔다. 따라서 이들은 proactive query/help-seeking literature로는 유용하지만, 우리 시스템의 state-level symbolic hypothesis query와는 구분해서 써야 한다.

우리 연구와의 핵심 차별점은 다음과 같이 정리할 수 있다.

- 기존 연구 상당수는 action choice, expert intervention, clarification request를 묻는다.
- 우리 시스템은 symbolic KB 안의 task-level state hypothesis를 묻고, 답변을 belief update와 subsequent planning에 직접 반영한다.
- 기존 연구는 "when to ask"에 강하고, 우리는 "when to ask + what state fact to ask + how the answer changes planning"을 연결한다.

## Quick Comparison

| No. | Paper | Main Area | Help/Query Trigger | Query/Help Content | Relevance to Our Work |
|---:|---|---|---|---|---|
| 1 | ELBA: Learning by Asking for Embodied Visual Navigation and Task Completion | Embodied vision-language task completion | Action/object prediction confusion: entropy or gradient magnitude | Free-form/template QA about future subgoals, object location, appearance, direction | 질문 생성과 task ambiguity 해소 측면에서 관련. Symbolic belief/planning은 약함 |
| 2 | HULA: Decision Making for Human-in-the-loop Robotic Agents via Uncertainty-Aware RL | Human-in-the-loop RL | Return variance from current state | Expert-provided action | 도움 요청 timing 연구로 관련. 질문 내용은 state verification이 아님 |
| 3 | Learning When to Ask for Help | Interactive navigation | Navigation policy feature uncertainty / learned interaction policy | Human manual control intervention | selective help-seeking 근거로 유용. Natural-language/state-level query는 future work로 언급 |
| 4 | KnowNo: Robots That Ask For Help | LLM robot planning + conformal prediction | CP prediction set size > 1 | Human selects among LLM-generated action/plan options | 가장 직접적인 baseline. Action-level query와 우리 state-level query 비교에 중요 |
| 5 | Introspective Planning | LLM planning + RAG + conformal prediction | Introspective CP prediction set size > 1 | Follow-up instruction/clarification among candidate plans | KnowNo의 확장형에 가까움. 불필요한 질문과 unsafe options 감소 논리 참고 가능 |
| 6 | Conformalized Interactive Imitation Learning | Interactive imitation learning | IQT-calibrated prediction interval size | Expert action label/intervention | online conformal uncertainty + intermittent feedback 근거. Task planning query와는 다름 |
| 7 | Asking the Right Question at the Right Time | Collaborative dialogue clarification | Entropy over model-predicted attributes | Template clarification question | uncertainty-based clarification timing 근거. 로봇 논문은 아니지만 query timing에 유용 |
| 8 | 불확실성 해소를 위한 로봇의 질문 생성 전략에 대한 사용자 만족도 연구 | HRI user satisfaction | Predefined uncertain command condition | Robot asks user instead of commonsense inference | 질문 전략이 응답 적절성 만족도를 높인다는 사용자 연구 근거 |
| 9 | When to Seek Help: Trust-Aware Assistance-Seeking | Trust-aware HRI + POMDP | POMDP policy over hidden human trust and task complexity | Robot asks human assistance for object collection | 도움 요청이 trust dynamics와 team performance에 미치는 영향 근거 |

## Detailed Review Table

<table>
<thead>
<tr>
<th>No.</th>
<th>Paper</th>
<th>1. Motivation</th>
<th>2. 풀고자 하는 문제</th>
<th>3. 방법론</th>
<th>4. 기여</th>
<th>5. 한계</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td><b>ELBA: Learning by Asking for Embodied Visual Navigation and Task Completion</b><br><code>2302.04865v3.pdf</code></td>
<td>Embodied agent가 복잡한 household task를 수행할 때 instruction, visual observation, object interaction 사이에 ambiguity가 생긴다. 기존 vision-language navigation/task completion 모델은 주어진 instruction을 따르는 데 집중하고, 추가 정보를 능동적으로 질문하는 능력이 약하다.</td>
<td>에이전트가 task 수행 중 불확실한 순간을 감지하고, task completion에 도움이 되는 질문을 생성 및 선택하는 문제. 특히 단순 navigation이 아니라 object interaction이 포함된 vision-dialog task completion에서 언제 질문하고 무엇을 물어볼지 결정해야 한다.</td>
<td>ELBA는 네 개 모듈로 구성된다.<br><br><b>Actioner</b>: dialogue history, visual observation, action history를 multimodal encoder로 encoding하고 다음 action/object distribution을 예측한다.<br><br><b>Confusion module</b>: action/object distribution의 entropy 또는 hidden state gradient norm을 이용해 confusion을 측정한다. threshold를 넘으면 질문 생성을 시도한다. 질문-답변을 state에 넣었을 때 entropy/confusion이 줄어들면 실제로 질문한다.<br><br><b>Planner</b>: T5 기반 module로 future high-level subgoal sequence를 생성한다. 예: find pot, pickup potato 등.<br><br><b>QA Generator/Evaluator</b>: template QA와 free-form QA를 생성한다. 후보 QA pair는 DistilBERT 기반 evaluator가 state와 QA pair embedding similarity를 계산해 ranking한다. 최상위 QA를 선택해 dialogue history에 추가한다.</td>
<td>Embodied vision-language task completion에서 "when to ask"와 "what to ask"를 함께 학습하는 모델을 제안했다. TEACh benchmark에서 질문 없는 E.T. baseline보다 task success/goal-condition success를 개선했다. 고정 주기 질문보다 질문 수를 줄이면서도 성능을 높이는 결과를 보였다.</td>
<td>질문과 답변 품질이 항상 안정적이지 않다. 논문 내 failure case로 잘못된 visual attribute 예측, 잘못 형성된 QA pair, ill-timed repeated QA가 제시된다. 또한 oracle/template 정보와 future subgoal prediction에 의존하며, 실제 로봇에서 사람 답변 오류나 low-level action failure까지 통합하지는 않는다.</td>
</tr>

<tr>
<td>2</td>
<td><b>Decision Making for Human-in-the-loop Robotic Agents via Uncertainty-Aware Reinforcement Learning</b><br><code>2303.06710v2.pdf</code></td>
<td>HitL robot은 대부분 자율적으로 동작하되 어려운 상황에서 expert assistance를 요청할 수 있어야 한다. 너무 적게 요청하면 실패하고, 너무 많이 요청하면 expert workload가 커진다.</td>
<td>RL agent가 deployment에서 제한된 expert call budget을 효율적으로 쓰도록, 어떤 state에서 expert action을 요청해야 하는지 결정하는 문제. 학습 중에는 expert call 없이 uncertainty를 배워야 한다.</td>
<td>HULA는 return variance를 agent uncertainty로 사용한다.<br><br>표준 Q-learning/DQN이 expected return Q(s,a)를 추정하는 것처럼, return의 second moment M(s,a)=E[R^2]도 Bellman-like recursion으로 추정한다. 이후 Var(R)=M(s,a)-Q(s,a)^2를 계산한다.<br><br>Deployment에서는 현재 state의 return variance가 threshold 이상이면 expert action을 요청하고, 그렇지 않으면 agent policy를 따른다. Expert는 고품질 action을 제공한다고 가정한다.</td>
<td>학습 중 expert query 없이도 deployment에서 expert call budget을 사용할 수 있는 HitL RL 방법을 제안했다. Return variance가 도움 요청 시점 결정에 유용하다는 점을 보였고, fully/partially observable grid-world에서 penalty-based expert use보다 budget에 robust한 성능을 보였다.</td>
<td>실험이 discrete navigation/grid-world 중심이다. Expert가 항상 좋은 action을 준다는 가정이 강하다. 도움 요청은 action-level expert intervention이며, query 내용이나 사람이 판단하기 쉬운 state fact 설계는 다루지 않는다. Continuous control과 language-guided navigation은 future work로 남아 있다.</td>
</tr>

<tr>
<td>3</td>
<td><b>Learning When to Ask for Help: Efficient Interactive Navigation via Implicit Uncertainty Estimation</b><br><code>2305.16502v3.pdf</code></td>
<td>Simulation에서 학습한 navigation policy는 낯선 환경에서 쉽게 실패한다. 지속적인 teleoperation은 부담이 크지만, 적은 intervention만으로도 failure를 막을 수 있다.</td>
<td>Visual navigation agent가 unfamiliar environment에서 언제 autonomous execution을 계속하고 언제 human expert에게 control을 넘길지 결정하는 문제.</td>
<td>기존 point-goal navigation policy의 visual feature를 재사용해 별도의 lightweight interaction policy를 학습한다. Interaction policy는 embedded visual input과 goal 정보를 받아 human assistance 여부를 결정한다.<br><br>Reward는 task success, goal distance improvement, time penalty, human interaction penalty를 포함한다. Human expert는 simulation에서는 shortest-path planner로 대체하고, FlightGoggles 환경에서는 실제 human joystick input을 사용한다.</td>
<td>Part-time human interaction만으로 navigation success를 크게 개선했다. Habitat HM3D validation에서 baseline success 0.523에서 0.906으로 향상되었고 interaction rate는 0.30이었다. 다른 환경으로 zero-shot transfer했을 때 autonomous success가 0.1 미만이었으나 real human expert와 함께 0.92까지 개선되었다.</td>
<td>도움 방식이 manual control hand-off라 human attention 요구가 여전히 크다. 저자들도 자연어 질문/고수준 verbal command를 통한 더 효율적인 interaction을 future work로 언급한다. Task-level semantic uncertainty나 symbolic planning과 직접 연결되지는 않는다.</td>
</tr>

<tr>
<td>4</td>
<td><b>Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners</b> / KnowNo<br><code>2307.01928v2.pdf</code></td>
<td>LLM planner는 commonsense와 step-by-step planning에 강하지만 hallucination과 overconfidence 문제가 있다. 자연어 instruction은 본질적으로 ambiguous할 수 있으므로, robot은 모를 때 물어볼 수 있어야 한다.</td>
<td>LLM planner의 uncertainty를 calibrate하여, 사용자 지정 성공률을 보장하면서 human help를 최소화하는 문제. 즉, "calibrated confidence"와 "minimal help"를 동시에 달성해야 한다.</td>
<td>문제를 MCQA로 바꾼다. LLM이 context, observation, instruction으로부터 candidate next steps를 생성하고 각 option confidence를 출력한다.<br><br>Conformal Prediction(CP)을 calibration set에 적용해 threshold를 얻고, test scenario에서 confidence가 threshold 이상인 option들을 prediction set으로 만든다. Prediction set이 singleton이면 해당 action을 실행하고, non-singleton이면 human에게 option 선택을 요청한다.<br><br>Single-step과 multi-step planning 모두를 다루며, multi-step에서는 sequence-level calibration으로 CP의 i.i.d. 문제를 완화하려 한다.</td>
<td>LLM robot planner에서 "when to ask for help"를 conformal prediction으로 정식화했다. User-specified task completion level에 대한 통계적 보장을 제공하고, baseline 대비 human help를 10-24% 줄였다고 보고한다. Simulation과 real robot manipulation에서 다양한 ambiguity를 평가했다.</td>
<td>보장은 text input에 object/environment가 충분히 grounded되어 있고 LLM이 제안한 action이 실제로 실행 가능하다는 가정에 의존한다. Perception uncertainty와 low-level policy uncertainty는 calibration에 포함되지 않는다. Human이 항상 faithful help를 준다는 가정도 있다. Query는 주로 action/plan option 선택이어서, 사용자가 planner state를 추론해야 할 수 있다.</td>
</tr>

<tr>
<td>5</td>
<td><b>Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity</b><br><code>2402.06529v4.pdf</code></td>
<td>KnowNo류 CP는 uncertainty를 보정하지만, underlying LLM reasoning이 약하면 prediction set이 과하게 넓거나 unsafe option을 포함할 수 있다. LLM이 task ambiguity와 safety를 더 잘 introspect하도록 유도할 필요가 있다.</td>
<td>LLM planner가 task ambiguity, user intent, safety constraint를 더 잘 반영하여 uncertainty를 표현하고, 불필요한 clarification request와 unsafe action을 줄이는 문제.</td>
<td>Offline에서 human-selected safe/compliant plan에 대한 post-hoc rationale을 LLM으로 생성해 introspective reasoning knowledge base를 만든다. 각 entry는 instruction embedding을 key로 저장된다.<br><br>Deployment에서는 test instruction과 유사한 KB example을 retrieve하고, 이를 few-shot prompt로 넣어 candidate plan과 explanation을 생성한다. 이후 direct prediction 또는 conformal prediction을 적용한다.<br><br>Introspective CP는 explanation을 먼저 생성한 뒤 각 option의 confidence를 query하고, CP threshold로 prediction set을 만든다. Prediction set이 여러 개면 follow-up instruction을 요청한다.</td>
<td>Retrieval-augmented planning과 conformal prediction을 결합해 LLM planner의 uncertainty alignment를 개선했다. Safe Mobile Manipulation benchmark와 추가 metric을 제안했다. KnowNo 대비 overask, overstep, unsafe contamination을 줄이는 방향의 결과를 보였다.</td>
<td>Direct prediction과 conformal prediction 사이의 성능 gap이 여전히 있다. 현재 single-label CP는 option들이 mutually exclusive라고 가정하는데, 실제 ambiguity는 multi-label valid option일 수 있다. Multi-label CP 시도는 prediction set이 너무 conservative해지는 문제가 있었다.</td>
</tr>

<tr>
<td>6</td>
<td><b>Conformalized Interactive Imitation Learning: Handling Expert Shift &amp; Intermittent Feedback</b><br><code>2410.08852v2.pdf</code></td>
<td>Imitation learning policy는 deployment distribution shift에 취약하다. 기존 ensemble/dropout uncertainty는 deployment 중 받은 human feedback을 uncertainty update에 즉시 반영하지 못한다.</td>
<td>Interactive IL에서 expert feedback이 간헐적으로만 들어오는 상황에서도 uncertainty를 online calibration하고, expert shift가 발생하면 robot이 더 적극적으로 feedback을 요청하게 만드는 문제.</td>
<td>Online conformal prediction을 intermittent label setting으로 확장한 Intermittent Quantile Tracking(IQT)을 제안한다. Expert label이 관측될 때만 residual 기반 quantile interval을 update하고, 관측되지 않으면 유지한다.<br><br>ConformalDAgger는 robot policy의 predicted action 주변에 calibrated interval을 만들고, interval size가 uncertainty threshold를 넘으면 expert action label을 query한다. Human-gated intervention과 robot-gated query를 함께 observation model로 표현한다. Episode 후에는 observed expert labels를 DAgger 방식으로 dataset에 aggregate하고 policy를 retrain한다.</td>
<td>Expert shift/drift 상황에서 기존 EnsembleDAgger, SafeDAgger보다 빠르게 uncertainty를 키우고 expert feedback을 더 요청하여 expert intention에 빨리 적응했다. Simulation과 7-DOF manipulator hardware experiment를 모두 포함한다.</td>
<td>Task planning이나 semantic clarification이 아니라 low-level action label feedback 중심이다. Threshold tuning과 observation likelihood modeling이 필요하다. Expert feedback 품질과 human burden에 대한 더 복잡한 모델은 제한적이다.</td>
</tr>

<tr>
<td>7</td>
<td><b>Asking the Right Question at the Right Time: Human and Model Uncertainty Guidance to Ask Clarification Questions</b><br><code>2024.eacl-long.16.pdf</code></td>
<td>Clarification question은 dialogue에서 ambiguity와 underspecification을 해결하는 핵심 도구다. 하지만 human clarification behavior를 그대로 supervision으로 쓰는 것이 model uncertainty 해소에 항상 적합한지는 불분명하다.</td>
<td>Collaborative dialogue task에서 model uncertainty와 human clarification-seeking behavior가 얼마나 일치하는지 분석하고, model uncertainty를 기반으로 언제 clarification question을 해야 하는지 결정하는 문제.</td>
<td>CoDraw task를 사용한다. Drawer agent는 Teller instruction을 받아 clipart scene을 재구성한다. Agent가 object attribute를 예측할 때 entropy가 threshold를 넘으면 clarification question을 생성한다.<br><br>주된 실험은 size attribute에 집중한다. Template-based question을 사용하고, question-answer pair를 dialogue history에 추가한 뒤 drawing action을 수행한다. Human clarification behavior를 학습한 baseline과 model uncertainty 기반 QDrawer를 비교한다.</td>
<td>Human clarification decision과 model uncertainty의 관계가 약하다는 점을 보였다. 따라서 human clarification behavior를 직접 모방하는 것보다, model uncertainty를 기준으로 질문하는 방식이 task success에 더 효과적일 수 있음을 보였다.</td>
<td>로봇 실험은 아니며 collaborative drawing dialogue task이다. 질문은 주로 size attribute에 제한되고 template-based generation을 사용한다. 복잡한 visual grounding, 다양한 attribute, open-ended query generation으로 확장하려면 추가 연구가 필요하다.</td>
</tr>

<tr>
<td>8</td>
<td><b>불확실성 해소를 위한 로봇의 질문 생성 전략에 대한 사용자 만족도 연구</b><br><code>불확실성 해소를 위한 로봇의 질문 생성 전략에 대한 사용자 만족도 연구.pdf</code></td>
<td>로봇은 실환경 작업 중 사용자 명령 모호성, 사물 위치 불명확성, 사용자 선호도 등으로 인해 불확실성을 만난다. 모든 불확실성을 commonsense inference로 해결할 수 없으므로 질문 전략이 필요하다.</td>
<td>불확실성 상황에서 로봇이 commonsense 기반 추론으로 행동하는 경우와 사용자에게 질문하여 해소하는 경우, 사용자 만족도에 차이가 있는지 평가하는 문제.</td>
<td>온라인 사용자 평가를 수행했다. 시뮬레이션 환경에서 테이블 위 과일을 대상으로 한 작업 명령 영상을 제작했다.<br><br>Group A는 불확실한 명령을 질문 없이 commonsense inference로 수행한다. Group B는 질문-응답을 통해 불확실성을 해소한 후 수행한다. 두 그룹의 최종 작업 결과는 동일하다고 가정한다.<br><br>참가자는 영상을 보고 응답 속도 만족도와 응답 적절성 만족도를 5점 Likert scale로 평가했다. F-test와 t-test를 사용해 그룹 차이를 분석했다.</td>
<td>응답 속도 만족도는 두 그룹 간 유의한 차이가 없었지만, 응답 적절성 만족도는 질문 전략 그룹이 유의하게 높았다. 즉, 로봇이 불확실할 때 질문하는 것이 사용자가 느끼는 response appropriateness를 높일 수 있음을 보였다.</td>
<td>온라인 영상 기반 평가이며 실제 로봇 interaction이 아니다. 두 그룹의 final task outcome이 동일하다고 가정해 task performance 차이는 평가하지 않는다. 질문 수, timing, workload, fatigue, long-horizon planning 영향은 제한적으로 다룬다.</td>
</tr>

<tr>
<td>9</td>
<td><b>When to Seek Help: Trust-Aware Assistance-Seeking in Human-Supervised Autonomy</b><br><code>When to Seek Help- Trust-Aware Assistance-Seeking in Human-Supervised Autonomy.pdf</code></td>
<td>Human-supervised autonomy에서 robot의 도움 요청은 task performance뿐 아니라 human trust에도 영향을 준다. 적절한 trust calibration은 disuse와 misuse를 모두 줄이는 데 중요하다.</td>
<td>Robot이 object collection task에서 autonomous attempt와 human assistance request 중 무엇을 선택해야 하는지, 이때 hidden human trust와 task complexity를 고려해 team performance를 최적화하는 문제.</td>
<td>두 단계 접근이다.<br><br><b>IOHMM trust model</b>: human trust를 hidden state(high/low)로 두고, robot action, task complexity, previous experience가 trust transition과 human intervention probability에 미치는 영향을 학습한다.<br><br><b>POMDP assistance policy</b>: state는 trust, experience, complexity로 구성된다. Robot action은 autonomous collection 또는 ask assistance이다. Observation은 human rely/intervene action이다. Reward는 object collection success를 높이고 human effort/interruption을 줄이도록 설계된다.<br><br>첫 번째 user experiment로 IOHMM/POMDP parameter를 추정하고, 두 번째 experiment에서 trust-aware policy와 trust-agnostic policy를 비교했다.</td>
<td>Assistance-seeking이 human trust에 미치는 영향을 모델링하고, trust-aware POMDP policy가 trust-agnostic policy보다 team performance와 post-task trust를 개선함을 보였다. Behavioral data 기반 trust estimate가 self-report trust와 isomorphic함도 보였다.</td>
<td>Supervisory object collection task에 특화되어 있고, trust state를 high/low로 단순화한다. 사용자별 trust dynamics를 adaptive하게 학습하는 것은 future work이다. 질문 내용 자체보다는 assistance request decision에 초점이 있다.</td>
</tr>
</tbody>
</table>

## Cross-Paper Interpretation

### 1. "When to ask"와 "What to ask"는 다른 문제다

HULA, Learning When to Ask for Help, ConformalDAgger, Trust-Aware Assistance-Seeking은 대부분 "언제 도움을 요청할 것인가"에 집중한다. 이때 사람의 역할은 expert action 제공, manual control, intervention, assistance이다.

반면 ELBA와 EACL 2024 논문은 "무엇을 질문할 것인가"를 다룬다. 하지만 이들 역시 질문이 symbolic planning state와 직접 연결되지는 않는다. ELBA는 future subgoal과 QA relevance를 사용하고, EACL 논문은 attribute entropy 기반 clarification을 사용한다.

KnowNo와 Introspective Planning은 "when to ask"와 "which candidate plan to clarify"를 연결하지만, 질문은 주로 candidate action/plan set에 대한 disambiguation이다.

### 2. 우리 연구의 위치

우리 시스템은 기존 흐름과 다음 지점에서 다르다.

- KnowNo류: action-level candidate 중 무엇을 할지 묻는다.
- Interactive IL/RL류: expert action이나 intervention을 요청한다.
- Dialogue clarification류: model prediction uncertainty가 높은 attribute를 묻는다.
- 우리 시스템: symbolic KB의 task-level state hypothesis를 묻고, 답변을 particle belief filtering과 planning에 반영한다.

따라서 교수님 질문에 대한 답변으로는 "Symbolic KB와 Symbolic Planning이 왜 표의 장점인가"를 다음처럼 연결할 수 있다.

Symbolic KB가 있기 때문에 로봇은 uncertainty를 사람이 검증 가능한 task fact로 표현할 수 있다. 예를 들어 "이 객체가 plastic인가?", "이 tomato가 ripe인가?", "이 object가 존재하는가?"와 같은 state-level question이 가능하다. 이는 KnowNo처럼 사용자가 planner의 next action을 선택하게 하는 것보다 인지적으로 쉽다.

Symbolic Planning이 있기 때문에 그 state fact가 action precondition/effect를 통해 downstream decision에 직접 영향을 준다. 즉 질문은 단순한 clarification이 아니라 belief update와 reachable action pruning으로 이어진다. 이 점이 task success, query efficiency, workload reduction의 구조적 원인이 된다.

### 3. Related Work 표에 넣을 때의 권장 분류

기존 표의 "Uncertainty-aware query systems"에는 다음을 포함하는 것이 자연스럽다.

- KnowNo: `ren2023robots`
- Introspective Planning: `liang2024introspective`
- LBAP가 있다면 해당 계열

추가 후보로는 다음을 고려할 수 있다.

- ELBA: embodied learning-by-asking. Proactive query는 가능하지만 planning under uncertainty와 symbolic planning은 약함.
- Learning When to Ask for Help: selective help-seeking은 강하지만 query content는 manual control.
- ConformalDAgger: online conformal feedback query는 강하지만 imitation learning action label 중심.
- Trust-Aware Assistance-Seeking: human trust와 assistance request timing은 강하지만 uncertainty-aware state query는 아님.

### 4. 우리 논문에서 쓸 수 있는 문장 초안

Prior uncertainty-aware query systems have shown that robots can improve reliability by asking for human input when planner confidence is low. However, the requested input is often an action-level choice, an expert intervention, or a clarification among candidate plans. In contrast, our framework asks users to verify symbolic task-state hypotheses. Because these hypotheses are grounded in the knowledge base and linked to action preconditions and effects, user feedback directly updates the belief state and constrains subsequent planning.

Korean version:

기존 uncertainty-aware query 시스템들은 planner confidence가 낮을 때 사람에게 묻는 것이 로봇의 신뢰성과 효율성을 높일 수 있음을 보였다. 그러나 많은 방법은 다음 행동 후보 중 하나를 고르게 하거나, expert intervention을 요청하거나, candidate plan 수준의 clarification을 요구한다. 반면 본 연구는 symbolic KB에 표현된 task-level state hypothesis를 사용자에게 검증하게 한다. 이 hypothesis는 symbolic action의 precondition/effect와 연결되어 있으므로, 사용자의 답변은 belief update와 후속 planning에 직접 반영된다.

