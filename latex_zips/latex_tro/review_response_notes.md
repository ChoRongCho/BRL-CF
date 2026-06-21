# Review Response Notes

## 1. External Review Summary

The external review evaluates the paper as a clear and practical contribution. The main strength is the shift from action-level human guidance to state-fact verification. The reviewer also recognizes the value of combining symbolic STRIPS-style representations, POMDP/POMCP planning, entropy-based confidence, and information-gain query selection.

The review is generally positive about the breadth of evaluation. It notes that the paper includes simulation, real-robot tomato harvesting, baseline comparisons, threshold ablations, scalability tests, and a Wizard-of-Oz user study.

The main concerns are not about the core idea itself. They concern how strongly the paper interprets the evidence, especially in the user study and the KnowNo comparison.

## 2. Main Concerns Raised

### User Study Power

The user study uses 12 participants and one trial per condition. The reviewer sees this as too small for strong generalization. The concern is not simply the sample size, but the mismatch between a pilot-style study and some parts of the paper that may sound too conclusive.

### KnowNo Comparison

The reviewer points out a possible asymmetry between the system evaluation and the user study. In the system evaluation, KnowNo can fail because the correct action is absent from the generated candidate set. In the user study, the interface necessarily includes a correct option, because otherwise the human-facing task would not be well-defined.

This can make the KnowNo comparison look inconsistent unless the paper clearly separates the purpose of the two evaluations.

### Oracle Assumption

The system evaluation assumes an oracle user. This means the state-level answers are always correct. The reviewer worries that this may overestimate real-world performance because human answers can be noisy.

The user study provides some evidence about human responses, but it does not directly test how wrong human feedback affects belief filtering and downstream planning.

### Planner Limitations

Some failures of Ours occur after correct state feedback has already been provided. Typical examples include repeated actions that do not make progress and tomato-harvesting dead ends where the planner commits to place or discard before scan-related ambiguity is resolved.

The reviewer suggests separating the contribution of the query strategy from limitations of the POMCP-based planner.

### Threshold Selection

The choice of $\tau=0.8$ can look post-hoc because it is selected after observing the trade-off across evaluated thresholds. The paper should avoid presenting it as a universal setting.

## 3. Our Interpretation And Defense

### User Study As Pilot Evidence

The user study should be framed as pilot evidence for user-facing interaction trends. It is not intended to be a standalone proof of superiority. This defense is valid, but the paper must keep this framing consistent.

The paper should avoid making strong claims from non-significant or weakly powered user-study results. Numerical differences can be reported, but the interpretation should remain cautious.

### KnowNo Evaluations Ask Different Questions

The system evaluation and user study examine different aspects of KnowNo.

In the system evaluation, KnowNo is evaluated end-to-end. Its action candidates are generated from the robot's perceived state. If the perceived state is overconfident but wrong, the correct true-state action may be absent from the candidate set.

In the user study, the purpose is different. The goal is to examine the cognitive burden of action-level interaction when a valid action choice is available. For the interface to be meaningful, the correct option must be included.

Thus, the user study does not remove a weakness unfairly. It isolates a different question. The paper should state this distinction clearly in the user-study setup, system discussion, and conclusion.

### Noisy Feedback Remains A Limitation

The user study can show whether users tend to answer correctly under the presented interface. However, it does not directly measure how incorrect feedback affects the belief update or planner behavior.

This should be acknowledged as a limitation rather than over-defended. A future version could add a noisy-feedback sensitivity analysis.

### Ours And KnowNo Fail At Different Points

KnowNo can fail before action selection because the correct action may be absent from the candidate set.

Ours can fail after correct state feedback because the POMCP-based planner may still choose repeated actions or enter dead-end sequences.

This distinction helps avoid a misleading single-axis comparison. The system evaluation should explain that the methods expose different failure surfaces.

## 4. Writing Direction For The Paper

The paper should not sound defensive. The stronger framing is:

1. The framework keeps action selection with the robot.
2. Human feedback is used as evidence about the world state.
3. State verification is useful when the queried uncertainty affects planning.
4. Action-level feedback can be efficient, but it depends on the correct action being present in the candidate set.
5. The user study is exploratory and supports interaction trends, not definitive statistical claims.

## 5. Concrete Revision Checklist

- Keep the user study framed as exploratory or pilot evidence.
- Avoid strong superiority claims from the user study.
- Clearly separate system-level KnowNo failure from user-study KnowNo interaction.
- State that KnowNo candidate generation can omit the true-state action when perception is overconfident but wrong.
- State that Ours can still fail due to planner limitations after correct feedback.
- Add or preserve a limitation about noisy human feedback and its downstream effect on belief filtering.
- Make sure Random is described as a query-timing baseline, not a query-content baseline.
- Align Random query-rate numbers between text, table, and newly rerun experiments.
- Avoid presenting $\tau=0.8$ as a universal threshold.

## 6. Current Priority

The system evaluation section should make the baseline roles explicit:

- \textbf{All}: dense state verification.
- \textbf{No}: no user feedback.
- \textbf{Random}: random timing of state-level feedback with the same query rate as Ours.
- \textbf{KnowNo}: action-level feedback where the user chooses among candidate next actions.
- \textbf{Ours}: belief-driven timing and state-level verification.

The discussion should connect these results back to the paper's main philosophy. Human feedback should verify task-relevant state facts, and the robot planner should remain responsible for action selection.
