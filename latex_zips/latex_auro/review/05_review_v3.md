# Review — "When and What to Query in Robot Task Planning under Partial Observability"

**Journal:** Autonomous Robots
**Recommendation:** Major Revision (borderline; currently below the acceptance bar)

---

## Summary

The paper proposes a method for deciding *when* a robot should ask an external agent (VLM or human) for information during symbolic task planning and *what* to ask. The robot plans online with POMCP over STRIPS-style action models. After each action it builds a weighted belief over the "frontier", i.e. the symbolically reachable successors of the current knowledge state. A query is triggered when the normalized-entropy confidence of this belief falls below a threshold tau. The fact to ask about is chosen greedily by expected entropy reduction. Answers are treated as exact and used to filter the belief. The maximum a posteriori state is then committed to the knowledge base. The method is evaluated on two domains (waste sorting with a fixed UR5e; tomato harvesting with a mobile manipulator). The evaluation covers (i) simulation with an oracle against KnowNo, IntroPlan and a HARPS-inspired POMCP-Q, (ii) ablations of query timing and selection, (iii) a threshold sweep, and (iv) physical-robot experiments with a VLM (40 trials per condition) and with five human participants (10 trials per condition) against KnowNo.

## Strengths

1. **Relevance and real-robot validation.** The problem fits the journal's scope. The physical experiments on two platforms with both VLM and human responders are a real strength for Autonomous Robots.
2. **Clear decomposition.** Separating query *timing* from query *selection* is useful. The 2D ablation design (Table 1, Table 3) cleanly shows that timing mainly drives success while selection mainly drives query count.
3. **Simple, interpretable mechanism.** Binary fact-verification questions are easy for humans and VLMs to answer. The interview results suggest participants find them less demanding than action-selection queries.
4. **Reproducibility effort.** The appendices give planner parameters, baseline settings, VLM prompts, trial-reuse policy and counterbalancing details.

## Major Concerns

### M1. Limited novelty and positioning
Entropy-thresholded triggering and greedy expected-information-gain question selection are well-established ideas. Examples include active perception, information-gathering POMDPs / rho-POMDPs (Araya-Lopez et al. 2010), value-of-information querying, and ask-for-help work such as Tellex et al. 2014 ("Asking for help using inverse semantics"), Rosenthal & Veloso, and more recent LLM clarification work. The paper's contribution is therefore mainly the *integration* of these ideas with a symbolic frontier belief inside POMCP. The authors should (a) state precisely what is new relative to this literature and (b) compare against, or at least discuss, information-gain/VOI-based query policies, not only conformal-prediction and reward-based ones.

### M2. The "belief" is collapsed to a point every step
Line 19 of Algorithm 1 commits the MAP state to the knowledge base, and the next frontier is built only from that single state. Uncertainty is therefore represented only one step at a time and is never propagated. If the MAP state is wrong but confident (Conf >= tau), the error can never be recovered by later observations. This is effectively a most-likely-state approximation rather than POMDP belief tracking. The framing as "task planning as a POMDP" should be toned down, or the authors should show that the approximation is harmless (e.g., an analysis of how often confident-but-wrong commitments occur).

### M3. Query relevance and query cost are not modeled
The candidate set Q(B) contains *every* fact that differs across frontier states, regardless of whether it matters for the goal or for the next action's preconditions. Likewise, the trigger ignores whether the uncertainty affects task success. The threshold sweep (Fig. 7) supports this concern: tau = 1.0 roughly doubles the query count with almost no gain. Although the introduction motivates the work by query cost, cost never appears in the decision rule, and tau must be tuned per domain (the authors note the domains "require different confidence levels"). Please discuss task-relevant uncertainty (e.g., restricting Q to facts that affect preconditions or goal), explicit query costs, or how tau would be selected in a new domain.

### M4. Confounded baseline comparisons
- **Planner vs. query policy.** KnowNo and IntroPlan use an LLM (GPT-4o) to select actions, whereas Ours uses POMCP with complete STRIPS action models that enforce preconditions. Many KnowNo failures are described as "autonomous action selections despite unmet preconditions" or "skip required preceding actions". These are planner failures, not querying failures. The large success gaps (e.g., 92.5% vs 25.0% with VLM in Harvest; 100% vs 30% with humans) therefore cannot be attributed to the query strategy alone. T2 (conformal timing under the same planner) partially addresses this, but only in simulation. A physical-robot or VLM comparison that uses the same planner with a different query policy is needed to support the central claim.
- **POMCP-Q tuning.** POMCP-Q uses zero query cost and the same budget of 100 simulations per step, even though its action space is larger. Its low Harvest success (57.5%) may simply reflect an insufficient search budget. Please report POMCP-Q under equal wall-clock time or with a sweep over simulation budgets.
- **Hyper-parameter fairness.** POMCP-Q's gamma was tuned per domain (0.9 / 0.5), while Ours uses gamma = 0.2, which yields an effective depth of 4. Please clarify how Ours' parameters, including tau = 0.8, were chosen, and whether they were chosen on the same trials that are reported (Appendix A.4 indicates trial reuse).
- **Missing reference points.** The main comparison should include a no-query baseline (tau = 0 gives about 55% / 52%) and an always-verify upper bound.

### M5. Oracle simulation is favorable to the proposed method
The oracle experiments assume exact answers, and the belief update in Eq. (2) uses the same T and O models that, presumably, generate the simulated noise. The paper does not describe how observation noise is generated in simulation, or how O(o | s, a) is defined or calibrated for the real detectors (SAM3, YOLOv12). Please specify the observation model, its source, and the method's sensitivity to model mismatch.

### M6. Hard filtering on answers
Eq. (5) deletes all hypotheses inconsistent with the answer. The authors acknowledge this limitation, and the real-robot failures (all three VLM failures of Ours in Sort) come directly from it. A soft Bayesian update with a responder-accuracy likelihood is a small change that would make the method considerably more robust. Its absence is notable for a journal paper, especially since VLM log-probabilities are already collected (Appendix B).

### M7. Statistical evidence, particularly for the human study
- None of the comparisons report confidence intervals or significance tests on success rates.
- The human study has n = 5 participants, 2 trials per method and domain per participant, and 10 trials per condition. Recruitment (e.g., whether participants were lab members) and familiarity with the system are not reported. Counterbalancing is incomplete (P1 and P5 share the same order).
- Cost metrics are averaged over successful trials only. When KnowNo succeeds in only 2–3 of 10 trials, these averages are not comparable to Ours. Some entries are based on a single trial (Fig. 5 caption).
- Please add, for example, Wilson intervals or Fisher exact tests for success rates, and report costs over all trials or with clear caveats.

### M8. Scale and generality
Only two small domains are tested. Harvest, for instance, has 4 tomatoes on 2 stems. The paper does not report the number of facts, the typical frontier size j, or how computation and query count scale with the number of objects. A scaling study, even in simulation, would strengthen the claims of generality and the low-planning-time claim.

## Minor Comments

1. Query-count claims: Ours asks *more* queries than KnowNo in every physical experiment (e.g., 6.70 vs 3.15 in Sort, VLM) and has longer interaction time in Sort. The abstract and contributions should make clear that query reduction is shown relative to the ablations, not the baselines. A success-vs-query-count Pareto plot, sweeping tau for Ours and coverage for KnowNo, would present the trade-off more honestly.
2. Normalized-entropy confidence (Eq. 3) depends on j. With many low-weight hypotheses, confidence can be low even when one state dominates. Please justify this choice over, e.g., the maximum belief weight. Citing Qu et al. 2023 (an IoT edge-inference paper) for normalized entropy is odd.
3. The title and abstract emphasize "partial observability", but the paper does not report the sources of partial observability in the oracle simulation (e.g., noise rates).
4. Sec. 5: "The planned human evaluation is designed for five participants" reads like proposal text and should be revised.
5. The affiliation placeholder "Institution2" must be fixed.
6. The PDDL reference is listed as "Aeronautiques, C., et al." (a known BibTeX artifact); it should be McDermott et al. (1998). Kaelbling et al. (1998) is cited for "execution failures", which is not its subject.
7. Leftover highlighted text in the Introduction ("In simulation with correct oracle responses").
8. The VLM prompts were issued in Korean. Please comment on whether the prompt language affects VLM accuracy and how this affects reproducibility for other users.
9. Fig. 1 is generic. Consider showing a concrete example of a frontier belief, a selected query and the resulting update.
10. Define "query probability" more precisely: is it per step or per trial, and is it computed over successful trials only?

## Questions to the Authors

1. How often does Ours commit a wrong MAP state with Conf >= tau, and what happens afterwards?
2. What is the success rate of KnowNo when it is given the same STRIPS precondition checking as Ours?
3. How were tau = 0.8 and gamma = 0.2 selected, and on which data?
4. How does performance change if answer reliability is modeled (e.g., VLM accuracy 0.8–0.95) instead of exact filtering?

## Overall Assessment

The paper addresses a relevant problem, is clearly written, and includes valuable physical-robot experiments, which matter for this journal. However, the methodological novelty is modest (M1). The belief representation is a strong simplification of the POMDP framing (M2). Most importantly, the large success gains over the main baseline are confounded by the difference in underlying planners (M4). The human study is too small to support strong conclusions (M7). I recommend **Major Revision**. If the authors add a same-planner comparison on the physical or VLM setting, a soft answer-reliability update, statistical tests, and a clearer positioning against information-gain querying work, the paper could become a solid contribution to Autonomous Robots.
