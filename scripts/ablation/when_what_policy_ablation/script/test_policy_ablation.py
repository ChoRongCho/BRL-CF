from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.ablation.when_what_policy_ablation.script.cp_when import _current_facts
from scripts.ablation.when_what_policy_ablation.script.policies import (
    BeliefThresholdWhen,
    ConformalActionAmbiguityWhen,
    InformationGainWhat,
    QueryValueWhat,
    QueryValueWhen,
    build_policy_bundle,
    cp_trigger,
)
from scripts.ablation.when_what_policy_ablation.script.policy_types import PolicyContext
from scripts.ablation.when_what_policy_ablation.script.policy_types import WhatDecision, WhenDecision
from scripts.ablation.when_what_policy_ablation.script.query_episode import run_query_episode


class FakeState:
    def __init__(self, facts, fluents=None):
        self.facts = list(facts)
        self.fluents = fluents or {}

    def has_fact(self, fact):
        return fact in self.facts


class FakeBelief:
    def __init__(self, frontier=None, weights=None):
        self.knowledge = FakeState([])
        self.frontier = frontier or [FakeState(["p(a)"]), FakeState([])]
        self.frontier_weights = weights or [0.5, 0.5]


class FakeManager:
    def compute_confidence(self, weights):
        return 0.0

    def get_changed_facts(self, knowledge, frontier):
        facts = set().union(*(set(state.facts) for state in frontier))
        return sorted(
            fact for fact in facts
            if len({state.has_fact(fact) for state in frontier}) == 2
        )

    def normalize(self, weights):
        return weights

    def entropy(self, weights):
        return 1.0 if len(weights) == 2 else 0.0

    def expected_entropy_after_asking(self, frontier, weights, fact):
        return 0.0


class FakeValueResult:
    def __init__(self, *, physical_q, query_q, root_mode):
        self.physical_q = physical_q
        self.query_q = query_q
        self.root_mode = root_mode

    def as_dict(self):
        return {
            "physical_q": dict(self.physical_q),
            "query_q": dict(self.query_q),
            "root_mode": self.root_mode,
        }


class FakeValueEvaluator:
    def __init__(self, *, physical_q=None, query_q=None):
        self.physical_q = physical_q or {}
        self.query_q = query_q or {}
        self.calls = []

    def evaluate(self, context, *, query_facts, root_mode):
        self.calls.append((list(query_facts), root_mode))
        return FakeValueResult(
            physical_q=self.physical_q if root_mode == "compare" else {},
            query_q={fact: self.query_q[fact] for fact in query_facts},
            root_mode=root_mode,
        )


class FakeCPResult:
    def __init__(self, prediction_set, fallback_token="E"):
        self.prediction_set = prediction_set
        self.fallback_token = fallback_token

    def as_dict(self):
        return {
            "prediction_set": self.prediction_set,
            "fallback_token": self.fallback_token,
        }


class FakeCPEvaluator:
    def __init__(self, prediction_set, fallback_token="E"):
        self.result = FakeCPResult(prediction_set, fallback_token)

    def evaluate(self, policy_context):
        return self.result


def context(belief=None):
    return PolicyContext(
        belief=belief or FakeBelief(), feedback_manager=FakeManager(), env=None,
        observation_facts=[], action_history=[], step=1, query_index=0, seed=42,
    )


class PolicyAblationTest(unittest.TestCase):
    def test_cp_trigger_uses_action_set_and_fallback(self):
        self.assertEqual(cp_trigger([], "E"), (True, "empty"))
        self.assertEqual(cp_trigger(["A", "B"], "E"), (True, "multiple"))
        self.assertEqual(cp_trigger(["E"], "E"), (True, "contains_noopt"))
        self.assertEqual(cp_trigger(["A"], "E"), (False, "singleton_action"))

    def test_cp_when_uses_action_ambiguity_only_as_trigger(self):
        decision = ConformalActionAmbiguityWhen(
            FakeCPEvaluator(["A", "B"])
        ).should_start(context())
        self.assertTrue(decision.start)
        self.assertEqual(decision.policy, "cp_action_ambiguity")
        self.assertEqual(decision.reason, "multiple")

    def test_cp_when_keeps_ours_eig_as_what_policy(self):
        bundle = build_policy_bundle(
            "cp_when", threshold=0.8,
            cp_evaluator=FakeCPEvaluator(["A", "B"]),
        )
        self.assertIsInstance(bundle.when, ConformalActionAmbiguityWhen)
        self.assertIsInstance(bundle.what, InformationGainWhat)

    def test_action_cp_reads_current_posterior_map(self):
        belief = FakeBelief(
            [FakeState(["old"]), FakeState(["new"])],
            [0.1, 0.9],
        )
        belief.knowledge = FakeState(["stale"])
        self.assertEqual(list(_current_facts(context(belief))), ["new"])

    def test_threshold_when(self):
        decision = BeliefThresholdWhen(0.8).should_start(context())
        self.assertTrue(decision.start)
        self.assertEqual(decision.reason, "below_threshold")

    def test_information_gain_what(self):
        decision = InformationGainWhat().select_fact(context())
        self.assertIsNotNone(decision)
        self.assertEqual(decision.fact, "p(a)")

    def test_value_when_compares_all_query_actions_and_only_triggers(self):
        belief = FakeBelief(
            [FakeState(["p(a)"]), FakeState(["q(b)"])],
            [0.5, 0.5],
        )
        evaluator = FakeValueEvaluator(
            physical_q={"pick": 3.0},
            query_q={"p(a)": 2.0, "q(b)": 5.0},
        )
        decision = QueryValueWhen(evaluator).should_start(context(belief))
        self.assertTrue(decision.start)
        self.assertIsNone(decision.query_fact)
        self.assertEqual(decision.diagnostics["trigger_query_fact"], "q(b)")
        self.assertEqual(evaluator.calls, [(["p(a)", "q(b)"], "compare")])

    def test_value_what_selects_highest_query_value(self):
        belief = FakeBelief(
            [FakeState(["p(a)"]), FakeState(["q(b)"])],
            [0.5, 0.5],
        )
        evaluator = FakeValueEvaluator(
            query_q={"p(a)": 2.0, "q(b)": 5.0},
        )
        decision = QueryValueWhat(evaluator).select_fact(context(belief))
        self.assertEqual(decision.fact, "q(b)")
        self.assertEqual(evaluator.calls, [(["p(a)", "q(b)"], "query_only")])

    def test_value_when_stops_after_exactly_one_question(self):
        belief = FakeBelief(
            [
                FakeState(["p(a)", "q(b)"]),
                FakeState(["q(b)"]),
                FakeState([]),
            ],
            [0.4, 0.3, 0.3],
        )
        belief.sync_knowledge_to_map = lambda: None
        belief.reset_belief = lambda: None
        manager = FakeManager()
        manager.num_of_query = 0
        manager.query_log = []
        manager.call_feedback = lambda *args, **kwargs: True

        def apply_answer(current, fact, answer):
            current.frontier = [current.frontier[0], current.frontier[1]]
            current.frontier_weights = [0.8, 0.2]
            return current

        manager.apply_fact_answer_to_belief = apply_answer

        class AlwaysWhen:
            def should_start(self, ctx):
                return WhenDecision(True, "query_value", reason="query_value_higher")

        class FirstAvailableWhat:
            def select_fact(self, ctx):
                return WhatDecision(ctx.ambiguous_facts()[0], "information_gain")

        bundle = SimpleNamespace(
            condition="value_when",
            when=AlwaysWhen(),
            what=FirstAvailableWhat(),
            max_questions_per_step=1,
        )
        _, trace = run_query_episode(
            belief=belief,
            bundle=bundle,
            feedback_manager=manager,
            env=SimpleNamespace(true_state=FakeState(["p(a)", "q(b)"])),
            observation_facts=[],
            action_history=[],
            step=1,
            seed=42,
            action=SimpleNamespace(name="detect"),
        )
        self.assertEqual(len(trace["questions"]), 1)
        self.assertEqual(trace["stop_reason"], "question_limit_reached")

    def test_when_is_rechecked_after_one_question(self):
        belief = FakeBelief()
        belief.sync_knowledge_to_map = lambda: None
        belief.reset_belief = lambda: None
        manager = FakeManager()
        manager.num_of_query = 0
        manager.query_log = []
        manager.call_feedback = lambda *args, **kwargs: True

        def apply_answer(current, fact, answer):
            current.frontier = [current.frontier[0]]
            current.frontier_weights = [1.0]
            return current

        manager.apply_fact_answer_to_belief = apply_answer

        class RecheckingWhen:
            calls = 0

            def should_start(self, ctx):
                self.calls += 1
                return WhenDecision(self.calls == 1, "test_when", reason="test")

        class OneFactWhat:
            def select_fact(self, ctx):
                return WhatDecision("p(a)", "test_what")

        when = RecheckingWhen()
        bundle = SimpleNamespace(
            condition="test", when=when, what=OneFactWhat()
        )
        _, trace = run_query_episode(
            belief=belief,
            bundle=bundle,
            feedback_manager=manager,
            env=SimpleNamespace(true_state=FakeState(["p(a)"])),
            observation_facts=[],
            action_history=[],
            step=1,
            seed=42,
            action=SimpleNamespace(name="detect"),
        )
        self.assertEqual(when.calls, 2)
        self.assertEqual(len(trace["questions"]), 1)
        self.assertEqual(len(trace["when_decisions"]), 2)


if __name__ == "__main__":
    unittest.main()
