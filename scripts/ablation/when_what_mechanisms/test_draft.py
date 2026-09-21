from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts.ablation.when_what_mechanisms.cp_when import KnowNoCPWhenEvaluator
from scripts.ablation.when_what_mechanisms.policies import (
    BeliefThresholdWhen,
    InformationGainWhat,
    cp_trigger,
)
from scripts.ablation.when_what_mechanisms.policy_types import PolicyContext


class FakeState:
    def __init__(self, facts):
        self.facts = list(facts)

    def has_fact(self, fact):
        return fact in self.facts


class FakeBelief:
    def __init__(self):
        self.knowledge = FakeState([])
        self.frontier = [FakeState(["p(a)"]), FakeState([])]
        self.frontier_weights = [0.5, 0.5]


class FakeTomatoEnvironment:
    obj_type = {
        "tomato(T)": ["tomato1", "tomato2", "tomato3", "tomato4"],
        "robot(R)": ["brl_robot"],
        "location(L)": ["dock_station", "stem_01", "stem_02"],
    }


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


def context():
    return PolicyContext(
        belief=FakeBelief(),
        feedback_manager=FakeManager(),
        env=None,
        observation_facts=[],
        action_history=[],
        step=1,
        query_index=0,
        seed=42,
    )


class PolicyDraftTest(unittest.TestCase):
    def test_cp_trigger(self):
        self.assertEqual(cp_trigger([], "E"), (True, "empty"))
        self.assertEqual(cp_trigger(["A", "B"], "E"), (True, "multiple"))
        self.assertEqual(cp_trigger(["E"], "E"), (True, "contains_noopt"))
        self.assertEqual(cp_trigger(["A"], "E"), (False, "singleton_action"))

    def test_threshold_when(self):
        decision = BeliefThresholdWhen(0.8).should_start(context())
        self.assertTrue(decision.start)
        self.assertEqual(decision.reason, "below_threshold")

    def test_information_gain_what(self):
        decision = InformationGainWhat().select_fact(context())
        self.assertIsNotNone(decision)
        self.assertEqual(decision.fact, "p(a)")

    def test_cp_evaluator_builds_prediction_set_without_hidden_state(self):
        ctx = context()
        ctx.env = FakeTomatoEnvironment()
        ctx.belief.knowledge = FakeState([
            "located(brl_robot,dock_station)",
            "handempty(brl_robot)",
        ])
        generation = "A) navigate to stem_01\nB) navigate to stem_02\nC) detect stem_01\nD) detect stem_02"
        scoring_response = {
            "choices": [{
                "text": "A",
                "logprobs": {"top_logprobs": [{
                    " A": -0.1,
                    " B": -1.0,
                    " C": -2.0,
                    " D": -3.0,
                    " E": -4.0,
                }]},
            }]
        }
        evaluator = KnowNoCPWhenEvaluator.__new__(KnowNoCPWhenEvaluator)
        evaluator.domain = "tomato"
        evaluator.qhat = 0.8404
        evaluator.score_temperature = 5.0
        evaluator.settings = {}
        with patch(
            "scripts.ablation.when_what_mechanisms.cp_when.call_llm",
            side_effect=[({}, generation), (scoring_response, "A")],
        ):
            result = evaluator.evaluate(ctx)
        self.assertTrue(result.options)
        self.assertIn(result.fallback_token, {"A", "B", "C", "D", "E"})
        self.assertTrue(result.prediction_set)


if __name__ == "__main__":
    unittest.main()
