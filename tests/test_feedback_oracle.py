from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models.action import Action
from models.belief import Belief
from models.feedback_manager import FeedbackManger
from models.state import State
from models.transition import NextStateOutcome


class StubTransitionModel:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def get_next_state_distribution(self, state, action):
        return self.outcomes


def make_manager(domain, outcomes):
    args = SimpleNamespace(
        domain=domain,
        answer_type="auto",
        initial_state="unused.yaml",
    )
    manager = FeedbackManger(
        args,
        conf_threshold=1.0,
        transition_model=StubTransitionModel(outcomes),
    )
    manager._true_init_facts = set()
    return manager


class FeedbackOracleTests(unittest.TestCase):
    def test_feedback_uses_executed_successor_without_resampling(self):
        manager = make_manager("tomato", [])
        failed_state = State(["handempty(brl_robot)", "at(tomato1,stem1)"])
        succeeded_state = State(["holding(brl_robot,tomato1)"])
        belief = Belief(State(), [failed_state, succeeded_state], np.array([0.5, 0.5]))
        action = Action(name="pick(brl_robot,tomato1,stem1)")

        with patch.object(
            manager,
            "_sample_oracle_successor_facts",
            side_effect=AssertionError("executed action must not be sampled again"),
        ):
            result = manager.get_new_observation(
                belief,
                action_name=action.name,
                action=action,
                oracle_prior_state=State(),
                oracle_state_facts=failed_state.facts,
                oracle_successor_facts=failed_state.facts,
            )

        self.assertTrue(result.knowledge.has_fact("handempty(brl_robot)"))
        self.assertTrue(result.knowledge.has_fact("at(tomato1,stem1)"))
        self.assertFalse(result.knowledge.has_fact("holding(brl_robot,tomato1)"))

    def test_tomato_pick_success_maps_deleted_and_added_facts(self):
        outcomes = [
            NextStateOutcome(
                State(["holding(brl_robot,tomato1)"]),
                0.95,
            ),
            NextStateOutcome(
                State(["handempty(brl_robot)", "at(tomato1,stem1)"]),
                0.05,
            ),
        ]
        manager = make_manager("tomato", outcomes)
        action = Action(name="pick(brl_robot,tomato1,stem1)")

        with patch("numpy.random.choice", return_value=0) as choice:
            successor = manager._sample_oracle_successor_facts(action, State())

        np.testing.assert_allclose(choice.call_args.kwargs["p"], [0.95, 0.05])
        self.assertIs(manager.query_oracle(
            "holding(brl_robot,tomato1)", action.name,
            oracle_successor_facts=successor,
        ), True)
        self.assertIs(manager.query_oracle(
            "handempty(brl_robot)", action.name,
            oracle_successor_facts=successor,
        ), False)
        self.assertIs(manager.query_oracle(
            "at(tomato1,stem1)", action.name,
            oracle_successor_facts=successor,
        ), False)

    def test_waste_pick_failure_keeps_handempty_and_waste_location(self):
        outcomes = [
            NextStateOutcome(State(["holding(brl_robot,waste1)"]), 0.90),
            NextStateOutcome(
                State(["handempty(brl_robot)", "at(waste1,table)"]),
                0.10,
            ),
        ]
        manager = make_manager("wastesorting", outcomes)
        action = Action(name="pick(brl_robot,waste1)")

        with patch("numpy.random.choice", return_value=1):
            successor = manager._sample_oracle_successor_facts(action, State())

        self.assertIs(manager.query_oracle(
            "holding(brl_robot,waste1)", action.name,
            oracle_successor_facts=successor,
        ), False)
        self.assertIs(manager.query_oracle(
            "handempty(brl_robot)", action.name,
            oracle_successor_facts=successor,
        ), True)
        self.assertIs(manager.query_oracle(
            "at(waste1,table)", action.name,
            oracle_successor_facts=successor,
        ), True)

    def test_detect_and_scan_do_not_sample_a_virtual_successor(self):
        outcomes = [NextStateOutcome(State(["wrong_fact"]), 1.0)]
        tomato_manager = make_manager("tomato", outcomes)
        waste_manager = make_manager("wastesorting", outcomes)

        self.assertIsNone(tomato_manager._sample_oracle_successor_facts(
            Action(name="detect(brl_robot,stem1)"), State()
        ))
        self.assertIsNone(tomato_manager._sample_oracle_successor_facts(
            Action(name="scan(brl_robot,tomato1)"), State()
        ))
        self.assertIsNone(waste_manager._sample_oracle_successor_facts(
            Action(name="detect_waste(brl_robot)"), State()
        ))

if __name__ == "__main__":
    unittest.main()
