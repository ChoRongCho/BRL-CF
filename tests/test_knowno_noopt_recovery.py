from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np


KNOWNO_DIR = Path(__file__).resolve().parents[1] / "scripts" / "baseline" / "knowno"
KNOWNO_SCRIPTS_DIR = KNOWNO_DIR / "scripts"
for path in (KNOWNO_DIR, KNOWNO_SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.auto_answer import (
    recover_tomato_noopt_action,
    recover_waste_noopt_action,
    select_tomato_answer,
)
from scripts.knowno_action_validation import validate_tomato_action, validate_waste_action
from scripts.knowno_multistep_tomato import select_tomato_action
from scripts.knowno_multistep_wastesorting import select_waste_action
from scripts.utils.utils import PlanningResult, execute_tomato_action, execute_waste_action
from tomato_utils import parse_tomato_action
from wastesorting_utils import parse_waste_action


class StubLogger:
    def __init__(self):
        self.records = []

    def console(self, *values):
        self.records.append(("console", values))

    def json(self, *values):
        self.records.append(("json", values))


class KnowNoNoOptRecoveryTests(unittest.TestCase):
    def test_tomato_noopt_recovery_is_validated_and_executed(self):
        tokens = ["A", "B", "C"]
        planning = PlanningResult(
            options_text="",
            options=["pick tomato1", "done", "an option not listed here"],
            fallback_token="C",
            scored_tokens=tokens,
            logprobs=[-1.0, -1.0, -0.1],
            scores=np.array([0.1, 0.1, 0.8]),
            prediction_set=["C"],
        )
        logger = StubLogger()
        args = SimpleNamespace(auto_answer=True, navigate_failure_prob=0.0)
        state = dict(
            robot_location="dock_station",
            active_tomatoes=["tomato1"],
            hidden_ripeness={"tomato1": "ripe"},
            hidden_freshness={"tomato1": "fresh"},
            hidden_locations={"tomato1": "stem_02"},
            observed_properties={},
            observed_locations={},
            scanned_properties={},
            held_tomato=None,
            loaded_tomatoes=[],
            discarded_tomatoes=[],
        )

        selection = select_tomato_action(
            planning,
            args=args,
            step=1,
            tokens=tokens,
            logger=logger,
            **state,
        )
        self.assertTrue(selection.oracle_provided_action)
        self.assertEqual(selection.action, "navigate to stem_02")

        action_type, action_arg = parse_tomato_action(selection.action)
        valid, _ = validate_tomato_action(
            action_type,
            action_arg,
            state["robot_location"],
            state["active_tomatoes"],
            state["hidden_ripeness"],
            state["hidden_freshness"],
            state["observed_properties"],
            state["observed_locations"],
            state["scanned_properties"],
            state["held_tomato"],
            state["loaded_tomatoes"],
            state["discarded_tomatoes"],
        )
        self.assertTrue(valid)
        robot_location, _, _, failures, error = execute_tomato_action(
            action_type,
            action_arg,
            args=args,
            step=1,
            detected_stems=set(),
            action_history=[],
            console=logger.console,
            log_json=logger.json,
            **state,
        )
        self.assertEqual(robot_location, "stem_02")
        self.assertEqual(failures, 0)
        self.assertIsNone(error)

    def test_waste_noopt_recovery_is_validated_and_executed(self):
        tokens = ["A", "B"]
        planning = PlanningResult(
            options_text="",
            options=["place waste1 into can bin", "an option not listed here"],
            fallback_token="B",
            scored_tokens=tokens,
            logprobs=[-1.0, -0.1],
            scores=np.array([0.1, 0.9]),
            prediction_set=["B"],
        )
        logger = StubLogger()
        args = SimpleNamespace(auto_answer=True, place_failure_prob=0.0)
        remaining_objects = ["waste1"]
        hidden_attributes = {"waste1": "paper"}
        observed_attributes = {"waste1": "paper"}
        placed_objects = {}

        selection = select_waste_action(
            planning,
            args=args,
            step=1,
            tokens=tokens,
            remaining_objects=remaining_objects,
            hidden_attributes=hidden_attributes,
            observed_attributes=observed_attributes,
            held_object="waste1",
            occlusions={},
            logger=logger,
        )
        self.assertTrue(selection.oracle_provided_action)
        self.assertEqual(selection.action, "place waste1 into paper bin")

        action_type, action_arg = parse_waste_action(selection.action)
        valid, _ = validate_waste_action(
            action_type,
            action_arg,
            remaining_objects,
            hidden_attributes,
            observed_attributes,
            "waste1",
            {},
        )
        self.assertTrue(valid)
        held, _, failures, error = execute_waste_action(
            action_type,
            action_arg,
            args=args,
            step=1,
            remaining_objects=remaining_objects,
            hidden_attributes=hidden_attributes,
            observed_attributes=observed_attributes,
            held_object="waste1",
            placed_objects=placed_objects,
            occlusions={},
            action_history=[],
            console=logger.console,
            log_json=logger.json,
        )
        self.assertIsNone(held)
        self.assertEqual(placed_objects, {"waste1": "paper bin"})
        self.assertEqual(failures, 0)
        self.assertIsNone(error)

    def test_oracle_generates_action_only_when_noopt_is_the_correct_answer(self):
        common = dict(
            options=["detect stem_02", "pick tomato4", "an option not listed here"],
            tokens=["A", "B", "C"],
            add_mc_prefix="C",
            robot_location="stem_02",
            active_tomatoes=["tomato4"],
            hidden_ripeness={"tomato4": "ripe"},
            hidden_freshness={"tomato4": "fresh"},
            hidden_locations={"tomato4": "stem_02"},
            observed_properties={},
            observed_locations={},
            held_tomato=None,
            loaded_tomatoes=[],
            discarded_tomatoes=[],
        )

        noopt_answer = select_tomato_answer(**common, allowed_tokens=["B", "C"])
        self.assertEqual(noopt_answer["selected_token"], "C")
        self.assertEqual(noopt_answer["provided_action"], "detect stem_02")

        regular_answer = select_tomato_answer(**common, allowed_tokens=["A", "C"])
        self.assertEqual(regular_answer["selected_token"], "A")
        self.assertNotIn("provided_action", regular_answer)

        no_noopt_answer = select_tomato_answer(**common, allowed_tokens=["B"])
        self.assertEqual(no_noopt_answer["selected_token"], "C")
        self.assertNotIn("provided_action", no_noopt_answer)

    def test_tomato_noopt_action_comes_directly_from_hidden_state(self):
        recovery = recover_tomato_noopt_action(
            robot_location="stem_02",
            active_tomatoes=["tomato2", "tomato3", "tomato4"],
            hidden_ripeness={"tomato2": "unripe", "tomato3": "ripe", "tomato4": "ripe"},
            hidden_freshness={"tomato2": "fresh", "tomato3": "fresh", "tomato4": "fresh"},
            hidden_locations={"tomato2": "stem_01", "tomato3": "stem_02", "tomato4": "stem_02"},
            observed_properties={"tomato2": "unripe", "tomato3": "unripe"},
            observed_locations={"tomato2": "stem_01", "tomato3": "stem_02"},
            held_tomato=None,
            loaded_tomatoes=[],
            discarded_tomatoes=["tomato1"],
            scanned_properties={"tomato1": "rotten"},
        )

        self.assertEqual(recovery["provided_action"], "detect stem_02")
        self.assertEqual(recovery["source"], "hidden-state oracle")

    def test_tomato_noopt_action_navigates_to_hidden_target_location(self):
        recovery = recover_tomato_noopt_action(
            robot_location="dock_station",
            active_tomatoes=["tomato1"],
            hidden_ripeness={"tomato1": "ripe"},
            hidden_freshness={"tomato1": "fresh"},
            hidden_locations={"tomato1": "stem_02"},
            observed_properties={},
            observed_locations={},
            held_tomato=None,
            loaded_tomatoes=[],
            discarded_tomatoes=[],
        )

        self.assertEqual(recovery["provided_action"], "navigate to stem_02")
        self.assertEqual(recovery["source"], "hidden-state oracle")

    def test_tomato_noopt_prefers_stem_with_more_remaining_work(self):
        recovery = recover_tomato_noopt_action(
            robot_location="dock_station",
            active_tomatoes=["tomato1", "tomato2", "tomato3"],
            hidden_ripeness={"tomato1": "ripe", "tomato2": "ripe", "tomato3": "ripe"},
            hidden_freshness={"tomato1": "fresh", "tomato2": "fresh", "tomato3": "fresh"},
            hidden_locations={"tomato1": "stem_01", "tomato2": "stem_02", "tomato3": "stem_02"},
            observed_properties={},
            observed_locations={},
            held_tomato=None,
            loaded_tomatoes=[],
            discarded_tomatoes=[],
        )

        self.assertEqual(recovery["provided_action"], "navigate to stem_02")

    def test_waste_noopt_action_comes_directly_from_hidden_state(self):
        recovery = recover_waste_noopt_action(
            remaining_objects=["waste1"],
            hidden_attributes={"waste1": "paper"},
            observed_attributes={"waste1": "paper"},
            held_object="waste1",
            occlusions={},
        )

        self.assertEqual(recovery["provided_action"], "place waste1 into paper bin")
        self.assertEqual(recovery["source"], "hidden-state oracle")

    def test_waste_noopt_prefers_pick_that_reveals_more_objects(self):
        recovery = recover_waste_noopt_action(
            remaining_objects=["waste1", "waste2", "waste3", "waste4"],
            hidden_attributes={obj: "paper" for obj in ("waste1", "waste2", "waste3", "waste4")},
            observed_attributes={"waste1": "paper", "waste2": "paper"},
            held_object=None,
            occlusions={"waste3": "waste2", "waste4": "waste3"},
        )

        self.assertEqual(recovery["provided_action"], "pick waste2")


if __name__ == "__main__":
    unittest.main()
