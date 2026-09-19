from pathlib import Path
import sys
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.baseline.targeted_query_pomdp.environment import Environment
from models.action import Action
from models.state import State


class HiddenExecutionPreconditionTests(unittest.TestCase):
    def make_env(self, domain, runtime_facts, hidden_facts):
        env = Environment.__new__(Environment)
        env.domain_name = domain
        env.state = State(runtime_facts)
        env.true_state = State(hidden_facts)
        return env

    def test_tomato_pick_at_wrong_hidden_stem_cannot_execute(self):
        env = self.make_env(
            "tomato",
            [
                "located(brl_robot,stem_01)",
                "handempty(brl_robot)",
                "observed(tomato3)",
                "at(tomato3,stem_01)",
                "ripe(tomato3)",
            ],
            ["at(tomato3,stem_02)", "ripe(tomato3)"],
        )
        action = Action(name="pick(brl_robot,tomato3,stem_01)")
        self.assertFalse(env._is_physically_executable(action))

    def test_tomato_pick_at_true_hidden_stem_can_execute(self):
        env = self.make_env(
            "tomato",
            ["located(brl_robot,stem_02)", "handempty(brl_robot)"],
            ["at(tomato3,stem_02)", "ripe(tomato3)"],
        )
        action = Action(name="pick(brl_robot,tomato3,stem_02)")
        self.assertTrue(env._is_physically_executable(action))

    def test_place_requires_the_target_object_to_be_held(self):
        env = self.make_env(
            "tomato",
            ["handempty(brl_robot)", "fresh(tomato1)"],
            ["fresh(tomato1)"],
        )
        action = Action(name="place(brl_robot,tomato1)")
        self.assertFalse(env._is_physically_executable(action))

    def test_occluded_waste_cannot_be_picked_before_blocker_is_cleared(self):
        env = self.make_env(
            "wastesorting",
            ["handempty(brl_robot)", "detected(waste4)"],
            ["occ(waste3,waste4)"],
        )
        action = Action(name="pick(brl_robot,waste4)")
        self.assertFalse(env._is_physically_executable(action))


if __name__ == "__main__":
    unittest.main()
