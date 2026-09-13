from __future__ import annotations

import unittest

from models.state import State
from scripts.baseline.targeted_query_pomdp.query_actions import (
    QueryAction,
    build_query_actions,
    fact_is_ambiguous,
)


class QueryActionTest(unittest.TestCase):
    def test_tomato_queries_are_fixed_and_grounded(self):
        obj_type = {
            "tomato(T)": ["tomato1"],
            "robot(R)": ["brl_robot"],
            "location(L)": ["dock_station", "stem_01"],
            "stem(S)": ["stem_01"],
        }
        actions = build_query_actions("tomato", obj_type, cost=0.25)
        by_name = {action.name: action for action in actions}
        action = by_name["query_ripeness(tomato1)"]
        self.assertEqual(action.target_fact, "ripe(tomato1)")
        self.assertEqual(action.cost, 0.25)
        self.assertNotIn("Ask", action.name)
        self.assertIn(
            "query_robot_loc(brl_robot,stem_01)",
            by_name,
        )

    def test_boolean_fact_ambiguity(self):
        fact = "fresh(tomato1)"
        states = [State([fact]), State([])]
        self.assertTrue(fact_is_ambiguous(states, fact))
        self.assertFalse(fact_is_ambiguous([State([fact])], fact))

    def test_query_action_is_a_no_effect_boolean_action(self):
        state = State(["tomato(tomato1)", "ripe(tomato1)"])
        action = QueryAction(
            name="query_ripeness(tomato1)",
            target_fact="ripe(tomato1)",
            query_schema="query_ripeness",
            preconditions=["tomato(tomato1)"],
            cost=0.1,
        )
        self.assertTrue(action.is_applicable(state))
        next_state = action.apply_action(state)
        self.assertEqual(next_state.facts, state.facts)
        self.assertIsNot(next_state, state)


if __name__ == "__main__":
    unittest.main()
