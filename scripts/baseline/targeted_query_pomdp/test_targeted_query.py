from __future__ import annotations

import unittest

from models.action import Action
from models.state import State
from planners.tree import POMDPTree
from scripts.baseline.targeted_query_pomdp.planner import QueryAsActionPOMCPPlanner
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

    def test_hidden_precondition_failure_is_penalized(self):
        planner = object.__new__(QueryAsActionPOMCPPlanner)
        planner.failure_penalty = 10.0
        action = Action(
            name="place_can_bin(robot,waste1,can_bin)",
            preconditions=["can(waste1)"],
        )

        next_state, observation, reward, terminal = planner._generate(
            State(["paper(waste1)"]),
            action,
        )

        self.assertEqual(next_state.facts, ["paper(waste1)"])
        self.assertEqual(observation, ("invalid_action", action.name))
        self.assertEqual(reward, -10.0)
        self.assertTrue(terminal)

    def test_history_uses_union_action_set_across_particles(self):
        planner = object.__new__(QueryAsActionPOMCPPlanner)
        planner.tree = POMDPTree()
        can_action = Action(
            name="place_can_bin(robot,waste1,can_bin)",
            preconditions=["can(waste1)"],
        )
        paper_action = Action(
            name="place_paper_bin(robot,waste1,paper_bin)",
            preconditions=["paper(waste1)"],
        )
        query = QueryAction(
            name="query_can(waste1)",
            target_fact="can(waste1)",
            query_schema="query_can",
            preconditions=["waste(waste1)"],
            cost=1.0,
        )
        planner.task_actions = [can_action, paper_action]
        planner.query_actions = [query]
        planner.actions = planner.task_actions + planner.query_actions
        planner.root_candidate_names = set()

        dummy = Action(name="dummy")
        action_node = planner.tree.expand_tree_from(
            planner.tree.root_id,
            dummy,
            is_action=True,
        )
        history = planner.tree.get_observation_node(action_node, "observation")
        planner.tree.add_particle(
            history,
            State(["waste(waste1)", "can(waste1)"]),
        )
        planner.tree.add_particle(
            history,
            State(["waste(waste1)", "paper(waste1)"]),
        )

        names = {action.name for action in planner._history_candidates(history)}
        self.assertEqual(
            names,
            {can_action.name, paper_action.name, query.name},
        )


if __name__ == "__main__":
    unittest.main()
