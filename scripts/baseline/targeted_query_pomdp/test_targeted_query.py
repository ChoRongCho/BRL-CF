from __future__ import annotations

import unittest
from types import SimpleNamespace

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
    def test_boolean_fact_ambiguity(self):
        fact = "fresh(tomato1)"
        states = [State([fact]), State([])]
        self.assertTrue(fact_is_ambiguous(states, fact))
        self.assertFalse(fact_is_ambiguous([State([fact])], fact))

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

    def test_only_impossible_static_groundings_are_removed(self):
        state = State([
            "robot(robot)",
            "waste(waste1)",
            "paper_bin(paper_bin)",
        ])
        valid = Action(
            name="place_paper_bin(robot,waste1,paper_bin)",
            preconditions=[
                "robot(robot)",
                "waste(waste1)",
                "paper_bin(paper_bin)",
                "holding(robot,waste1)",
            ],
        )
        wrong_bin_type = Action(
            name="place_paper_bin(robot,waste1,can_bin)",
            preconditions=[
                "robot(robot)",
                "waste(waste1)",
                "paper_bin(can_bin)",
                "holding(robot,waste1)",
            ],
        )
        static_predicates = {"robot", "waste", "paper_bin"}

        self.assertTrue(
            QueryAsActionPOMCPPlanner._has_valid_static_binding(
                valid, state, static_predicates
            )
        )
        self.assertFalse(
            QueryAsActionPOMCPPlanner._has_valid_static_binding(
                wrong_bin_type, state, static_predicates
            )
        )

    def test_queries_require_ambiguity_while_task_actions_require_applicability(self):
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
        unavailable_action = Action(
            name="place_plastic_bin(robot,waste1,plastic_bin)",
            preconditions=["plastic(waste1)"],
        )
        query = QueryAction(
            name="query_can(waste1)",
            target_fact="can(waste1)",
            query_schema="query_can",
            preconditions=["waste(waste1)"],
            cost=1.0,
        )
        planner.task_actions = [can_action, paper_action, unavailable_action]
        planner.query_actions = [query]
        planner.actions = planner.task_actions + planner.query_actions

        # The root includes a question only while both answers remain possible.
        belief = SimpleNamespace(
            knowledge=State(["waste(waste1)", "can(waste1)"]),
            frontier=[
                State(["waste(waste1)", "can(waste1)"]),
                State(["waste(waste1)", "paper(waste1)"]),
            ],
        )
        self.assertEqual(
            {action.name for action in planner._root_candidates(belief)},
            {can_action.name, query.name},
        )
        belief.frontier = [State(["waste(waste1)", "can(waste1)"])]
        self.assertEqual(
            {action.name for action in planner._root_candidates(belief)},
            {can_action.name},
        )

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

    def test_new_observation_history_is_initialized_by_rollout(self):
        planner = object.__new__(QueryAsActionPOMCPPlanner)
        planner.tree = POMDPTree()
        planner.task_actions = [Action(name="physical_action")]
        planner.query_actions = []
        planner.actions = list(planner.task_actions)
        planner.max_node_particles = 10
        planner.max_depth = 20
        planner.gamma = 0.95
        planner.epsilon = 0.005
        planner.rollout = lambda state, depth: 7.0

        root_action = Action(name="root_action")
        action_node = planner.tree.expand_tree_from(
            planner.tree.root_id,
            root_action,
            is_action=True,
        )
        history = planner.tree.get_observation_node(action_node, "observed")

        value = planner.simulate(State(), history, 1)

        self.assertEqual(value, 7.0)
        self.assertEqual(planner.tree.get_visit(history), 1)
        children = planner.tree.get_action_children(history)
        self.assertEqual([action.name for action, _ in children], ["physical_action"])
        self.assertEqual(planner.tree.get_visit(children[0][1]), 0)


if __name__ == "__main__":
    unittest.main()
