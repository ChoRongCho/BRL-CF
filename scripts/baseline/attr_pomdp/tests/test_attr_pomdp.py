import unittest

import numpy as np

from baseline.attr_pomdp.scripts.belief import ParticleBeliefModel
from baseline.attr_pomdp.scripts.initial_belief import initialize_attribute_belief
from baseline.attr_pomdp.scripts.planner import AttrPOMDPPlanner
from baseline.attr_pomdp.scripts.providers import make_provider
from baseline.attr_pomdp.scripts.query import QueryAction, QueryFactory
from models.action import Action
from models.belief import Belief
from models.state import State
from models.transition import NextStateOutcome


class _DeterministicProvider:
    def likelihood(self, answer, query, state):
        return 1.0 if state.has_fact(query.target_fact) == answer else 0.0


class _BeliefModel:
    @staticmethod
    def materialize(belief):
        return belief.particles, belief.particle_weights

    @staticmethod
    def transition_outcomes(state, action):
        next_state = state.copy()
        next_state.remove_fact("alive")
        return [NextStateOutcome(next_state, 1.0)]

    @classmethod
    def branches(cls, belief, action):
        particles = [cls.transition_outcomes(state, action)[0].next_state for state in belief.particles]
        return [(1.0, Belief(belief.knowledge.copy(), particles, belief.particle_weights))]

    @classmethod
    def evaluate_task(cls, belief, action, reward_model, *, include_branches=True):
        reward = sum(
            float(weight)
            * reward_model.get_reward(
                state,
                action,
                cls.transition_outcomes(state, action)[0].next_state,
            )
            for state, weight in zip(belief.particles, belief.particle_weights)
            if action.is_applicable(state)
        )
        return reward, cls.branches(belief, action) if include_branches else []

    @staticmethod
    def query_branches(belief, query, provider):
        result = []
        for answer in (True, False):
            mask = np.array(
                [provider.likelihood(answer, query, state) for state in belief.particles]
            )
            probability = float(np.dot(belief.particle_weights, mask))
            if probability:
                result.append(
                    (
                        probability,
                        answer,
                        Belief(
                            belief.knowledge.copy(),
                            [state.copy() for state in belief.particles],
                            belief.particle_weights * mask,
                        ),
                    )
                )
        return result


class _RewardModel:
    @staticmethod
    def get_reward(state, action, next_state):
        correct = (
            action.name == "choose_paper" and state.has_fact("paper(waste1)")
        ) or (
            action.name == "choose_plastic" and state.has_fact("plastic(waste1)")
        )
        return 10.0 if correct else -10.0


class _Env:
    def __init__(self):
        self.actions = [
            Action("choose_paper", preconditions=["alive"]),
            Action("choose_plastic", preconditions=["alive"]),
        ]
        self.reward_model = _RewardModel()


class _Args:
    gamma = 0.95
    max_depth = 2


class AttrPOMDPTest(unittest.TestCase):
    def test_wastesorting_initial_belief_enumerates_categories(self):
        belief = initialize_attribute_belief(
            "wastesorting",
            State(["robot(brl_robot)", "waste(waste1)", "waste(waste2)"]),
            {"waste(W)": ["waste1", "waste2"]},
        )
        self.assertEqual(len(belief.particles), 16)
        self.assertAlmostEqual(float(belief.particle_weights.sum()), 1.0)

    def test_query_factory_only_returns_uncertain_attributes(self):
        belief = Belief(
            State(["waste(waste1)", "alive"]),
            [
                State(["waste(waste1)", "paper(waste1)", "alive"]),
                State(["waste(waste1)", "plastic(waste1)", "alive"]),
            ],
            np.array([0.5, 0.5]),
        )
        facts = {
            query.target_fact
            for query in QueryFactory("wastesorting").candidates(belief, None)
        }
        self.assertEqual(facts, {"paper(waste1)", "plastic(waste1)"})

    def test_query_update_filters_particles_without_world_transition(self):
        state_yes = State(["paper(waste1)"])
        state_no = State(["plastic(waste1)"])
        belief = Belief(State(), [state_yes, state_no], np.array([0.5, 0.5]))
        model = ParticleBeliefModel(None, None)
        query = QueryAction("paper(waste1)", None)
        updated = model.query_update(belief, query, True, _DeterministicProvider())
        self.assertAlmostEqual(updated.particle_weights[0], 1.0)
        self.assertAlmostEqual(updated.particle_weights[1], 0.0)
        self.assertEqual(updated.particles[0].facts, state_yes.facts)
        self.assertIn("paper(waste1)", updated.knowledge.facts)

    def test_unimplemented_providers_fail_explicitly(self):
        for name in ("vlm", "human"):
            provider = make_provider(name, "tomato")
            with self.assertRaises(NotImplementedError):
                provider.answer(QueryAction("ripe(tomato1)", None), State(), State())

    def test_oracle_adapter_matches_domain_answer_function(self):
        from models.wastesorting.answer import answer_question

        query = QueryAction("paper(waste1)", "detect_waste(brl_robot)")
        true_state = State(["paper(waste1)"])
        current_state = State(["waste(waste1)"])
        expected = answer_question(
            "oracle",
            query.target_fact,
            query.context_action_name,
            set(true_state.facts),
            set(current_state.facts),
        )
        actual = make_provider("oracle", "wastesorting").answer(
            query,
            true_state,
            current_state,
        )
        self.assertEqual(actual, expected)

    def test_planner_selects_query_when_it_improves_task_choice(self):
        belief = Belief(
            State(["waste(waste1)", "alive"]),
            [
                State(["waste(waste1)", "paper(waste1)", "alive"]),
                State(["waste(waste1)", "plastic(waste1)", "alive"]),
            ],
            np.array([0.5, 0.5]),
        )
        planner = AttrPOMDPPlanner(
            _Args(),
            _Env(),
            _BeliefModel(),
            QueryFactory("wastesorting", query_cost=1.0),
            _DeterministicProvider(),
        )
        result = planner.search(belief)
        self.assertIsInstance(result.action, QueryAction)
        self.assertGreater(result.value, 0.0)


if __name__ == "__main__":
    unittest.main()
