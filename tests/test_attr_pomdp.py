import numpy as np

from scripts.baseline.attr_pomdp.planner import (
    AttrPOMDPPlanner,
    binary_attribute_question,
    pointing_question,
)


def test_confident_belief_commits_without_asking():
    planner = AttrPOMDPPlanner(depth=3)
    question = binary_attribute_question("red", [True, False])
    decision = planner.plan([0.999, 0.001], [question])
    assert decision.kind == "commit"
    assert decision.candidate_index == 0


def test_ambiguous_belief_asks_discriminative_attribute():
    planner = AttrPOMDPPlanner(depth=3)
    question = binary_attribute_question("red", [True, False])
    decision = planner.plan([0.5, 0.5], [question])
    assert decision.kind == "ask"
    assert decision.question == question


def test_binary_answer_updates_belief_with_paper_likelihood():
    planner = AttrPOMDPPlanner(depth=3)
    question = binary_attribute_question("red", [True, False])
    posterior = planner.update_belief([0.5, 0.5], question, True)
    assert np.allclose(posterior, [0.99, 0.01])


def test_pointing_question_uses_candidate_identity():
    question = pointing_question(1, 3)
    assert question.kind == "point"
    assert np.allclose(question.likelihoods[1], [0.01, 0.99])
    assert np.allclose(question.likelihoods[0], [0.99, 0.01])
    assert question.cost == 0.3
