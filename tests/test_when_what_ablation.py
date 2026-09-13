from types import SimpleNamespace

import numpy as np

from when_what_ablation_main import (
    ABLATION_POLICIES,
    select_question,
    should_start_query,
)


class StubRng:
    def __init__(self, random_value=0.0, choice_value=None):
        self.random_value = random_value
        self.choice_value = choice_value

    def random(self):
        return self.random_value

    def choice(self, candidates):
        assert self.choice_value in candidates
        return self.choice_value


class StubFeedbackManager:
    def get_changed_facts(self, knowledge, frontier):
        return ["a", "b"]

    def select_best_fact_to_ask(self, belief):
        return "best"


def test_policy_matrix_is_the_requested_two_by_two():
    assert {
        key: (value.when_policy, value.what_policy)
        for key, value in ABLATION_POLICIES.items()
    } == {
        "random": ("random", "random"),
        "ours-when-only": ("proposed", "random"),
        "ours-what-only": ("random", "proposed"),
        "ours": ("proposed", "proposed"),
    }


def test_proposed_when_uses_confidence_threshold():
    policy = ABLATION_POLICIES["ours"]
    rng = StubRng(random_value=0.0)
    assert should_start_query(policy, 0.79, 0.8, 0.0, rng)
    assert not should_start_query(policy, 0.8, 0.8, 1.0, rng)


def test_random_when_uses_probability_not_confidence():
    policy = ABLATION_POLICIES["random"]
    assert should_start_query(policy, 1.0, 0.8, 0.3, StubRng(0.29))
    assert not should_start_query(policy, 0.0, 0.8, 0.3, StubRng(0.30))


def test_random_and_proposed_what_are_separate():
    manager = StubFeedbackManager()
    belief = SimpleNamespace(
        knowledge=object(),
        frontier=[object(), object()],
        frontier_weights=np.array([0.5, 0.5]),
    )
    assert select_question(
        ABLATION_POLICIES["ours-when-only"],
        manager,
        belief,
        StubRng(choice_value="b"),
    ) == "b"
    assert select_question(
        ABLATION_POLICIES["ours-what-only"],
        manager,
        belief,
        StubRng(choice_value="b"),
    ) == "best"
