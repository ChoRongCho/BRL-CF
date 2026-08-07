"""Feedback providers used by the adapted Attr-POMDP baseline."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol

from models.state import State

from .query import QueryAction


class FeedbackProvider(Protocol):
    name: str

    def answer(self, query: QueryAction, true_state: State, current_state: State) -> bool:
        ...

    def likelihood(self, answer: bool, query: QueryAction, state: State) -> float:
        ...


class OracleProvider:
    """Thin adapter around ``models.<domain>.answer.answer_question``.

    No truth rules are duplicated here. Both runtime answers and hypothetical
    particle answers call the domain's existing public oracle function.
    """

    name = "oracle"

    def __init__(self, domain: str):
        self.domain = domain
        module = import_module(f"models.{domain}.answer")
        self._answer_question = module.answer_question

    def _call(self, query: QueryAction, true_facts: set[str], current_facts: set[str]) -> bool:
        return bool(
            self._answer_question(
                "oracle",
                query.target_fact,
                query.context_action_name,
                true_facts,
                current_facts,
            )
        )

    def answer(self, query: QueryAction, true_state: State, current_state: State) -> bool:
        return self._call(query, set(true_state.facts), set(current_state.facts))

    def likelihood(self, answer: bool, query: QueryAction, state: State) -> float:
        predicted = self._call(query, set(state.facts), set(state.facts))
        return 1.0 if predicted is bool(answer) else 0.0


class _PlaceholderProvider:
    def __init__(self, name: str):
        self.name = name

    def _raise(self):
        raise NotImplementedError(
            f"Attr-POMDP feedback provider '{self.name}' is a placeholder and is not implemented."
        )

    def answer(self, query: QueryAction, true_state: State, current_state: State) -> bool:
        self._raise()

    def likelihood(self, answer: bool, query: QueryAction, state: State) -> float:
        self._raise()


def make_provider(name: str, domain: str) -> FeedbackProvider:
    if name == "oracle":
        return OracleProvider(domain)
    if name in {"vlm", "human"}:
        return _PlaceholderProvider(name)
    raise ValueError(f"Unknown feedback provider: {name}")
