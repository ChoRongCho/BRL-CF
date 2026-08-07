"""Grounded fact query actions for the adapted Attr-POMDP baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.action import normalize_fact
from models.belief import Belief


ATTRIBUTE_PREDICATES = {
    "tomato": frozenset({"ripe", "unripe", "rotten", "at"}),
    "wastesorting": frozenset({"plastic", "can", "paper", "general"}),
}


def predicate_of(fact: str) -> str:
    return normalize_fact(fact).split("(", 1)[0]


@dataclass(frozen=True, slots=True)
class QueryAction:
    """A binary question about one grounded symbolic fact."""

    target_fact: str
    context_action_name: str | None
    provider: str = "oracle"
    cost: float = 1.0

    @property
    def name(self) -> str:
        return f"query_fact({self.target_fact})"

    @property
    def action_type(self) -> str:
        return "query"


class QueryFactory:
    """Create Attr-POMDP-style questions from uncertain attribute facts."""

    def __init__(self, domain: str, provider: str = "oracle", query_cost: float = 1.0):
        if domain not in ATTRIBUTE_PREDICATES:
            raise ValueError(f"Unsupported Attr-POMDP domain: {domain}")
        self.domain = domain
        self.provider = provider
        self.query_cost = float(query_cost)

    def candidates(
        self,
        belief: Belief,
        context_action_name: str | None,
        asked_facts: Iterable[str] = (),
    ) -> list[QueryAction]:
        asked = {normalize_fact(fact) for fact in asked_facts}
        allowed = ATTRIBUTE_PREDICATES[self.domain]
        facts = {
            normalize_fact(fact)
            for state in belief.particles
            for fact in state.facts
            if predicate_of(fact) in allowed
        }

        candidates = []
        for fact in sorted(facts):
            if fact in asked:
                continue
            probability = sum(
                float(weight)
                for state, weight in zip(belief.particles, belief.particle_weights)
                if state.has_fact(fact)
            )
            if 1e-12 < probability < 1.0 - 1e-12:
                candidates.append(
                    QueryAction(
                        target_fact=fact,
                        context_action_name=context_action_name,
                        provider=self.provider,
                        cost=self.query_cost,
                    )
                )
        return candidates
