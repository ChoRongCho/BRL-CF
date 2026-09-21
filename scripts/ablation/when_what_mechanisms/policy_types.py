from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PolicyContext:
    belief: Any
    feedback_manager: Any
    env: Any
    observation_facts: list[str]
    action_history: list[str]
    step: int
    query_index: int
    seed: int
    excluded_facts: set[str] = field(default_factory=set)

    def ambiguous_facts(self) -> list[str]:
        facts = self.feedback_manager.get_changed_facts(
            self.belief.knowledge,
            self.belief.frontier,
        )
        return [fact for fact in facts if fact not in self.excluded_facts]


@dataclass(frozen=True)
class WhenDecision:
    start: bool
    policy: str
    score: Optional[float] = None
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WhatDecision:
    fact: str
    policy: str
    score: Optional[float] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

