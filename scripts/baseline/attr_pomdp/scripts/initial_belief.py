"""Domain-sized initial beliefs for the adapted Attr-POMDP experiments."""

from __future__ import annotations

from itertools import product

import numpy as np

from models.belief import Belief
from models.state import State


def _objects(type_map: dict, declaration: str) -> list[str]:
    return list(type_map.get(declaration, []))


def _uniform_belief(knowledge: State, particles: list[State]) -> Belief:
    if not particles:
        particles = [knowledge.copy()]
    weights = np.full(len(particles), 1.0 / len(particles), dtype=float)
    return Belief(knowledge.copy(), particles, weights)


def _tomato_belief(knowledge: State, type_map: dict) -> Belief:
    tomatoes = _objects(type_map, "tomato(T)")
    stems = _objects(type_map, "stem(S)")
    assignments = product(product(("ripe", "unripe", "rotten"), stems), repeat=len(tomatoes))
    particles = []
    for assignment in assignments:
        state = knowledge.copy()
        for tomato, (quality, stem) in zip(tomatoes, assignment):
            state.add_fact(f"{quality}({tomato})")
            state.add_fact(f"at({tomato},{stem})")
        particles.append(state)
    return _uniform_belief(knowledge, particles)


def _wastesorting_belief(knowledge: State, type_map: dict) -> Belief:
    wastes = _objects(type_map, "waste(W)")
    particles = []
    for categories in product(("plastic", "can", "paper", "general"), repeat=len(wastes)):
        state = knowledge.copy()
        for waste, category in zip(wastes, categories):
            state.add_fact(f"{category}({waste})")
        particles.append(state)
    return _uniform_belief(knowledge, particles)


def initialize_attribute_belief(domain: str, knowledge: State, type_map: dict) -> Belief:
    if domain == "tomato":
        return _tomato_belief(knowledge, type_map)
    if domain == "wastesorting":
        return _wastesorting_belief(knowledge, type_map)
    raise ValueError(f"Unsupported Attr-POMDP domain: {domain}")
