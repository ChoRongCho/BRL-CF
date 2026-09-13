"""Fixed query schemas and their domain-specific grounding.

The baseline deliberately does not expose a generic ``Ask(fact)`` action.
Every action name below belongs to a predefined query vocabulary; grounding
only substitutes objects declared by the selected scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

from models.action import Action
from models.state import State


@dataclass(frozen=True)
class QuerySchema:
    name: str
    parameters: tuple[str, ...]
    target_template: str


class QueryAction(Action):
    """A grounded, Boolean question that is also a POMCP action."""

    __slots__ = ("target_fact", "query_schema")

    def __init__(
        self,
        *,
        name: str,
        target_fact: str,
        query_schema: str,
        preconditions: Sequence[str],
        cost: float,
    ) -> None:
        super().__init__(
            name=name,
            preconditions=list(preconditions),
            cost=float(cost),
        )
        self.target_fact = target_fact.replace(" ", "")
        self.query_schema = query_schema


# Parameters use the same one-letter type symbols as the domain YAML files.
QUERY_SCHEMAS: Mapping[str, tuple[QuerySchema, ...]] = {
    "tomato": (
        QuerySchema("query_ripeness", ("T",), "ripe({T})"),
        QuerySchema("query_fresh", ("T",), "fresh({T})"),
        QuerySchema("query_robot_loc", ("R", "L"), "located({R},{L})"),
        QuerySchema("query_tomato_loc", ("T", "S"), "at({T},{S})"),
        QuerySchema("query_observed", ("T",), "observed({T})"),
        QuerySchema("query_holding", ("R", "T"), "holding({R},{T})"),
        QuerySchema("query_handempty", ("R",), "handempty({R})"),
        QuerySchema("query_scanned", ("T",), "scanned({T})"),
        QuerySchema("query_loaded", ("T", "R"), "loaded({T},{R})"),
        QuerySchema("query_discarded", ("T",), "discarded({T})"),
    ),
    "wastesorting": (
        QuerySchema("query_plastic", ("W",), "plastic({W})"),
        QuerySchema("query_can", ("W",), "can({W})"),
        QuerySchema("query_paper", ("W",), "paper({W})"),
        QuerySchema("query_general", ("W",), "general({W})"),
        QuerySchema("query_detected", ("W",), "detected({W})"),
        QuerySchema("query_holding", ("R", "W"), "holding({R},{W})"),
        QuerySchema("query_handempty", ("R",), "handempty({R})"),
        QuerySchema("query_in_bin", ("W", "B"), "in_bin({W},{B})"),
    ),
}


def _type_map(obj_type: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for declaration, objects in obj_type.items():
        if "(" not in declaration or ")" not in declaration:
            continue
        symbol = declaration.split("(", 1)[1].split(")", 1)[0].strip()
        if not symbol:
            continue
        bucket = result.setdefault(symbol[0], [])
        for obj in objects or []:
            if obj not in bucket:
                bucket.append(str(obj))
    return result


def build_query_actions(domain: str, obj_type, cost: float) -> list[QueryAction]:
    """Ground the domain's fixed query schemas against scene objects."""
    if domain not in QUERY_SCHEMAS:
        raise ValueError(f"No predefined query schemas for domain: {domain}")
    types = _type_map(obj_type)
    grounded: list[QueryAction] = []

    for schema in QUERY_SCHEMAS[domain]:
        try:
            domains = [types[param[0]] for param in schema.parameters]
        except KeyError as exc:
            raise ValueError(
                f"Query schema {schema.name} requires undeclared type {exc.args[0]}"
            ) from exc
        for values in product(*domains):
            binding = dict(zip(schema.parameters, values))
            target = schema.target_template.format(**binding).replace(" ", "")
            args = ",".join(values)
            preconditions = [
                _type_fact(obj_type, param[0], value)
                for param, value in zip(schema.parameters, values)
            ]
            grounded.append(QueryAction(
                name=f"{schema.name}({args})",
                target_fact=target,
                query_schema=schema.name,
                preconditions=preconditions,
                cost=cost,
            ))
    return grounded


def _type_fact(obj_type, symbol: str, value: str) -> str:
    for declaration, objects in obj_type.items():
        declared = declaration.split("(", 1)[1].split(")", 1)[0].strip()
        if declared.startswith(symbol) and value in (objects or []):
            return f"{declaration.split('(', 1)[0]}({value})"
    raise ValueError(f"Cannot build type fact for {symbol}={value}")


def fact_is_ambiguous(states: Sequence[State], fact: str) -> bool:
    if len(states) < 2:
        return False
    values = {state.has_fact(fact) for state in states}
    return len(values) == 2

