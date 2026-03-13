from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import yaml
import clingo


@dataclass
class PossibleWorld:
    world_id: int
    atoms: List[str] = field(default_factory=list)

    def filter_by_prefix(self, prefixes: Sequence[str]) -> List[str]:
        result = []
        for atom in self.atoms:
            if any(atom.startswith(p) for p in prefixes):
                result.append(atom)
        return result


class DomainRuleBridge:
    """
    Load domain_rule.yaml, merge runtime facts, build ASP program,
    run clingo, and extract answer sets as possible worlds.
    """

    def __init__(self, yaml_path: Union[str, Path]) -> None:
        self.yaml_path = Path(yaml_path)
        self.raw_data: Dict[str, Any] = {}
        self.domain_name: str = "unknown"
        self.meta_data: Dict[str, Any] = {}

        self.facts: List[str] = []
        self.rules: List[str] = []
        self.constraints: List[str] = []
        self.choice_rules: List[str] = []
        self.derived_rules: List[str] = []
        self.shows: List[str] = []

        self.runtime_facts: List[str] = []
        self.runtime_rules: List[str] = []
        self.runtime_constraints: List[str] = []
        self.runtime_shows: List[str] = []

    def load(self) -> None:
        with self.yaml_path.open("r", encoding="utf-8") as f:
            self.raw_data = yaml.safe_load(f) or {}

        self.domain_name = self.raw_data.get("domain", "unknown")
        self.meta_data = self.raw_data.get("meta_data", {}) or {}

        self.facts = self._parse_fact_section(self.raw_data.get("fact", []))
        self.rules = self._parse_rule_section(self.raw_data.get("rule", []))
        self.constraints = self._parse_constraint_section(self.raw_data.get("constraint", []))
        self.choice_rules = self._parse_choice_rule_section(self.raw_data.get("choice_rules", []))
        self.derived_rules = self._parse_rule_section(self.raw_data.get("derived_rules", []))
        self.shows = self._parse_show_section(self.raw_data.get("show", []))

    def add_runtime_facts(self, facts: Sequence[str]) -> None:
        self.runtime_facts.extend(self._normalize_statements(facts))

    def add_runtime_rules(self, rules: Sequence[str]) -> None:
        self.runtime_rules.extend(self._normalize_statements(rules))

    def add_runtime_constraints(self, constraints: Sequence[str]) -> None:
        result = []
        for c in constraints:
            c = c.strip()
            if not c.startswith(":-"):
                c = f":- {c}"
            if not c.endswith("."):
                c += "."
            result.append(c)
        self.runtime_constraints.extend(result)

    def add_runtime_shows(self, shows: Sequence[str]) -> None:
        result = []
        for s in shows:
            s = s.strip()
            if s.startswith("show "):
                s = "#show " + s[len("show "):]
            elif not s.startswith("#show"):
                s = "#show " + s
            if not s.endswith("."):
                s += "."
            result.append(s)
        self.runtime_shows.extend(result)

    def build_program(self) -> str:
        lines: List[str] = []
        lines.append(f"% Domain: {self.domain_name}")
        lines.append("")

        if self.runtime_facts:
            lines.append("% Runtime facts")
            lines.extend(self.runtime_facts)
            lines.append("")

        if self.facts:
            lines.append("% Static domain facts")
            lines.extend(self.facts)
            lines.append("")

        if self.derived_rules:
            lines.append("% Derived rules")
            lines.extend(self.derived_rules)
            lines.append("")

        if self.choice_rules:
            lines.append("% Choice rules")
            lines.extend(self.choice_rules)
            lines.append("")

        if self.rules:
            lines.append("% Rules")
            lines.extend(self.rules)
            lines.append("")

        if self.runtime_rules:
            lines.append("% Runtime rules")
            lines.extend(self.runtime_rules)
            lines.append("")

        if self.constraints:
            lines.append("% Constraints")
            lines.extend(self.constraints)
            lines.append("")

        if self.runtime_constraints:
            lines.append("% Runtime constraints")
            lines.extend(self.runtime_constraints)
            lines.append("")

        show_lines = self.shows + self.runtime_shows
        if show_lines:
            lines.append("% Show")
            lines.extend(show_lines)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def save_program(self, output_path: Union[str, Path]) -> Path:
        output_path = Path(output_path)
        output_path.write_text(self.build_program(), encoding="utf-8")
        return output_path

    def solve(
        self,
        max_models: int = 0,
        yield_: bool = True,
    ) -> List[PossibleWorld]:
        """
        max_models=0 means enumerate all models.
        """
        program = self.build_program()

        ctl = clingo.Control(arguments=[str(max_models)])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        worlds: List[PossibleWorld] = []

        with ctl.solve(yield_=yield_) as handle:
            idx = 0
            for model in handle:
                idx += 1
                atoms = [str(sym) for sym in model.symbols(shown=True)]
                atoms.sort()
                worlds.append(PossibleWorld(world_id=idx, atoms=atoms))

        return worlds

    def solve_as_dicts(
        self,
        max_models: int = 0,
        prefixes: Sequence[str] = ("on(", "visible(", "hidden("),
    ) -> List[Dict[str, Any]]:
        worlds = self.solve(max_models=max_models)
        result = []

        for w in worlds:
            filtered = w.filter_by_prefix(prefixes)
            result.append(
                {
                    "world_id": w.world_id,
                    "atoms": filtered,
                }
            )

        return result

    def _parse_fact_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                result.append(self._ensure_period(item))
            elif isinstance(item, dict):
                pred = item.get("predicate")
                if pred:
                    result.append(self._ensure_period(pred))
            else:
                raise TypeError(f"Unsupported fact item: {item}")
        return result

    def _parse_rule_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                result.append(self._ensure_period(item))
            elif isinstance(item, dict):
                head = item.get("head", "").strip()
                body = item.get("body", []) or []
                if not head:
                    raise ValueError(f"Rule missing head: {item}")

                if body:
                    stmt = f"{head} :- {', '.join(x.strip() for x in body)}"
                else:
                    stmt = head
                result.append(self._ensure_period(stmt))
            else:
                raise TypeError(f"Unsupported rule item: {item}")
        return result

    def _parse_constraint_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                stmt = item.strip()
                if not stmt.startswith(":-"):
                    stmt = f":- {stmt}"
                result.append(self._ensure_period(stmt))
            elif isinstance(item, dict):
                body = item.get("body", []) or []
                if not body:
                    raise ValueError(f"Constraint missing body: {item}")
                stmt = f":- {', '.join(x.strip() for x in body)}"
                result.append(self._ensure_period(stmt))
            else:
                raise TypeError(f"Unsupported constraint item: {item}")
        return result

    def _parse_choice_rule_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                result.append(self._ensure_period(item))
            elif isinstance(item, dict):
                head = item.get("head", "").strip()
                body = item.get("body", []) or []
                cardinality = str(item.get("cardinality", "0..1")).strip()

                if not head:
                    raise ValueError(f"Choice rule missing head: {item}")
                if ".." not in cardinality:
                    raise ValueError(f"Choice rule cardinality must be like 1..1: {item}")

                lo, hi = [x.strip() for x in cardinality.split("..", 1)]

                if body:
                    stmt = f"{lo} {{ {head} : {', '.join(x.strip() for x in body)} }} {hi}"
                else:
                    stmt = f"{lo} {{ {head} }} {hi}"

                result.append(self._ensure_period(stmt))
            else:
                raise TypeError(f"Unsupported choice rule item: {item}")
        return result

    def _parse_show_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError(f"Unsupported show item: {item}")
            stmt = item.strip()
            if stmt.startswith("show "):
                stmt = "#show " + stmt[len("show "):]
            elif not stmt.startswith("#show"):
                stmt = "#show " + stmt
            result.append(self._ensure_period(stmt))
        return result

    @staticmethod
    def _ensure_period(stmt: str) -> str:
        stmt = stmt.strip()
        if not stmt.endswith("."):
            stmt += "."
        return stmt

    @staticmethod
    def _normalize_statements(items: Sequence[str]) -> List[str]:
        result = []
        for x in items:
            x = x.strip()
            if not x.endswith("."):
                x += "."
            result.append(x)
        return result