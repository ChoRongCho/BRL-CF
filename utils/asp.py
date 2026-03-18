from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import re

import clingo
import yaml


@dataclass
class PossibleWorld:
    world_id: int
    atoms: List[str] = field(default_factory=list)

    def filter_by_prefix(self, prefixes: Sequence[str]) -> List[str]:
        if not prefixes:
            return list(self.atoms)
        return [atom for atom in self.atoms if any(atom.startswith(p) for p in prefixes)]

    def print_atoms(self):
        for atom in self.atoms:
            print(atom)



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
        self.predicate_schemas: List[Dict[str, Any]] = []

        self.facts: List[str] = []
        self.rules: List[str] = []
        self.constraints: List[str] = []
        self.choice_rules: List[str] = []
        self.shows: List[str] = []

        self.runtime_facts: List[str] = []
        self.runtime_rules: List[str] = []
        self.runtime_constraints: List[str] = []
        self.runtime_shows: List[str] = []

    def load(self) -> None:
        with self.yaml_path.open("r", encoding="utf-8") as f:
            self.raw_data = yaml.safe_load(f) or {}

        self.domain_name = str(self.raw_data.get("domain", "unknown"))
        self.meta_data = self.raw_data.get("meta_data", {}) or {}
        self.predicate_schemas = self.raw_data.get("predicates", []) or []

        self.facts = self._parse_fact_section(self.raw_data.get("facts", []))
        self.rules = self._parse_rule_section(self.raw_data.get("rule", []))
        self.constraints = self._parse_constraint_section(self.raw_data.get("constraint", []))
        self.choice_rules = self._parse_choice_rule_section(self.raw_data.get("choice_rules", []))
        self.shows = self._parse_show_section(self.raw_data.get("show", []))

    def clear_runtime(self):
        self.runtime_facts.clear()
        self.runtime_rules.clear()
        self.runtime_constraints.clear()
        self.runtime_shows.clear()
    
    def add_runtime_facts(self, facts: Sequence[str]) -> None:
        self.runtime_facts.extend(self._normalize_statements(facts))

    def add_runtime_rules(self, rules: Sequence[str]) -> None:
        self.runtime_rules.extend(self._normalize_statements(rules))

    def add_runtime_constraints(self, constraints: Sequence[str]) -> None:
        result: List[str] = []
        for c in constraints or []:
            if c is None:
                continue
            c = str(c).strip()
            if not c:
                continue
            if not c.startswith(":-"):
                c = f":- {c}"
            result.append(self._ensure_period(c))
        self.runtime_constraints.extend(result)

    def add_runtime_shows(self, shows: Sequence[str]) -> None:
        result: List[str] = []
        for s in shows or []:
            if s is None:
                continue
            s = str(s).strip()
            if not s:
                continue

            if s.startswith("show "):
                s = "#show " + s[len("show "):].strip()
            elif not s.startswith("#show"):
                s = "#show " + s

            result.append(self._ensure_period(s))
        self.runtime_shows.extend(result)  
        
        
    def build_certain_worlds(self) -> str:
        lines: List[str] = [f"% Domain: {self.domain_name}", ""]

        if self.runtime_facts:
            lines.append("% Runtime facts")
            lines.extend(self.runtime_facts)
            lines.append("")

        if self.facts:
            lines.append("% Static domain facts")
            lines.extend(self.facts)
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
    
    
    def build_possible_worlds(self) -> str:
        lines: List[str] = [f"% Domain: {self.domain_name}", ""]

        if self.runtime_facts:
            lines.append("% Runtime facts")
            lines.extend(self.runtime_facts)
            lines.append("")

        if self.facts:
            lines.append("% Static domain facts")
            lines.extend(self.facts)
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
        output_path.write_text(self.build_possible_worlds(), encoding="utf-8")
        return output_path

    def solve(self, max_models: int = 0, yield_: bool = True) -> List[PossibleWorld]:
        """
        max_models=0 means enumerate all models.
        """
        program = self.build_possible_worlds()

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

    def _parse_fact_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []

        for item in items or []:
            if self._is_empty_item(item):
                continue

            if isinstance(item, str):
                stmt = item.strip()
                if stmt and self._is_ground_atom(stmt):
                    result.append(self._ensure_period(stmt))
                continue

            if isinstance(item, dict):
                pred = item.get("predicate")
                if pred is None:
                    continue
                pred = str(pred).strip()
                if pred and self._is_ground_atom(pred):
                    result.append(self._ensure_period(pred))
                continue

        return result

    def _parse_rule_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []

        for item in items or []:
            if self._is_empty_item(item):
                continue

            if isinstance(item, str):
                stmt = item.strip()
                if stmt:
                    result.append(self._ensure_period(stmt))
                continue

            if isinstance(item, dict):
                head = str(item.get("head", "") or "").strip()
                body = self._clean_str_list(item.get("body", []))

                if not head:
                    continue

                stmt = f"{head} :- {', '.join(body)}" if body else head
                result.append(self._ensure_period(stmt))
                continue

        return result

    def _parse_constraint_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []

        for item in items or []:
            if self._is_empty_item(item):
                continue

            if isinstance(item, str):
                stmt = item.strip()
                if not stmt:
                    continue
                if not stmt.startswith(":-"):
                    stmt = f":- {stmt}"
                result.append(self._ensure_period(stmt))
                continue

            if isinstance(item, dict):
                body = self._clean_str_list(item.get("body", []))
                if not body:
                    continue
                stmt = f":- {', '.join(body)}"
                result.append(self._ensure_period(stmt))
                continue

        return result

    def _parse_choice_rule_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []

        for item in items or []:
            if self._is_empty_item(item):
                continue

            if isinstance(item, str):
                stmt = item.strip()
                if stmt:
                    result.append(self._ensure_period(stmt))
                continue

            if not isinstance(item, dict):
                continue

            given = self._clean_str_list(item.get("given", []))
            cardinality = str(item.get("cardinality", "0..1") or "0..1").strip()

            if ".." not in cardinality:
                continue

            lo, hi = [x.strip() for x in cardinality.split("..", 1)]

            inner_parts: List[str] = []

            elements = item.get("elements", None)
            if elements is not None:
                if not isinstance(elements, list):
                    continue

                for elem in elements:
                    if self._is_empty_item(elem):
                        continue
                    if not isinstance(elem, dict):
                        continue

                    atom = str(elem.get("atom", "") or "").strip()
                    if not atom:
                        continue

                    over = self._clean_str_list(elem.get("over", []))

                    part = atom
                    if over:
                        part += " : " + ", ".join(over)

                    inner_parts.append(part)

            else:
                choose = str(item.get("choose", "") or "").strip()
                if not choose:
                    continue

                over = self._clean_str_list(item.get("over", []))

                part = choose
                if over:
                    part += " : " + ", ".join(over)

                inner_parts.append(part)

            if not inner_parts:
                continue

            inner = "; ".join(inner_parts)
            stmt = f"{lo} {{ {inner} }} {hi}"

            if given:
                stmt += " :- " + ", ".join(given)

            result.append(self._ensure_period(stmt))

        return result

    def _parse_show_section(self, items: Sequence[Any]) -> List[str]:
        result: List[str] = []

        for item in items or []:
            if self._is_empty_item(item):
                continue

            if not isinstance(item, str):
                continue

            stmt = item.strip()
            if not stmt:
                continue

            if stmt.startswith("show "):
                stmt = stmt[len("show "):].strip()

            if stmt.startswith("#show"):
                stmt = stmt[len("#show"):].strip()

            if not stmt:
                continue

            if "/" in stmt and "(" not in stmt and ")" not in stmt:
                result.append(self._ensure_period(f"#show {stmt}"))
                continue

            if "(" in stmt and ")" in stmt:
                result.append(self._ensure_period(f"#show {stmt} : {stmt}"))
                continue

            result.append(self._ensure_period(f"#show {stmt}"))

        return result

    @staticmethod
    def _is_ground_atom(stmt: str) -> bool:
        stmt = stmt.strip().rstrip(".")
        if not stmt:
            return False
        terms = re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", stmt)
        return len(terms) == 0

    @staticmethod
    def _is_empty_item(item: Any) -> bool:
        if item is None:
            return True
        if isinstance(item, str) and not item.strip():
            return True
        if isinstance(item, dict) and not item:
            return True
        if isinstance(item, (list, tuple, set)) and len(item) == 0:
            return True
        return False

    @staticmethod
    def _clean_str_list(items: Any) -> List[str]:
        if items is None:
            return []

        if isinstance(items, str):
            items = [items]

        result: List[str] = []
        for x in items:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                result.append(s)
        return result

    @staticmethod
    def _ensure_period(stmt: str) -> str:
        stmt = stmt.strip()
        if not stmt.endswith("."):
            stmt += "."
        return stmt

    @staticmethod
    def _normalize_statements(items: Sequence[str]) -> List[str]:
        result: List[str] = []
        for x in items or []:
            if x is None:
                continue
            s = str(x).strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            result.append(s)
        return result
    
    
    
    
# ======================= Running Method =======================
def solve_asp(program, max_models: int = 0, yield_: bool = True) -> List[PossibleWorld]:
    """
    max_models=0 means enumerate all models.
    """

    ctl = clingo.Control(
        arguments=[str(max_models),
        "--warn=no-atom-undefined",]
    )       
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


def solve_as_dicts(self, max_models: int = 0, prefixes: Sequence[str] = ()) -> List[Dict[str, Any]]:
    worlds = self.solve(max_models=max_models)
    result: List[Dict[str, Any]] = []

    for w in worlds:
        atoms = w.filter_by_prefix(prefixes) if prefixes else list(w.atoms)
        result.append(
            {
                "world_id": w.world_id,
                "atoms": atoms,
            }
        )

    return result


def save_program(program: str, output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.write_text(program, encoding="utf-8")
    return output_path