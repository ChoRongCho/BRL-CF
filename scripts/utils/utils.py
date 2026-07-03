

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import re



def _format_fact(pred: str, args: List[str]) -> str:
    if not args:
        return pred
    return f"{pred}({','.join(args)})"


def _parse_fact(fact: str) -> Tuple[str, List[str]]:
    fact = fact.replace(" ", "")
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", fact):
        return fact, []

    m = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", fact)
    if not m:
        raise ValueError(f"Invalid fact format: {fact}")
    pred = m.group(1)
    args_text = m.group(2)
    args = [] if args_text == "" else [x.strip() for x in args_text.split(",")]
    return pred, args


def _dedup_facts(facts: List[str]) -> List[str]:
    return list(dict.fromkeys(f.replace(" ", "") for f in facts))
