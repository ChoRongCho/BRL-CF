
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def _fmt_seconds(value: float) -> str:
    return f"{value:.4f}s"


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def logger_exp(result: Dict[str, Any], log_dir: str | Path = "logs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domain = result.get("meta", {}).get("domain", "unknown")
    path = Path(log_dir) / f"exp_{domain}_{timestamp}.txt"

    meta = result.get("meta", {})
    timing = result.get("timing", {})
    reward = result.get("reward", {})
    actions = result.get("actions", [])
    questions = result.get("questions", [])
    knowledge = result.get("final_knowledge", {})

    lines = [
        "Planning Result Log",
        "=" * 80,
        "",
        "[Meta]",
    ]
    for key in sorted(meta):
        lines.append(f"{key}: {meta[key]}")

    lines.extend([
        "",
        "[Plan Summary]",
        f"success: {result.get('success')}",
        f"steps: {result.get('steps')}",
        f"cumulated_reward: {reward.get('cumulated', 0.0)}",
        f"total_questions: {result.get('total_questions', 0)}",
        "",
        "[Timing]",
    ])
    for key in ("search_time", "execute_time", "update_time", "interaction_time", "pruning_time"):
        value = timing.get(key, {})
        lines.append(
            f"{key}: total={_fmt_seconds(value.get('total', 0.0))}, "
            f"avg={_fmt_seconds(value.get('avg', 0.0))}"
        )
    lines.append(f"total_time: {_fmt_seconds(timing.get('total_time', 0.0))}")

    lines.extend(["", "[Full Plan]"])
    if actions:
        for item in actions:
            lines.append(
                f"STEP {item['step']}: {item['action']} "
                f"(search={_fmt_seconds(item.get('search_time', 0.0))}, "
                f"execute={_fmt_seconds(item.get('execute_time', 0.0))}, "
                f"update={_fmt_seconds(item.get('update_time', 0.0))}, "
                f"interaction={_fmt_seconds(item.get('interaction_time', 0.0))}, "
                f"pruning={_fmt_seconds(item.get('pruning_time', 0.0))}, "
                f"step_reward={item.get('step_reward', 0.0)}, "
                f"cumulated_reward={item.get('cumulated_reward', 0.0)})"
            )
    else:
        lines.append("-")

    lines.extend(["", "[Questions]"])
    if questions:
        for idx, item in enumerate(questions, start=1):
            lines.append(
                f"Q{idx}: step={item.get('step')}, action={item.get('action')}, "
                f"question='{item.get('question')} is True?', answer={item.get('answer')}, "
                f"confidence={item.get('confidence_before', 0.0):.4f}"
                f"->{item.get('confidence_after', 0.0):.4f}"
            )
    else:
        lines.append("-")

    lines.extend(["", "[Final Knowledge]", "facts:"])
    facts = knowledge.get("facts", [])
    lines.extend(f"  - {fact}" for fact in facts) if facts else lines.append("  -")
    lines.append("fluents:")
    fluents = knowledge.get("fluents", {})
    if fluents:
        for obj in sorted(fluents):
            lines.append(f"  {obj}: {fluents[obj]}")
    else:
        lines.append("  -")

    _write_lines(path, lines)
    return path
