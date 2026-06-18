
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def _fmt_seconds(value: float) -> str:
    return f"{value:.4f}s"


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def logger_exp(result: Dict[str, Any], log_dir: str | Path = "experiments_logs/system_log") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    domain = result.get("meta", {}).get("domain", "unknown")
    path = Path(log_dir) / f"exp_{domain}_{timestamp}.txt"

    meta = dict(result.get("meta", {}))
    if "scene" not in meta and meta.get("initial_state"):
        meta["scene"] = Path(meta["initial_state"]).stem
    timing = result.get("timing", {})
    scale_metrics = result.get("scale_metrics", {})
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
        f"end_reason: {result.get('end_reason')}",
        f"steps: {result.get('steps')}",
        f"cumulated_reward: {reward.get('cumulated', 0.0)}",
        f"total_questions: {result.get('total_questions', 0)}",
        "",
        "[Timing]",
    ])
    for key in ("search_time", "execute_time", "update_time", "interaction_time", "pruning_time", "step_total_time"):
        value = timing.get(key, {})
        lines.append(
            f"{key}: total={_fmt_seconds(value.get('total', 0.0))}, "
            f"avg={_fmt_seconds(value.get('avg', 0.0))}"
        )
    lines.append(f"total_time: {_fmt_seconds(timing.get('total_time', 0.0))}")

    if scale_metrics:
        lines.extend(["", "[Scale Metrics]"])
        for key in sorted(scale_metrics):
            value = scale_metrics.get(key, {})
            if isinstance(value, dict):
                lines.append(
                    f"{key}: total={value.get('total', 0.0):.4f}, "
                    f"avg={value.get('avg', 0.0):.4f}, "
                    f"max={value.get('max', 0.0):.4f}"
                )
            else:
                lines.append(f"{key}: {value}")

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
                f"step_total={_fmt_seconds(item.get('step_total_time', 0.0))}, "
                f"tree_nodes={item.get('tree_node_count', '-')}, "
                f"expanded_nodes={item.get('tree_nodes_expanded_this_step', '-')}, "
                f"root_actions={item.get('root_action_count', '-')}, "
                f"max_tree_depth={item.get('max_tree_depth', '-')}, "
                f"belief_frontier={item.get('post_update_belief_frontier_size', '-')}, "
                f"step_reward={item.get('step_reward', 0.0)}, "
                f"cumulated_reward={item.get('cumulated_reward', 0.0)})"
            )
    else:
        lines.append("-")

    lines.extend(["", "[Action Schema Summary]"])
    action_schema_summary = result.get("action_schema_summary", {})
    if action_schema_summary:
        lines.extend([
            "action_count: number of executed actions for this schema.",
            "question_count: number of questions asked during executions of this schema.",
            "expected_questions_per_action: expected number of questions per one action execution.",
            "",
            f"{'action_schema':<18} {'action_count':>12} {'question_count':>14} {'expected_questions_per_action':>30}",
            f"{'-' * 18} {'-' * 12} {'-' * 14} {'-' * 30}",
        ])
        for schema, stats in action_schema_summary.items():
            lines.append(
                f"{schema:<18} "
                f"{stats.get('action_count', 0):>12} "
                f"{stats.get('question_count', 0):>14} "
                f"{stats.get('questions_per_action', 0.0):>30.4f}"
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
