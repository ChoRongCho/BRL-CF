"""Shared implementation for the isolated E1-E4 evaluation pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SYSTEM_EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SYSTEM_EVAL_ROOT.parents[1]
LOG_ROOT = PROJECT_ROOT / "experiments_logs" / "system_log"
EXPERIMENT_DIRS = {
    "e1": "e1_threshold",
    "e2": "e2_ablation",
    "e3": "e3_baselines",
    "e4": "e4_feedback",
}
RUN_FIELDS = (
    "experiment", "domain", "scene", "condition", "method", "feedback_source",
    "threshold", "seed", "success", "end_reason", "steps", "reward",
    "question_count", "query_step_count", "query_rate", "elapsed_seconds",
    "source_file",
)
METRICS = {
    "success_rate": "Success Rate",
    "average_steps": "Average Steps",
    "average_questions": "Average Query Count",
    "query_rate": "Query Rate",
    "average_reward": "Average Reward",
    "elapsed_seconds": "Elapsed Time (s)",
}
OUTCOMES = ("all", "success_only", "failure_only")
CONDITION_ORDERS = {
    "e2": ("ours", "ours-random-when", "ours-random-what"),
    "e3": ("ours", "knowno", "active-search"),
    "e4": ("ours-vlm", "ours-human", "knowno-vlm", "knowno-human"),
}


def _section(text: str, name: str) -> dict[str, str]:
    match = re.search(rf"^\[{re.escape(name)}\]\s*$([\s\S]*?)(?=^\[|\Z)", text, re.MULTILINE)
    if not match:
        return {}
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().rstrip("s"))
    except (TypeError, ValueError):
        return default


def _scene(parts: tuple[str, ...], meta: dict[str, Any]) -> str:
    for part in parts:
        match = re.match(r"(scene_\d+)", part)
        if match:
            return match.group(1)
    value = str(meta.get("scene") or meta.get("initial_state") or "")
    match = re.search(r"scene_\d+", value)
    return match.group(0) if match else "unknown"


def _condition(experiment: str, parts: tuple[str, ...], meta: dict[str, Any]) -> tuple[str, str, str, str]:
    method = str(meta.get("method") or "")
    feedback = str(meta.get("feedback_source") or "")
    threshold = str(meta.get("threshold") or "")
    if experiment == "e1":
        tau = next((p.removeprefix("tau_").replace("-", ".") for p in parts if p.startswith("tau_")), threshold)
        return f"tau_{tau}", method or "ours", feedback or "oracle", tau
    if experiment in {"e2", "e3"}:
        allowed = {
            "e2": ("ours", "ours-random-when", "ours-random-what"),
            "e3": ("ours", "knowno", "active-search"),
        }[experiment]
        path_method = next((p for p in parts if p in allowed), "")
        method = method or path_method
        return method, method, feedback or "oracle", threshold
    allowed_methods = ("ours", "knowno")
    allowed_feedback = ("vlm", "human")
    method = method or next((p for p in parts if p in allowed_methods), "")
    feedback = feedback or next((p for p in parts if p in allowed_feedback), "")
    return f"{method}-{feedback}", method, feedback, threshold


def _parse_txt(path: Path, experiment: str, root: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    summary = _section(text, "Plan Summary")
    meta = _section(text, "Meta")
    if not summary:
        return None
    timing = _section(text, "Timing")
    rel_parts = path.relative_to(root).parts
    condition, method, feedback, threshold = _condition(experiment, rel_parts, meta)
    steps = int(_number(summary.get("steps")))
    questions = int(_number(summary.get("total_questions")))
    query_steps = len(set(re.findall(r"^Q\d+:\s*step=(\d+)", text, re.MULTILINE)))
    return {
        "experiment": experiment,
        "domain": meta.get("domain") or rel_parts[0],
        "scene": _scene(rel_parts, meta),
        "condition": condition,
        "method": method,
        "feedback_source": feedback,
        "threshold": threshold,
        "seed": meta.get("seed", ""),
        "success": _bool(summary.get("success")),
        "end_reason": summary.get("end_reason", ""),
        "steps": steps,
        "reward": _number(summary.get("cumulated_reward")),
        "question_count": questions,
        "query_step_count": query_steps,
        "query_rate": query_steps / steps if steps else "",
        "elapsed_seconds": _number(timing.get("total_time")),
        "source_file": str(path),
    }


def _extract_embedded_summary(text: str) -> dict[str, Any]:
    match = re.search(r"Summary:\s*", text)
    if not match:
        return {}
    try:
        value, _end = json.JSONDecoder().raw_decode(text[match.end():].lstrip())
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _parse_json(path: Path, experiment: str, root: Path) -> dict[str, Any] | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _extract_embedded_summary(text)
    if not isinstance(data, dict) or not data:
        return None
    meta = data.get("meta") or data.get("metadata") or {}
    summary = data.get("summary") or data
    rel_parts = path.relative_to(root).parts
    condition, method, feedback, threshold = _condition(experiment, rel_parts, meta)
    steps = int(_number(summary.get("steps", summary.get("planning_length"))))
    questions_obj = summary.get("questions", [])
    questions = int(_number(summary.get("total_questions", summary.get("question_count", len(questions_obj) if isinstance(questions_obj, list) else 0))))
    if isinstance(questions_obj, list):
        query_steps = len({str(q.get("step")) for q in questions_obj if isinstance(q, dict) and q.get("step") is not None})
    else:
        query_steps = min(questions, steps)
    reward_obj = summary.get("reward", 0)
    reward = reward_obj.get("cumulated", 0) if isinstance(reward_obj, dict) else reward_obj
    timing = summary.get("timing") or {}
    return {
        "experiment": experiment,
        "domain": meta.get("domain") or rel_parts[0],
        "scene": _scene(rel_parts, meta),
        "condition": condition,
        "method": method,
        "feedback_source": feedback,
        "threshold": threshold,
        "seed": meta.get("seed", summary.get("seed", "")),
        "success": _bool(summary.get("success")),
        "end_reason": summary.get("end_reason", summary.get("stop_reason", "")),
        "steps": steps,
        "reward": _number(reward),
        "question_count": questions,
        "query_step_count": query_steps,
        "query_rate": query_steps / steps if steps else "",
        "elapsed_seconds": _number(timing.get("total_time", summary.get("total_elapsed_seconds", 0))),
        "source_file": str(path),
    }


def analyze(experiment: str, logs_root: Path, output: Path) -> None:
    root = logs_root / EXPERIMENT_DIRS[experiment]
    if not root.exists():
        raise FileNotFoundError(f"Experiment log directory does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.name.startswith("console_") or path.suffix not in {".txt", ".json"}:
            continue
        # Active Search writes equivalent txt and JSON logs; prefer its JSON.
        if path.suffix == ".txt" and path.with_suffix(".active_search.json").exists():
            continue
        parsed = _parse_txt(path, experiment, root) if path.suffix == ".txt" else _parse_json(path, experiment, root)
        if parsed and parsed["condition"]:
            rows.append(parsed)
    # A rerun with the same experimental seed replaces the older result.
    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    unseeded = 0
    for row in rows:
        seed = str(row["seed"])
        if not seed:
            unseeded += 1
            seed = f"unseeded-{unseeded}"
        deduped[(str(row["domain"]), str(row["scene"]), str(row["condition"]), seed)] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(deduped.values())
    print(f"Wrote {len(deduped)} runs to {output}")


def _metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    field = {
        "success_rate": "success", "average_steps": "steps", "average_questions": "question_count",
        "query_rate": "query_rate", "average_reward": "reward", "elapsed_seconds": "elapsed_seconds",
    }[metric]
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        values.append(1.0 if metric == "success_rate" and _bool(value) else 0.0 if metric == "success_rate" else float(value))
    return values


def refine(input_path: Path, output: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        runs = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        grouped[(row["condition"], row["domain"])].append(row)
    fields = ("condition", "domain", "outcome", "metric", "metric_label", "mean", "std", "n")
    rows = []
    for (condition, domain), group in sorted(grouped.items()):
        outcome_groups = {
            "all": group,
            "success_only": [row for row in group if _bool(row.get("success"))],
            "failure_only": [row for row in group if not _bool(row.get("success"))],
        }
        for outcome, outcome_group in outcome_groups.items():
            for metric, label in METRICS.items():
                values = _metric_values(outcome_group, metric)
                rows.append({
                    "condition": condition, "domain": domain, "outcome": outcome,
                    "metric": metric, "metric_label": label,
                    "mean": mean(values) if values else "", "std": stdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} summary rows to {output}")


def _value_label(value: float) -> str:
    """Format plot annotations compactly without hiding meaningful decimals."""
    absolute = abs(value)
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def plot(summary_path: Path, output_root: Path, experiment: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No summary rows in {summary_path}")
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    if experiment == "e1":
        conditions.sort(key=lambda x: _number(x.removeprefix("tau_")))
    elif experiment in CONDITION_ORDERS:
        order = {condition: index for index, condition in enumerate(CONDITION_ORDERS[experiment])}
        conditions.sort(key=lambda condition: (order.get(condition, len(order)), condition))
    domains = sorted({row["domain"] for row in rows})
    lookup = {(r["outcome"], r["metric"], r["condition"], r["domain"]): r for r in rows}
    outcome_titles = {"all": "All Runs", "success_only": "Successful Runs", "failure_only": "Failed Runs"}
    for outcome in OUTCOMES:
        for metric, label in METRICS.items():
            figsize = (6, 4) if experiment == "e1" else (max(6.4, len(conditions) * 1.25), 4.5)
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            if experiment == "e1":
                for domain_index, domain in enumerate(domains):
                    values = [_number(lookup.get((outcome, metric, c, domain), {}).get("mean"), math.nan) for c in conditions]
                    ax.plot(conditions, values, marker="o", linewidth=2, label=domain)
                    label_offset = 7 if domain_index % 2 == 0 else -12
                    for condition, value in zip(conditions, values):
                        if math.isfinite(value):
                            ax.annotate(
                                _value_label(value),
                                (condition, value),
                                xytext=(0, label_offset),
                                textcoords="offset points",
                                ha="center",
                                va="bottom" if label_offset > 0 else "top",
                                fontsize=8,
                            )
                # Keep the series compact while leaving room for edge/top labels.
                ax.margins(x=0.04, y=0.08)
            else:
                width = 0.8 / max(len(domains), 1)
                x = list(range(len(conditions)))
                for index, domain in enumerate(domains):
                    values = [_number(lookup.get((outcome, metric, c, domain), {}).get("mean"), math.nan) for c in conditions]
                    bars = ax.bar([v + (index - (len(domains) - 1) / 2) * width for v in x], values, width=width, label=domain)
                    labels = [_value_label(value) if math.isfinite(value) else "" for value in values]
                    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
                ax.set_xticks(x, conditions)
            ax.set_title(outcome_titles[outcome])
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            ax.tick_params(axis="x", rotation=20)
            for suffix in ("png", "pdf"):
                fig.savefig(output_dir / f"{metric}_{outcome}.{suffix}", dpi=200)
            plt.close(fig)
    print(f"Wrote figures to {output_dir}")


def analysis_main(experiment: str, script_dir: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=LOG_ROOT)
    parser.add_argument("--output", type=Path, default=SYSTEM_EVAL_ROOT / "data" / experiment / "runs.csv")
    args = parser.parse_args()
    analyze(experiment, args.logs_root, args.output)


def read_main(experiment: str, script_dir: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SYSTEM_EVAL_ROOT / "data" / experiment / "runs.csv")
    parser.add_argument("--output", type=Path, default=SYSTEM_EVAL_ROOT / "data" / experiment / "summary.csv")
    args = parser.parse_args()
    refine(args.input, args.output)


def plot_main(experiment: str, script_dir: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=SYSTEM_EVAL_ROOT / "data" / experiment / "summary.csv")
    parser.add_argument("--output-root", type=Path, default=SYSTEM_EVAL_ROOT / "figure" / experiment)
    args = parser.parse_args()
    plot(args.input, args.output_root, experiment)
