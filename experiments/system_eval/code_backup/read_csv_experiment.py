"""Stage 2: build refined condition-comparison CSVs for paper tables.

Input:
    <script_dir>/data/raw_runs/<experiment>/all_domains.csv

Output:
    <script_dir>/data/refined/01_threshold.csv
    <script_dir>/data/refined/02_when.csv
    <script_dir>/data/refined/03_scale.csv
    <script_dir>/data/refined/04_answer_mode.csv
    <script_dir>/data/refined/05_active_search.csv
    <script_dir>/data/refined/06_knowno.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
POLICY_CONDITIONS = ("all", "no", "ours", "random", "knowno_gpt4", "knowno_gpt35turbo")
THRESHOLD_CONDITIONS = (
    "threshold_0_0",
    "threshold_0_1",
    "threshold_0_2",
    "threshold_0_3",
    "threshold_0_4",
    "threshold_0_5",
    "threshold_0_6",
    "threshold_0_7",
    "threshold_0_8",
    "threshold_0_9",
    "threshold_1_0",
)
ANSWER_MODE_CONDITIONS = (
    "oracle",
    "noisy_oracle_0_5",
    "noisy_oracle_0_7",
    "noisy_oracle_0_9",
    "human_proxy",
    "knowno_oracle",
    "knowno_noisy_oracle_0_5",
    "knowno_noisy_oracle_0_7",
    "knowno_noisy_oracle_0_9",
    "knowno_human_proxy",
)
ACTIVE_SEARCH_CONDITIONS = ("active_search", "ours")
KNOWNO_CONDITIONS = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt35turbo_raw98",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "gpt4_raw98",
    "ours",
)
KNOWNO_CALIBRATED_CONDITIONS = (
    "gpt35turbo_75",
    "gpt35turbo_85",
    "gpt35turbo_95",
    "gpt4_75",
    "gpt4_85",
    "gpt4_95",
    "ours",
)
DOMAINS = ("waste", "tomato", "all")
KNOWNO_DOMAINS = ("tomato", "wastesorting")
SCALE_CONDITIONS = ("original", "scaled")
SCALE_DOMAINS = ("waste", "tomato")
DOMAIN_ALIASES = {"wastesorting": "waste", "tomato": "tomato"}
KNOWNO_QHAT_CONDITIONS = {
    "0.8938": "75",
    "0.9082": "85",
    "0.9243": "95",
    "0.8512": "75",
    "0.8851": "85",
    "0.9028": "95",
    "0.7322": "75",
    "0.7779": "85",
    "0.8404": "95",
    "0.7084": "75",
    "0.7369": "85",
    "0.8704": "95",
    "0.98": "raw98",
    "0.9800": "raw98",
}
METRICS = (
    "success_rate",
    "average_step",
    "average_step_success_only",
    "average_step_failure_only",
    "average_question",
    "average_question_success_only",
    "average_question_failure_only",
    "query_probability_per_step",
    "query_probability_per_step_success_only",
    "query_probability_per_step_failure_only",
    "average_reward",
    "average_reward_success_only",
    "average_reward_failure_only",
    "elapsed_time",
    "elapsed_time_success_only",
    "elapsed_time_failure_only",
    "prediction_set_size_when_asked",
    "prediction_set_size_when_asked_success_only",
    "prediction_set_size_when_asked_failure_only",
)
KNOWNO_METRICS = (
    "success_rate",
    "average_step",
    "average_step_success_only",
    "average_step_failure_only",
    "average_question",
    "average_question_success_only",
    "average_question_failure_only",
    "query_probability_per_step",
    "elapsed_time",
    "prediction_set_size_when_asked",
    "token_overall",
)
METRIC_LABELS = {
    "success_rate": "Success Rate",
    "average_step": "Average Step",
    "average_step_success_only": "Average Step",
    "average_step_failure_only": "Average Step",
    "average_question": "Average Query Number",
    "average_question_success_only": "Average Query Number",
    "average_question_failure_only": "Average Query Number",
    "query_probability_per_step": "Query Probability per Step",
    "query_probability_per_step_success_only": "Query Probability per Step",
    "query_probability_per_step_failure_only": "Query Probability per Step",
    "average_reward": "Average Reward",
    "average_reward_success_only": "Average Reward",
    "average_reward_failure_only": "Average Reward",
    "elapsed_time": "Elapsed Time",
    "elapsed_time_success_only": "Elapsed Time",
    "elapsed_time_failure_only": "Elapsed Time",
    "prediction_set_size_when_asked": "Prediction Set Size When Asked",
    "prediction_set_size_when_asked_success_only": "Prediction Set Size When Asked",
    "prediction_set_size_when_asked_failure_only": "Prediction Set Size When Asked",
    "token_overall": "Token Usage",
}
SCALE_METRICS = (
    "success_rate",
    "average_step_success_only",
    "average_question",
    "query_probability_per_step",
    "elapsed_time",
    "search_time_avg",
    "update_time_avg",
    "step_total_time_avg",
    "tree_nodes_expanded_this_step_avg",
    "belief_frontier_size_avg",
    "belief_frontier_size_max",
)
SCALE_METRIC_LABELS = {
    "success_rate": "Success Rate",
    "average_step_success_only": "Average Step",
    "average_question": "Average Query Number",
    "query_probability_per_step": "Query Probability per Step",
    "elapsed_time": "Elapsed Time",
    "search_time_avg": "Search Time per Step",
    "update_time_avg": "Update Time per Step",
    "step_total_time_avg": "Step Time",
    "tree_nodes_expanded_this_step_avg": "Expanded Nodes",
    "belief_frontier_size_avg": "Belief Frontier Size",
    "belief_frontier_size_max": "Max Belief Frontier Size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw run CSV into wide condition-comparison CSV.")
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--experiment",
        choices=("threshold", "policy", "answer_mode", "active_search", "knowno"),
        default="answer_mode",
        help="Condition axis to build. Defaults to the unified system_log raw runs.",
    )
    parser.add_argument("--exclude-raw98", action="store_true", help="For --experiment knowno, omit raw qhat=0.98 conditions.")
    parser.add_argument("--exclude-ours", action="store_true", help="For --experiment knowno, omit the 00_ours reference.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_input_rows(path: Path, user_supplied_input: bool) -> list[dict[str, str]]:
    if path.exists():
        return read_rows(path)
    if user_supplied_input:
        raise FileNotFoundError(path)
    print(f"Missing input {path}; writing placeholder output.")
    return []


def as_float(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def mean_or_blank(values: list[float]) -> str:
    valid = [value for value in values if not math.isnan(value)]
    return f"{mean(valid):.6f}" if valid else ""


def rows_for_outcome(rows: list[dict[str, str]], outcome: str) -> list[dict[str, str]]:
    if outcome == "success":
        return [row for row in rows if is_success(row)]
    if outcome == "failure":
        return [row for row in rows if not is_success(row)]
    return rows


def query_probability_or_blank(rows: list[dict[str, str]]) -> str:
    query_steps = 0.0
    planning_steps = 0.0
    for row in rows:
        query_step_count = as_float(row.get("query_step_count"))
        planning_length = as_float(row.get("planning_length"))
        if math.isnan(query_step_count) or math.isnan(planning_length) or planning_length <= 0:
            continue
        query_steps += query_step_count
        planning_steps += planning_length
    if planning_steps <= 0:
        return ""
    return f"{query_steps / planning_steps:.6f}"


def metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    if metric == "success_rate":
        return [1.0 if is_success(row) else 0.0 for row in rows]
    if metric == "average_step":
        return [as_float(row["planning_length"]) for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row["planning_length"]) for row in rows if is_success(row)]
    if metric == "average_step_failure_only":
        return [as_float(row["planning_length"]) for row in rows if not is_success(row)]
    if metric == "average_question":
        return [as_float(row["question_count"]) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row["question_count"]) for row in rows if is_success(row)]
    if metric == "average_question_failure_only":
        return [as_float(row["question_count"]) for row in rows if not is_success(row)]
    if metric == "average_reward":
        return [as_float(row["reward"]) for row in rows]
    if metric == "average_reward_success_only":
        return [as_float(row["reward"]) for row in rows if is_success(row)]
    if metric == "average_reward_failure_only":
        return [as_float(row["reward"]) for row in rows if not is_success(row)]
    if metric == "elapsed_time":
        return [as_float(row["elapsed_seconds"]) for row in rows]
    if metric == "elapsed_time_success_only":
        return [as_float(row["elapsed_seconds"]) for row in rows if is_success(row)]
    if metric == "elapsed_time_failure_only":
        return [as_float(row["elapsed_seconds"]) for row in rows if not is_success(row)]
    if metric == "token_overall":
        return [as_float(row.get("token_overall")) for row in rows]
    if metric in {
        "search_time_avg",
        "update_time_avg",
        "step_total_time_avg",
        "tree_nodes_expanded_this_step_avg",
        "belief_frontier_size_avg",
        "belief_frontier_size_max",
    }:
        return [as_float(row.get(metric)) for row in rows]
    if metric.startswith("prediction_set_size_when_asked"):
        if metric.endswith("_success_only"):
            rows = rows_for_outcome(rows, "success")
        elif metric.endswith("_failure_only"):
            rows = rows_for_outcome(rows, "failure")
        weighted: list[float] = []
        for row in rows:
            question_count = int(float(row.get("question_count") or 0))
            value = as_float(row.get("average_prediction_set_size_when_asked"))
            if question_count > 0 and not math.isnan(value):
                weighted.extend([value] * question_count)
        return weighted
    raise ValueError(f"unknown metric: {metric}")


def metric_result(rows: list[dict[str, str]], metric: str) -> str:
    if metric == "query_probability_per_step":
        return query_probability_or_blank(rows)
    if metric == "query_probability_per_step_success_only":
        return query_probability_or_blank(rows_for_outcome(rows, "success"))
    if metric == "query_probability_per_step_failure_only":
        return query_probability_or_blank(rows_for_outcome(rows, "failure"))
    return mean_or_blank(metric_values(rows, metric))


def conditions_for_experiment(experiment: str) -> tuple[str, ...]:
    if experiment == "threshold":
        return THRESHOLD_CONDITIONS
    if experiment == "answer_mode":
        return ANSWER_MODE_CONDITIONS
    if experiment == "active_search":
        return ACTIVE_SEARCH_CONDITIONS
    if experiment == "knowno":
        return KNOWNO_CONDITIONS
    return POLICY_CONDITIONS


def metrics_for_experiment(experiment: str) -> tuple[str, ...]:
    if experiment == "knowno":
        return KNOWNO_METRICS
    return METRICS


def domains_for_experiment(experiment: str) -> tuple[str, ...]:
    if experiment == "knowno":
        return KNOWNO_DOMAINS
    return DOMAINS


def knowno_condition_for_row(row: dict[str, str]) -> str:
    if row.get("experiment") == "answer_mode" and row.get("policy") == "oracle":
        return "ours"

    policy = row.get("policy", "")
    if policy == "knowno_gpt35turbo":
        model = "gpt35turbo"
    elif policy == "knowno_gpt4":
        model = "gpt4"
    else:
        return ""

    qhat = str(row.get("qhat", "")).strip()
    qhat_key = f"{as_float(qhat):.4f}" if qhat else ""
    target = KNOWNO_QHAT_CONDITIONS.get(qhat_key) or KNOWNO_QHAT_CONDITIONS.get(qhat)
    return f"{model}_{target}" if target else ""


def threshold_condition_for_row(row: dict[str, str]) -> str:
    if row.get("experiment") != "threshold":
        return ""
    threshold = row.get("threshold", "")
    return f"threshold_{threshold.replace('.', '_')}" if threshold else ""


def active_search_condition_for_row(row: dict[str, str]) -> str:
    if row.get("experiment") == "answer_mode" and row.get("policy") == "oracle":
        return "ours"
    if row.get("experiment") == "active_search":
        return "active_search"
    return ""


def condition_for_row(row: dict[str, str], experiment: str) -> str:
    if experiment == "threshold":
        return threshold_condition_for_row(row)
    if experiment == "answer_mode" and row["experiment"] == "answer_mode":
        return row.get("policy", "")
    if experiment == "active_search":
        return active_search_condition_for_row(row)
    if experiment == "knowno":
        return knowno_condition_for_row(row)
    if row["experiment"] == "knowno":
        return row.get("policy", "")
    if row["experiment"] == "when":
        return row.get("policy", "")
    return ""


def collect_policy_rows(
    rows: list[dict[str, str]],
    conditions: tuple[str, ...],
    experiment: str,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        condition = condition_for_row(row, experiment)
        domain = row.get("domain", "") if experiment == "knowno" else DOMAIN_ALIASES.get(row.get("domain", ""))
        if condition not in conditions or domain not in domains_for_experiment(experiment):
            continue
        grouped[(condition, domain)].append(row)
        if experiment != "knowno":
            grouped[(condition, "all")].append(row)
    return grouped


def output_column(condition: str, domain: str, experiment: str) -> str:
    if experiment == "knowno":
        return f"{domain}_{condition}"
    return f"{condition}_{domain}"


def output_fields(conditions: tuple[str, ...], experiment: str) -> list[str]:
    fields = ["metric", "metric_label"]
    if experiment == "knowno":
        for domain in domains_for_experiment(experiment):
            for condition in conditions:
                fields.append(output_column(condition, domain, experiment))
        return fields
    for condition in conditions:
        for domain in domains_for_experiment(experiment):
            fields.append(output_column(condition, domain, experiment))
    return fields


def build_output(
    rows: list[dict[str, str]],
    conditions: tuple[str, ...],
    experiment: str,
    metrics: tuple[str, ...],
) -> list[dict[str, str]]:
    grouped = collect_policy_rows(rows, conditions, experiment)
    output: list[dict[str, str]] = []
    for metric in metrics:
        row = {"metric": metric, "metric_label": METRIC_LABELS[metric]}
        for condition in conditions:
            for domain in domains_for_experiment(experiment):
                row[output_column(condition, domain, experiment)] = metric_result(grouped[(condition, domain)], metric)
        output.append(row)
    return output


def scene_number(row: dict[str, str]) -> int:
    scene = row.get("scene", "")
    try:
        return int(scene.removeprefix("scene_"))
    except ValueError:
        return -1


def scale_condition_for_row(row: dict[str, str]) -> str:
    scene = scene_number(row)
    if row.get("experiment") == "scalability" and row.get("policy") == "ours" and 1 <= scene <= 5:
        return "original"
    if row.get("experiment") == "scalability" and row.get("policy") == "ours" and 6 <= scene <= 10:
        return "scaled"
    return ""


def collect_scale_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        condition = scale_condition_for_row(row)
        domain = DOMAIN_ALIASES.get(row.get("domain", ""))
        if condition not in SCALE_CONDITIONS or domain not in SCALE_DOMAINS:
            continue
        grouped[(condition, domain)].append(row)
    return grouped


def scale_output_fields() -> list[str]:
    fields = ["metric", "metric_label"]
    for condition in SCALE_CONDITIONS:
        for domain in SCALE_DOMAINS:
            fields.append(f"{condition}_{domain}")
    return fields


def build_scale_output(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped = collect_scale_rows(rows)
    output: list[dict[str, str]] = []
    for metric in SCALE_METRICS:
        row = {"metric": metric, "metric_label": SCALE_METRIC_LABELS[metric]}
        for condition in SCALE_CONDITIONS:
            for domain in SCALE_DOMAINS:
                row[f"{condition}_{domain}"] = metric_result(grouped[(condition, domain)], metric)
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    default_inputs = {
        "threshold": SCRIPT_DIR / "data" / "raw_runs" / "01_threshold" / "all_domains.csv",
        "answer_mode": SCRIPT_DIR / "data" / "raw_runs" / "04_answer_mode" / "all_domains.csv",
        "policy": SCRIPT_DIR / "data" / "raw_runs" / "02_when" / "all_domains.csv",
        "active_search": SCRIPT_DIR / "data" / "raw_runs" / "05_active_search" / "all_domains.csv",
        "knowno": SCRIPT_DIR / "data" / "raw_runs" / "06_knowno" / "all_domains.csv",
    }
    default_outputs = {
        "threshold": SCRIPT_DIR / "data" / "refined" / "01_threshold.csv",
        "answer_mode": SCRIPT_DIR / "data" / "refined" / "04_answer_mode.csv",
        "policy": SCRIPT_DIR / "data" / "refined" / "02_when.csv",
        "active_search": SCRIPT_DIR / "data" / "refined" / "05_active_search.csv",
        "knowno": SCRIPT_DIR / "data" / "refined" / "06_knowno.csv",
    }
    input_path = Path(args.input) if args.input else default_inputs[args.experiment]
    raw_rows = read_input_rows(input_path, bool(args.input))
    conditions = conditions_for_experiment(args.experiment)
    if args.experiment == "knowno":
        conditions = KNOWNO_CALIBRATED_CONDITIONS if args.exclude_raw98 else conditions
        if args.exclude_ours:
            conditions = tuple(condition for condition in conditions if condition != "ours")
    rows = build_output(raw_rows, conditions, args.experiment, metrics_for_experiment(args.experiment))
    output_path = Path(args.output) if args.output else default_outputs[args.experiment]
    write_csv(output_path, rows, output_fields(conditions, args.experiment))
    print(f"Wrote {len(rows)} rows to {output_path}")
    if not args.output and args.experiment == "policy":
        scale_rows = build_scale_output(raw_rows)
        scale_output_path = SCRIPT_DIR / "data" / "refined" / "03_scale.csv"
        write_csv(scale_output_path, scale_rows, scale_output_fields())
        print(f"Wrote {len(scale_rows)} rows to {scale_output_path}")


if __name__ == "__main__":
    main()
