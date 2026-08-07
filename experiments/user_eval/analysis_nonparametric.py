"""Stage 2: run nonparametric user-study statistics.

Usage:
    python3 experiments/user_eval/analysis_nonparametric.py

Input:
    experiments/user_eval/data/raw_user_study/participant_metric_long.csv

Outputs:
    experiments/user_eval/data/nonparametric_user_study/friedman_results.csv
    experiments/user_eval/data/nonparametric_user_study/posthoc_pairwise_bonferroni.csv
    experiments/user_eval/data/nonparametric_user_study/condition_summary.csv
    experiments/user_eval/data/nonparametric_user_study/long_data.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "raw_user_study" / "participant_metric_long.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "nonparametric_user_study"

CONDITIONS = ("C1", "C2", "C3", "C4", "C5")
CONDITION_LABELS = {
    "C1": "C1: All",
    "C2": "C2: No",
    "C3": "C3: Ours1",
    "C4": "C4: Ours2",
    "C5": "C5: KnowNo",
}
BINARY_METRICS = {"조작성공률 (%)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Friedman/Cochran and posthoc tests for user study data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_long(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def stars(p_value: float | None) -> str:
    if p_value is None or math.isnan(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def build_values(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    values: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        value = as_float(row.get("value", ""))
        if value is None:
            continue
        values[
            (
                row["subject"].strip().lower(),
                row["domain"].strip(),
                row["condition_short"].strip(),
                row["metric"].strip(),
            )
        ] = value
    return values


def available_conditions(
    values: dict[tuple[str, str, str, str], float],
    subjects: list[str],
    domain: str,
    metric: str,
) -> list[str]:
    conditions: list[str] = []
    for condition in CONDITIONS:
        count = sum((subject, domain, condition, metric) in values for subject in subjects)
        if count > 0:
            conditions.append(condition)
    return conditions


def complete_subjects(
    values: dict[tuple[str, str, str, str], float],
    subjects: list[str],
    domain: str,
    metric: str,
    conditions: list[str],
) -> list[str]:
    return [
        subject
        for subject in subjects
        if all((subject, domain, condition, metric) in values for condition in conditions)
    ]


def matrix_for(
    values: dict[tuple[str, str, str, str], float],
    subjects: list[str],
    domain: str,
    metric: str,
    conditions: list[str],
) -> list[list[float]]:
    return [
        [values[(subject, domain, condition, metric)] for condition in conditions]
        for subject in subjects
    ]


def cochran_q(matrix: list[list[float]]) -> tuple[float, float]:
    n_subjects = len(matrix)
    n_conditions = len(matrix[0]) if matrix else 0
    if n_subjects == 0 or n_conditions < 3:
        return math.nan, math.nan
    binary = [[1.0 if value > 0 else 0.0 for value in row] for row in matrix]
    col_sums = [sum(row[col] for row in binary) for col in range(n_conditions)]
    row_sums = [sum(row) for row in binary]
    total = sum(col_sums)
    denominator = n_conditions * total - sum(row_sum * row_sum for row_sum in row_sums)
    if denominator == 0:
        return math.nan, math.nan
    numerator = (n_conditions - 1) * (n_conditions * sum(col * col for col in col_sums) - total * total)
    q_stat = numerator / denominator
    p_value = stats.chi2.sf(q_stat, n_conditions - 1)
    return q_stat, p_value


def rank_biserial_from_diffs(diffs: list[float]) -> float:
    nonzero = [diff for diff in diffs if diff != 0]
    if not nonzero:
        return 0.0
    abs_values = [abs(diff) for diff in nonzero]
    ranks = stats.rankdata(abs_values)
    positive = sum(rank for rank, diff in zip(ranks, nonzero) if diff > 0)
    negative = sum(rank for rank, diff in zip(ranks, nonzero) if diff < 0)
    denom = positive + negative
    return 0.0 if denom == 0 else (positive - negative) / denom


def mcnemar_exact(a_values: list[float], b_values: list[float]) -> tuple[float, str]:
    b_over_c = 0
    c_over_b = 0
    for a_value, b_value in zip(a_values, b_values):
        a_bin = a_value > 0
        b_bin = b_value > 0
        if a_bin and not b_bin:
            b_over_c += 1
        elif b_bin and not a_bin:
            c_over_b += 1
    discordant = b_over_c + c_over_b
    if discordant == 0:
        return 1.0, f"{b_over_c}/{c_over_b}"
    p_value = stats.binomtest(min(b_over_c, c_over_b), discordant, 0.5, alternative="two-sided").pvalue
    return p_value, f"{b_over_c}/{c_over_b}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(values_for_condition: list[float]) -> dict[str, str]:
    n = len(values_for_condition)
    if n == 0:
        return {"n": "0", "mean": "nan", "sd": "nan", "se": "nan", "median": "nan", "min": "nan", "max": "nan"}
    sd = stdev(values_for_condition) if n >= 2 else 0.0
    return {
        "n": str(n),
        "mean": fmt(mean(values_for_condition)),
        "sd": fmt(sd),
        "se": fmt(sd / math.sqrt(n)),
        "median": fmt(median(values_for_condition)),
        "min": fmt(min(values_for_condition)),
        "max": fmt(max(values_for_condition)),
    }


def main() -> None:
    args = parse_args()
    rows = read_long(args.input)
    values = build_values(rows)
    subjects = sorted({row["subject"].strip().lower() for row in rows})
    domains = sorted({row["domain"].strip() for row in rows})
    metrics = sorted({row["metric"].strip() for row in rows})

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.input, output_dir / "long_data.csv")

    omnibus_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for domain in domains:
        for metric in metrics:
            conditions = available_conditions(values, subjects, domain, metric)
            complete = complete_subjects(values, subjects, domain, metric, conditions)

            for condition in CONDITIONS:
                condition_values = [
                    values[(subject, domain, condition, metric)]
                    for subject in subjects
                    if (subject, domain, condition, metric) in values
                ]
                summary = summarize_condition(condition_values)
                summary_rows.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "condition": CONDITION_LABELS[condition],
                        "condition_short": condition,
                        **summary,
                    }
                )

            if len(conditions) < 3 or len(complete) < 2:
                continue

            matrix = matrix_for(values, complete, domain, metric, conditions)
            if metric in BINARY_METRICS:
                statistic, p_value = cochran_q(matrix)
                test = "Cochran's Q"
                statistic_name = "Q"
            else:
                columns = [[row[index] for row in matrix] for index in range(len(conditions))]
                try:
                    statistic, p_value = stats.friedmanchisquare(*columns)
                except ValueError:
                    statistic, p_value = math.nan, math.nan
                test = "Friedman"
                statistic_name = "chi-square"

            n_subjects = len(complete)
            kendalls_w = statistic / (n_subjects * (len(conditions) - 1)) if not math.isnan(statistic) else math.nan
            omnibus_rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "test": test,
                    "statistic_name": statistic_name,
                    "statistic": fmt(statistic),
                    "df": len(conditions) - 1,
                    "p": fmt(p_value),
                    "kendalls_w": fmt(kendalls_w),
                    "n_complete_subjects": n_subjects,
                    "complete_subjects": ",".join(complete),
                    "n_conditions": len(conditions),
                    "conditions_included": ",".join(conditions),
                }
            )

            comparisons = list(itertools.combinations(conditions, 2))
            correction = len(comparisons)
            for condition_a, condition_b in comparisons:
                a_values = [values[(subject, domain, condition_a, metric)] for subject in complete]
                b_values = [values[(subject, domain, condition_b, metric)] for subject in complete]
                diffs = [a - b for a, b in zip(a_values, b_values)]
                nonzero = [diff for diff in diffs if diff != 0]

                if metric in BINARY_METRICS:
                    p_raw, mcnemar_counts = mcnemar_exact(a_values, b_values)
                    test_name = "McNemar exact"
                    w_value: float | None = None
                elif nonzero:
                    result = stats.wilcoxon(a_values, b_values, zero_method="wilcox", alternative="two-sided")
                    p_raw = float(result.pvalue)
                    w_value = float(result.statistic)
                    mcnemar_counts = ""
                    test_name = "Wilcoxon signed-rank"
                else:
                    p_raw = 1.0
                    w_value = 0.0
                    mcnemar_counts = ""
                    test_name = "Wilcoxon signed-rank"

                p_adj = min(1.0, p_raw * correction)
                pairwise_rows.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "test": test_name,
                        "comparison": f"{condition_a} vs {condition_b}",
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "n_complete_subjects": len(complete),
                        "n_nonzero_pairs": len(nonzero),
                        "W": fmt(w_value),
                        "p_raw": fmt(p_raw),
                        "p_bonferroni": fmt(p_adj),
                        "significant_p_lt_05": "yes" if p_adj < 0.05 else "no",
                        "stars": stars(p_adj),
                        "rank_biserial": fmt(rank_biserial_from_diffs(diffs)),
                        "mean_difference_a_minus_b": fmt(mean(diffs)),
                        "median_difference_a_minus_b": fmt(median(diffs)),
                        "n_conditions": len(conditions),
                        "conditions_included": ",".join(conditions),
                        "mcnemar_b_over_c": mcnemar_counts,
                    }
                )

    write_csv(
        output_dir / "friedman_results.csv",
        omnibus_rows,
        [
            "domain",
            "metric",
            "test",
            "statistic_name",
            "statistic",
            "df",
            "p",
            "kendalls_w",
            "n_complete_subjects",
            "complete_subjects",
            "n_conditions",
            "conditions_included",
        ],
    )
    write_csv(
        output_dir / "posthoc_pairwise_bonferroni.csv",
        pairwise_rows,
        [
            "domain",
            "metric",
            "test",
            "comparison",
            "condition_a",
            "condition_b",
            "n_complete_subjects",
            "n_nonzero_pairs",
            "W",
            "p_raw",
            "p_bonferroni",
            "significant_p_lt_05",
            "stars",
            "rank_biserial",
            "mean_difference_a_minus_b",
            "median_difference_a_minus_b",
            "n_conditions",
            "conditions_included",
            "mcnemar_b_over_c",
        ],
    )
    write_csv(
        output_dir / "condition_summary.csv",
        summary_rows,
        ["domain", "metric", "condition", "condition_short", "n", "mean", "sd", "se", "median", "min", "max"],
    )

    print(f"input: {args.input}")
    print(f"output_dir: {output_dir}")
    print(f"omnibus_rows: {len(omnibus_rows)}")
    print(f"pairwise_rows: {len(pairwise_rows)}")
    print(f"summary_rows: {len(summary_rows)}")


if __name__ == "__main__":
    main()
