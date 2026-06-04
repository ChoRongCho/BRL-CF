from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from itertools import product
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSE_PATH = BASE_DIR / "response.csv"
DEFAULT_CSV_OUTPUT_PATH = BASE_DIR / "stats_results.csv"
DEFAULT_TEXT_OUTPUT_PATH = BASE_DIR / "stats_summary.txt"

CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_SHORT = {
    "C1: All": "C1",
    "C2: No": "C2",
    "C3: Ours1": "C3",
    "C4: Ours2": "C4",
    "C5: KnowNo": "C5",
}
PRIMARY_CONDITION = "C4: Ours2"
PLANNED_COMPARISONS = (
    ("C4: Ours2", "C1: All"),
    ("C4: Ours2", "C2: No"),
    ("C4: Ours2", "C3: Ours1"),
    ("C4: Ours2", "C5: KnowNo"),
)

METRICS = {
    "Workload_NASA_low": {
        "items": ("N1", "N2", "N3", "N5", "N6"),
        "direction": "lower",
        "description": "NASA workload excluding self-rated performance",
    },
    "Fatigue_low": {
        "items": ("F",),
        "direction": "lower",
        "description": "Fatigue",
    },
    "Performance_high": {
        "items": ("N4",),
        "direction": "higher",
        "description": "Self-rated task success/performance",
    },
    "SART_demand_low": {
        "items": ("S1", "S2", "S3"),
        "direction": "lower",
        "description": "Situation-awareness demand",
    },
    "SART_supply_high": {
        "items": ("S4", "S5", "S6", "S7"),
        "direction": "higher",
        "description": "Situation-awareness supply; S6 is not reverse-coded here",
    },
    "SART_understanding_high": {
        "items": ("S8", "S9", "S10"),
        "direction": "higher",
        "description": "Situation understanding",
    },
}

CSV_FIELDS = [
    "metric",
    "metric_description",
    "direction",
    "test",
    "comparison",
    "n",
    "C1_mean",
    "C2_mean",
    "C3_mean",
    "C4_mean",
    "C5_mean",
    "statistic",
    "df",
    "p",
    "p_holm",
    "effect",
    "mean_difference",
]


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def condition_value(condition: str) -> str:
    for expected in CONDITION_ORDER:
        if condition == expected or condition.startswith(expected.split(":", 1)[0]):
            return expected
    return condition


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"\d+", subject)
    if match:
        return int(match.group(0)), subject
    return 999999, subject


def item_id_from_scale_column(column: str) -> str | None:
    item_id = column.split(".", 1)[0].strip()
    if item_id == "F":
        return item_id
    if len(item_id) >= 2 and item_id[0] in {"N", "S"} and item_id[1:].isdigit():
        return item_id
    return None


def scale_question_columns(response_rows: list[dict[str, str]]) -> dict[str, str]:
    if not response_rows:
        return {}
    columns: dict[str, str] = {}
    for column in response_rows[0]:
        item_id = item_id_from_scale_column(column)
        if item_id is not None:
            columns[item_id] = column
    return columns


def parse_float(value: str | None) -> float | None:
    normalized = normalize(value)
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = avg_rank
        index = end
    return ranks


def chi_square_sf_df4(value: float) -> float:
    if value <= 0:
        return 1.0
    return math.exp(-value / 2) * (1 + value / 2)


def friedman_test(matrix: list[list[float]]) -> tuple[float, float, float]:
    n = len(matrix)
    k = len(matrix[0])
    rank_sums = [0.0] * k
    tie_sum = 0.0

    for row in matrix:
        ranks = rank_values(row)
        for index, rank in enumerate(ranks):
            rank_sums[index] += rank

        counts: dict[float, int] = defaultdict(int)
        for value in row:
            counts[value] += 1
        tie_sum += sum(count**3 - count for count in counts.values() if count > 1)

    statistic = (12 / (n * k * (k + 1))) * sum(rank_sum**2 for rank_sum in rank_sums) - 3 * n * (k + 1)
    tie_correction = 1 - tie_sum / (n * (k**3 - k))
    if tie_correction > 0:
        statistic /= tie_correction

    p_value = chi_square_sf_df4(statistic)
    kendalls_w = statistic / (n * (k - 1))
    return statistic, p_value, kendalls_w


def wilcoxon_exact(values_a: list[float], values_b: list[float]) -> tuple[int, float, float, float, float]:
    diffs = [a - b for a, b in zip(values_a, values_b) if a != b]
    n = len(diffs)
    if n == 0:
        return 0, 0.0, 0.0, 1.0, 0.0

    ranks = rank_values([abs(diff) for diff in diffs])
    w_plus = sum(rank for diff, rank in zip(diffs, ranks) if diff > 0)
    w_minus = sum(rank for diff, rank in zip(diffs, ranks) if diff < 0)
    observed_signed_rank = abs(w_plus - w_minus)

    more_extreme = 0
    total = 2**n
    for signs in product((-1, 1), repeat=n):
        signed_rank = abs(sum(sign * rank for sign, rank in zip(signs, ranks)))
        if signed_rank >= observed_signed_rank - 1e-12:
            more_extreme += 1

    p_value = min(1.0, more_extreme / total)
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus)
    return n, min(w_plus, w_minus), w_plus - w_minus, p_value, rank_biserial


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, value)
        adjusted[original_index] = running_max
    return adjusted


def build_subject_condition_scores(
    response_rows: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, float]]]:
    columns = scale_question_columns(response_rows)
    row_scores: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for row in response_rows:
        subject = normalize(row.get("피험자"))
        condition = condition_value(normalize(row.get("조건 선택")))
        if not subject or condition not in CONDITION_ORDER:
            continue

        for metric_name, metric in METRICS.items():
            values = [
                value
                for item_id in metric["items"]
                if (value := parse_float(row.get(columns.get(item_id, "")))) is not None
            ]
            if values:
                row_scores[subject][condition][metric_name].append(mean(values))

    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for subject, condition_scores in row_scores.items():
        for condition, metric_scores in condition_scores.items():
            for metric_name, values in metric_scores.items():
                scores[subject][condition][metric_name] = mean(values)
    return scores


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def analyze(response_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scores = build_subject_condition_scores(response_rows)
    subjects = sorted(scores, key=subject_sort_key)
    result_rows: list[dict[str, str]] = []

    for metric_name, metric in METRICS.items():
        complete_subjects = [
            subject
            for subject in subjects
            if all(metric_name in scores[subject].get(condition, {}) for condition in CONDITION_ORDER)
        ]
        matrix = [
            [scores[subject][condition][metric_name] for condition in CONDITION_ORDER]
            for subject in complete_subjects
        ]
        means_by_condition = [mean([row[index] for row in matrix]) for index in range(len(CONDITION_ORDER))]

        statistic, p_value, kendalls_w = friedman_test(matrix)
        base = {
            "metric": metric_name,
            "metric_description": metric["description"],
            "direction": metric["direction"],
            "n": str(len(complete_subjects)),
            "C1_mean": fmt(means_by_condition[0]),
            "C2_mean": fmt(means_by_condition[1]),
            "C3_mean": fmt(means_by_condition[2]),
            "C4_mean": fmt(means_by_condition[3]),
            "C5_mean": fmt(means_by_condition[4]),
        }
        result_rows.append(
            {
                **base,
                "test": "Friedman",
                "comparison": "C1-C5",
                "statistic": fmt(statistic),
                "df": "4",
                "p": fmt(p_value),
                "p_holm": "",
                "effect": fmt(kendalls_w),
                "mean_difference": "",
            }
        )

        pairwise_rows: list[dict[str, str]] = []
        pairwise_p_values: list[float] = []
        for condition_a, condition_b in PLANNED_COMPARISONS:
            index_a = CONDITION_ORDER.index(condition_a)
            index_b = CONDITION_ORDER.index(condition_b)
            values_a = [row[index_a] for row in matrix]
            values_b = [row[index_b] for row in matrix]
            n_pairs, statistic_w, signed_rank_sum, pairwise_p, rank_biserial = wilcoxon_exact(values_a, values_b)
            mean_difference = means_by_condition[index_a] - means_by_condition[index_b]
            pairwise_p_values.append(pairwise_p)
            pairwise_rows.append(
                {
                    **base,
                    "test": "Wilcoxon signed-rank exact",
                    "comparison": f"{CONDITION_SHORT[condition_a]}-{CONDITION_SHORT[condition_b]}",
                    "n": str(n_pairs),
                    "statistic": fmt(statistic_w),
                    "df": "",
                    "p": fmt(pairwise_p),
                    "p_holm": "",
                    "effect": fmt(rank_biserial),
                    "mean_difference": fmt(mean_difference),
                }
            )

        for row, adjusted_p in zip(pairwise_rows, holm_adjust(pairwise_p_values)):
            row["p_holm"] = fmt(adjusted_p)
            result_rows.append(row)

    return result_rows


def p_label(value: str) -> str:
    if not value:
        return ""
    p_value = float(value)
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def build_summary_text(rows: list[dict[str, str]]) -> str:
    lines = [
        "Statistical Analysis Summary",
        "============================",
        "",
        "Design: within-subject, five conditions.",
        "Omnibus test: Friedman test across C1-C5.",
        "Planned pairwise tests: exact Wilcoxon signed-rank tests comparing C4 against C1, C2, C3, and C5.",
        "Correction: Holm adjustment within each metric's four planned comparisons.",
        "Effect sizes: Kendall's W for Friedman; matched-pairs rank-biserial correlation for Wilcoxon.",
        "",
        "Condition labels: C1=All, C2=No, C3=Ours1, C4=Ours2, C5=KnowNo.",
        "",
    ]

    by_metric: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)

    for metric_name, metric_rows in by_metric.items():
        friedman = next(row for row in metric_rows if row["test"] == "Friedman")
        lines.append(metric_name)
        lines.append("-" * len(metric_name))
        lines.append(
            "Means: "
            f"C1={float(friedman['C1_mean']):.2f}, "
            f"C2={float(friedman['C2_mean']):.2f}, "
            f"C3={float(friedman['C3_mean']):.2f}, "
            f"C4={float(friedman['C4_mean']):.2f}, "
            f"C5={float(friedman['C5_mean']):.2f}"
        )
        lines.append(
            f"Friedman: chi2(4) = {float(friedman['statistic']):.2f}, "
            f"{p_label(friedman['p'])}, Kendall's W = {float(friedman['effect']):.2f}."
        )
        for row in metric_rows:
            if row["test"] == "Friedman":
                continue
            lines.append(
                f"{row['comparison']}: mean diff = {float(row['mean_difference']):.2f}, "
                f"raw {p_label(row['p'])}, Holm {p_label(row['p_holm'])}, "
                f"rank-biserial r = {float(row['effect']):.2f}."
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-measures statistics for response.csv.")
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response_rows = read_csv(args.response)
    result_rows = analyze(response_rows)
    write_csv(args.csv_output, result_rows)
    args.text_output.write_text(build_summary_text(result_rows), encoding="utf-8")
    print(f"response: {args.response}")
    print(f"csv output: {args.csv_output}")
    print(f"text output: {args.text_output}")
    print(f"rows: {len(result_rows)}")


if __name__ == "__main__":
    main()
