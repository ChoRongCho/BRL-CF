from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

from process_responses import (
    CONDITION_ORDER,
    DEFAULT_ANSWER_PATH,
    DEFAULT_RESPONSE_PATH,
    DEFAULT_SURVEY_PATH,
    DOMAIN_ORDER,
    QUESTIONS,
    build_answer_lookup,
    condition_value,
    normalize,
    read_csv_dicts,
    read_csv_table,
    read_survey_option_codes,
    score_sagat_rows,
    unique_header_indices,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_OUTPUT_PATH = BASE_DIR / "stats_results.csv"
DEFAULT_MATRIX_OUTPUT_PATH = BASE_DIR / "stats_pairwise_matrix.csv"
DEFAULT_SCORES_OUTPUT_PATH = BASE_DIR / "participant_condition_scores.csv"
DEFAULT_TEXT_OUTPUT_PATH = BASE_DIR / "stats_summary.txt"

DOMAIN_PANELS: tuple[tuple[str, str | None], ...] = (
    ("토마토 수확", "토마토 수확"),
    ("분리수거", "분리수거"),
    ("통합", None),
)
CONDITION_SHORT = {
    "C1: All": "C1",
    "C2: No": "C2",
    "C3: Ours1": "C3",
    "C4: Ours2": "C4",
    "C5: KnowNo": "C5",
}
PLANNED_COMPARISONS = (
    ("C3: Ours1", "C1: All"),
    ("C3: Ours1", "C2: No"),
    ("C3: Ours1", "C5: KnowNo"),
    ("C4: Ours2", "C1: All"),
    ("C4: Ours2", "C2: No"),
    ("C4: Ours2", "C3: Ours1"),
    ("C4: Ours2", "C5: KnowNo"),
)

SCALE_METRICS: dict[str, dict[str, object]] = {
    "NASA_TLX_workload": {
        "items": ("N1", "N2", "N3", "N5", "N6"),
        "direction": "lower",
        "description": "NASA-TLX workload excluding N4 performance",
    },
    "NASA_N1_mental": {"items": ("N1",), "direction": "lower", "description": "NASA-TLX mental demand"},
    "NASA_N2_physical": {"items": ("N2",), "direction": "lower", "description": "NASA-TLX physical demand"},
    "NASA_N3_temporal": {"items": ("N3",), "direction": "lower", "description": "NASA-TLX temporal demand"},
    "NASA_N4_performance": {"items": ("N4",), "direction": "higher", "description": "NASA-TLX perceived performance"},
    "NASA_N5_effort": {"items": ("N5",), "direction": "lower", "description": "NASA-TLX effort"},
    "NASA_N6_frustration": {"items": ("N6",), "direction": "lower", "description": "NASA-TLX frustration"},
    "Fatigue": {"items": ("F",), "direction": "lower", "description": "Fatigue"},
    "SART_demand": {"items": ("S1", "S2", "S3"), "direction": "lower", "description": "SART demand"},
    "SART_supply": {"items": ("S4", "S5", "S6", "S7"), "direction": "higher", "description": "SART supply"},
    "SART_understanding": {"items": ("S8", "S9", "S10"), "direction": "higher", "description": "SART understanding"},
}

STAT_FIELDS = [
    "measure",
    "description",
    "domain_panel",
    "direction",
    "test_family",
    "test",
    "comparison",
    "n",
    "C1_mean",
    "C1_sd",
    "C2_mean",
    "C2_sd",
    "C3_mean",
    "C3_sd",
    "C4_mean",
    "C4_sd",
    "C5_mean",
    "C5_sd",
    "statistic",
    "df",
    "p",
    "p_holm",
    "effect_size",
    "mean_difference",
    "median_difference",
]

MATRIX_FIELDS = [
    "measure",
    "description",
    "domain_panel",
    "direction",
    "value_type",
    "row_condition",
    *[CONDITION_SHORT[condition] for condition in CONDITION_ORDER],
]

SCORE_FIELDS = ["subject", "domain_panel", "condition", "measure", "value"]


def row_get(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index]


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"\d+", subject)
    return (int(match.group(0)) if match else 999999, subject)


def scale_item_id(column: str) -> str | None:
    item = normalize(column).split(".", 1)[0]
    if item == "F":
        return item
    if len(item) >= 2 and item[0] in {"N", "S"} and item[1:].isdigit():
        return item
    return None


def scale_column_indices(header: list[str]) -> dict[str, int]:
    columns: dict[str, int] = {}
    for index, column in enumerate(header):
        item = scale_item_id(column)
        if item is not None:
            columns[item] = index
    return columns


def parse_float(value: str | None) -> float | None:
    value = normalize(value)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def chi_square_sf(value: float, df: int) -> float:
    if value <= 0:
        return 1.0
    if df == 4:
        return math.exp(-value / 2) * (1 + value / 2)
    # Wilson-Hilferty normal approximation fallback for unexpected dfs.
    z = ((value / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def friedman_test(matrix: list[list[float]]) -> tuple[float, int, float, float]:
    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n == 0 or k < 2:
        return 0.0, max(0, k - 1), 1.0, 0.0

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
    df = k - 1
    p_value = chi_square_sf(statistic, df)
    kendalls_w = statistic / (n * df) if n and df else 0.0
    return statistic, df, p_value, kendalls_w


def wilcoxon_exact(values_a: list[float], values_b: list[float]) -> tuple[int, float, float, float, float, float]:
    diffs = [a - b for a, b in zip(values_a, values_b) if a != b]
    n = len(diffs)
    if n == 0:
        return 0, 0.0, 0.0, 1.0, 0.0, 0.0

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
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0
    return n, min(w_plus, w_minus), w_plus - w_minus, p_value, rank_biserial, median(diffs)


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


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_sagat_item_rows(
    header: list[str],
    response_rows: list[list[str]],
    answer_sheet: Path,
    survey: Path,
) -> list[dict[str, str]]:
    answer_lookup = build_answer_lookup(read_csv_dicts(answer_sheet))
    option_codes = read_survey_option_codes(survey)
    sagat_rows, warnings = score_sagat_rows(header, response_rows, answer_lookup, option_codes)
    if warnings:
        for warning in warnings:
            print(f"warning: {warning}")
    return sagat_rows


def build_measure_observations(
    header: list[str],
    response_rows: list[list[str]],
    sagat_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], list[float]]:
    indices = unique_header_indices(header)
    scale_columns = scale_column_indices(header)
    observations: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for row in response_rows:
        subject = normalize(row_get(row, indices.get("피험자")))
        domain = normalize(row_get(row, indices.get("평가 도메인")))
        condition = condition_value(normalize(row_get(row, indices.get("조건 선택"))))
        if not subject or domain not in DOMAIN_ORDER or condition not in CONDITION_ORDER:
            continue

        for metric_name, metric in SCALE_METRICS.items():
            values = [
                value
                for item in metric["items"]  # type: ignore[index]
                if (value := parse_float(row_get(row, scale_columns.get(item)))) is not None
            ]
            if values:
                observations[(subject, domain, condition, metric_name)].append(mean(values))

    for row in sagat_rows:
        subject = row["피험자"]
        domain = row["평가 도메인"]
        condition = row["조건 선택"]
        question = row["문항"]
        value = float(row["값"])
        observations[(subject, domain, condition, "SAGAT_accuracy")].append(value)
        observations[(subject, domain, condition, f"SAGAT_{question}")].append(value)

    return observations


def measure_info(measure: str) -> tuple[str, str]:
    if measure == "SAGAT_accuracy":
        return "SAGAT mean accuracy across Q1-Q6", "higher"
    if measure.startswith("SAGAT_Q"):
        return f"SAGAT correctness for {measure.removeprefix('SAGAT_')}", "higher"
    metric = SCALE_METRICS[measure]
    return str(metric["description"]), str(metric["direction"])


def build_subject_condition_scores(
    observations: dict[tuple[str, str, str, str], list[float]],
) -> list[dict[str, str]]:
    subjects = sorted({key[0] for key in observations}, key=subject_sort_key)
    measures = sorted({key[3] for key in observations}, key=measure_sort_key)
    rows: list[dict[str, str]] = []

    for subject in subjects:
        for panel_label, domain_filter in DOMAIN_PANELS:
            for condition in CONDITION_ORDER:
                for measure_name in measures:
                    values: list[float] = []
                    for domain in DOMAIN_ORDER:
                        if domain_filter is not None and domain != domain_filter:
                            continue
                        values.extend(observations.get((subject, domain, condition, measure_name), []))
                    if not values:
                        continue
                    rows.append(
                        {
                            "subject": subject,
                            "domain_panel": panel_label,
                            "condition": condition,
                            "measure": measure_name,
                            "value": fmt(mean(values)),
                        }
                    )
    return rows


def measure_sort_key(measure: str) -> tuple[int, int, str]:
    if measure == "SAGAT_accuracy":
        return (0, 0, measure)
    if measure.startswith("SAGAT_Q"):
        return (0, int(measure.rsplit("Q", 1)[1]), measure)
    order = list(SCALE_METRICS)
    return (1, order.index(measure) if measure in SCALE_METRICS else 999, measure)


def score_lookup(score_rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    return {
        (row["subject"], row["domain_panel"], row["condition"], row["measure"]): float(row["value"])
        for row in score_rows
    }


def analyze_measure_panel(
    measure: str,
    panel: str,
    scores: dict[tuple[str, str, str, str], float],
    test_family: str,
    pairwise_pairs: tuple[tuple[str, str], ...],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    subjects = sorted(
        {
            subject
            for subject, domain_panel, _condition, item_measure in scores
            if domain_panel == panel and item_measure == measure
        },
        key=subject_sort_key,
    )
    complete_subjects = [
        subject
        for subject in subjects
        if all((subject, panel, condition, measure) in scores for condition in CONDITION_ORDER)
    ]
    matrix = [
        [scores[(subject, panel, condition, measure)] for condition in CONDITION_ORDER]
        for subject in complete_subjects
    ]
    if not matrix:
        return [], []

    description, direction = measure_info(measure)
    condition_values = [[row[index] for row in matrix] for index in range(len(CONDITION_ORDER))]
    means = [mean(values) for values in condition_values]
    sds = [sd(values) for values in condition_values]

    base = {
        "measure": measure,
        "description": description,
        "domain_panel": panel,
        "direction": direction,
        "test_family": test_family,
        "n": str(len(complete_subjects)),
        "C1_mean": fmt(means[0]),
        "C1_sd": fmt(sds[0]),
        "C2_mean": fmt(means[1]),
        "C2_sd": fmt(sds[1]),
        "C3_mean": fmt(means[2]),
        "C3_sd": fmt(sds[2]),
        "C4_mean": fmt(means[3]),
        "C4_sd": fmt(sds[3]),
        "C5_mean": fmt(means[4]),
        "C5_sd": fmt(sds[4]),
    }

    stat_rows: list[dict[str, str]] = []
    statistic, df, p_value, kendalls_w = friedman_test(matrix)
    stat_rows.append(
        {
            **base,
            "test": "Friedman",
            "comparison": "C1-C5",
            "statistic": fmt(statistic),
            "df": str(df),
            "p": fmt(p_value),
            "p_holm": "",
            "effect_size": fmt(kendalls_w),
            "mean_difference": "",
            "median_difference": "",
        }
    )

    pair_rows: list[dict[str, str]] = []
    p_values: list[float] = []
    for condition_a, condition_b in pairwise_pairs:
        index_a = CONDITION_ORDER.index(condition_a)
        index_b = CONDITION_ORDER.index(condition_b)
        values_a = condition_values[index_a]
        values_b = condition_values[index_b]
        n_pairs, w_stat, _signed_sum, p_pair, rank_biserial, median_diff = wilcoxon_exact(values_a, values_b)
        mean_diff = means[index_a] - means[index_b]
        p_values.append(p_pair)
        pair_rows.append(
            {
                **base,
                "test": "Wilcoxon signed-rank exact",
                "comparison": f"{CONDITION_SHORT[condition_a]}-{CONDITION_SHORT[condition_b]}",
                "n": str(n_pairs),
                "statistic": fmt(w_stat),
                "df": "",
                "p": fmt(p_pair),
                "p_holm": "",
                "effect_size": fmt(rank_biserial),
                "mean_difference": fmt(mean_diff),
                "median_difference": fmt(median_diff),
            }
        )

    for row, p_adjusted in zip(pair_rows, holm_adjust(p_values)):
        row["p_holm"] = fmt(p_adjusted)
        stat_rows.append(row)

    matrix_rows = build_pairwise_matrix_rows(measure, description, panel, direction, pair_rows)
    return stat_rows, matrix_rows


def build_pairwise_matrix_rows(
    measure: str,
    description: str,
    panel: str,
    direction: str,
    pair_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    values_by_pair = {row["comparison"]: row for row in pair_rows}
    rows: list[dict[str, str]] = []
    for value_type, field in (("p_holm", "p_holm"), ("mean_difference", "mean_difference"), ("effect_size", "effect_size")):
        for row_condition in CONDITION_ORDER:
            matrix_row = {
                "measure": measure,
                "description": description,
                "domain_panel": panel,
                "direction": direction,
                "value_type": value_type,
                "row_condition": CONDITION_SHORT[row_condition],
            }
            for column_condition in CONDITION_ORDER:
                row_short = CONDITION_SHORT[row_condition]
                column_short = CONDITION_SHORT[column_condition]
                if row_condition == column_condition:
                    matrix_row[column_short] = "-"
                    continue

                comparison = f"{row_short}-{column_short}"
                reverse = f"{column_short}-{row_short}"
                pair = values_by_pair.get(comparison)
                sign = 1.0
                if pair is None:
                    pair = values_by_pair.get(reverse)
                    sign = -1.0
                if pair is None:
                    matrix_row[column_short] = ""
                    continue

                value = pair[field]
                if value_type in {"mean_difference", "effect_size"} and value:
                    value = fmt(sign * float(value))
                matrix_row[column_short] = value
            rows.append(matrix_row)
    return rows


def analyze(score_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scores = score_lookup(score_rows)
    measures = sorted({row["measure"] for row in score_rows}, key=measure_sort_key)
    panels = [panel for panel, _ in DOMAIN_PANELS]
    stat_rows: list[dict[str, str]] = []
    matrix_rows: list[dict[str, str]] = []

    all_pairwise = tuple(combinations(CONDITION_ORDER, 2))
    for measure in measures:
        for panel in panels:
            rows, matrices = analyze_measure_panel(measure, panel, scores, "planned", PLANNED_COMPARISONS)
            stat_rows.extend(rows)
            rows, matrices = analyze_measure_panel(measure, panel, scores, "all_pairwise", all_pairwise)
            matrix_rows.extend(matrices)

    return stat_rows, matrix_rows


def p_label(value: str) -> str:
    if not value:
        return ""
    p_value = float(value)
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def build_summary_text(rows: list[dict[str, str]]) -> str:
    planned_rows = [row for row in rows if row["test_family"] == "planned"]
    lines = [
        "Statistical Analysis Summary",
        "============================",
        "",
        "Design: within-subject, five conditions.",
        "Primary analysis unit: participant x condition score.",
        "Omnibus test: Friedman test across C1-C5.",
        "Planned pairwise tests: exact Wilcoxon signed-rank tests centered on C3 and C4.",
        "Correction: Holm adjustment within each measure/domain panel.",
        "Effect sizes: Kendall's W for Friedman; matched-pairs rank-biserial correlation for Wilcoxon.",
        "",
        "Condition labels: C1=All, C2=No, C3=Ours1, C4=Ours2, C5=KnowNo.",
        "",
    ]

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in planned_rows:
        grouped[(row["measure"], row["domain_panel"])].append(row)

    for (measure, panel), group in sorted(grouped.items(), key=lambda item: (measure_sort_key(item[0][0]), item[0][1])):
        friedman = next(row for row in group if row["test"] == "Friedman")
        lines.append(f"{measure} [{panel}]")
        lines.append("-" * len(lines[-1]))
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
            f"{p_label(friedman['p'])}, Kendall's W = {float(friedman['effect_size']):.2f}."
        )
        for row in group:
            if row["test"] == "Friedman":
                continue
            lines.append(
                f"{row['comparison']}: mean diff = {float(row['mean_difference']):.2f}, "
                f"median diff = {float(row['median_difference']):.2f}, "
                f"raw {p_label(row['p'])}, Holm {p_label(row['p_holm'])}, "
                f"rank-biserial r = {float(row['effect_size']):.2f}."
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run within-subject condition statistics for the user study.")
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--answer-sheet", type=Path, default=DEFAULT_ANSWER_PATH)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY_PATH)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT_PATH)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX_OUTPUT_PATH)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES_OUTPUT_PATH)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header, response_rows = read_csv_table(args.response)
    sagat_rows = build_sagat_item_rows(header, response_rows, args.answer_sheet, args.survey)
    observations = build_measure_observations(header, response_rows, sagat_rows)
    score_rows = build_subject_condition_scores(observations)
    stat_rows, matrix_rows = analyze(score_rows)

    write_csv(args.scores_output, score_rows, SCORE_FIELDS)
    write_csv(args.csv_output, stat_rows, STAT_FIELDS)
    write_csv(args.matrix_output, matrix_rows, MATRIX_FIELDS)
    args.text_output.write_text(build_summary_text(stat_rows), encoding="utf-8")

    print(f"response: {args.response}")
    print(f"answer sheet: {args.answer_sheet}")
    print(f"survey: {args.survey}")
    print(f"participant-condition scores: {args.scores_output} ({len(score_rows)} rows)")
    print(f"stats csv: {args.csv_output} ({len(stat_rows)} rows)")
    print(f"pairwise matrix csv: {args.matrix_output} ({len(matrix_rows)} rows)")
    print(f"summary: {args.text_output}")


if __name__ == "__main__":
    main()
