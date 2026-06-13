from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "users_raw_data" / "피험자별결과.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "domain_condition_friedman"

CONDITIONS = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_SHORT = {
    "C1: All": "C1",
    "C2: No": "C2",
    "C3: Ours1": "C3",
    "C4: Ours2": "C4",
    "C5: KnowNo": "C5",
}
DOMAINS = ("Waste", "Tomato")
DVS = ("조작성공률 (%)", "조작시간 (s)", "SAGAT1", "SAGAT2", "SAGAT3", "Fatigue", "NASA-RTLX", "SART")
BINARY_DVS = {"조작성공률 (%)"}
METRIC_LABELS = {
    "조작성공률 (%)": "Task success (%)",
    "조작시간 (s)": "Operation time (s)",
    "SAGAT1": "SAGAT1",
    "SAGAT2": "SAGAT2",
    "SAGAT3": "SAGAT3",
    "Fatigue": "Fatigue",
    "NASA-RTLX": "NASA-RTLX",
    "SART": "SART",
}
METRIC_SLUGS = {
    "조작성공률 (%)": "task_success",
    "조작시간 (s)": "operation_time",
    "SAGAT1": "sagat1",
    "SAGAT2": "sagat2",
    "SAGAT3": "sagat3",
    "Fatigue": "fatigue",
    "NASA-RTLX": "nasa_rtlx",
    "SART": "sart",
}
BONFERRONI_ALPHA = 0.05


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def parse_float(value: str | None) -> float | None:
    text = normalize(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def se(values: list[float]) -> float:
    if not values:
        return math.nan
    return sd(values) / math.sqrt(len(values))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2
        for original_index, _value in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def chi_square_sf(value: float, df: int) -> float:
    if value <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(value / 2))
    if df == 2:
        return math.exp(-value / 2)
    if df == 4:
        return math.exp(-value / 2) * (1 + value / 2)

    # Wilson-Hilferty normal approximation. This script uses df=4 for C1-C5,
    # but the fallback keeps the helper usable if the condition count changes.
    z = ((value / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def friedman_test(matrix: list[list[float]]) -> tuple[float, int, float, float]:
    n_subjects = len(matrix)
    n_conditions = len(matrix[0]) if matrix else 0
    if n_subjects == 0 or n_conditions < 2:
        return math.nan, max(0, n_conditions - 1), math.nan, math.nan

    rank_sums = [0.0] * n_conditions
    tie_sum = 0.0
    for row in matrix:
        ranks = rank_values(row)
        for index, rank in enumerate(ranks):
            rank_sums[index] += rank

        counts: dict[float, int] = defaultdict(int)
        for value in row:
            counts[value] += 1
        tie_sum += sum(count**3 - count for count in counts.values() if count > 1)

    statistic = (
        12 / (n_subjects * n_conditions * (n_conditions + 1)) * sum(rank_sum**2 for rank_sum in rank_sums)
        - 3 * n_subjects * (n_conditions + 1)
    )
    tie_correction = 1 - tie_sum / (n_subjects * (n_conditions**3 - n_conditions))
    if tie_correction > 0:
        statistic /= tie_correction

    df = n_conditions - 1
    p_value = chi_square_sf(statistic, df)
    kendalls_w = statistic / (n_subjects * df)
    return statistic, df, p_value, kendalls_w


def wilcoxon_exact(values_a: list[float], values_b: list[float]) -> tuple[int, float, float, float, float, float]:
    diffs = [a - b for a, b in zip(values_a, values_b) if a != b]
    n_nonzero = len(diffs)
    if n_nonzero == 0:
        return 0, 0.0, 1.0, 0.0, 0.0, 0.0

    ranks = rank_values([abs(diff) for diff in diffs])
    w_plus = sum(rank for diff, rank in zip(diffs, ranks) if diff > 0)
    w_minus = sum(rank for diff, rank in zip(diffs, ranks) if diff < 0)
    observed = abs(w_plus - w_minus)

    more_extreme = 0
    total = 2**n_nonzero
    for signs in product((-1, 1), repeat=n_nonzero):
        signed_rank = abs(sum(sign * rank for sign, rank in zip(signs, ranks)))
        if signed_rank >= observed - 1e-12:
            more_extreme += 1

    p_value = min(1.0, more_extreme / total)
    rank_biserial = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0
    return n_nonzero, min(w_plus, w_minus), p_value, rank_biserial, mean(diffs), median(diffs)


def cochran_q_test(matrix: list[list[float]]) -> tuple[float, int, float, float]:
    n_subjects = len(matrix)
    n_conditions = len(matrix[0]) if matrix else 0
    if n_subjects == 0 or n_conditions < 2:
        return math.nan, max(0, n_conditions - 1), math.nan, math.nan

    binary_matrix = [[1 if value > 0 else 0 for value in row] for row in matrix]
    column_totals = [sum(row[index] for row in binary_matrix) for index in range(n_conditions)]
    row_totals = [sum(row) for row in binary_matrix]
    total = sum(column_totals)
    denominator = n_conditions * total - sum(value**2 for value in row_totals)
    if denominator == 0:
        return 0.0, n_conditions - 1, 1.0, 0.0

    statistic = (n_conditions - 1) * (n_conditions * sum(value**2 for value in column_totals) - total**2) / denominator
    df = n_conditions - 1
    p_value = chi_square_sf(statistic, df)
    kendalls_w = statistic / (n_subjects * df) if n_subjects and df else math.nan
    return statistic, df, p_value, kendalls_w


def binomial_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def mcnemar_exact(values_a: list[float], values_b: list[float]) -> tuple[int, int, int, float, float, float]:
    binary_a = [1 if value > 0 else 0 for value in values_a]
    binary_b = [1 if value > 0 else 0 for value in values_b]
    b = sum(1 for a, b_value in zip(binary_a, binary_b) if a == 1 and b_value == 0)
    c = sum(1 for a, b_value in zip(binary_a, binary_b) if a == 0 and b_value == 1)
    discordant = b + c
    p_value = binomial_two_sided_p(min(b, c), discordant)
    paired_proportion_diff = mean(values_a) - mean(values_b)
    discordant_effect = (b - c) / discordant if discordant else 0.0
    return discordant, b, c, p_value, discordant_effect, paired_proportion_diff


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def parse_subject_blocks(path: Path) -> list[dict[str, str]]:
    rows = read_rows(path)
    long_rows: list[dict[str, str]] = []

    row_index = 0
    while row_index < len(rows):
        header = rows[row_index]
        if len(header) < 3 or normalize(header[0]) != "조건" or normalize(header[1]) != "도메인":
            row_index += 1
            continue
        if row_index + 1 >= len(rows):
            break

        metric_header = rows[row_index + 1]
        subject_starts = [
            index
            for index, value in enumerate(header)
            if index >= 2 and normalize(value).upper().startswith("P") and normalize(value).upper() != "P_ALL"
        ]

        for data_row in rows[row_index + 2 : row_index + 12]:
            if not any(normalize(cell) for cell in data_row):
                break
            condition = normalize(data_row[0])
            if condition:
                current_condition = condition
            domain = normalize(data_row[1] if len(data_row) > 1 else "")
            if domain not in DOMAINS:
                continue

            for start in subject_starts:
                subject = normalize(header[start]).lower()
                if not subject:
                    continue
                for offset in range(8):
                    column = start + offset
                    metric = normalize(metric_header[column] if column < len(metric_header) else "")
                    if metric not in DVS:
                        continue
                    value = parse_float(data_row[column] if column < len(data_row) else "")
                    if value is None:
                        continue
                    long_rows.append(
                        {
                            "subject": subject,
                            "domain": domain,
                            "condition": current_condition,
                            "condition_short": CONDITION_SHORT[current_condition],
                            "metric": metric,
                            "value": fmt(value),
                        }
                    )

        row_index += 1

    return long_rows


def build_score_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        lookup[(row["subject"], row["domain"], row["condition"], row["metric"])] = float(row["value"])
    return lookup


def complete_matrix(
    scores: dict[tuple[str, str, str, str], float],
    domain: str,
    metric: str,
    conditions: tuple[str, ...] = CONDITIONS,
) -> tuple[list[str], list[list[float]]]:
    subjects = sorted({subject for subject, item_domain, _condition, item_metric in scores if item_domain == domain and item_metric == metric})
    complete_subjects = [
        subject
        for subject in subjects
        if all((subject, domain, condition, metric) in scores for condition in conditions)
    ]
    matrix = [
        [scores[(subject, domain, condition, metric)] for condition in conditions]
        for subject in complete_subjects
    ]
    return complete_subjects, matrix


def significance(p_adjusted: float) -> str:
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < BONFERRONI_ALPHA:
        return "*"
    return ""


def analyze(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    scores = build_score_lookup(rows)
    friedman_rows: list[dict[str, str]] = []
    pairwise_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for domain in DOMAINS:
        for metric in DVS:
            active_conditions = tuple(
                condition
                for condition in CONDITIONS
                if any(
                    item_domain == domain and item_condition == condition and item_metric == metric
                    for _subject, item_domain, item_condition, item_metric in scores
                )
            )
            analysis_conditions = active_conditions if metric in BINARY_DVS else CONDITIONS
            subjects, matrix = complete_matrix(scores, domain, metric, analysis_conditions)
            condition_values = [[subject_row[index] for subject_row in matrix] for index in range(len(analysis_conditions))]
            condition_value_map = {
                condition: values
                for condition, values in zip(analysis_conditions, condition_values)
            }

            for condition in CONDITIONS:
                values = condition_value_map.get(condition, [])
                summary_rows.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "condition": condition,
                        "condition_short": CONDITION_SHORT[condition],
                        "n": str(len(values)),
                        "mean": fmt(mean(values) if values else math.nan),
                        "sd": fmt(sd(values) if values else math.nan),
                        "se": fmt(se(values) if values else math.nan),
                    }
                )

            if metric in BINARY_DVS:
                statistic, df, p_value, kendalls_w = cochran_q_test(matrix)
                test_name = "Cochran's Q"
                statistic_name = "Q"
            else:
                statistic, df, p_value, kendalls_w = friedman_test(matrix)
                test_name = "Friedman"
                statistic_name = "chi-square"

            friedman_rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "test": test_name,
                    "statistic_name": statistic_name,
                    "chi_square": fmt(statistic),
                    "F": "",
                    "df": str(df),
                    "p": fmt(p_value),
                    "kendalls_w": fmt(kendalls_w),
                    "n_complete_subjects": str(len(subjects)),
                    "complete_subjects": ",".join(subjects),
                    "n_conditions": str(len(analysis_conditions)),
                    "conditions_included": ",".join(CONDITION_SHORT[condition] for condition in analysis_conditions),
                }
            )

            raw_pair_rows: list[dict[str, str]] = []
            pair_count = math.comb(len(analysis_conditions), 2) if len(analysis_conditions) >= 2 else 0
            for index_a, index_b in combinations(range(len(analysis_conditions)), 2):
                condition_a = analysis_conditions[index_a]
                condition_b = analysis_conditions[index_b]
                values_a = condition_values[index_a]
                values_b = condition_values[index_b]
                if metric in BINARY_DVS:
                    n_nonzero, b_count, c_count, p_raw, rank_biserial, mean_diff = mcnemar_exact(values_a, values_b)
                    w_stat = math.nan
                    median_diff = median([a - b for a, b in zip(values_a, values_b)]) if values_a else math.nan
                    test = "McNemar exact"
                    discordant = f"{b_count}/{c_count}"
                else:
                    n_nonzero, w_stat, p_raw, rank_biserial, mean_diff, median_diff = wilcoxon_exact(values_a, values_b)
                    test = "Wilcoxon signed-rank exact"
                    discordant = ""
                raw_pair_rows.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "test": test,
                        "comparison": f"{CONDITION_SHORT[condition_a]} vs {CONDITION_SHORT[condition_b]}",
                        "condition_a": CONDITION_SHORT[condition_a],
                        "condition_b": CONDITION_SHORT[condition_b],
                        "n_complete_subjects": str(len(subjects)),
                        "n_nonzero_pairs": str(n_nonzero),
                        "W": fmt(w_stat),
                        "p_raw": fmt(p_raw),
                        "p_bonferroni": fmt(min(1.0, p_raw * pair_count)) if pair_count else "",
                        "significant_p_lt_05": "",
                        "stars": "",
                        "rank_biserial": fmt(rank_biserial),
                        "mean_difference_a_minus_b": fmt(mean_diff),
                        "median_difference_a_minus_b": fmt(median_diff),
                        "n_conditions": str(len(analysis_conditions)),
                        "conditions_included": ",".join(CONDITION_SHORT[condition] for condition in analysis_conditions),
                        "mcnemar_b_over_c": discordant,
                    }
                )

            for row in raw_pair_rows:
                p_adjusted = float(row["p_bonferroni"])
                row["stars"] = significance(p_adjusted)
                row["significant_p_lt_05"] = "yes" if row["stars"] else "no"
                pairwise_rows.append(row)

    return friedman_rows, pairwise_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    domain: str,
    metric: str,
    summary_rows: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
    output_dir: Path,
) -> Path:
    rows = [row for row in summary_rows if row["domain"] == domain and row["metric"] == metric]
    means = [float(next(row["mean"] for row in rows if row["condition"] == condition)) for condition in CONDITIONS]
    ses = [float(next(row["se"] for row in rows if row["condition"] == condition)) for condition in CONDITIONS]
    labels = [CONDITION_SHORT[condition] for condition in CONDITIONS]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    x_values = list(range(len(CONDITIONS)))
    colors = ["#4C78A8" if domain == "Waste" else "#D66B4D"] * len(CONDITIONS)
    ax.bar(x_values, means, yerr=ses, capsize=5, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x_values, labels)
    ax.set_title(f"{domain} - {METRIC_LABELS[metric]}")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean +/- SE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    if metric.startswith("SAGAT"):
        ax.set_ylim(0, 1.25)
    elif metric in BINARY_DVS:
        ax.set_ylim(0, 125)
    else:
        upper = max(mean + error for mean, error in zip(means, ses))
        ax.set_ylim(0, max(upper * 1.35, 1.0))

    significant_pairs = [
        row
        for row in pairwise_rows
        if row["domain"] == domain and row["metric"] == metric and row["stars"]
    ]
    y_min, y_max = ax.get_ylim()
    step = (y_max - y_min) * 0.07
    current_y = max(mean + error for mean, error in zip(means, ses)) + step
    for row in significant_pairs:
        idx_a = labels.index(row["condition_a"])
        idx_b = labels.index(row["condition_b"])
        ax.plot([idx_a, idx_a, idx_b, idx_b], [current_y, current_y + step * 0.25, current_y + step * 0.25, current_y], color="#222222", linewidth=1)
        ax.text((idx_a + idx_b) / 2, current_y + step * 0.3, row["stars"], ha="center", va="bottom", fontsize=11)
        current_y += step
    if significant_pairs:
        ax.set_ylim(y_min, max(y_max, current_y + step))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{domain.lower()}_{METRIC_SLUGS[metric]}.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def write_text_report(
    path: Path,
    friedman_rows: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
    figures: list[Path],
) -> None:
    lines = [
        "Domain-wise C1-C5 Friedman test report",
        "",
        "Note: Continuous/ordinal variables use non-parametric Friedman tests, so the omnibus statistic is chi-square rather than RM-ANOVA F.",
        "Binary task success uses Cochran's Q and exact McNemar post-hoc tests.",
        "Post-hoc tests use Bonferroni correction across the available condition pairs per domain/metric.",
        "",
        "Omnibus results",
    ]
    for row in friedman_rows:
        lines.append(
            f"- {row['domain']} / {row['metric']}: {row['test']} {row['statistic_name']}({row['df']}) = {row['chi_square']}, "
            f"p = {row['p']}, effect = {row['kendalls_w']}, n = {row['n_complete_subjects']}, "
            f"conditions = {row.get('conditions_included', '')}"
        )

    lines.extend(["", "Significant post-hoc pairs (Bonferroni p < .05)"])
    significant = [row for row in pairwise_rows if row["stars"]]
    if significant:
        for row in significant:
            lines.append(
                f"- {row['domain']} / {row['metric']} / {row['comparison']}: "
                f"p_bonferroni = {row['p_bonferroni']} {row['stars']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "Figures"])
    lines.extend(f"- {path}" for path in figures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run domain-wise Friedman tests and Bonferroni post-hoc analyses for C1-C5.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_subject_blocks(args.input)
    friedman_rows, pairwise_rows, summary_rows = analyze(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "long_data.csv", rows, ["subject", "domain", "condition", "condition_short", "metric", "value"])
    write_csv(
        args.output_dir / "friedman_results.csv",
        friedman_rows,
        [
            "domain",
            "metric",
            "test",
            "statistic_name",
            "chi_square",
            "F",
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
        args.output_dir / "posthoc_pairwise_bonferroni.csv",
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
        args.output_dir / "condition_mean_se.csv",
        summary_rows,
        ["domain", "metric", "condition", "condition_short", "n", "mean", "sd", "se"],
    )

    figure_dir = args.output_dir / "figures"
    figures = [
        plot_metric(domain, metric, summary_rows, pairwise_rows, figure_dir)
        for domain in DOMAINS
        for metric in DVS
    ]
    write_text_report(args.output_dir / "summary_report.txt", friedman_rows, pairwise_rows, figures)

    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"long_rows: {len(rows)}")
    print(f"friedman_tests: {len(friedman_rows)}")
    print(f"pairwise_tests: {len(pairwise_rows)}")
    print(f"figures: {len(figures)}")


if __name__ == "__main__":
    main()
