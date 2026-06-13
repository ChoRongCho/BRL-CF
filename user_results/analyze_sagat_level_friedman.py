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
DEFAULT_OUTPUT_DIR = BASE_DIR / "sagat_level_friedman"

CONDITIONS = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_SHORT = {
    "C1: All": "C1",
    "C2: No": "C2",
    "C3: Ours1": "C3",
    "C4: Ours2": "C4",
    "C5: KnowNo": "C5",
}
DOMAINS = ("Waste", "Tomato")
LEVELS = ("Lv1", "Lv2", "Lv3")
LEVEL_TO_METRIC = {
    "Lv1": "SAGAT1",
    "Lv2": "SAGAT2",
    "Lv3": "SAGAT3",
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
    z = ((value / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def friedman_test(matrix: list[list[float]]) -> tuple[float, int, float, float]:
    n_subjects = len(matrix)
    n_levels = len(matrix[0]) if matrix else 0
    if n_subjects == 0 or n_levels < 2:
        return math.nan, max(0, n_levels - 1), math.nan, math.nan

    rank_sums = [0.0] * n_levels
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
        12 / (n_subjects * n_levels * (n_levels + 1)) * sum(rank_sum**2 for rank_sum in rank_sums)
        - 3 * n_subjects * (n_levels + 1)
    )
    tie_correction = 1 - tie_sum / (n_subjects * (n_levels**3 - n_levels))
    if tie_correction > 0:
        statistic /= tie_correction

    df = n_levels - 1
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


def significance(p_adjusted: float) -> str:
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < BONFERRONI_ALPHA:
        return "*"
    return ""


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

        current_condition = ""
        for data_row in rows[row_index + 2 : row_index + 12]:
            if not any(normalize(cell) for cell in data_row):
                break
            condition = normalize(data_row[0])
            if condition:
                current_condition = condition
            domain = normalize(data_row[1] if len(data_row) > 1 else "")
            if domain not in DOMAINS or current_condition not in CONDITIONS:
                continue

            for start in subject_starts:
                subject = normalize(header[start]).lower()
                if not subject:
                    continue
                for offset in range(8):
                    column = start + offset
                    metric = normalize(metric_header[column] if column < len(metric_header) else "")
                    level = next((level for level, item_metric in LEVEL_TO_METRIC.items() if item_metric == metric), "")
                    if not level:
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
                            "level": level,
                            "metric": metric,
                            "value": fmt(value),
                        }
                    )

        row_index += 1

    return long_rows


def build_score_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        lookup[(row["subject"], row["domain"], row["condition"], row["level"])] = float(row["value"])
    return lookup


def complete_matrix(
    scores: dict[tuple[str, str, str, str], float],
    domain: str,
    condition: str,
) -> tuple[list[str], list[list[float]]]:
    subjects = sorted({subject for subject, item_domain, item_condition, _level in scores if item_domain == domain and item_condition == condition})
    complete_subjects = [
        subject
        for subject in subjects
        if all((subject, domain, condition, level) in scores for level in LEVELS)
    ]
    matrix = [
        [scores[(subject, domain, condition, level)] for level in LEVELS]
        for subject in complete_subjects
    ]
    return complete_subjects, matrix


def analyze(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    scores = build_score_lookup(rows)
    friedman_rows: list[dict[str, str]] = []
    pairwise_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for domain in DOMAINS:
        for condition in CONDITIONS:
            subjects, matrix = complete_matrix(scores, domain, condition)
            level_values = [[subject_row[index] for subject_row in matrix] for index in range(len(LEVELS))]

            for level, values in zip(LEVELS, level_values):
                summary_rows.append(
                    {
                        "domain": domain,
                        "condition": condition,
                        "condition_short": CONDITION_SHORT[condition],
                        "level": level,
                        "metric": LEVEL_TO_METRIC[level],
                        "n": str(len(values)),
                        "mean": fmt(mean(values) if values else math.nan),
                        "sd": fmt(sd(values) if values else math.nan),
                        "se": fmt(se(values) if values else math.nan),
                    }
                )

            statistic, df, p_value, kendalls_w = friedman_test(matrix)
            friedman_rows.append(
                {
                    "domain": domain,
                    "condition": condition,
                    "condition_short": CONDITION_SHORT[condition],
                    "test": "Friedman",
                    "statistic_name": "chi-square",
                    "chi_square": fmt(statistic),
                    "df": str(df),
                    "p": fmt(p_value),
                    "kendalls_w": fmt(kendalls_w),
                    "n_complete_subjects": str(len(subjects)),
                    "complete_subjects": ",".join(subjects),
                    "levels_included": ",".join(LEVELS),
                }
            )

            pair_count = math.comb(len(LEVELS), 2)
            for index_a, index_b in combinations(range(len(LEVELS)), 2):
                level_a = LEVELS[index_a]
                level_b = LEVELS[index_b]
                values_a = level_values[index_a]
                values_b = level_values[index_b]
                n_nonzero, w_stat, p_raw, rank_biserial, mean_diff, median_diff = wilcoxon_exact(values_a, values_b)
                p_adjusted = min(1.0, p_raw * pair_count)
                pairwise_rows.append(
                    {
                        "domain": domain,
                        "condition": condition,
                        "condition_short": CONDITION_SHORT[condition],
                        "test": "Wilcoxon signed-rank exact",
                        "comparison": f"{level_a} vs {level_b}",
                        "level_a": level_a,
                        "level_b": level_b,
                        "metric_a": LEVEL_TO_METRIC[level_a],
                        "metric_b": LEVEL_TO_METRIC[level_b],
                        "n_complete_subjects": str(len(subjects)),
                        "n_nonzero_pairs": str(n_nonzero),
                        "W": fmt(w_stat),
                        "p_raw": fmt(p_raw),
                        "p_bonferroni": fmt(p_adjusted),
                        "significant_p_lt_05": "yes" if significance(p_adjusted) else "no",
                        "stars": significance(p_adjusted),
                        "rank_biserial": fmt(rank_biserial),
                        "mean_difference_a_minus_b": fmt(mean_diff),
                        "median_difference_a_minus_b": fmt(median_diff),
                    }
                )

    return friedman_rows, pairwise_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_condition(
    domain: str,
    condition: str,
    summary_rows: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
    output_dir: Path,
) -> Path:
    rows = [row for row in summary_rows if row["domain"] == domain and row["condition"] == condition]
    means = [float(next(row["mean"] for row in rows if row["level"] == level)) for level in LEVELS]
    ses = [float(next(row["se"] for row in rows if row["level"] == level)) for level in LEVELS]

    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=180)
    x_values = list(range(len(LEVELS)))
    color = "#4C78A8" if domain == "Waste" else "#D66B4D"
    ax.bar(x_values, means, yerr=ses, capsize=5, color=[color] * len(LEVELS), edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(x_values, [f"{level}\n{LEVEL_TO_METRIC[level]}" for level in LEVELS])
    ax.set_title(f"{domain} - {CONDITION_SHORT[condition]} SAGAT levels")
    ax.set_xlabel("SAGAT level")
    ax.set_ylabel("Mean correctness +/- SE")
    ax.set_ylim(0, 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    significant_pairs = [
        row
        for row in pairwise_rows
        if row["domain"] == domain and row["condition"] == condition and row["stars"]
    ]
    step = 0.07
    current_y = max(mean_value + error for mean_value, error in zip(means, ses)) + step
    for row in significant_pairs:
        idx_a = LEVELS.index(row["level_a"])
        idx_b = LEVELS.index(row["level_b"])
        ax.plot([idx_a, idx_a, idx_b, idx_b], [current_y, current_y + step * 0.25, current_y + step * 0.25, current_y], color="#222222", linewidth=1)
        ax.text((idx_a + idx_b) / 2, current_y + step * 0.3, row["stars"], ha="center", va="bottom", fontsize=11)
        current_y += step
    if significant_pairs:
        ax.set_ylim(0, max(1.25, current_y + step))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{domain.lower()}_{CONDITION_SHORT[condition].lower()}_sagat_levels.png"
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
        "SAGAT level-wise Friedman test report",
        "",
        "Within each domain-condition cell, Lv1(SAGAT1), Lv2(SAGAT2), and Lv3(SAGAT3) were compared.",
        "Post-hoc tests use exact Wilcoxon signed-rank tests with Bonferroni correction across 3 level pairs.",
        "",
        "Friedman results",
    ]
    for row in friedman_rows:
        lines.append(
            f"- {row['domain']} / {row['condition_short']}: chi-square({row['df']}) = {row['chi_square']}, "
            f"p = {row['p']}, Kendall's W = {row['kendalls_w']}, n = {row['n_complete_subjects']}"
        )

    lines.extend(["", "Significant post-hoc pairs (Bonferroni p < .05)"])
    significant = [row for row in pairwise_rows if row["stars"]]
    if significant:
        for row in significant:
            lines.append(
                f"- {row['domain']} / {row['condition_short']} / {row['comparison']}: "
                f"p_bonferroni = {row['p_bonferroni']} {row['stars']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "Figures"])
    lines.extend(f"- {path}" for path in figures)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAGAT level-wise Friedman tests within each domain-condition cell.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_subject_blocks(args.input)
    friedman_rows, pairwise_rows, summary_rows = analyze(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "long_data.csv", rows, ["subject", "domain", "condition", "condition_short", "level", "metric", "value"])
    write_csv(
        args.output_dir / "friedman_results.csv",
        friedman_rows,
        ["domain", "condition", "condition_short", "test", "statistic_name", "chi_square", "df", "p", "kendalls_w", "n_complete_subjects", "complete_subjects", "levels_included"],
    )
    write_csv(
        args.output_dir / "posthoc_pairwise_bonferroni.csv",
        pairwise_rows,
        [
            "domain",
            "condition",
            "condition_short",
            "test",
            "comparison",
            "level_a",
            "level_b",
            "metric_a",
            "metric_b",
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
        ],
    )
    write_csv(
        args.output_dir / "level_mean_se.csv",
        summary_rows,
        ["domain", "condition", "condition_short", "level", "metric", "n", "mean", "sd", "se"],
    )

    figure_dir = args.output_dir / "figures"
    figures = [
        plot_condition(domain, condition, summary_rows, pairwise_rows, figure_dir)
        for domain in DOMAINS
        for condition in CONDITIONS
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
