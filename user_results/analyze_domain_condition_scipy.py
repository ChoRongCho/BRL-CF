from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

try:
    from scipy.stats import friedmanchisquare, wilcoxon
except ImportError as exc:
    raise SystemExit(
        "SciPy is required for this script. Install it with: pip install scipy"
    ) from exc

from analyze_domain_condition_friedman import (
    BASE_DIR,
    CONDITIONS,
    CONDITION_SHORT,
    DEFAULT_INPUT,
    DOMAINS,
    DVS,
    build_score_lookup,
    complete_matrix,
    fmt,
    mean,
    median,
    parse_subject_blocks,
    plot_metric,
    sd,
    se,
    significance,
    write_csv,
    write_text_report,
)


DEFAULT_OUTPUT_DIR = BASE_DIR / "domain_condition_scipy"


def scipy_wilcoxon_exact(values_a: list[float], values_b: list[float]) -> tuple[int, float, float, float, float, float]:
    diffs = [a - b for a, b in zip(values_a, values_b)]
    nonzero_diffs = [diff for diff in diffs if diff != 0]
    n_nonzero = len(nonzero_diffs)
    if n_nonzero == 0:
        return 0, 0.0, 1.0, 0.0, 0.0, 0.0

    try:
        result = wilcoxon(
            values_a,
            values_b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
    except TypeError:
        # Older SciPy versions used "mode" before the "method" parameter.
        result = wilcoxon(
            values_a,
            values_b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="exact",
        )

    w_stat = float(result.statistic)
    p_value = float(result.pvalue)

    rank_biserial = rank_biserial_effect(nonzero_diffs)
    return n_nonzero, w_stat, p_value, rank_biserial, mean(nonzero_diffs), median(nonzero_diffs)


def rank_biserial_effect(diffs: list[float]) -> float:
    abs_values = [abs(diff) for diff in diffs]
    ranks = average_ranks(abs_values)
    w_plus = sum(rank for diff, rank in zip(diffs, ranks) if diff > 0)
    w_minus = sum(rank for diff, rank in zip(diffs, ranks) if diff < 0)
    return (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) else 0.0


def average_ranks(values: list[float]) -> list[float]:
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


def analyze_with_scipy(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    scores = build_score_lookup(rows)
    friedman_rows: list[dict[str, str]] = []
    pairwise_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for domain in DOMAINS:
        for metric in DVS:
            subjects, matrix = complete_matrix(scores, domain, metric)
            condition_values = [[subject_row[index] for subject_row in matrix] for index in range(len(CONDITIONS))]

            for condition, values in zip(CONDITIONS, condition_values):
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

            if len(subjects) == 0:
                statistic = math.nan
                p_value = math.nan
            else:
                result = friedmanchisquare(*condition_values)
                statistic = float(result.statistic)
                p_value = float(result.pvalue)

            df = len(CONDITIONS) - 1
            kendalls_w = statistic / (len(subjects) * df) if subjects and df else math.nan
            friedman_rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "test": "Friedman",
                    "statistic_name": "chi-square",
                    "chi_square": fmt(statistic),
                    "F": "",
                    "df": str(df),
                    "p": fmt(p_value),
                    "kendalls_w": fmt(kendalls_w),
                    "n_complete_subjects": str(len(subjects)),
                    "complete_subjects": ",".join(subjects),
                }
            )

            for index_a, index_b in combinations(range(len(CONDITIONS)), 2):
                condition_a = CONDITIONS[index_a]
                condition_b = CONDITIONS[index_b]
                values_a = condition_values[index_a]
                values_b = condition_values[index_b]
                n_nonzero, w_stat, p_raw, rank_biserial, mean_diff, median_diff = scipy_wilcoxon_exact(values_a, values_b)
                p_bonferroni = min(1.0, p_raw * 10)
                stars = significance(p_bonferroni)
                pairwise_rows.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "test": "Wilcoxon signed-rank exact (SciPy)",
                        "comparison": f"{CONDITION_SHORT[condition_a]} vs {CONDITION_SHORT[condition_b]}",
                        "condition_a": CONDITION_SHORT[condition_a],
                        "condition_b": CONDITION_SHORT[condition_b],
                        "n_complete_subjects": str(len(subjects)),
                        "n_nonzero_pairs": str(n_nonzero),
                        "W": fmt(w_stat),
                        "p_raw": fmt(p_raw),
                        "p_bonferroni": fmt(p_bonferroni),
                        "significant_p_lt_05": "yes" if stars else "no",
                        "stars": stars,
                        "rank_biserial": fmt(rank_biserial),
                        "mean_difference_a_minus_b": fmt(mean_diff),
                        "median_difference_a_minus_b": fmt(median_diff),
                    }
                )

    return friedman_rows, pairwise_rows, summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SciPy-based Friedman and Wilcoxon-Bonferroni analyses for C1-C5.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = parse_subject_blocks(args.input)
    friedman_rows, pairwise_rows, summary_rows = analyze_with_scipy(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "long_data.csv", rows, ["subject", "domain", "condition", "condition_short", "metric", "value"])
    write_csv(
        args.output_dir / "friedman_results.csv",
        friedman_rows,
        ["domain", "metric", "test", "statistic_name", "chi_square", "F", "df", "p", "kendalls_w", "n_complete_subjects", "complete_subjects"],
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
