"""Stage 3: prepare final CSV tables for user-study figures.

Usage:
    python3 experiments/user_eval/read_csv_user_study.py

Inputs:
    experiments/user_eval/data/nonparametric_user_study/condition_summary.csv
    experiments/user_eval/data/nonparametric_user_study/posthoc_pairwise_bonferroni.csv

Outputs:
    experiments/user_eval/data/figure_user_study/user_study_metric_summary.csv
    experiments/user_eval/data/figure_user_study/user_study_significance_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATS_DIR = SCRIPT_DIR / "data" / "nonparametric_user_study"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "figure_user_study"

CONDITION_ORDER = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5}
DOMAIN_ORDER = {"Waste": 1, "Tomato": 2}
METRIC_ORDER = {
    "조작성공률 (%)": 1,
    "조작시간 (s)": 2,
    "SAGAT1": 3,
    "SAGAT2": 4,
    "SAGAT3": 5,
    "Fatigue": 6,
    "NASA-RTLX": 7,
    "SART": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final plotting CSVs for user-study analysis.")
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_slug(metric: str) -> str:
    return (
        metric.lower()
        .replace(" (%)", "")
        .replace(" (s)", "")
        .replace("조작성공률", "task_success")
        .replace("조작시간", "operation_time")
        .replace("-", "_")
    )


def main() -> None:
    args = parse_args()
    summary_rows = read_rows(args.stats_dir / "condition_summary.csv")
    pairwise_rows = read_rows(args.stats_dir / "posthoc_pairwise_bonferroni.csv")

    final_summary: list[dict[str, str]] = []
    for row in summary_rows:
        metric = row["metric"]
        domain = row["domain"]
        condition = row["condition_short"]
        final_summary.append(
            {
                "figure_group": f"{domain.lower()}_{metric_slug(metric)}",
                "domain": domain,
                "metric": metric,
                "condition": condition,
                "condition_label": row["condition"],
                "n": row["n"],
                "mean": row["mean"],
                "sd": row["sd"],
                "se": row["se"],
                "median": row["median"],
                "min": row["min"],
                "max": row["max"],
            }
        )

    final_summary.sort(
        key=lambda row: (
            DOMAIN_ORDER.get(row["domain"], 99),
            METRIC_ORDER.get(row["metric"], 99),
            CONDITION_ORDER.get(row["condition"], 99),
        )
    )

    final_pairs = [
        {
            "figure_group": f"{row['domain'].lower()}_{metric_slug(row['metric'])}",
            "domain": row["domain"],
            "metric": row["metric"],
            "condition_a": row["condition_a"],
            "condition_b": row["condition_b"],
            "comparison": row["comparison"],
            "p_raw": row["p_raw"],
            "p_bonferroni": row["p_bonferroni"],
            "significant_p_lt_05": row["significant_p_lt_05"],
            "stars": row["stars"],
            "test": row["test"],
        }
        for row in pairwise_rows
    ]
    final_pairs.sort(
        key=lambda row: (
            DOMAIN_ORDER.get(row["domain"], 99),
            METRIC_ORDER.get(row["metric"], 99),
            CONDITION_ORDER.get(row["condition_a"], 99),
            CONDITION_ORDER.get(row["condition_b"], 99),
        )
    )

    write_rows(
        args.output_dir / "user_study_metric_summary.csv",
        final_summary,
        ["figure_group", "domain", "metric", "condition", "condition_label", "n", "mean", "sd", "se", "median", "min", "max"],
    )
    write_rows(
        args.output_dir / "user_study_significance_pairs.csv",
        final_pairs,
        [
            "figure_group",
            "domain",
            "metric",
            "condition_a",
            "condition_b",
            "comparison",
            "p_raw",
            "p_bonferroni",
            "significant_p_lt_05",
            "stars",
            "test",
        ],
    )

    print(f"output_dir: {args.output_dir}")
    print(f"summary_rows: {len(final_summary)}")
    print(f"pairwise_rows: {len(final_pairs)}")


if __name__ == "__main__":
    main()
