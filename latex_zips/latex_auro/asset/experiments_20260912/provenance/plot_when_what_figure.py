"""Summarize and plot the paired 2x2 When/What ablation.

Run after ``analysis_experiment.py``.  Outputs a long-form summary, a compact
paper table, paired success contrasts, and PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "raw_runs" / "domain_compare" / "raw_runs.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figure" / "when_what"
POLICIES = ("random", "ours_when_only", "ours_what_only", "ours")
POLICY_LABELS = {
    "random": "Random",
    "ours_when_only": "Ours-when-only",
    "ours_what_only": "Ours-what-only",
    "ours": "Ours",
}
DOMAINS = ("wastesorting", "tomato")
DOMAIN_LABELS = {"wastesorting": "Waste", "tomato": "Tomato"}
DOMAIN_COLORS = {"wastesorting": "#f2d675", "tomato": "#b8a1d9"}
METRICS = {
    "success_rate": "Success Rate",
    "average_step": "Average Step",
    "average_step_success_only": "Average Step (Success Only)",
    "average_question": "Average Query Number",
    "average_question_success_only": "Average Query Number (Success Only)",
    "query_probability_per_step": "Query Probability per Step",
    "average_reward": "Average Reward",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the paired When/What ablation.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--metric", default="")
    return parser.parse_args()


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_success(row):
    return str(row.get("success", "")).lower() == "true"


def metric_values(rows, metric):
    if metric == "success_rate":
        return [float(is_success(row)) for row in rows]
    if metric == "average_step":
        return [as_float(row["planning_length"]) for row in rows]
    if metric == "average_step_success_only":
        return [as_float(row["planning_length"]) for row in rows if is_success(row)]
    if metric == "average_question":
        return [as_float(row["question_count"]) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row["question_count"]) for row in rows if is_success(row)]
    if metric == "average_reward":
        return [as_float(row["reward"]) for row in rows]
    if metric == "query_probability_per_step":
        questions = sum(as_float(row["query_step_count"]) for row in rows)
        steps = sum(as_float(row["planning_length"]) for row in rows)
        return [questions / steps] if steps else []
    raise ValueError("Unknown metric: " + metric)


def fmt(value):
    return "" if math.isnan(value) else f"{value:.6f}"


def load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("experiment") == "when_what"]
    if not rows:
        raise ValueError("No when_what rows found in " + str(path))
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("domain") in DOMAINS and row.get("policy") in POLICIES:
            grouped[(row["domain"], row["policy"])].append(row)
    expected = {(domain, policy) for domain in DOMAINS for policy in POLICIES}
    if set(grouped) != expected:
        raise ValueError("Missing When/What cells: " + str(sorted(expected - set(grouped))))
    sizes = {len(grouped[key]) for key in expected}
    if len(sizes) != 1:
        raise ValueError("Inconsistent cell sizes: " + str(sorted(sizes)))
    return grouped


def build_summary(grouped):
    summary = []
    for metric, metric_label in METRICS.items():
        for domain in DOMAINS:
            for policy in POLICIES:
                group = grouped[(domain, policy)]
                values = [value for value in metric_values(group, metric) if not math.isnan(value)]
                value_mean = mean(values) if values else math.nan
                value_std = stdev(values) if len(values) > 1 else math.nan
                summary.append({
                    "metric": metric,
                    "metric_label": metric_label,
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "n": str(len(group)),
                    "valid_n": str(len(values)),
                    "mean": fmt(value_mean),
                    "std": fmt(value_std),
                    "min": fmt(min(values) if values else math.nan),
                    "max": fmt(max(values) if values else math.nan),
                })
    return summary


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def summary_lookup(summary):
    return {(row["metric"], row["domain"], row["policy"]): as_float(row["mean"]) for row in summary}


def build_paper_rows(summary):
    lookup = summary_lookup(summary)
    rows = []
    for policy in POLICIES:
        row = {"condition": POLICY_LABELS[policy]}
        for domain, prefix in (("wastesorting", "waste"), ("tomato", "tomato")):
            row[prefix + "_success_rate_pct"] = f"{100 * lookup[('success_rate', domain, policy)]:.1f}"
            row[prefix + "_avg_queries"] = f"{lookup[('average_question', domain, policy)]:.1f}"
            row[prefix + "_query_rate"] = f"{lookup[('query_probability_per_step', domain, policy)]:.2f}"
            row[prefix + "_avg_steps_success"] = f"{lookup[('average_step_success_only', domain, policy)]:.1f}"
        rows.append(row)
    return rows


def write_paper_table(output_dir, summary):
    rows = build_paper_rows(summary)
    fields = ["condition"]
    for prefix in ("waste", "tomato"):
        fields.extend([
            prefix + "_success_rate_pct", prefix + "_avg_queries",
            prefix + "_query_rate", prefix + "_avg_steps_success",
        ])
    write_csv(output_dir / "when_what_paper_table.csv", rows, fields)
    episodes = int(next(row["n"] for row in summary if row["metric"] == "success_rate"))
    lines = [
        "% Generated by experiments/system_eval/plot_when_what_figure.py",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{When/What query-policy ablation with Oracle answers. Success rates are percentages, query counts are averages per episode, query rate is the fraction of execution steps with at least one question, and average steps use successful runs only. Each condition-domain cell contains $%d$ episodes.}" % episodes,
        "\\label{tab:when_what_ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccc}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Condition}} & \\multicolumn{4}{c}{\\textbf{Waste Sorting}} & \\multicolumn{4}{c}{\\textbf{Tomato Harvesting}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "& \\textbf{Success (\\%)} & \\textbf{Avg. Queries} & \\textbf{Query Rate} & \\textbf{Avg. Steps} & \\textbf{Success (\\%)} & \\textbf{Avg. Queries} & \\textbf{Query Rate} & \\textbf{Avg. Steps} \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row[field] for field in fields) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table*}", ""])
    path = output_dir / "when_what_paper_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


def exact_mcnemar_p(improved, worsened):
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(improved, worsened) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def write_paired_contrasts(output_dir, grouped):
    comparisons = (
        ("random", "ours_when_only"), ("random", "ours_what_only"),
        ("random", "ours"), ("ours_when_only", "ours"),
        ("ours_what_only", "ours"),
    )
    output = []
    for domain in DOMAINS:
        paired = defaultdict(dict)
        for policy in POLICIES:
            for row in grouped[(domain, policy)]:
                paired[(row["scene"], row["seed"])][policy] = row
        if not paired or any(len(unit) != len(POLICIES) for unit in paired.values()):
            raise ValueError("Incomplete paired seeds for " + domain)
        for baseline, comparison in comparisons:
            outcomes = [(is_success(unit[baseline]), is_success(unit[comparison])) for unit in paired.values()]
            improved = sum(not before and after for before, after in outcomes)
            worsened = sum(before and not after for before, after in outcomes)
            difference = 100.0 * sum(float(after) - float(before) for before, after in outcomes) / len(outcomes)
            output.append({
                "domain": domain, "baseline": POLICY_LABELS[baseline],
                "comparison": POLICY_LABELS[comparison], "paired_n": str(len(outcomes)),
                "success_difference_pp": f"{difference:.1f}",
                "improved_pairs": str(improved), "worsened_pairs": str(worsened),
                "mcnemar_exact_p": f"{exact_mcnemar_p(improved, worsened):.8g}",
            })
    fields = ["domain", "baseline", "comparison", "paired_n", "success_difference_pp", "improved_pairs", "worsened_pairs", "mcnemar_exact_p"]
    write_csv(output_dir / "when_what_paired_success_contrasts.csv", output, fields)


def plot_metric(summary, metric, output_dir):
    lookup = summary_lookup(summary)
    positions = list(range(len(POLICIES)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    for index, domain in enumerate(DOMAINS):
        values = [lookup[(metric, domain, policy)] for policy in POLICIES]
        if metric.endswith("_rate"):
            values = [100.0 * value for value in values]
        ax.bar(
            [position + (index - 0.5) * width for position in positions], values,
            width=width, color=DOMAIN_COLORS[domain], edgecolor="#1e293b",
            linewidth=0.8, label=DOMAIN_LABELS[domain],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([POLICY_LABELS[policy] for policy in POLICIES], fontsize=13)
    ylabel = METRICS[metric]
    if metric.endswith("_rate"):
        ylabel += " (%)"
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", ncol=2, fontsize=15, frameon=True, fancybox=False, edgecolor="#1e293b")
    for suffix in (".png", ".pdf"):
        path = output_dir / (metric + suffix)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


def main():
    args = parse_args()
    metrics = [args.metric] if args.metric else list(METRICS)
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise ValueError("Unknown metric(s): " + ", ".join(unknown))
    grouped = group_rows(load_rows(Path(args.input)))
    output_dir = Path(args.output_root) / datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = build_summary(grouped)
    fields = ["metric", "metric_label", "domain", "domain_label", "policy", "policy_label", "n", "valid_n", "mean", "std", "min", "max"]
    write_csv(output_dir / "when_what_summary.csv", summary, fields)
    write_paper_table(output_dir, summary)
    write_paired_contrasts(output_dir, grouped)
    for metric in metrics:
        plot_metric(summary, metric, output_dir)


if __name__ == "__main__":
    main()
