"""Summarize and plot the paired query-strategy baseline comparison.

The comparison uses the same environment seeds for Ours, KnowNo, and the
Query-Action POMCP baseline.  Run ``analysis_experiment.py`` first to refresh
the normalized run-level CSV.
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
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "figure" / "query_baselines"

METHODS = ("knowno", "query_action_pomcp", "ours")
METHOD_LABELS = {
    "knowno": "KnowNo",
    "query_action_pomcp": "Query-Action\nPOMCP",
    "ours": "Ours",
}
DOMAINS = ("wastesorting", "tomato")
DOMAIN_LABELS = {"wastesorting": "Waste", "tomato": "Tomato"}
DOMAIN_COLORS = {"wastesorting": "#f2d675", "tomato": "#b8a1d9"}
METRICS = {
    "success_rate": "Success Rate (%)",
    "average_question": "Average Query Number",
    "average_question_success_only": "Average Query Number (Success Only)",
    "average_step_success_only": "Average Step (Success Only)",
    "wall_clock_time": "Episode Wall-Clock Time (s)",
    "wall_clock_time_success_only": "Episode Wall-Clock Time (Success Only, s)",
}
MAIN_METRICS = (
    "success_rate",
    "average_question",
    "average_step_success_only",
    "wall_clock_time",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the paired query-baseline comparison.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_success(row):
    return str(row.get("success", "")).strip().lower() == "true"


def method_for_row(row):
    if row.get("experiment") == "knowno" and row.get("policy") == "knowno_gpt4":
        return "knowno"
    if row.get("experiment") == "query_baseline" and row.get("policy") == "query_action_pomcp":
        return "query_action_pomcp"
    if row.get("experiment") == "when_what" and row.get("policy") == "ours":
        return "ours"
    return ""


def load_grouped(path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            method = method_for_row(row)
            domain = row.get("domain", "")
            if method in METHODS and domain in DOMAINS:
                grouped[(domain, method)].append(row)

    expected = {(domain, method) for domain in DOMAINS for method in METHODS}
    missing = expected - set(grouped)
    if missing:
        raise ValueError("Missing baseline cells: " + str(sorted(missing)))
    sizes = {key: len(grouped[key]) for key in expected}
    if any(size != 200 for size in sizes.values()):
        raise ValueError("Expected 200 episodes per domain/method, found: " + str(sizes))
    return grouped


def metric_values(rows, metric):
    if metric == "success_rate":
        return [float(is_success(row)) for row in rows]
    if metric == "average_question":
        return [as_float(row.get("question_count")) for row in rows]
    if metric == "average_question_success_only":
        return [as_float(row.get("question_count")) for row in rows if is_success(row)]
    if metric == "average_step_success_only":
        return [as_float(row.get("planning_length")) for row in rows if is_success(row)]
    if metric == "wall_clock_time":
        return [as_float(row.get("elapsed_seconds")) for row in rows]
    if metric == "wall_clock_time_success_only":
        return [as_float(row.get("elapsed_seconds")) for row in rows if is_success(row)]
    raise ValueError("Unknown metric: " + metric)


def valid(values):
    return [value for value in values if not math.isnan(value)]


def build_summary(grouped):
    rows = []
    for domain in DOMAINS:
        for method in METHODS:
            episodes = grouped[(domain, method)]
            for metric, label in METRICS.items():
                values = valid(metric_values(episodes, metric))
                value_mean = mean(values) if values else math.nan
                value_std = stdev(values) if len(values) > 1 else math.nan
                rows.append({
                    "metric": metric,
                    "metric_label": label,
                    "domain": domain,
                    "domain_label": DOMAIN_LABELS[domain],
                    "method": method,
                    "method_label": METHOD_LABELS[method].replace("\n", " "),
                    "n": len(episodes),
                    "valid_n": len(values),
                    "mean": "" if math.isnan(value_mean) else f"{value_mean:.6f}",
                    "std": "" if math.isnan(value_std) else f"{value_std:.6f}",
                })
    return rows


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def lookup(summary):
    return {
        (row["metric"], row["domain"], row["method"]): as_float(row["mean"])
        for row in summary
    }


def write_paper_table(output_dir, summary):
    values = lookup(summary)
    rows = []
    for method in METHODS:
        row = {"method": METHOD_LABELS[method].replace("\n", " ")}
        for domain, prefix in (("wastesorting", "waste"), ("tomato", "tomato")):
            row[prefix + "_success_rate_pct"] = f"{100 * values[('success_rate', domain, method)]:.1f}"
            row[prefix + "_avg_queries"] = f"{values[('average_question', domain, method)]:.2f}"
            row[prefix + "_avg_steps_success"] = f"{values[('average_step_success_only', domain, method)]:.2f}"
            row[prefix + "_wall_time"] = f"{values[('wall_clock_time', domain, method)]:.2f}"
            row[prefix + "_wall_time_success"] = f"{values[('wall_clock_time_success_only', domain, method)]:.2f}"
        rows.append(row)

    fields = [
        "method", "waste_success_rate_pct", "waste_avg_queries", "waste_avg_steps_success",
        "waste_wall_time", "waste_wall_time_success",
        "tomato_success_rate_pct", "tomato_avg_queries", "tomato_avg_steps_success",
        "tomato_wall_time", "tomato_wall_time_success",
    ]
    write_csv(output_dir / "query_baseline_paper_table.csv", rows, fields)

    lines = [
        "% Generated by experiments/system_eval/plot_query_baseline_figure.py",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Query-strategy baseline comparison with Boolean Oracle answers. Each domain-method cell contains 200 episodes (five scenes $\\times$ 40 runs). Average steps and wall-clock times in this table are computed over successful episodes only. KnowNo wall-clock time includes API latency.}",
        "\\label{tab:query_baselines}", "\\footnotesize",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "\\textbf{Method} & \\textbf{Success (\\%)} & \\textbf{Avg. Queries} & \\textbf{Avg. Steps} & \\textbf{Wall Time (s)} \\\\",
        "\\midrule", "\\multicolumn{5}{l}{\\textit{Waste Sorting}} \\\\",
    ]
    for row in rows:
        lines.append(f"{row['method']} & {row['waste_success_rate_pct']} & {row['waste_avg_queries']} & {row['waste_avg_steps_success']} & {row['waste_wall_time_success']} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{l}{\\textit{Tomato Harvesting}} \\\\")
    for row in rows:
        lines.append(f"{row['method']} & {row['tomato_success_rate_pct']} & {row['tomato_avg_queries']} & {row['tomato_avg_steps_success']} & {row['tomato_wall_time_success']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path = output_dir / "query_baseline_paper_table.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


def exact_mcnemar(improved, worsened):
    discordant = improved + worsened
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(improved, worsened) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def write_paired_contrasts(output_dir, grouped):
    rows = []
    for domain in DOMAINS:
        paired = defaultdict(dict)
        for method in METHODS:
            for row in grouped[(domain, method)]:
                paired[(row["scene"], row["seed"])][method] = is_success(row)
        incomplete = [key for key, unit in paired.items() if set(unit) != set(METHODS)]
        if incomplete or len(paired) != 200:
            raise ValueError(f"Incomplete paired seeds for {domain}: {len(paired)} units, {len(incomplete)} incomplete")
        for baseline in ("knowno", "query_action_pomcp"):
            outcomes = [(unit[baseline], unit["ours"]) for unit in paired.values()]
            improved = sum(not before and after for before, after in outcomes)
            worsened = sum(before and not after for before, after in outcomes)
            difference = 100 * sum(float(after) - float(before) for before, after in outcomes) / len(outcomes)
            rows.append({
                "domain": domain,
                "baseline": METHOD_LABELS[baseline].replace("\n", " "),
                "comparison": "Ours",
                "paired_n": len(outcomes),
                "success_difference_pp": f"{difference:.1f}",
                "improved_pairs": improved,
                "worsened_pairs": worsened,
                "mcnemar_exact_p": f"{exact_mcnemar(improved, worsened):.8g}",
            })
    fields = ["domain", "baseline", "comparison", "paired_n", "success_difference_pp", "improved_pairs", "worsened_pairs", "mcnemar_exact_p"]
    write_csv(output_dir / "query_baseline_paired_success_contrasts.csv", rows, fields)


def write_scene_summary(output_dir, grouped):
    rows = []
    for domain in DOMAINS:
        for method in METHODS:
            by_scene = defaultdict(list)
            for episode in grouped[(domain, method)]:
                by_scene[episode["scene"]].append(episode)
            for scene, episodes in sorted(by_scene.items()):
                successes = sum(is_success(episode) for episode in episodes)
                rows.append({
                    "domain": domain,
                    "method": METHOD_LABELS[method].replace("\n", " "),
                    "scene": scene,
                    "n": len(episodes),
                    "successes": successes,
                    "success_rate_pct": f"{100 * successes / len(episodes):.1f}",
                })
    fields = ["domain", "method", "scene", "n", "successes", "success_rate_pct"]
    write_csv(output_dir / "query_baseline_scene_summary.csv", rows, fields)


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_metric(ax, values, metric, show_legend=False):
    positions = list(range(len(METHODS)))
    width = 0.34
    for index, domain in enumerate(DOMAINS):
        heights = [values[(metric, domain, method)] for method in METHODS]
        if metric == "success_rate":
            heights = [100 * height for height in heights]
        bars = ax.bar(
            [position + (index - 0.5) * width for position in positions], heights,
            width=width, color=DOMAIN_COLORS[domain], edgecolor="#1e293b",
            linewidth=0.8, label=DOMAIN_LABELS[domain],
        )
        for bar, height in zip(bars, heights):
            label = f"{height:.1f}" if metric == "success_rate" else f"{height:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHODS])
    style_axis(ax, METRICS[metric])
    if metric == "success_rate":
        ax.set_ylim(0, 108)
    if show_legend:
        ax.legend(ncol=2, fontsize=11, frameon=True, fancybox=False, edgecolor="#1e293b")


def plot_figures(output_dir, summary):
    values = lookup(summary)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
        fig.patch.set_facecolor("#f8fafc")
        ax.set_facecolor("#ffffff")
        draw_metric(ax, values, metric, show_legend=True)
        for suffix in (".png", ".pdf"):
            path = output_dir / (metric + suffix)
            fig.savefig(path, dpi=200)
            print(path)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.2), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for index, (ax, metric) in enumerate(zip(axes.flat, MAIN_METRICS)):
        ax.set_facecolor("#ffffff")
        draw_metric(ax, values, metric, show_legend=(index == 0))
    for suffix in (".png", ".pdf"):
        path = output_dir / ("query_baseline_comparison" + suffix)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


def main():
    args = parse_args()
    grouped = load_grouped(Path(args.input))
    output_dir = Path(args.output_root) / datetime.now().strftime("00_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = build_summary(grouped)
    fields = ["metric", "metric_label", "domain", "domain_label", "method", "method_label", "n", "valid_n", "mean", "std"]
    write_csv(output_dir / "query_baseline_summary.csv", summary, fields)
    write_paper_table(output_dir, summary)
    write_paired_contrasts(output_dir, grouped)
    write_scene_summary(output_dir, grouped)
    plot_figures(output_dir, summary)


if __name__ == "__main__":
    main()
