#!/usr/bin/env python3
"""Group human_summary.csv by domain/condition and print an aggregate table.

By default, read BRL_EXP/01_BASELINE/01_processed/human_summary.csv and
write human_aggregate.csv in the same directory.

All avg_* columns except avg_interaction_per_query are arithmetic means
across runs (the *_suc columns use
successful runs only). query_rate columns pool question-bearing steps / steps.
avg_interaction_per_query = total interaction time / total query count,
including successful and failed runs with at least one query. Empty subsets or zero denominators produce blank cells.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from summarize_human import display


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / '01_BASELINE/01_processed/human_summary.csv'
COLUMNS = [
    "domain", "condition", "runs", "successes", "success_rate", "avg_query",
    "avg_query_suc", "avg_steps", "avg_step_suc", "query_rate_suc",
    "avg_time_suc", "avg_interaction_suc", "query_rate", "avg_time",
    "avg_interaction", "avg_interaction_per_query",
]
REQUIRED = {
    "domain", "condition", "success", "steps", "query", "query_rate",
    "total_time", "interaction_time", "interaction_time_per_query",
}


def number(row, key, optional=False):
    raw = row[key].strip()
    if optional and not raw:
        return None
    result = float(raw)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid %s: %s" % (key, raw))
    return result


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        for line, raw in enumerate(reader, start=2):
            try:
                row = {key: raw[key].strip() for key in ("domain", "condition")}
                if row["domain"] not in {"tomato", "waste"} or row["condition"] not in {"knowno", "pomdp"}:
                    raise ValueError("Invalid domain/condition")
                success = raw["success"].strip().upper()
                if success not in {"TRUE", "FALSE"}:
                    raise ValueError("Invalid success: " + success)
                row["success"] = success == "TRUE"
                for key in ("steps", "query", "total_time", "interaction_time"):
                    row[key] = number(raw, key)
                for key in ("steps", "query"):
                    if not row[key].is_integer():
                        raise ValueError(key + " must be an integer")
                    row[key] = int(row[key])
                rate = number(raw, "query_rate", optional=True)
                if rate is None:
                    if row["steps"]:
                        raise ValueError("Missing query_rate for nonzero steps")
                    row["query_steps"] = 0
                else:
                    if rate > 1:
                        raise ValueError("query_rate must be between zero and one")
                    # The input CSV rounds rates to six decimals. Recover the
                    # integer count before pooling to avoid rounding drift.
                    count = rate * row["steps"]
                    row["query_steps"] = round(count)
                    tolerance = max(0.00001, row["steps"] * 0.000001)
                    if abs(count - row["query_steps"]) > tolerance:
                        raise ValueError("query_rate does not represent an integer number of query steps")
                if row["query_steps"] > row["query"]:
                    raise ValueError("Query steps exceed total questions")
                row["interaction_time_per_query"] = number(raw, "interaction_time_per_query", optional=True)
                if row["query"] == 0:
                    row["interaction_time_per_query"] = None
                elif row["interaction_time_per_query"] is None:
                    row["interaction_time_per_query"] = row["interaction_time"] / row["query"]
                rows.append(row)
            except (ValueError, AttributeError, TypeError) as error:
                raise ValueError("%s:%d: %s" % (path, line, error)) from error
    return rows


def average(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return mean(values) if values else None


def pooled_query_rate(rows):
    steps = sum(row["steps"] for row in rows)
    return sum(row["query_steps"] for row in rows) / steps if steps else None


def pooled_interaction_per_query(rows):
    queried = [row for row in rows if row['query'] > 0]
    queries = sum(row['query'] for row in queried)
    return sum(row['interaction_time'] for row in queried) / queries if queries else None


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["domain"], row["condition"]].append(row)
    results = []
    for (domain, condition), group in sorted(groups.items()):
        successful = [row for row in group if row["success"]]
        results.append({
            "domain": domain, "condition": condition, "runs": len(group),
            "successes": len(successful), "success_rate": len(successful) / len(group),
            "avg_query": average(group, "query"),
            "avg_query_suc": average(successful, "query"),
            "avg_steps": average(group, "steps"),
            "avg_step_suc": average(successful, "steps"),
            "query_rate_suc": pooled_query_rate(successful),
            "avg_time_suc": average(successful, "total_time"),
            "avg_interaction_suc": average(successful, "interaction_time"),
            "query_rate": pooled_query_rate(group),
            "avg_time": average(group, "total_time"),
            "avg_interaction": average(group, "interaction_time"),
            "avg_interaction_per_query": pooled_interaction_per_query(group),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Summary CSV (default: 01_BASELINE/01_processed/human_summary.csv)")
    parser.add_argument("--output", type=Path, help="Default: <input directory>/human_aggregate.csv")
    args = parser.parse_args()
    try:
        rows = aggregate(read_rows(args.input))
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    output = args.output or args.input.parent / "human_aggregate.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    formatted = [{key: display(row[key]) for key in COLUMNS} for row in rows]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(formatted)
    print("| " + " | ".join(COLUMNS) + " |")
    print("| " + " | ".join(":---:" for _ in COLUMNS) + " |")
    for row in formatted:
        print("| " + " | ".join(row[key] for key in COLUMNS) + " |")
    print("\nSaved %d groups: %s" % (len(rows), output), file=sys.stderr)


if __name__ == "__main__":
    main()
