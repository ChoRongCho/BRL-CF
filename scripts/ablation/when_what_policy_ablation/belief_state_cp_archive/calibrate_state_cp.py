#!/usr/bin/env python3
"""Calibrate State-CP qhat using the finite-sample KnowNo rank rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .state_cp_dataset import load_dataset


def finite_sample_qhat(scores: list[float], coverage: float) -> tuple[float, int]:
    """KnowNo-style split conformal quantile: ceil((n+1)*coverage)."""
    if not scores:
        raise ValueError("No calibration scores")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    rank = int(math.ceil((len(scores) + 1) * coverage))
    qhat = sorted(scores)[rank - 1] if rank <= len(scores) else 1.0
    return float(qhat), rank


def prediction_set(record: dict[str, Any], qhat: float) -> list[tuple[str, ...]]:
    cutoff = 1.0 - qhat
    return [
        tuple(row["signature"])
        for row in record["hypotheses"]
        if float(row["probability"]) >= cutoff
    ]


def evaluate(records: list[dict[str, Any]], qhat: float) -> dict[str, Any]:
    sizes = []
    covered = prediction_set_covered = belief_support_covered = singleton_correct = 0
    for record in records:
        predicted = prediction_set(record, qhat)
        truth = tuple(record["true_signature"])
        sizes.append(len(predicted))
        # Split-conformal coverage is score <= qhat. Keep support coverage
        # separate because a misspecified belief can assign the truth zero mass.
        covered += float(record["nonconformity_score"]) <= qhat
        prediction_set_covered += truth in predicted
        belief_support_covered += float(record["p_true"]) > 0.0
        singleton_correct += len(predicted) == 1 and predicted[0] == truth
    n = len(records)
    return {
        "n": n,
        "empirical_coverage": covered / n if n else None,
        "truth_in_prediction_set_rate": prediction_set_covered / n if n else None,
        "truth_in_belief_support_rate": belief_support_covered / n if n else None,
        "truth_outside_belief_support_rate": (
            sum(float(record["p_true"]) == 0.0 for record in records) / n
            if n else None
        ),
        "average_set_size": sum(sizes) / n if n else None,
        "query_trigger_rate": sum(size != 1 for size in sizes) / n if n else None,
        "empty_set_rate": sum(size == 0 for size in sizes) / n if n else None,
        "multiple_set_rate": sum(size > 1 for size in sizes) / n if n else None,
        "singleton_correct_rate": singleton_correct / n if n else None,
    }


def calibrate(dataset: dict[str, Any], coverage: float) -> dict[str, Any]:
    result = {
        "method": "split_conformal_finite_sample_higher",
        "coverage_target": coverage,
        "dataset": dataset["source_path"],
        "dataset_metadata": dataset.get("metadata", {}),
        "domains": {},
        "qhats": {},
    }
    domains = sorted({record["domain"] for record in dataset["records"]})
    for domain in domains:
        calibration = [
            record for record in dataset["records"]
            if record["domain"] == domain and record["split"] == "calibration"
        ]
        test = [
            record for record in dataset["records"]
            if record["domain"] == domain and record["split"] == "test"
        ]
        scores = [float(record["nonconformity_score"]) for record in calibration]
        qhat, rank = finite_sample_qhat(scores, coverage)
        result["qhats"][domain] = qhat
        result["domains"][domain] = {
            "qhat": qhat,
            "calibration_n": len(calibration),
            "test_n": len(test),
            "finite_sample_rank": rank,
            "calibration": evaluate(calibration, qhat),
            "test": evaluate(test, qhat),
        }
    return result


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "domain", "scene", "episode", "seed", "split", "step", "action",
        "p_true", "nonconformity_score", "candidate_count", "hypothesis_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **{key: record[key] for key in fields[:9]},
                "candidate_count": len(record["candidate_facts"]),
                "hypothesis_count": len(record["hypotheses"]),
            })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    result = calibrate(dataset, args.coverage)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.output_csv:
        write_records_csv(Path(args.output_csv).expanduser().resolve(), dataset["records"])
    for domain, details in result["domains"].items():
        print(
            f"{domain}: qhat={details['qhat']:.12g}, "
            f"calibration_n={details['calibration_n']}, test_n={details['test_n']}, "
            f"test_coverage={details['test']['empirical_coverage']}, "
            f"test_query_rate={details['test']['query_trigger_rate']}"
        )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
