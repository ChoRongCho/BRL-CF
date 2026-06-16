from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


BASELINE_DIR = Path(__file__).resolve().parent
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from scripts.calibration import (  # noqa: E402
    calibrate_qhat,
    load_calibration_dataset,
    prepare_calibration_choices,
    score_calibration_choices,
)
from scripts.llm import configure_openai  # noqa: E402
from scripts.prompt import temperature_scaling  # noqa: E402
from tomato_utils import TOMATO_BACKGROUND, build_tomato_calibration_prompt  # noqa: E402
from wastesorting_utils import WASTE_BACKGROUND, build_waste_calibration_prompt  # noqa: E402


DOMAIN_CONFIG = {
    "tomato": {
        "background": TOMATO_BACKGROUND,
        "builder": build_tomato_calibration_prompt,
        "default_file": BASELINE_DIR / "data" / "tomato-mc-gen-prompt.txt",
    },
    "waste": {
        "background": WASTE_BACKGROUND,
        "builder": build_waste_calibration_prompt,
        "default_file": BASELINE_DIR / "data" / "waste-mc-gen-prompt.txt",
    },
    "wastesorting": {
        "background": WASTE_BACKGROUND,
        "builder": build_waste_calibration_prompt,
        "default_file": BASELINE_DIR / "data" / "waste-mc-gen-prompt.txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KnowNo qhat from calibration records.")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIG), default="tomato")
    parser.add_argument("--calibration-file", default="")
    parser.add_argument("--settings", default=str(BASELINE_DIR / "llm_setting.json"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--num-calibration", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument(
        "--score-with-llm",
        action="store_true",
        help="Call the LLM to fill top-token logprobs before computing qhat.",
    )
    parser.add_argument(
        "--scored-json",
        default="",
        help="Optional JSON file with already-scored records. Avoids LLM calls.",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_scored_json(path: str) -> list[dict]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = records.get("records", [])
    if not isinstance(records, list):
        raise ValueError("scored JSON must be a list or {'records': [...]} object.")
    return records


def true_probability(record: dict, temperature: float) -> tuple[float, str]:
    top_tokens = [str(token).strip().upper() for token in record["top_tokens"]]
    top_logprobs = [float(value) for value in record["top_logprobs"]]
    true_options = [str(token).strip().upper() for token in record["true_options"]]
    probs = temperature_scaling(top_logprobs, temperature=temperature)

    candidates = [
        (token, float(prob))
        for token, prob in zip(top_tokens, probs)
        if token in true_options
    ]
    if not candidates:
        return 0.0, "<missing>"
    true_token, prob = max(candidates, key=lambda item: item[1])
    return prob, true_token


def add_scores(records: list[dict], temperature: float) -> list[dict]:
    rows = []
    for index, record in enumerate(records, start=1):
        prob, true_token = true_probability(record, temperature)
        score = 1.0 - prob
        record["qhat_score"] = score
        rows.append(
            {
                "index": index,
                "true_options": ",".join(str(token).strip().upper() for token in record["true_options"]),
                "selected_true_option": true_token,
                "p_true": prob,
                "nonconformity_score": score,
                "top_tokens": ",".join(str(token).strip() for token in record["top_tokens"]),
                "top_logprobs": ",".join(str(value) for value in record["top_logprobs"]),
            }
        )
    return rows


def qhat_from_scores(records: list[dict], target_success: float) -> tuple[float, float]:
    n = len(records)
    if n == 0:
        raise ValueError("No calibration records.")
    q_level = min(1.0, float(np.ceil((n + 1) * target_success) / n))
    qhat = float(np.quantile([record["qhat_score"] for record in records], q_level, method="higher"))
    return qhat, q_level


def write_csv(path: str, rows: list[dict]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, records: list[dict], qhat: float, q_level: float, args: argparse.Namespace) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "domain": args.domain,
                "target_success": args.target_success,
                "temperature": args.temperature,
                "q_level": q_level,
                "qhat": qhat,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.scored_json:
        records = load_scored_json(args.scored_json)
    else:
        config = DOMAIN_CONFIG[args.domain]
        calibration_file = args.calibration_file or str(config["default_file"])
        calibration_set, _ = load_calibration_dataset(
            calibration_file,
            args.num_calibration,
            args.num_test,
            domain_name=args.domain,
        )
        records = calibration_set
        if args.score_with_llm:
            configure_openai(args.api_key, args.settings)
            prepare_calibration_choices(
                records,
                config["background"],
                config["builder"],
                generate=False,
            )
            score_calibration_choices(records)
        elif not all("top_tokens" in record and "top_logprobs" in record for record in records):
            raise ValueError(
                "Calibration records do not contain top_tokens/top_logprobs. "
                "Use --score-with-llm, or pass --scored-json with saved LLM scores."
            )

    rows = add_scores(records, args.temperature)
    qhat, q_level = qhat_from_scores(records, args.target_success)

    # Cross-check the shared calibration implementation when temperature is unchanged.
    if args.temperature == 5.0:
        shared_qhat = calibrate_qhat(records, args.target_success)
        if not np.isclose(qhat, shared_qhat):
            raise RuntimeError(f"qhat mismatch: local={qhat}, shared={shared_qhat}")

    print("domain:", args.domain)
    print("num_calibration:", len(records))
    print("target_success:", args.target_success)
    print("temperature:", args.temperature)
    print("q_level:", q_level)
    print("qhat:", qhat)
    write_csv(args.output_csv, rows)
    write_json(args.output_json, records, qhat, q_level, args)
    if args.output_csv:
        print("wrote_csv:", args.output_csv)
    if args.output_json:
        print("wrote_json:", args.output_json)


if __name__ == "__main__":
    main()
