from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parents[2]
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
        "default_prompt_file": BASELINE_DIR / "data" / "tomato-mc-gen-prompt.txt",
        "default_info_file": BASELINE_DIR / "data" / "tomato-tasks-info.txt",
    },
    "waste": {
        "background": WASTE_BACKGROUND,
        "builder": build_waste_calibration_prompt,
        "default_prompt_file": BASELINE_DIR / "data" / "waste-mc-gen-prompt.txt",
        "default_info_file": BASELINE_DIR / "data" / "waste-tasks-info.txt",
    },
    "wastesorting": {
        "background": WASTE_BACKGROUND,
        "builder": build_waste_calibration_prompt,
        "default_prompt_file": BASELINE_DIR / "data" / "waste-mc-gen-prompt.txt",
        "default_info_file": BASELINE_DIR / "data" / "waste-tasks-info.txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute KnowNo qhat from calibration records.")
    parser.add_argument("--domain", choices=sorted(DOMAIN_CONFIG), default="tomato")
    parser.add_argument("--calibration-file", default="", help="Backward-compatible alias for --prompt-file.")
    parser.add_argument("--prompt-file", default="", help="MC generation prompt file split by --0000--.")
    parser.add_argument("--info-file", default="", help="Indexed scenario/label metadata file paired with --prompt-file.")
    parser.add_argument("--settings", default=str(BASELINE_DIR / "llm_setting.json"))
    parser.add_argument("--model", default="", help="Override the model name in the LLM settings file.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--num-calibration", type=int, default=1)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument(
        "--target-successes",
        default="",
        help="Comma-separated target success levels. Overrides --target-success when set.",
    )
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
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for calibration logs. Defaults to experiments_logs/calibration_log/<domain>/<model>/<timestamp>.",
    )
    return parser.parse_args()


def model_slug_from_name(model_name: str) -> str:
    model = model_name.lower()
    if "palm-2l" in model or "palm2l" in model:
        return "palm2l"
    if "gpt-3.5" in model or "gpt-35" in model:
        return "gpt35turbo"
    if "gpt-4" in model:
        return "gpt4"
    return "".join(ch for ch in model if ch.isalnum()) or "unknown"


def model_slug(settings_path: str, model_override: str = "") -> str:
    if model_override.strip():
        return model_slug_from_name(model_override)
    try:
        settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    model = str(settings.get("model") or settings.get("model_name") or "")
    return model_slug_from_name(model)


def target_successes(args: argparse.Namespace) -> list[float]:
    if args.target_successes.strip():
        return [float(item.strip()) for item in args.target_successes.split(",") if item.strip()]
    return [float(args.target_success)]


def default_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiments_logs" / "calibration_log" / args.domain / model_slug(args.settings, args.model) / timestamp


def calibration_paths(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    prompt_file = args.prompt_file or args.calibration_file or str(config["default_prompt_file"])
    info_file = args.info_file or str(config["default_info_file"])
    return prompt_file, info_file


def load_runtime_settings(settings_path: Path) -> dict:
    merged: dict = {}
    for path in (PROJECT_ROOT / "llm_setting_dummy.json", settings_path):
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    merged.update(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return merged


def runtime_settings_path(args: argparse.Namespace) -> tuple[str, Path | None]:
    if not args.model.strip():
        return args.settings, None
    settings_path = Path(args.settings).expanduser().resolve()
    settings = load_runtime_settings(settings_path)
    settings["model"] = args.model.strip()
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="knowno_llm_setting_",
        suffix=".json",
        delete=False,
    )
    with handle:
        json.dump(settings, handle, indent=2)
        handle.write("\n")
    return handle.name, Path(handle.name)


def write_error(output_dir: Path, args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "domain": args.domain,
        "model": args.model or "",
        "model_slug": model_slug(args.settings, args.model),
        "settings": str(Path(args.settings).expanduser().resolve()),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    (output_dir / "calibration_error.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "calibration_error.txt").write_text(
        f"{payload['error_type']}: {payload['error']}\n",
        encoding="utf-8",
    )


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


def write_json(path: str, records: list[dict], summaries: list[dict], args: argparse.Namespace) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = DOMAIN_CONFIG[args.domain]
    prompt_file, info_file = calibration_paths(args, config)
    output_path.write_text(
        json.dumps(
            {
                "domain": args.domain,
                "model": args.model or "",
                "model_slug": model_slug(args.settings, args.model),
                "settings": str(Path(args.settings).expanduser().resolve()),
                "calibration_file": str(Path(prompt_file).expanduser().resolve()),
                "prompt_file": str(Path(prompt_file).expanduser().resolve()),
                "info_file": str(Path(info_file).expanduser().resolve()),
                "num_calibration": len(records),
                "temperature": args.temperature,
                "summaries": summaries,
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
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_for_run, temp_settings_path = runtime_settings_path(args)

    try:
        if args.scored_json:
            records = load_scored_json(args.scored_json)
        else:
            config = DOMAIN_CONFIG[args.domain]
            prompt_file, info_file = calibration_paths(args, config)
            calibration_set, _ = load_calibration_dataset(
                prompt_file,
                args.num_calibration,
                args.num_test,
                domain_name=args.domain,
                info_path=info_file,
            )
            records = calibration_set
            if args.score_with_llm:
                configure_openai(args.api_key, settings_for_run)
                prepare_calibration_choices(
                    records,
                    config["background"],
                    config["builder"],
                    generate=False,
                )
                try:
                    score_calibration_choices(records)
                except Exception as exc:
                    write_error(output_dir, args, exc)
                    print("calibration_error_log_dir:", output_dir)
                    raise
            elif not all("top_tokens" in record and "top_logprobs" in record for record in records):
                raise ValueError(
                    "Calibration records do not contain top_tokens/top_logprobs. "
                    "Use --score-with-llm, or pass --scored-json with saved LLM scores."
                )

        rows = add_scores(records, args.temperature)
        summaries = []
        for target_success in target_successes(args):
            qhat, q_level = qhat_from_scores(records, target_success)

            # Cross-check the shared calibration implementation when temperature is unchanged.
            if args.temperature == 5.0:
                shared_qhat = calibrate_qhat(records, target_success)
                if not np.isclose(qhat, shared_qhat):
                    raise RuntimeError(f"qhat mismatch: local={qhat}, shared={shared_qhat}")

            summaries.append(
                {
                    "target_success": target_success,
                    "q_level": q_level,
                    "qhat": qhat,
                    "threshold": 1 - qhat,
                }
            )

        print("domain:", args.domain)
        print("num_calibration:", len(records))
        print("model_slug:", model_slug(args.settings, args.model))
        print("targets:", ", ".join(str(item["target_success"]) for item in summaries))
        print("temperature:", args.temperature)
        for item in summaries:
            print(
                f"target_success={item['target_success']} "
                f"q_level={item['q_level']} qhat={item['qhat']} threshold={item['threshold']}"
            )

        output_csv = args.output_csv or str(output_dir / "calibration_scores.csv")
        output_json = args.output_json or str(output_dir / "calibration_summary.json")
        write_csv(output_csv, rows)
        write_json(output_json, records, summaries, args)
        print("wrote_csv:", output_csv)
        print("wrote_json:", output_json)
        print("calibration_log_dir:", output_dir)
    finally:
        if temp_settings_path is not None:
            temp_settings_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
