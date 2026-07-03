from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "experiments_logs" / "calibration_log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KnowNo calibration logs as Markdown.")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Include every successful run. By default, keep only the latest run per domain/model.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_timestamp(path: Path) -> str:
    return path.parent.name


def latest_successful_runs(log_root: Path, all_runs: bool) -> list[tuple[Path, dict]]:
    runs = []
    for path in sorted(log_root.glob("*/*/*/calibration_summary.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append((path, payload))

    if all_runs:
        return runs

    latest: dict[tuple[str, str], tuple[Path, dict]] = {}
    for path, payload in runs:
        key = (str(payload.get("domain", "")), str(payload.get("model_slug", "")))
        prev = latest.get(key)
        if prev is None or run_timestamp(path) > run_timestamp(prev[0]):
            latest[key] = (path, payload)
    return sorted(latest.values(), key=lambda item: (str(item[1].get("domain", "")), str(item[1].get("model_slug", ""))))


def latest_error_runs(log_root: Path) -> list[tuple[Path, dict]]:
    latest: dict[tuple[str, str], tuple[Path, dict]] = {}
    for path in sorted(log_root.glob("*/*/*/calibration_error.json")):
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        key = (str(payload.get("domain", "")), str(payload.get("model_slug", "")))
        prev = latest.get(key)
        if prev is None or run_timestamp(path) > run_timestamp(prev[0]):
            latest[key] = (path, payload)
    return sorted(latest.values(), key=lambda item: (str(item[1].get("domain", "")), str(item[1].get("model_slug", ""))))


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def summary_by_target(payload: dict) -> dict[float, dict]:
    summaries = {}
    for item in payload.get("summaries", []):
        try:
            summaries[float(item["target_success"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return summaries


def calibration_dataset(payload: dict) -> str:
    calibration_file = payload.get("prompt_file") or payload.get("calibration_file")
    if not calibration_file:
        return "-"
    path = Path(str(calibration_file))
    try:
        return f"`{path.relative_to(PROJECT_ROOT)}`"
    except ValueError:
        return f"`{path}`"


def calibration_dataset_dir(payload: dict) -> str:
    calibration_file = payload.get("prompt_file") or payload.get("calibration_file")
    if not calibration_file:
        return "-"
    path = Path(str(calibration_file)).parent
    try:
        return f"`{path.relative_to(PROJECT_ROOT)}`"
    except ValueError:
        return f"`{path}`"


def calibration_info_file(payload: dict) -> str:
    info_file = payload.get("info_file")
    if not info_file:
        prompt_file = payload.get("prompt_file") or payload.get("calibration_file")
        if not prompt_file:
            return "-"
        prompt_path = Path(str(prompt_file))
        inferred = prompt_path.with_name(prompt_path.name.replace("-mc-gen-prompt", "-tasks-info"))
        info_file = str(inferred)
    path = Path(str(info_file))
    try:
        return f"`{path.relative_to(PROJECT_ROOT)}`"
    except ValueError:
        return f"`{path}`"


def dataset_summary_table(runs: list[tuple[Path, dict]]) -> str:
    by_domain: dict[str, dict[str, set[str]]] = {}
    for _, payload in runs:
        domain = str(payload.get("domain", "-"))
        item = by_domain.setdefault(domain, {"prompts": set(), "infos": set(), "dirs": set(), "sizes": set()})
        item["prompts"].add(calibration_dataset(payload))
        item["infos"].add(calibration_info_file(payload))
        item["dirs"].add(calibration_dataset_dir(payload))
        item["sizes"].add(str(payload.get("num_calibration", "-")))

    rows = []
    for domain, item in sorted(by_domain.items()):
        rows.append(
            [
                domain,
                ", ".join(sorted(item["dirs"])),
                ", ".join(sorted(item["prompts"])),
                ", ".join(sorted(item["infos"])),
                ", ".join(sorted(item["sizes"])),
            ]
        )
    return markdown_table(rows, ["Domain", "Source directory", "Prompt file", "Info file", "Calibration size"])


def dataset_source_note() -> str:
    return (
        "The calibration records are loaded by "
        "`scripts/baseline/knowno/compute_qhat.py` from the domain-specific files in "
        "`scripts/baseline/knowno/data/`."
    )


def markdown_table(rows: list[list[str]], header: list[str]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(log_root: Path, all_runs: bool) -> str:
    successful_runs = latest_successful_runs(log_root, all_runs)
    error_runs = latest_error_runs(log_root)

    lines = [
        "# KnowNo Calibration Summary",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Log root: `{log_root}`",
        "- Targets: 95%, 85%, 75% success",
        "- Calibration size: number of calibration examples used to compute qhat",
        "",
    ]

    if successful_runs:
        lines.extend(["## Calibration Datasets", "", dataset_source_note(), "", dataset_summary_table(successful_runs), ""])

    rows = []
    for path, payload in successful_runs:
        by_target = summary_by_target(payload)
        rows.append(
            [
                str(payload.get("domain", "-")),
                str(payload.get("model") or payload.get("model_slug", "-")),
                str(payload.get("num_calibration", "-")),
                fmt_float(payload.get("temperature"), digits=1),
                fmt_float(by_target.get(0.95, {}).get("qhat")),
                fmt_float(by_target.get(0.85, {}).get("qhat")),
                fmt_float(by_target.get(0.75, {}).get("qhat")),
                f"`{path.parent.relative_to(PROJECT_ROOT)}`",
            ]
        )

    lines.append("## Successful Runs")
    lines.append("")
    if rows:
        lines.append(
            markdown_table(
                rows,
                [
                    "Domain",
                    "Model",
                    "Calibration size",
                    "Temp.",
                    "qhat@95%",
                    "qhat@85%",
                    "qhat@75%",
                    "Run",
                ],
            )
        )
    else:
        lines.append("No successful calibration summaries found.")

    if error_runs:
        lines.extend(["", "## Failed Runs", ""])
        error_rows = []
        for path, payload in error_runs:
            error_rows.append(
                [
                    str(payload.get("domain", "-")),
                    str(payload.get("model") or payload.get("model_slug", "-")),
                    str(payload.get("error_type", "-")),
                    str(payload.get("error", "-")).replace("\n", " ")[:180],
                    f"`{path.parent.relative_to(PROJECT_ROOT)}`",
                ]
            )
        lines.append(markdown_table(error_rows, ["Domain", "Model", "Error type", "Error", "Run"]))

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_root = Path(args.log_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else log_root / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(log_root, args.all_runs), encoding="utf-8")
    print(f"wrote_summary: {output}")


if __name__ == "__main__":
    main()
