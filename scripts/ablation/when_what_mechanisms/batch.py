from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_ONE = ROOT / "run" / "run_when_what_mechanism.sh"
VALID_CONDITIONS = {"ours", "cp_when", "value_when", "value_what"}


def env(name, default):
    return os.environ.get("WW_" + name, default)


def flag(name, default="false"):
    value = env(name, default)
    if value not in {"true", "false"}:
        raise ValueError(f"WW_{name} must be true or false")
    return value == "true"


def absolute(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def paired_seeds(path, domains, scenes, iterations):
    expected = {
        (domain, scene, iteration)
        for domain in domains
        for scene in scenes
        for iteration in range(1, iterations + 1)
    }
    rows = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"domain", "condition", "scene", "iteration", "seed"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Paired seed CSV is missing required columns")
        for row in reader:
            if row["condition"] != "ours":
                continue
            key = (row["domain"], int(row["scene"]), int(row["iteration"]))
            if key not in expected:
                continue
            if key in rows:
                raise ValueError(f"Duplicate paired seed: {key}")
            rows[key] = int(row["seed"])
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} paired seeds, e.g. {sorted(missing)[:3]}")
    return rows


def main():
    conditions = env("CONDITIONS", "ours cp_when value_when value_what").split()
    domains = env("DOMAINS", "tomato wastesorting").split()
    scenes = [int(value) for value in env("SCENES", "1 2 3 4 5").split()]
    iterations = int(env("ITERATIONS", "40"))
    if not conditions or set(conditions) - VALID_CONDITIONS:
        raise ValueError(f"Invalid conditions: {conditions}")
    if not domains or set(domains) - {"tomato", "wastesorting"}:
        raise ValueError(f"Invalid domains: {domains}")
    if iterations < 1 or not scenes or min(scenes) < 1:
        raise ValueError("Iterations and scenes must be positive")

    seed_path = absolute(env(
        "PAIRED_SEED_LOG",
        "experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv",
    ))
    seeds = paired_seeds(seed_path, domains, scenes, iterations)
    dry_run = flag("DRY_RUN")
    resume = flag("RESUME")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    configured_run_root = env("RUN_ROOT", "").strip()
    run_root = (
        absolute(configured_run_root)
        if configured_run_root
        else absolute(env("LOG_ROOT", "experiments_logs/when_what_mechanisms")) / timestamp
    )
    plans = [
        (condition, domain, scene, iteration, seeds[domain, scene, iteration])
        for condition in conditions
        for domain in domains
        for scene in scenes
        for iteration in range(1, iterations + 1)
    ]
    print(
        f"Conditions: {conditions}\nDomains: {domains}; scenes: {scenes}\n"
        f"Repetitions: {iterations}\nTotal episodes: {len(plans)}",
        flush=True,
    )
    if dry_run:
        sample = [
            next(
                plan for plan in plans
                if plan[0] == condition and plan[1] == domain
            )
            for condition in conditions
            for domain in domains
        ]
        for condition, domain, scene, iteration, seed in sample:
            subprocess.run([
                "bash", str(RUN_ONE), "--condition", condition,
                "--domain", domain, "--scene", str(scene),
                "--seed", str(seed), "--dry-run",
            ], cwd=ROOT, check=True)
        print("Dry run validated paired seeds and representative commands.")
        return 0

    if resume:
        if not run_root.is_dir():
            raise ValueError("--resume requires an existing --run-root")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "conditions": conditions,
        "domains": domains,
        "scenes": scenes,
        "iterations": iterations,
        "paired_seed_log": str(seed_path),
        "paired_seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    manifest_path = run_root / "manifest.json"
    if resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest does not match the requested matrix")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    runs_csv = run_root / "runs.csv"
    fields = ["index", "condition", "domain", "scene", "iteration", "seed", "status", "log_dir"]
    if not resume:
        with runs_csv.open("w", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()

    failures = skipped = 0
    for index, (condition, domain, scene, iteration, seed) in enumerate(plans, 1):
        log_dir = run_root / domain / f"scene_{scene:02}" / condition / f"run_{iteration:02}_seed_{seed}"
        marker = log_dir / ".complete"
        status = "complete"
        if resume and marker.exists():
            status = "skipped"
            skipped += 1
        else:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "console.log").open("w") as output:
                result = subprocess.run([
                    "bash", str(RUN_ONE), "--condition", condition,
                    "--domain", domain, "--scene", str(scene),
                    "--seed", str(seed), "--log-dir", str(log_dir),
                ], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                marker.write_text("complete\n")
            else:
                status = "failed"
                failures += 1
        with runs_csv.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow({
                "index": index,
                "condition": condition,
                "domain": domain,
                "scene": f"{scene:02}",
                "iteration": iteration,
                "seed": seed,
                "status": status,
                "log_dir": str(log_dir),
            })
        print(f"\rProgress: {index}/{len(plans)} {condition} ({status})", end="", flush=True)
    print(f"\nCompleted: {len(plans)-failures-skipped}, skipped: {skipped}, failed: {failures}")
    print(f"Run root: {run_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
