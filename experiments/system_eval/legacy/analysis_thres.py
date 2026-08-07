"""Aggregate threshold-sweep raw logs into scene_metrics CSV files."""

from __future__ import annotations

import re
from pathlib import Path

from analysis_common import aggregate_folders

DOMAINS = ("tomato", "wastesorting")
SCENE_IDS = ("01", "02", "03", "04", "05")
THRESHOLD_PATTERN = re.compile(r"thres_(\d-\d)$")


def scene_dir(domain: str, scene_id: str) -> Path:
    return Path("logs") / domain / f"scene_{scene_id}_step50"


def common_thresholds(domain: str) -> list[str]:
    threshold_sets: list[set[str]] = []
    for scene_id in SCENE_IDS:
        root = scene_dir(domain, scene_id)
        labels = {
            match.group(1)
            for path in root.iterdir()
            if path.is_dir() and (match := THRESHOLD_PATTERN.fullmatch(path.name))
        }
        threshold_sets.append(labels)

    common = set.intersection(*threshold_sets) if threshold_sets else set()
    return sorted(common, key=lambda value: float(value.replace("-", ".")))


def threshold_folders(domain: str, threshold: str) -> list[Path]:
    return [
        scene_dir(domain, scene_id) / f"thres_{threshold}"
        for scene_id in SCENE_IDS
    ]


def main() -> None:
    for domain in DOMAINS:
        thresholds = common_thresholds(domain)
        if not thresholds:
            print(f"no common threshold folders found for {domain}")
            continue

        for threshold in thresholds:
            output_dir = Path("logs") / domain / "scene_metrics" / f"step_50_thres_{threshold}"
            written = aggregate_folders(threshold_folders(domain, threshold), output_dir)
            print(f"{domain} threshold {threshold}: wrote {len(written)} files to {output_dir}")


if __name__ == "__main__":
    main()
