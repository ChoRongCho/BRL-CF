"""Aggregate question-policy raw logs into scene_metrics CSV files."""

from __future__ import annotations

import re
from pathlib import Path

from analysis_common import aggregate_folders

DOMAINS = ("tomato", "wastesorting")
SCENE_IDS = ("01", "02", "03", "04", "05")
WHEN_PATTERN = re.compile(r"when_(all|no|ours|random)_rand_0-3$")
STRATEGIES = ("all", "no", "ours", "random")


def scene_dir(domain: str, scene_id: str) -> Path:
    return Path("logs") / domain / f"scene_{scene_id}_step50"


def available_strategies(domain: str) -> list[str]:
    strategy_sets: list[set[str]] = []
    for scene_id in SCENE_IDS:
        root = scene_dir(domain, scene_id)
        labels = {
            match.group(1)
            for path in root.iterdir()
            if path.is_dir() and (match := WHEN_PATTERN.fullmatch(path.name))
        }
        strategy_sets.append(labels)

    common = set.intersection(*strategy_sets) if strategy_sets else set()
    return [strategy for strategy in STRATEGIES if strategy in common]


def strategy_folders(domain: str, strategy: str) -> list[Path]:
    return [
        scene_dir(domain, scene_id) / f"when_{strategy}_rand_0-3"
        for scene_id in SCENE_IDS
    ]


def main() -> None:
    for domain in DOMAINS:
        strategies = available_strategies(domain)
        if not strategies:
            print(f"no common when_* folders found for {domain}")
            continue

        for strategy in strategies:
            output_dir = Path("logs") / domain / "scene_metrics" / f"step_50_thres_{strategy}"
            written = aggregate_folders(strategy_folders(domain, strategy), output_dir)
            print(f"{domain} strategy {strategy}: wrote {len(written)} files to {output_dir}")


if __name__ == "__main__":
    main()
