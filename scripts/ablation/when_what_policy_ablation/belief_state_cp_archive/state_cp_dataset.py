"""Dataset schema shared by State-CP collection and calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cp_when import (
    aggregate_state_probabilities,
    ambiguous_facts,
    state_key,
    state_label,
)


SCHEMA_VERSION = 3


def make_belief_record(
    belief: Any,
    oracle_true_facts: set[str],
    *,
    domain: str,
    scene: int,
    episode: int,
    seed: int,
    step: int,
    action: str,
    observation_facts: list[str],
) -> dict[str, Any] | None:
    """Serialize one post-observation belief and its hidden true assignment."""
    candidates = ambiguous_facts(belief)
    if not candidates:
        return None
    probabilities = aggregate_state_probabilities(belief, candidates)
    if len(probabilities) < 2:
        return None
    truth = tuple(fact for fact in candidates if fact in oracle_true_facts)
    p_true = float(probabilities.get(truth, 0.0))
    hypotheses = [
        {
            "signature": list(signature),
            "label": state_label(signature),
            "probability": float(probability),
        }
        for signature, probability in sorted(
            probabilities.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    return {
        "domain": domain,
        "scene": int(scene),
        "episode": int(episode),
        "seed": int(seed),
        "step": int(step),
        "action": action,
        "observation_facts": sorted(map(str, observation_facts)),
        "candidate_facts": candidates,
        "hypotheses": hypotheses,
        "true_signature": list(truth),
        "true_label": state_label(truth),
        "p_true": p_true,
        "nonconformity_score": 1.0 - p_true,
        "truth_source": "domain_oracle_resolved_full_belief_state",
    }


def load_dataset(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported State-CP dataset schema: {data.get('schema_version')}"
        )
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("State-CP dataset contains no records")
    required = {
        "domain", "scene", "episode", "seed", "split", "candidate_facts",
        "hypotheses", "true_signature", "p_true", "nonconformity_score",
        "truth_source", "truth_resolution", "matching_truth_hypotheses",
    }
    for index, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"Record {index} is missing fields: {sorted(missing)}")
        total = sum(float(row["probability"]) for row in record["hypotheses"])
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Record {index} hypothesis probabilities sum to {total}")
    data["source_path"] = str(source)
    return data
