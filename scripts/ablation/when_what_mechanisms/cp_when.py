from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import random
import sys
from typing import Any

import numpy as np

from scripts.baseline.knowno.scripts.llm import call_llm, configure_openai
from scripts.baseline.knowno.scripts.prompt import (
    process_mc_raw,
    process_mc_raw_preserve_duplicates,
    temperature_scaling,
    top_choice_logprobs,
)
# The KnowNo prompt module was written for direct-script execution and imports
# its sibling as ``scripts.structured_prompts``. Register that exact sibling
# under the expected legacy name without changing the original baseline files.
from scripts.baseline.knowno.scripts import structured_prompts as _knowno_prompts_v1
sys.modules.setdefault("scripts.structured_prompts", _knowno_prompts_v1)
from scripts.baseline.knowno.scripts.structured_prompts_v2 import (
    build_tomato_generation_prompt_text,
    build_tomato_score_prompt_text,
    build_waste_generation_prompt_text,
    build_waste_score_prompt_text,
)
from scripts.utils.utils import _parse_fact

from .policy_types import PolicyContext


NOOPT = "an option not listed here"
LETTERS = ("A", "B", "C", "D", "E")


@contextmanager
def isolated_python_rng(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


@dataclass(frozen=True)
class CPResult:
    prediction_set: list[str]
    fallback_token: str
    options: list[str]
    option_text: str
    option_logprobs: dict[str, float]
    scores: dict[str, float]
    generation_text: str
    score_text: str
    generation_prompt: str
    score_prompt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_set": list(self.prediction_set),
            "fallback_token": self.fallback_token,
            "options": list(self.options),
            "option_text": self.option_text,
            "option_logprobs": dict(self.option_logprobs),
            "scores": dict(self.scores),
            "generation_text": self.generation_text,
            "score_text": self.score_text,
            "generation_prompt": self.generation_prompt,
            "score_prompt": self.score_prompt,
            "prediction_set_size": len(self.prediction_set),
            "contains_noopt": self.fallback_token in self.prediction_set,
        }


def _objects(obj_type, declaration_prefix: str) -> list[str]:
    for declaration, values in obj_type.items():
        if declaration.startswith(declaration_prefix + "("):
            return [str(value) for value in values or []]
    return []


def _fact_index(facts) -> dict[str, list[tuple[str, ...]]]:
    result: dict[str, list[tuple[str, ...]]] = {}
    for fact in facts:
        try:
            pred, args = _parse_fact(str(fact))
        except ValueError:
            continue
        result.setdefault(pred, []).append(tuple(args))
    return result


def _first(index, predicate, position=0, default=None):
    rows = index.get(predicate, [])
    if not rows or len(rows[0]) <= position:
        return default
    return rows[0][position]


def _history_text(context: PolicyContext) -> str:
    lines = [
        f"{i}. {action}"
        for i, action in enumerate(context.action_history, 1)
    ]
    observation = ", ".join(sorted(map(str, context.observation_facts)))
    if observation:
        lines.append(f"Latest observation: {observation}")
    return "\n".join(lines) or "None"


def _tomato_prompts(context: PolicyContext) -> tuple[str, Any]:
    index = _fact_index(context.belief.knowledge.facts)
    tomatoes = _objects(context.env.obj_type, "tomato")
    robot_location = _first(index, "located", 1, "dock_station")
    held = _first(index, "holding", 1, None)
    loaded = sorted(row[0] for row in index.get("loaded", []) if row)
    discarded = sorted(row[0] for row in index.get("discarded", []) if row)
    completed = set(loaded) | set(discarded)
    active = [tomato for tomato in tomatoes if tomato not in completed]

    observed_set = {row[0] for row in index.get("observed", []) if row}
    scanned_set = {row[0] for row in index.get("scanned", []) if row}
    states = []
    for tomato in tomatoes:
        if tomato in loaded:
            status = "loaded"
        elif tomato in discarded:
            status = "discarded"
        elif tomato == held:
            status = "held"
        elif tomato in observed_set:
            status = "detected"
        else:
            status = "unknown"
        ripeness = "unknown"
        for label in ("ripe", "unripe"):
            if (tomato,) in index.get(label, []):
                ripeness = label
                break
        freshness = "unknown"
        if tomato in scanned_set:
            for label in ("fresh", "rotten"):
                if (tomato,) in index.get(label, []):
                    freshness = label
                    break
        states.append(
            f"{tomato}: {status}, observed {ripeness}, scanned {freshness}"
        )

    instruction = "Harvest all ripe tomatoes and discard rotten tomatoes."
    history = _history_text(context)
    kwargs = dict(
        instruction=instruction,
        robot_location=robot_location,
        active_tomatoes=active,
        tomato_state_text="\n".join(states),
        held_tomato=held,
        loaded_tomatoes=loaded,
        discarded_tomatoes=discarded,
        history_text=history,
        required_next_action_text="",
    )
    generation = build_tomato_generation_prompt_text(**kwargs)

    def score(options_text):
        return build_tomato_score_prompt_text(mc_gen_full=options_text, **kwargs)

    return generation, score


def _waste_prompts(context: PolicyContext) -> tuple[str, Any]:
    index = _fact_index(context.belief.knowledge.facts)
    objects = _objects(context.env.obj_type, "waste")
    placed = {row[0] for row in index.get("in_bin", []) if row}
    remaining = [obj for obj in objects if obj not in placed]
    held = _first(index, "holding", 1, None)
    detected = {row[0] for row in index.get("detected", []) if row}
    observed = {}
    for obj in detected:
        for label in ("general", "plastic", "paper", "can"):
            if (obj,) in index.get(label, []):
                observed[obj] = label
                break
    observed_text = (
        ", ".join(f"{obj}: {value}" for obj, value in sorted(observed.items()))
        if observed else "None"
    )
    history = _history_text(context)
    instruction = "Discard all waste."
    kwargs = dict(
        instruction=instruction,
        remaining_objects=remaining,
        observed_text=observed_text,
        held_text=held or "None",
        history_text=history,
        available_bins=["general bin", "plastic bin", "paper bin", "can bin"],
        occlusion_text="None",
    )
    generation = build_waste_generation_prompt_text(**kwargs)

    def score(options_text):
        return build_waste_score_prompt_text(mc_gen_full=options_text, **kwargs)

    return generation, score


class KnowNoCPWhenEvaluator:
    """Use KnowNo-style action scoring only to produce a binary query trigger."""

    def __init__(
        self,
        *,
        domain: str,
        qhat: float,
        score_temperature: float,
        settings_path: str,
        api_key: str = "",
    ):
        self.domain = domain
        self.qhat = float(qhat)
        self.score_temperature = float(score_temperature)
        self.settings = configure_openai(api_key or None, settings_path)

    def evaluate(self, context: PolicyContext) -> CPResult:
        if self.domain == "tomato":
            generation_prompt, score_builder = _tomato_prompts(context)
            parser = process_mc_raw
        elif self.domain == "wastesorting":
            generation_prompt, score_builder = _waste_prompts(context)
            parser = process_mc_raw_preserve_duplicates
        else:
            raise ValueError(f"Unsupported CP domain: {self.domain}")

        local_seed = (int(context.seed) ^ (context.step * 0x4350)) & 0xFFFFFFFF
        with isolated_python_rng(local_seed):
            _, generation_text = call_llm(
                generation_prompt,
                stop_seq=["We:"],
                logit_bias={},
            )
        option_text, options, fallback_token = parser(generation_text.strip())

        score_prompt = score_builder(option_text)
        response, score_text = call_llm(
            score_prompt,
            max_tokens=1,
            logprobs=5,
            logit_bias={},
        )
        _, _, raw = top_choice_logprobs(response)
        logprobs: dict[str, float] = {}
        for raw_token, value in raw.items():
            token = raw_token.strip().strip("'\"").upper()
            if token in LETTERS:
                logprobs[token] = max(float(value), logprobs.get(token, -np.inf))
        if not logprobs:
            raise RuntimeError(f"LLM returned no A-E score tokens: {raw}")

        if self.domain == "wastesorting":
            generated = dict(zip(logprobs, temperature_scaling(
                list(logprobs.values()), self.score_temperature
            )))
            unique_options: list[str] = []
            combined: list[float] = []
            indexes: dict[str, int] = {}
            for token, option in zip(LETTERS, options):
                normalized = option.lower().strip().rstrip(".")
                if normalized not in indexes:
                    indexes[normalized] = len(unique_options)
                    unique_options.append(option)
                    combined.append(0.0)
                combined[indexes[normalized]] += float(generated.get(token, 0.0))
            options = unique_options
            tokens = list(LETTERS[:len(options)])
            fallback_token = tokens[options.index(NOOPT)]
            score_map = dict(zip(tokens, combined))
        else:
            tokens = list(logprobs)
            scaled = temperature_scaling(
                [logprobs[token] for token in tokens],
                self.score_temperature,
            )
            score_map = dict(zip(tokens, map(float, scaled)))

        prediction_set = [
            token for token, score in score_map.items()
            if score >= 1.0 - self.qhat
        ]
        return CPResult(
            prediction_set=prediction_set,
            fallback_token=fallback_token,
            options=options,
            option_text=option_text,
            option_logprobs=logprobs,
            scores=score_map,
            generation_text=generation_text,
            score_text=score_text,
            generation_prompt=generation_prompt,
            score_prompt=score_prompt,
        )
