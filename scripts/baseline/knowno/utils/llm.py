from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None


DEFAULT_LOGIT_BIAS = {
    317: 100.0,  # A, with leading space
    347: 100.0,  # B, with leading space
    327: 100.0,  # C, with leading space
    360: 100.0,  # D, with leading space
    412: 100.0,  # E, with leading space
}

GOOGLE_ALIASES = {"palm-2l", "palm2l", "palm_2l"}
DEFAULT_GOOGLE_MODEL = "gemini-1.5-pro"

_ACTIVE_SETTINGS: Dict[str, Any] = {}
_ACTIVE_SETTINGS_PATH: Optional[Path] = None
COMPLETIONS_MODELS = {"gpt-3.5-turbo-instruct"}
DEFAULT_COMPLETIONS_MODEL = "gpt-3.5-turbo-instruct"


class timeout:
    def __init__(self, seconds: int = 1, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def load_llm_settings(path: Optional[str | Path] = None) -> Dict[str, Any]:
    setting_path = Path(path) if path else Path(__file__).resolve().parents[1] / "llm_setting.json"
    project_root = Path(__file__).resolve().parents[4]
    merged: Dict[str, Any] = {}
    for candidate in (project_root / "llm_setting_dummy.json", setting_path):
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                merged.update(payload)
        except (OSError, json.JSONDecodeError):
            continue
    return merged


def configure_openai(api_key: Optional[str] = None, settings_path: Optional[str | Path] = None) -> Dict[str, Any]:
    global _ACTIVE_SETTINGS, _ACTIVE_SETTINGS_PATH

    settings = load_llm_settings(settings_path)
    _ACTIVE_SETTINGS = settings
    _ACTIVE_SETTINGS_PATH = Path(settings_path) if settings_path else Path(__file__).resolve().parents[1] / "llm_setting.json"

    key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or settings.get("openai_api_key")
        or settings.get("api_key")
    )
    if key and key != "your-api-key":
        os.environ["OPENAI_API_KEY"] = key
    if openai is not None and key and key != "your-api-key":
        openai.api_key = key

    google_key = (
        os.environ.get("GOOGLE_API_KEY")
        or settings.get("google_api_key")
        or settings.get("google_api_key_path")
    )
    if google_key and google_key != "my-google-api-key":
        os.environ["GOOGLE_API_KEY"] = google_key
    return settings


def _ensure_openai():
    global openai
    if openai is not None:
        return openai
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai as openai_module

    openai = openai_module
    return openai


def _ensure_google_genai():
    global genai, genai_types
    if genai is not None and genai_types is not None:
        return genai, genai_types
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
    from google import genai as genai_module
    from google.genai import types as genai_types_module

    genai = genai_module
    genai_types = genai_types_module
    return genai, genai_types


def _chat_completion_to_legacy_dict(response) -> Dict[str, Any]:
    data = response.model_dump() if hasattr(response, "model_dump") else response
    choice = data["choices"][0]
    text = choice.get("message", {}).get("content") or ""

    legacy_logprobs = None
    logprobs = choice.get("logprobs")
    if logprobs and logprobs.get("content"):
        first_token = logprobs["content"][0]
        top_logprobs = {}
        for item in first_token.get("top_logprobs", []):
            top_logprobs[item["token"]] = item["logprob"]
        legacy_logprobs = {"top_logprobs": [top_logprobs]}

    return {"choices": [{"text": text, "logprobs": legacy_logprobs}], "usage": data.get("usage")}


def _response_to_dict(response) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict_recursive"):
        return response.to_dict_recursive()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return dict(response)


def _is_completions_model(model: str) -> bool:
    return model.strip() in COMPLETIONS_MODELS


def _completion_create(openai_module, api_key: Optional[str], **kwargs) -> Dict[str, Any]:
    model = kwargs["model"]
    use_legacy_completions = _is_completions_model(model)

    if hasattr(openai_module, "OpenAI"):
        client = openai_module.OpenAI(api_key=api_key) if api_key else openai_module.OpenAI()
        if use_legacy_completions:
            response = client.completions.create(**kwargs)
            return _response_to_dict(response)

        stop = kwargs.get("stop")
        chat_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": kwargs["prompt"]}],
            "temperature": kwargs.get("temperature", 0),
        }
        if model.startswith(("gpt-5", "o3", "o4")):
            chat_kwargs["max_completion_tokens"] = kwargs.get("max_tokens", 256)
        else:
            chat_kwargs["max_tokens"] = kwargs.get("max_tokens", 256)
        if stop is not None:
            chat_kwargs["stop"] = stop
        if kwargs.get("logprobs") is not None:
            chat_kwargs["logprobs"] = True
            chat_kwargs["top_logprobs"] = kwargs["logprobs"]
        if kwargs.get("logit_bias"):
            chat_kwargs["logit_bias"] = kwargs["logit_bias"]

        response = client.chat.completions.create(**chat_kwargs)
        return _chat_completion_to_legacy_dict(response)

    if api_key:
        openai_module.api_key = api_key
    return _response_to_dict(openai_module.Completion.create(**kwargs))


def _resolve_model_for_request(model: str, stop_seq, logprobs, logit_bias) -> str:
    return model


def _is_google_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized in GOOGLE_ALIASES or normalized.startswith("gemini") or normalized.startswith("models/")


def _google_model_name(settings: Dict[str, Any], model: str) -> str:
    configured = str(settings.get("google_model") or "").strip()
    if configured:
        return configured.removeprefix("models/")
    normalized = model.strip().lower()
    if normalized in GOOGLE_ALIASES:
        return DEFAULT_GOOGLE_MODEL
    return model.strip().removeprefix("models/")


def _google_api_key(settings: Dict[str, Any]) -> str:
    key = os.environ.get("GOOGLE_API_KEY") or str(settings.get("google_api_key") or "").strip()
    if not key or key == "my-google-api-key":
        raise RuntimeError(
            "Google API key is required for PaLM-2L/Gemini models. "
            "Set google_api_key in the settings JSON or export GOOGLE_API_KEY."
        )
    return key


def _google_generate_content(
    settings: Dict[str, Any],
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    logprobs: Optional[int],
    stop_seq: Optional[Iterable[str]],
) -> Dict[str, Any]:
    api_key = _google_api_key(settings)
    google_model = _google_model_name(settings, model)

    if settings.get("google_backend", "sdk") != "rest":
        try:
            return _google_generate_content_sdk(
                api_key,
                google_model,
                prompt,
                max_tokens,
                temperature,
                logprobs,
                stop_seq,
            )
        except Exception as exc:
            if settings.get("google_backend") == "sdk":
                raise
            print(f"Google SDK call failed, falling back to REST: {exc}")

    return _google_generate_content_rest(
        api_key,
        google_model,
        prompt,
        max_tokens,
        temperature,
        logprobs,
        stop_seq,
    )


def _google_generate_content_sdk(
    api_key: str,
    google_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    logprobs: Optional[int],
    stop_seq: Optional[Iterable[str]],
) -> Dict[str, Any]:
    genai_module, types_module = _ensure_google_genai()
    client = genai_module.Client(api_key=api_key)
    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if stop_seq is not None:
        config_kwargs["stop_sequences"] = list(stop_seq)
    if logprobs is not None:
        config_kwargs["response_logprobs"] = True
        config_kwargs["logprobs"] = int(logprobs)

    response = client.models.generate_content(
        model=google_model,
        contents=prompt,
        config=types_module.GenerateContentConfig(**config_kwargs),
    )
    data = response.model_dump() if hasattr(response, "model_dump") else response.to_json_dict()
    return _google_response_to_legacy_dict(data)


def _google_response_to_legacy_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Google LLM API returned no candidates: {data}")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    text = "".join(str(part.get("text", "")) for part in parts)

    legacy_logprobs = None
    logprobs_result = candidate.get("logprobsResult") or candidate.get("logprobs_result") or {}
    top_candidates = logprobs_result.get("topCandidates") or logprobs_result.get("top_candidates") or []
    if top_candidates:
        first_top_candidate = top_candidates[0]
        token_candidates = first_top_candidate.get("candidates") or []
        top_logprobs = {}
        for item in token_candidates:
            token = item.get("token")
            if token is None:
                continue
            logprob = item.get("logProbability", item.get("log_probability", item.get("logprob", float("-inf"))))
            top_logprobs[str(token)] = float(logprob)
        if top_logprobs:
            legacy_logprobs = {"top_logprobs": [top_logprobs]}

    usage = data.get("usageMetadata") or data.get("usage_metadata")
    return {"choices": [{"text": text, "logprobs": legacy_logprobs}], "usage": usage, "google_raw": data}


def _google_generate_content_rest(
    api_key: str,
    google_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    logprobs: Optional[int],
    stop_seq: Optional[Iterable[str]],
) -> Dict[str, Any]:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(google_model, safe='')}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )

    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if stop_seq is not None:
        generation_config["stopSequences"] = list(stop_seq)
    if logprobs is not None:
        generation_config["responseLogprobs"] = True
        generation_config["logprobs"] = int(logprobs)

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google LLM API error: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Google LLM API error: {exc}") from exc

    return _google_response_to_legacy_dict(data)


def call_llm(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0,
    logprobs: Optional[int] = None,
    stop_seq: Optional[Iterable[str]] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    timeout_seconds: int = 20,
    model: Optional[str] = None,
    max_attempts: int = 5,
) -> Tuple[Any, str]:
    settings = _ACTIVE_SETTINGS or load_llm_settings(_ACTIVE_SETTINGS_PATH)
    key = (
        os.environ.get("OPENAI_API_KEY")
        or settings.get("openai_api_key")
        or settings.get("api_key")
    )
    model = str(model or settings.get("model") or settings.get("model_name") or DEFAULT_COMPLETIONS_MODEL).strip()
    if logit_bias is None:
        logit_bias = DEFAULT_LOGIT_BIAS
    model = _resolve_model_for_request(model, stop_seq, logprobs, logit_bias)

    if _is_google_model(model):
        response = _google_generate_content(
            settings,
            model,
            prompt,
            max_tokens,
            temperature,
            logprobs,
            stop_seq,
        )
        if logprobs is not None and response["choices"][0]["logprobs"] is None:
            raise RuntimeError(
                "Google model did not return token logprobs. KnowNo scoring/calibration requires "
                "top-token log probabilities for A/B/C/D/E."
            )
        return response, response["choices"][0]["text"].strip()

    openai_module = _ensure_openai()
    if key and key != "your-api-key":
        openai_module.api_key = key

    last_error = None
    response = None
    for attempt in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                response = _completion_create(
                    openai_module,
                    key if key and key != "your-api-key" else None,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    logit_bias=logit_bias,
                    stop=list(stop_seq) if stop_seq is not None else None,
                )
            break
        except TimeoutError as exc:
            last_error = exc
            print("Timeout, retrying...")
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:
            last_error = exc
            message = str(exc)
            retryable = any(token in message.lower() for token in ["timeout", "rate limit", "temporarily", "server error", "503", "502", "500"])
            if not retryable:
                raise RuntimeError(f"LLM API error: {message}") from exc
            print(f"Retryable API error, retrying... ({message})")
            time.sleep(min(2 ** attempt, 8))
    if response is None:
        raise RuntimeError("LLM call failed after retries") from last_error
    
    return response, response["choices"][0]["text"].strip()
