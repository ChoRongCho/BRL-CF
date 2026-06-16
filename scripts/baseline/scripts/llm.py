from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None


DEFAULT_LOGIT_BIAS = {
    317: 100.0,  # A, with leading space
    347: 100.0,  # B, with leading space
    327: 100.0,  # C, with leading space
    360: 100.0,  # D, with leading space
    412: 100.0,  # E, with leading space
}

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
    if not setting_path.exists():
        return {}
    with setting_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    return settings


def _ensure_openai():
    global openai
    if openai is not None:
        return openai
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai as openai_module

    openai = openai_module
    return openai


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
    openai_module = _ensure_openai()

    settings = _ACTIVE_SETTINGS or load_llm_settings(_ACTIVE_SETTINGS_PATH)
    key = (
        os.environ.get("OPENAI_API_KEY")
        or settings.get("openai_api_key")
        or settings.get("api_key")
    )
    if key and key != "your-api-key":
        openai_module.api_key = key
    model = str(model or settings.get("model") or settings.get("model_name") or DEFAULT_COMPLETIONS_MODEL).strip()
    if logit_bias is None:
        logit_bias = DEFAULT_LOGIT_BIAS
    model = _resolve_model_for_request(model, stop_seq, logprobs, logit_bias)

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
