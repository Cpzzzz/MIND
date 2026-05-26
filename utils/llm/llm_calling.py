import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


PROVIDER_CONFIGS = {
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key_env": "SF_KEY",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}

CHAT_EXTRA_BODY_BY_PROVIDER = {
    "openrouter": {"reasoning": {"enabled": False}},
    "siliconflow": {"enable_thinking": False},
}

DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_CALL_MAX_ATTEMPTS = 5

_CLIENT_CACHE: Dict[tuple, Any] = {}
_CLIENT_CACHE_LOCK = threading.Lock()
_THREAD_STATE = threading.local()


def _new_call_stats():
    return {
        "chat_calls": 0,
        "embedding_calls": 0,
        "latency_sec": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _call_stats():
    if not hasattr(_THREAD_STATE, "call_stats"):
        _THREAD_STATE.call_stats = _new_call_stats()
    return _THREAD_STATE.call_stats


@dataclass
class ModelSpec:
    provider: str = "openrouter"
    model: str = ""
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_api_base(self) -> str:
        provider_cfg = PROVIDER_CONFIGS.get(self.provider, {})
        return self.api_base or provider_cfg.get("api_base") or PROVIDER_CONFIGS["openrouter"]["api_base"]

    @property
    def resolved_api_key_env(self) -> str:
        provider_cfg = PROVIDER_CONFIGS.get(self.provider, {})
        return self.api_key_env or provider_cfg.get("api_key_env") or "OPENROUTER_API_KEY"

def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_model_spec(model: Union[str, Dict[str, Any], ModelSpec]) -> ModelSpec:
    """
    Normalize model strings and dict configs into a ModelSpec.

    Supported examples:
    - "openrouter/meta-llama/llama-3.1-8b-instruct"
    - "siliconflow/Qwen/Qwen3.6-27B"
    - {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash"}
    """
    if isinstance(model, ModelSpec):
        return model

    if isinstance(model, dict):
        provider = model.get("provider", "openrouter")
        return ModelSpec(
            provider=provider,
            model=model.get("model") or model.get("model_id") or "",
            api_base=model.get("api_base"),
            api_key_env=model.get("api_key_env"),
            extra_body=model.get("extra_body", {}) or {},
        )

    if not isinstance(model, str):
        raise TypeError(f"Unsupported model spec type: {type(model)}")

    provider = "openrouter"
    model_id = model
    # Only explicit infrastructure prefixes are parsed from strings. Use a dict
    # {"provider": "openai", "model": "gpt-4o-mini"} for the direct OpenAI API.
    for known_provider in ("openrouter", "siliconflow"):
        prefix = f"{known_provider}/"
        if model.startswith(prefix):
            provider = known_provider
            model_id = model[len(prefix):]
            break

    return ModelSpec(provider=provider, model=model_id)


def _get_api_key(spec: ModelSpec) -> Optional[str]:
    api_key = os.getenv(spec.resolved_api_key_env)
    if not api_key and spec.provider == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
    return api_key


def _get_client(spec: ModelSpec):
    if OpenAI is None:
        return None
    api_key = _get_api_key(spec)
    cache_key = (spec.resolved_api_base, spec.resolved_api_key_env, api_key)
    with _CLIENT_CACHE_LOCK:
        if cache_key not in _CLIENT_CACHE:
            _CLIENT_CACHE[cache_key] = OpenAI(base_url=spec.resolved_api_base, api_key=api_key)
        return _CLIENT_CACHE[cache_key]


def _headers(spec: ModelSpec) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = _get_api_key(spec)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(spec: ModelSpec, endpoint: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    url = f"{spec.resolved_api_base.rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.post(url, headers=_headers(spec), json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _build_chat_extra_body(spec: ModelSpec, kwargs_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body = deepcopy(CHAT_EXTRA_BODY_BY_PROVIDER.get(spec.provider, {}))
    body = _merge_dicts(body, spec.extra_body)
    body = _merge_dicts(body, kwargs_body or {})
    return body


def _build_embedding_extra_body(spec: ModelSpec, kwargs_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    body = _merge_dicts(body, spec.extra_body)
    body = _merge_dicts(body, kwargs_body or {})
    return body


def reset_call_stats():
    _THREAD_STATE.call_stats = _new_call_stats()


def get_call_stats() -> Dict[str, Any]:
    return dict(_call_stats())


def _record_usage(response: Any, duration: float, kind: str):
    stats = _call_stats()
    if kind == "chat":
        stats["chat_calls"] += 1
    elif kind == "embedding":
        stats["embedding_calls"] += 1
    stats["latency_sec"] += duration

    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage:
        if isinstance(usage, dict):
            stats["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            stats["completion_tokens"] += usage.get("completion_tokens", 0) or 0
            stats["total_tokens"] += usage.get("total_tokens", 0) or 0
        else:
            stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
            stats["total_tokens"] += getattr(usage, "total_tokens", 0) or 0


def call_model(model: Union[str, Dict[str, Any], ModelSpec], prompt: str, task: str = None, **kwargs):
    spec = normalize_model_spec(model)
    print(f"[LLM] Calling: {spec.provider}/{spec.model} | Task: {task}")

    messages = [{"role": "user", "content": prompt}]
    if task and "[Optimizer]" in task:
        messages.insert(0, {
            "role": "system",
            "content": (
                "You are a prompt optimizer. Return exactly one XML tag in this format: "
                "<prompt>optimized prompt text</prompt>. Do not include markdown, labels, "
                "analysis, alternatives, or any text outside the XML tag. Keep the optimized "
                "prompt close to the original length unless the user prompt requires a safer concise rewrite."
            )
        })
    elif task and "[Evaluator]" in task:
        messages.insert(0, {
            "role": "system",
            "content": "Return only the requested JSON object. Do not include reasoning, markdown, or extra text."
        })

    extra_body = _build_chat_extra_body(spec, kwargs.pop("extra_body", {}))
    for attempt in range(1, DEFAULT_CALL_MAX_ATTEMPTS + 1):
        try:
            call_kwargs = dict(kwargs)
            call_extra_body = deepcopy(extra_body)
            if attempt > 1:
                print(f"[LLM] Retry {attempt}/{DEFAULT_CALL_MAX_ATTEMPTS}: {spec.provider}/{spec.model} | Task: {task}")

            start = time.perf_counter()
            client = _get_client(spec)
            if client is not None:
                create_kwargs = dict(call_kwargs)
                create_kwargs["timeout"] = DEFAULT_REQUEST_TIMEOUT
                response = client.chat.completions.create(
                    model=spec.model,
                    messages=messages,
                    extra_body=call_extra_body,
                    **create_kwargs
                )
                _record_usage(response, time.perf_counter() - start, "chat")
                content = response.choices[0].message.content
                if content:
                    return content
                raise ValueError("Chat response content is empty.")

            payload = {"model": spec.model, "messages": messages}
            payload.update(call_extra_body)
            payload.update(call_kwargs)
            response = _post_json(spec, "/chat/completions", payload, timeout=DEFAULT_REQUEST_TIMEOUT)
            _record_usage(response, time.perf_counter() - start, "chat")
            content = response["choices"][0]["message"].get("content")
            if content:
                return content
            raise ValueError("Chat response content is empty.")
        except Exception as e:
            print(f"[Error] Chat Error ({attempt}/{DEFAULT_CALL_MAX_ATTEMPTS}): {e}")
            if attempt < DEFAULT_CALL_MAX_ATTEMPTS:
                time.sleep(min(2 ** (attempt - 1), 4))
    return None


def get_embedding(text: str, model: Union[str, Dict[str, Any], ModelSpec], **kwargs):
    spec = normalize_model_spec(model)
    try:
        extra_body = _build_embedding_extra_body(spec, kwargs.pop("extra_body", {}))
        start = time.perf_counter()
        client = _get_client(spec)
        if client is not None:
            create_kwargs = dict(kwargs)
            create_kwargs["timeout"] = DEFAULT_REQUEST_TIMEOUT
            response = client.embeddings.create(
                model=spec.model,
                input=text,
                extra_body=extra_body,
                **create_kwargs
            )
            _record_usage(response, time.perf_counter() - start, "embedding")
            return response.data[0].embedding

        payload = {"model": spec.model, "input": text}
        payload.update(extra_body)
        payload.update(kwargs)
        response = _post_json(spec, "/embeddings", payload, timeout=DEFAULT_REQUEST_TIMEOUT)
        _record_usage(response, time.perf_counter() - start, "embedding")
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"[Error] Embed Error: {e}")
        return []
