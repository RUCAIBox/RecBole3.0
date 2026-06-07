from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Hashable, Mapping, Protocol, Sequence, TypeVar


class OpenAICompatibleConfig(Protocol):
    """Configuration contract used by the shared chat-completions client."""

    api_model_name: str
    api_base_url: str
    api_key_env: str
    temperature: float
    max_output_tokens: int
    request_retries: int
    retry_backoff_sec: float
    request_timeout_sec: float
    api_response_cache_path: str
    refresh_api_response_cache: bool


class OpenAICompatibleClient:
    """Small cached client for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key_env: str,
        temperature: float,
        max_output_tokens: int,
        request_retries: int,
        retry_backoff_sec: float,
        request_timeout_sec: float,
        cache_path: str | Path,
        refresh_cache: bool,
    ) -> None:
        self.endpoint = str(endpoint)
        self.model = str(model)
        self.api_key_env = str(api_key_env)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.request_retries = max(1, int(request_retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.request_timeout_sec = float(request_timeout_sec)
        self.cache_path = Path(cache_path)
        self.refresh_cache = bool(refresh_cache)
        self._cache: dict[str, str] | None = None
        self._cache_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: OpenAICompatibleConfig) -> OpenAICompatibleClient:
        return cls(
            endpoint=config.api_base_url,
            model=config.api_model_name,
            api_key_env=config.api_key_env,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            request_retries=config.request_retries,
            retry_backoff_sec=config.retry_backoff_sec,
            request_timeout_sec=config.request_timeout_sec,
            cache_path=config.api_response_cache_path,
            refresh_cache=config.refresh_api_response_cache,
        )

    def request(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> str:
        payload = self._build_payload(prompt, system_prompt=system_prompt, extra_body=extra_body)
        cache_key = self._cache_key(payload)
        cached_response = self._lookup_cache(cache_key)
        if cached_response is not None:
            return cached_response
        response = self._request_uncached(payload)
        self._store_cache(cache_key, response)
        return response

    def _build_payload(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        extra_body: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(prompt)})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        if extra_body:
            payload.update(dict(extra_body))
        return payload

    def _request_uncached(self, payload: Mapping[str, Any]) -> str:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        backoff = self.retry_backoff_sec
        last_error: Exception | None = None
        for attempt in range(self.request_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                    content = json.loads(response.read().decode("utf-8"))
                return str(content["choices"][0]["message"]["content"])
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.request_retries:
                    time.sleep(backoff)
                    backoff *= 2.0
        raise RuntimeError(f"Failed to call the configured OpenAI-compatible endpoint: {last_error}") from last_error

    def _cache_key(self, payload: Mapping[str, Any]) -> str:
        value = {"endpoint": self.endpoint, "payload": dict(payload)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()

    def _lookup_cache(self, key: str) -> str | None:
        if self.refresh_cache:
            return None
        return self._load_cache().get(key)

    def _load_cache(self) -> dict[str, str]:
        with self._cache_lock:
            if self._cache is not None:
                return self._cache
            cache: dict[str, str] = {}
            if self.cache_path.exists():
                with self.cache_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        key = str(record.get("key", "")).strip()
                        response = record.get("response")
                        if key and isinstance(response, str):
                            cache[key] = response
            self._cache = cache
            return cache

    def _store_cache(self, key: str, response: str) -> None:
        cache = self._load_cache()
        with self._cache_lock:
            cache[key] = response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


TTask = TypeVar("TTask", bound=Hashable)
TResult = TypeVar("TResult")


def dispatch_requests(
    tasks: Sequence[TTask],
    request_fn: Callable[[TTask], TResult],
    *,
    max_concurrency: int = 1,
) -> list[TResult]:
    """Dispatch unique requests concurrently and restore the original task order."""

    if not tasks:
        return []
    unique_tasks = list(dict.fromkeys(tasks))

    workers = min(max(1, int(max_concurrency)), len(unique_tasks))
    if workers == 1:
        unique_results = [request_fn(task) for task in unique_tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            unique_results = list(executor.map(request_fn, unique_tasks))

    results_by_task = dict(zip(unique_tasks, unique_results, strict=True))
    return [results_by_task[task] for task in tasks]


__all__ = [
    "OpenAICompatibleClient",
    "OpenAICompatibleConfig",
    "dispatch_requests",
]
