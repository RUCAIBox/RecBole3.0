from __future__ import annotations

import asyncio
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any


class LLMClient:
    """Async OpenAI-compatible HTTP client with retry and batching.

    Self-contained copy for the AgentCF++ package (no dependency on the
    `agentcf` package or the `openai` SDK).
    """

    def __init__(
        self,
        *,
        api_base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        temperature: float = 0.2,
        max_tokens: int = 2000,
        request_retries: int = 3,
        retry_backoff_sec: float = 20.0,
        request_timeout_sec: float = 120.0,
        concurrency: int = 10,
    ):
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key_env
        self._model_name = model_name
        self._embedding_model = embedding_model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_retries = request_retries
        self._retry_backoff_sec = retry_backoff_sec
        self._request_timeout_sec = request_timeout_sec
        self._concurrency = concurrency
        self._ssl_context = ssl.create_default_context()

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _request_sync(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = self._build_headers()

        for attempt in range(self._request_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(
                    req,
                    timeout=self._request_timeout_sec,
                    context=self._ssl_context,
                ) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt == self._request_retries - 1:
                    raise RuntimeError(f"LLM API request failed after {self._request_retries} attempts: {e}") from e
                wait = self._retry_backoff_sec * (attempt + 1)
                print(f"[agentcfpp:llm_client] request failed (attempt {attempt + 1}): {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
        raise RuntimeError("Unreachable")

    async def _request_async(self, url: str, payload: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._request_sync, url, payload)

    def chat_completion(self, messages: list[dict[str, str]], *, temperature: float | None = None) -> str:
        url = f"{self._api_base_url}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": self._max_tokens,
        }
        response = self._request_sync(url, payload)
        return response["choices"][0]["message"]["content"]

    def chat_completion_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        temperature: float | None = None,
    ) -> list[str]:
        if not messages_list:
            return []

        url = f"{self._api_base_url}/chat/completions"
        temp = temperature if temperature is not None else self._temperature

        async def _run() -> list[str]:
            semaphore = asyncio.Semaphore(self._concurrency)
            tasks = []
            for messages in messages_list:
                payload = {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": self._max_tokens,
                }
                tasks.append(self._request_async(url, payload, semaphore))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for resp in responses:
                if isinstance(resp, Exception):
                    print(f"[agentcfpp:llm_client] batch request failed: {resp}")
                    results.append("")
                else:
                    results.append(resp["choices"][0]["message"]["content"])
            return results

        return asyncio.run(_run())

    def embedding_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        url = f"{self._api_base_url}/embeddings"

        async def _run() -> list[list[float]]:
            semaphore = asyncio.Semaphore(self._concurrency)
            tasks = []
            for text in texts:
                payload = {
                    "model": self._embedding_model,
                    "input": text,
                }
                tasks.append(self._request_async(url, payload, semaphore))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for resp in responses:
                if isinstance(resp, Exception):
                    print(f"[agentcfpp:llm_client] embedding request failed: {resp}")
                    results.append([])
                else:
                    results.append(resp["data"][0]["embedding"])
            return results

        return asyncio.run(_run())


__all__ = ["LLMClient"]
