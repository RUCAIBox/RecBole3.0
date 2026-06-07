from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recbole3.model.openai import OpenAICompatibleClient, dispatch_requests


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"choices": [{"message": {"content": self.content}}]}).encode("utf-8")


def test_openai_compatible_client_caches_complete_payload(tmp_path: Path, monkeypatch) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_urlopen(request, *, timeout: float):
        assert timeout == 3.0
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse("A B C")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        endpoint="http://localhost/v1/chat/completions",
        model="test-model",
        api_key_env="MISSING_TEST_API_KEY",
        temperature=0.0,
        max_output_tokens=10,
        request_retries=1,
        retry_backoff_sec=0.0,
        request_timeout_sec=3.0,
        cache_path=tmp_path / "responses.jsonl",
        refresh_cache=False,
    )

    first = client.request("rank these", system_prompt="letters only", extra_body={"stop": "\n"})
    second = client.request("rank these", system_prompt="letters only", extra_body={"stop": "\n"})

    assert first == second == "A B C"
    assert len(payloads) == 1
    assert payloads[0]["messages"][0] == {"role": "system", "content": "letters only"}
    assert payloads[0]["stop"] == "\n"


def test_dispatch_requests_deduplicates_and_restores_order() -> None:
    requested: list[str] = []

    def request(value: str) -> str:
        requested.append(value)
        return value.upper()

    results = dispatch_requests(["a", "b", "a"], request, max_concurrency=1)

    assert requested == ["a", "b"]
    assert results == ["A", "B", "A"]
