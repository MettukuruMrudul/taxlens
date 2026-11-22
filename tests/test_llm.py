"""LLM client tests with a fake HTTP layer (no network, no API key)."""
import pytest

from taxlens import llm


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


def _ok():
    return FakeResponse(200, {"choices": [{"message": {"content": "hello"}}]})


def test_rate_limited_call_is_retried_then_succeeds(monkeypatch):
    replies = [FakeResponse(429, headers={"Retry-After": "2"}), FakeResponse(503), _ok()]
    waits = []
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    client = llm.OpenAICompatible("m", "https://example.test/v1", "key")
    assert client.complete("sys", "user") == "hello"
    assert waits == [2.0, llm.BACKOFF_BASE_S * 2]     # Retry-After first, then backoff


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(429))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(llm.LLMError):
        llm.OpenAICompatible("m", "https://example.test/v1").complete("sys", "user")


def test_auth_errors_are_not_retried(monkeypatch):
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResponse(401, {"error": "bad key"})
    monkeypatch.setattr(llm.requests, "post", post)
    with pytest.raises(llm.LLMError):
        llm.OpenAICompatible("m", "https://example.test/v1", "wrong").complete("sys", "user")
    assert len(calls) == 1
