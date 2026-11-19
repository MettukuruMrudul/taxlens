"""Thin, swappable LLM client.

Two wire formats cover almost every model you might use:

* ``openai``    - the OpenAI chat-completions API. The same code talks to
                  OpenAI, a local vLLM server, or Ollama; only LLM_BASE_URL
                  and LLM_MODEL change.
* ``anthropic`` - the Anthropic messages API.

``none`` means no LLM: the answer generator falls back to extractive
answers built directly from retrieved IRS text, so the whole system runs
(and is testable) without an API key or a GPU.

Only ``requests`` is used, so there is no SDK dependency.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator, Protocol

import requests

from taxlens import config


class LLMError(RuntimeError):
    """The model call failed (network, auth, or bad response)."""


class LLMTimeout(LLMError):
    """The model did not answer within LLM_TIMEOUT_S."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.0) -> str: ...

    def stream(self, system: str, user: str, *, max_tokens: int = 800,
               temperature: float = 0.0) -> Iterator[str]: ...


def _sse_data(response) -> Iterator[dict]:
    """Yield the JSON payload of each `data:` line of a server-sent-event stream."""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _post(url: str, headers: dict, body: dict, timeout: float, stream: bool = False):
    try:
        r = requests.post(url, headers=headers, json=body, timeout=timeout, stream=stream)
    except requests.Timeout as e:
        raise LLMTimeout(str(e)) from e
    except requests.RequestException as e:
        raise LLMError(str(e)) from e
    if r.status_code != 200:
        raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r


@dataclass
class OpenAICompatible:
    model: str
    base_url: str
    api_key: str = ""
    timeout: float = 60.0
    name: str = "openai"

    def _request(self, system: str, user: str, max_tokens: int, temperature: float,
                 stream: bool):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "stream": stream,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        return _post(f"{self.base_url.rstrip('/')}/chat/completions", headers, body,
                     self.timeout, stream)

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.0) -> str:
        r = self._request(system, user, max_tokens, temperature, stream=False)
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response: {r.text[:300]}") from e

    def stream(self, system: str, user: str, *, max_tokens: int = 800,
               temperature: float = 0.0) -> Iterator[str]:
        r = self._request(system, user, max_tokens, temperature, stream=True)
        for event in _sse_data(r):
            choices = event.get("choices") or [{}]
            piece = (choices[0].get("delta") or {}).get("content")
            if piece:
                yield piece


@dataclass
class Anthropic:
    model: str
    api_key: str
    timeout: float = 60.0
    base_url: str = "https://api.anthropic.com/v1"
    name: str = "anthropic"

    def _request(self, system: str, user: str, max_tokens: int, temperature: float,
                 stream: bool):
        headers = {"Content-Type": "application/json", "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "system": system, "stream": stream,
                "messages": [{"role": "user", "content": user}]}
        return _post(f"{self.base_url}/messages", headers, body, self.timeout, stream)

    def complete(self, system: str, user: str, *, max_tokens: int = 800,
                 temperature: float = 0.0) -> str:
        r = self._request(system, user, max_tokens, temperature, stream=False)
        try:
            return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
        except (KeyError, ValueError) as e:
            raise LLMError(f"unexpected response: {r.text[:300]}") from e

    def stream(self, system: str, user: str, *, max_tokens: int = 800,
               temperature: float = 0.0) -> Iterator[str]:
        r = self._request(system, user, max_tokens, temperature, stream=True)
        for event in _sse_data(r):
            if event.get("type") == "content_block_delta":
                piece = (event.get("delta") or {}).get("text")
                if piece:
                    yield piece


def get_llm() -> LLM | None:
    """The configured model, or None for extractive (no-LLM) mode."""
    p = config.LLM_PROVIDER
    if p == "none":
        return None
    if p == "openai":
        return OpenAICompatible(config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_API_KEY,
                                config.LLM_TIMEOUT_S)
    if p == "anthropic":
        if not config.LLM_API_KEY:
            raise LLMError("LLM_PROVIDER=anthropic needs LLM_API_KEY in .env")
        return Anthropic(config.LLM_MODEL, config.LLM_API_KEY, config.LLM_TIMEOUT_S)
    raise LLMError(f"Unknown LLM_PROVIDER {p!r}; use openai, anthropic, or none")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json(text: str):
    """Parse a model's JSON reply, tolerating ``` fences and leading chatter."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
        if start < 0:
            raise
        closer = "}" if cleaned[start] == "{" else "]"
        end = cleaned.rfind(closer)
        return json.loads(cleaned[start:end + 1])
