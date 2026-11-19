"""Redis answer cache with index-version invalidation.

Key  = sha256("<normalised question>|<tax year or auto>|<index_version>")
Value = zlib-compressed JSON of the pipeline result, TTL 24 h.

Rebuilding the index writes a new ``taxlens:index_version``, so every
answer computed from the old index stops matching and ages out. If Redis
is down the cache is bypassed and ``unavailable`` is counted; answers are
never blocked by the cache.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib

from taxlens import config

VERSION_KEY = "taxlens:index_version"


def cache_key(question: str, tax_year: int | None, index_version: str) -> str:
    normalized_q = re.sub(r"\s+", " ", question.strip().lower())
    payload = f"{normalized_q}|{tax_year if tax_year is not None else 'auto'}|{index_version}"
    return "taxlens:answer:" + hashlib.sha256(payload.encode()).hexdigest()


class AnswerCache:
    def __init__(self, client=None, ttl: int | None = None):
        self.client = client
        self.ttl = ttl or config.CACHE_TTL_S
        self.hits = self.misses = self.unavailable = 0

    @classmethod
    def from_url(cls, url: str | None = None) -> "AnswerCache":
        try:
            import redis
            client = redis.from_url(url or config.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
            client.ping()
            return cls(client)
        except Exception as e:
            print(f"Redis unavailable ({e}); answer cache disabled")
            return cls(None)

    def index_version(self) -> str:
        if self.client is None:
            return "none"
        try:
            v = self.client.get(VERSION_KEY)
            return v.decode() if isinstance(v, bytes) else (v or "none")
        except Exception:
            self.unavailable += 1
            return "none"

    def get(self, question: str, tax_year: int | None) -> dict | None:
        if self.client is None:
            return None
        try:
            raw = self.client.get(cache_key(question, tax_year, self.index_version()))
        except Exception:
            self.unavailable += 1
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(zlib.decompress(raw))

    def set(self, question: str, tax_year: int | None, value: dict) -> None:
        if self.client is None:
            return
        try:
            blob = zlib.compress(json.dumps(value, default=str).encode(), 6)
            self.client.set(cache_key(question, tax_year, self.index_version()), blob, ex=self.ttl)
        except Exception:
            self.unavailable += 1

    def stats(self) -> dict:
        return {"enabled": self.client is not None, "hits": self.hits, "misses": self.misses,
                "unavailable": self.unavailable}
