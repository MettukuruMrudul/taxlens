"""API tests with an in-memory pipeline (no database, no Redis)."""
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("slowapi")
os.environ["TAXLENS_SKIP_STARTUP"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from taxlens.api import main  # noqa: E402
from taxlens.api.auth import create_token  # noqa: E402
from taxlens.api.cache import AnswerCache  # noqa: E402
from taxlens.pipeline import Pipeline  # noqa: E402


@pytest.fixture
def client(corpus, monkeypatch):
    monkeypatch.setattr(main.state, "pipeline", Pipeline(corpus))
    monkeypatch.setattr(main.state, "cache", AnswerCache(None))
    monkeypatch.setattr(main.state, "error", None)
    monkeypatch.setattr(main.state, "log_query", lambda *a, **k: 42)
    main.limiter.reset()
    with TestClient(main.app) as c:
        yield c


def auth(user="tester"):
    return {"Authorization": f"Bearer {create_token(user)}"}


def test_health_needs_no_auth(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_requires_a_valid_token(client):
    body = {"question": "What is line 30 used for in 2022?"}
    assert client.post("/v1/ask", json=body).status_code == 401
    r = client.post("/v1/ask", json=body, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.json()["code"] == "unauthorized"
    expired = {"Authorization": f"Bearer {create_token('x', minutes=-1)}"}
    assert client.post("/v1/ask", json=body, headers=expired).status_code == 401


def test_ask_returns_a_cited_answer(client):
    r = client.post("/v1/ask", json={"question": "What is line 30 of Form 1040 used for in 2022?"},
                    headers=auth())
    data = r.json()
    assert r.status_code == 200 and data["status"] == "answered" and data["query_id"] == 42
    assert data["citations"] and data["citations"][0]["tax_year"] == 2022


def test_unknown_tax_year_is_422(client):
    r = client.post("/v1/ask", json={"question": "line 30?", "tax_year": 2019}, headers=auth())
    assert r.status_code == 422 and "available" in r.json()["error"]


def test_malformed_body_is_400(client):
    assert client.post("/v1/ask", json={"q": "x"}, headers=auth()).status_code == 400


def test_stream_sends_sentences_then_done(client):
    with client.stream("POST", "/v1/ask/stream", headers=auth(),
                       json={"question": "What is line 30 of Form 1040 used for in 2023?"}) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(ln[6:]) for ln in lines[:-1]]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and "sentence" in kinds and kinds[-1] == "done"
    assert events[-1]["result"]["query_id"] == 42


def test_rate_limit_returns_429(client, monkeypatch):
    body = {"question": "best pizza in Boston?"}
    codes = [client.post("/v1/ask", json=body, headers=auth()).status_code for _ in range(25)]
    assert 429 in codes


def test_dev_token_endpoint(client):
    r = client.post("/v1/auth/dev-token", json={"username": "mrudul"})
    assert r.status_code == 200 and r.json()["token_type"] == "bearer"


def test_503_when_index_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main.state, "pipeline", None)
    monkeypatch.setattr(main.state, "error", "Index not yet built")
    r = client.post("/v1/ask", json={"question": "line 30 in 2023?"}, headers=auth())
    assert r.status_code == 503 and r.json()["code"] == "unavailable"
