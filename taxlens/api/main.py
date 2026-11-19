"""TaxLens HTTP API (component C12).

    uvicorn taxlens.api.main:app --reload --port 8000

Endpoints (all /v1 routes except /v1/meta and /v1/auth/dev-token need a
bearer token):

    GET  /health                              liveness, no auth
    GET  /v1/meta                             corpus years and which features are on
    POST /v1/auth/dev-token                   development login (ALLOW_DEV_TOKENS=true)
    POST /v1/ask                              full answer as JSON
    POST /v1/ask/stream                       server-sent events, one verified sentence at a time
    GET  /v1/documents/{doc_id}               document metadata and PDF link
    GET  /v1/documents/{doc_id}/pdf           the IRS PDF itself
    GET  /v1/documents/{doc_id}/pages/{page}  PNG of one page, cited boxes highlighted
    GET  /v1/history                          the user's past questions, newest first
    POST /v1/feedback                         thumbs up / down on an answer

Every error response is JSON: {"error": "...", "code": "..."}.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from taxlens import config, tracing
from taxlens.api.auth import User, create_token, get_current_user
from taxlens.api.cache import AnswerCache
from taxlens.api.models import AskRequest, FeedbackRequest, TokenRequest
from taxlens.api.render import parse_boxes, render_page
from taxlens.db import connect
from taxlens.llm import LLMTimeout

# Outcomes that don't depend on today's date are safe to cache.
CACHEABLE = {"answered", "abstained", "out_of_scope"}
ERROR_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
               422: "unprocessable", 429: "rate_limited", 503: "unavailable", 504: "llm_timeout"}

limiter = Limiter(key_func=get_remote_address,
                  storage_uri=os.getenv("RATE_LIMIT_STORAGE", "memory://"))


# ------------------------------------------------------------------------- state

def log_query(user_id: str, result: dict, cache_hit: bool, latency_ms: int) -> int | None:
    """Record a question in queries_log; returns its query_id (None if the DB write fails)."""
    try:
        with connect() as conn:
            row = conn.execute(
                "INSERT INTO queries_log (user_id, question, tax_year, intent_type, cache_hit,"
                " abstained, latency_ms) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING query_id",
                (user_id, result.get("question"), result.get("tax_year"), result.get("intent"),
                 cache_hit, result.get("status") != "answered", latency_ms)).fetchone()
            return row[0]
    except Exception as e:  # logging must never break answering
        print(f"queries_log write failed: {e}")
        return None


class State:
    pipeline = None
    cache = AnswerCache(None)
    error: str | None = "Starting up"
    log_query = staticmethod(log_query)


state = State()


def load_state() -> None:
    from taxlens.pipeline import build_pipeline
    state.cache = AnswerCache.from_url()
    try:
        pipeline = build_pipeline(connect)
    except Exception as e:
        state.pipeline, state.error = None, f"Could not load the index: {e}"
        return
    if not pipeline.corpus.units:
        state.pipeline, state.error = None, "Index not yet built. Run: python -m taxlens.index.build"
    else:
        state.pipeline, state.error = pipeline, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.JWT_SECRET == config.DEV_JWT_SECRET:
        if not config.ALLOW_DEV_TOKENS:
            # Production mode with a public secret would let anyone mint tokens.
            raise RuntimeError("Set JWT_SECRET in .env before running with ALLOW_DEV_TOKENS=false")
        print("Using the built-in development JWT secret. Set JWT_SECRET in .env before deploying.")
    tracing.setup()
    if os.getenv("TAXLENS_SKIP_STARTUP") != "1":       # tests inject their own pipeline
        await run_in_threadpool(load_state)
    yield


app = FastAPI(title="TaxLens API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------------- errors

def error_response(status: int, message: str, code: str | None = None) -> JSONResponse:
    return JSONResponse({"error": message, "code": code or ERROR_CODES.get(status, "error")},
                        status_code=status)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return error_response(exc.status_code, detail)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return error_response(400, f"{where}: {first.get('msg', 'invalid request')}".strip(": "))


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    return error_response(429, f"Too many requests ({exc.detail}). Try again in a minute.")


# ------------------------------------------------------------------------- helpers

def require_pipeline():
    if state.pipeline is None:
        raise HTTPException(503, state.error or "Index not loaded")
    return state.pipeline


def check_year(tax_year: int | None, pipeline) -> None:
    years = pipeline.corpus.years
    if tax_year is not None and tax_year not in years:
        raise HTTPException(422, f"Tax year {tax_year} is not in the corpus "
                                 f"(available: {years[0]}-{years[-1]})")


def sse(obj) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def _meta_from_result(result: dict, years: list[int]) -> dict:
    return {"type": "meta", "status": result["status"], "tax_year": result.get("tax_year"),
            "year_source": result.get("year_source"), "intent": result.get("intent"),
            "message": result.get("message"), "compare_years": result.get("compare_years"),
            "available_years": years}


def _document(doc_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT doc_id, form_code, doc_type, title, tax_year, source_url, local_path"
                           " FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(422, f"Unknown document {doc_id!r}")
    keys = ("doc_id", "form_code", "doc_type", "title", "tax_year", "source_url", "local_path")
    return dict(zip(keys, row))


def _pdf_path(doc: dict) -> Path:
    path = (config.PROJECT_ROOT / doc["local_path"]).resolve()
    raw = config.RAW_DIR.resolve()
    if raw not in path.parents or not path.exists():
        raise HTTPException(404, f"The PDF for {doc['doc_id']} is not on this server. "
                                 "Run: python -m taxlens.ingest.fetch")
    return path


# ------------------------------------------------------------------------- routes

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/meta")
async def meta():
    p = state.pipeline
    return {"index_loaded": p is not None, "error": state.error,
            "years": p.corpus.years if p else [],
            "llm": config.LLM_PROVIDER,
            "dense_retrieval": bool(p and p.corpus.encoder is not None),
            "reranker": bool(p and p.corpus.reranker is not None),
            "cache": state.cache.stats(),
            "allow_dev_tokens": config.ALLOW_DEV_TOKENS}


@app.post("/v1/auth/dev-token")
async def dev_token(body: TokenRequest):
    if not config.ALLOW_DEV_TOKENS:
        raise HTTPException(403, "Development tokens are disabled on this server")
    return {"access_token": create_token(body.username), "token_type": "bearer",
            "user": body.username}


@app.post("/v1/ask")
@limiter.limit(config.RATE_LIMIT)
async def ask(request: Request, body: AskRequest, user: User = Depends(get_current_user)):
    pipeline = require_pipeline()
    check_year(body.tax_year, pipeline)
    start = time.perf_counter()
    cached = state.cache.get(body.question, body.tax_year)
    if cached is not None:
        result, hit = cached, True
    else:
        try:
            res = await run_in_threadpool(pipeline.run, body.question, body.tax_year)
        except LLMTimeout:
            raise HTTPException(504, "The language model did not answer in time")
        result, hit = res.to_dict(), False
        if result["status"] in CACHEABLE:
            state.cache.set(body.question, body.tax_year, result)
    latency = int((time.perf_counter() - start) * 1000)
    query_id = await run_in_threadpool(state.log_query, user.id, result, hit, latency)
    return {**result, "query_id": query_id, "cache_hit": hit, "latency_ms": latency}


@app.post("/v1/ask/stream")
@limiter.limit(config.RATE_LIMIT)
async def ask_stream(request: Request, body: AskRequest, user: User = Depends(get_current_user)):
    pipeline = require_pipeline()
    check_year(body.tax_year, pipeline)

    def events():
        start = time.perf_counter()
        cached = state.cache.get(body.question, body.tax_year)
        final, hit = None, False
        if cached is not None:
            final, hit = cached, True
            yield sse(_meta_from_result(cached, pipeline.corpus.years))
            for s in cached.get("sentences", []):
                yield sse({"type": "sentence", "sentence": s,
                           "citations": [c for c in cached.get("citations", []) if c["n"] in s["citations"]]})
            if cached["status"] == "abstained":
                yield sse({"type": "abstain", "message": cached.get("message"),
                           "reason": cached.get("abstain_reason")})
        else:
            try:
                for event in pipeline.stream(body.question, body.tax_year):
                    if event["type"] == "done":
                        final = event["result"].to_dict()
                    else:
                        yield sse(event)
            except LLMTimeout:
                yield sse({"type": "error", "error": "The language model did not answer in time",
                           "code": "llm_timeout"})
                yield "data: [DONE]\n\n"
                return
            except Exception as e:
                print(f"stream failed: {e!r}")
                yield sse({"type": "error", "error": "Something went wrong while answering",
                           "code": "internal"})
                yield "data: [DONE]\n\n"
                return
            if final["status"] in CACHEABLE:
                state.cache.set(body.question, body.tax_year, final)
        latency = int((time.perf_counter() - start) * 1000)
        query_id = state.log_query(user.id, final, hit, latency)
        yield sse({"type": "done", "result": {**final, "query_id": query_id, "cache_hit": hit,
                                              "latency_ms": latency}})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/documents/{doc_id}")
async def document(doc_id: str, user: User = Depends(get_current_user)):
    doc = await run_in_threadpool(_document, doc_id)
    doc.pop("local_path")
    return {**doc, "pdf_url": f"/v1/documents/{doc_id}/pdf",
            "page_url_template": f"/v1/documents/{doc_id}/pages/{{page}}"}


@app.get("/v1/documents/{doc_id}/pdf")
async def document_pdf(doc_id: str, user: User = Depends(get_current_user)):
    doc = await run_in_threadpool(_document, doc_id)
    return FileResponse(_pdf_path(doc), media_type="application/pdf", filename=f"{doc_id}.pdf")


@app.get("/v1/documents/{doc_id}/pages/{page}")
async def document_page(doc_id: str, page: int, boxes: str | None = Query(None, max_length=2000),
                        user: User = Depends(get_current_user)):
    doc = await run_in_threadpool(_document, doc_id)
    path = _pdf_path(doc)
    try:
        png = await run_in_threadpool(render_page, str(path), page, parse_boxes(boxes))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/v1/history")
async def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  user: User = Depends(get_current_user)):
    def fetch():
        with connect() as conn:
            return conn.execute(
                "SELECT query_id, question, tax_year, intent_type, abstained, cache_hit, latency_ms,"
                " created_at FROM queries_log WHERE user_id = %s ORDER BY created_at DESC"
                " LIMIT %s OFFSET %s", (user.id, limit + 1, offset)).fetchall()
    rows = await run_in_threadpool(fetch)
    keys = ("query_id", "question", "tax_year", "intent_type", "abstained", "cache_hit",
            "latency_ms", "created_at")
    items = [dict(zip(keys, r)) for r in rows[:limit]]
    for it in items:
        it["created_at"] = it["created_at"].isoformat()
    return {"items": items, "has_more": len(rows) > limit, "offset": offset, "limit": limit}


@app.post("/v1/feedback")
async def feedback(body: FeedbackRequest, user: User = Depends(get_current_user)):
    def insert():
        with connect() as conn:
            return conn.execute(
                "INSERT INTO feedback (query_id, user_id, rating, comment)"
                " SELECT query_id, %s, %s, %s FROM queries_log WHERE query_id = %s AND user_id = %s"
                " RETURNING feedback_id",
                (user.id, body.rating, body.comment, body.query_id, user.id)).fetchone()
    row = await run_in_threadpool(insert)
    if row is None:
        raise HTTPException(422, f"No question {body.query_id} for this user")
    return {"feedback_id": row[0], "status": "recorded"}
