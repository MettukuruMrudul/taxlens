"""OpenTelemetry tracing (component C16), with a no-op fallback.

Every pipeline stage runs inside ``span("taxlens.<stage>", **attrs)``. If
the opentelemetry packages are installed and OTEL_EXPORTER_OTLP_ENDPOINT is
set, spans are exported over OTLP/HTTP to any compatible backend (Jaeger,
Grafana Tempo, Honeycomb, ...). Otherwise ``span`` still records how long
each stage took, so the API can return per-stage timings and the pipeline
never depends on telemetry being available.

Span tree for one request:

    taxlens.request
      taxlens.cache_lookup
      taxlens.year_detect
      taxlens.intent_classify
      taxlens.retrieval
      taxlens.answer_generate
      taxlens.verify
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from taxlens import config

_tracer = None
_configured = False


def setup(service_name: str = "taxlens-api") -> bool:
    """Configure the OTLP exporter once. Returns True if real tracing is on."""
    global _tracer, _configured
    if _configured:
        return _tracer is not None
    _configured = True
    if not config.OTEL_EXPORTER_OTLP_ENDPOINT:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        print("opentelemetry not installed; tracing disabled")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=config.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/") + "/v1/traces")))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("taxlens")
    return True


class SpanHandle:
    """What a ``with span(...)`` block gets: set attributes, read duration."""

    def __init__(self, name: str, otel_span=None):
        self.name = name
        self.attributes: dict = {}
        self._otel = otel_span
        self.start = time.perf_counter()
        self.ms: float = 0.0

    def set(self, key: str, value) -> None:
        self.attributes[key] = value
        if self._otel is not None and value is not None:
            self._otel.set_attribute(key, value if isinstance(value, (bool, int, float, str)) else str(value))


def _finish(handle: SpanHandle, timings: dict | None) -> None:
    handle.ms = (time.perf_counter() - handle.start) * 1000
    if timings is not None:
        timings[handle.name.removeprefix("taxlens.")] = round(handle.ms, 1)


@contextmanager
def span(name: str, timings: dict | None = None, **attrs) -> Iterator[SpanHandle]:
    """Trace a block. If ``timings`` is given, its duration in ms is stored there."""
    if _tracer is not None:
        with _tracer.start_as_current_span(name) as otel_span:
            handle = SpanHandle(name, otel_span)
            for k, v in attrs.items():
                handle.set(k, v)
            try:
                yield handle
            finally:
                _finish(handle, timings)
    else:
        handle = SpanHandle(name)
        for k, v in attrs.items():
            handle.set(k, v)
        try:
            yield handle
        finally:
            _finish(handle, timings)
