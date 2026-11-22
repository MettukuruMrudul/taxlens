"""Quick LLM connectivity and speed check.

    python -m taxlens.llm_check                         # the model in .env
    python -m taxlens.llm_check gemini-3.5-flash-lite gemini-flash-latest

For each model it sends one tiny prompt twice, as a normal request and as a
stream (TaxLens streams answers), with a 30-second limit, and prints the time
taken or the error. Uses LLM_BASE_URL, LLM_API_KEY and LLM_REASONING_EFFORT
from .env; the key itself is never printed.
"""
from __future__ import annotations

import sys
import time

from taxlens import config
from taxlens.llm import LLMError, OpenAICompatible

SYSTEM = "Reply with exactly one word."
USER = "Say OK."
LIMIT_S = 30.0


def check(model: str, effort: str) -> None:
    client = OpenAICompatible(model, config.LLM_BASE_URL, config.LLM_API_KEY, LIMIT_S,
                              reasoning_effort=effort)
    label = f"{model} (reasoning_effort={effort or 'not sent'})"
    for mode in ("normal", "stream"):
        start = time.perf_counter()
        try:
            if mode == "normal":
                text = client.complete(SYSTEM, USER, max_tokens=50)
            else:
                text = "".join(client.stream(SYSTEM, USER, max_tokens=50))
            secs = time.perf_counter() - start
            print(f"  {label:<55} {mode:<7} OK  {secs:5.1f}s  reply={text.strip()[:30]!r}")
        except LLMError as e:
            secs = time.perf_counter() - start
            print(f"  {label:<55} {mode:<7} FAIL {secs:5.1f}s  {str(e)[:150]}")


def main(argv: list[str]) -> int:
    if config.LLM_PROVIDER != "openai":
        print("This check is for LLM_PROVIDER=openai (OpenAI-compatible endpoints such as Gemini).")
        return 2
    models = argv or [config.LLM_MODEL]
    print(f"endpoint: {config.LLM_BASE_URL}")
    for model in models:
        check(model, config.LLM_REASONING_EFFORT)
        if config.LLM_REASONING_EFFORT:
            check(model, "")          # also try without the reasoning setting
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
