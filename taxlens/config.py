"""Central settings. Values come from the .env file, with safe local defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://taxlens:taxlens@localhost:5433/taxlens")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RAW_DIR = PROJECT_ROOT / os.getenv("RAW_DIR", "data/raw")
USER_AGENT = os.getenv("USER_AGENT", "TaxLens research project")

# --- Answer generation (LLM) -------------------------------------------------
# LLM_PROVIDER: "openai" (any OpenAI-compatible server: OpenAI, vLLM, Ollama),
#               "anthropic", or "none" (extractive answers, no LLM needed).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
# For "thinking" models: how much internal reasoning to allow before answering
# (e.g. low / medium / high). Empty = don't send the setting (provider default).
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip()

# --- Optional neural models (pip install -r requirements-ml.txt) -------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "")            # e.g. BAAI/bge-base-en-v1.5 (768-d)
RERANK_MODEL = os.getenv("RERANK_MODEL", "")          # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2
NLI_MODEL = os.getenv("NLI_MODEL", "")                # e.g. cross-encoder/nli-deberta-v3-base
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")

# --- API ----------------------------------------------------------------------
# The default is for local development only (long enough for HS256, but public).
DEV_JWT_SECRET = "taxlens-local-dev-secret-not-for-production"
JWT_SECRET = os.getenv("JWT_SECRET") or DEV_JWT_SECRET
JWT_TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "720"))
ALLOW_DEV_TOKENS = os.getenv("ALLOW_DEV_TOKENS", "true").lower() == "true"
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", str(24 * 3600)))
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
