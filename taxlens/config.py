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
