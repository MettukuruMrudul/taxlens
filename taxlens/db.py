"""Database connection helper."""
import psycopg

from taxlens.config import DATABASE_URL


def connect() -> psycopg.Connection:
    """Open a new connection to the TaxLens Postgres database."""
    return psycopg.connect(DATABASE_URL)
