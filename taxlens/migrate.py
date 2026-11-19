"""Apply database migrations.

Every .sql file in db/migrations is applied once, in filename order, and
recorded in the schema_migrations table. Re-running is safe.

Run from the project folder:
    python -m taxlens.migrate
"""
import sys

from taxlens.config import PROJECT_ROOT
from taxlens.db import connect

MIGRATIONS_DIR = PROJECT_ROOT / "db" / "migrations"


def main() -> int:
    with connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " filename TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        conn.commit()
        applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
        conn.commit()

        pending = [p for p in sorted(MIGRATIONS_DIR.glob("*.sql")) if p.name not in applied]
        if not pending:
            print("Database is up to date.")
            return 0
        for path in pending:
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            print(f"Applied {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
