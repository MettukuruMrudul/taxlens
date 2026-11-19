"""Parse every downloaded PDF into line_records.

Run from the project folder:
    python -m taxlens.ingest.run_parser            # parse new or changed documents
    python -m taxlens.ingest.run_parser --force    # re-parse everything
    python -m taxlens.ingest.run_parser --doc f1040-2024

A document is skipped when its file hash and the parser version both match
the last run, so re-running is cheap. Each document is replaced inside a
single transaction: you never end up with half a document in the database.
"""
import argparse
import sys
import time
from collections import Counter

from psycopg.types.json import Jsonb

from taxlens.config import PROJECT_ROOT
from taxlens.db import connect
from taxlens.ingest.parse import PARSER_VERSION, parse_pdf

INSERT_SQL = """
INSERT INTO line_records
    (doc_id, seq, record_type, line_label, heading_path, text, page, boxes)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

UPSERT_RUN_SQL = """
INSERT INTO parse_runs (doc_id, sha256, parser_version, record_count, seconds)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (doc_id) DO UPDATE SET
    sha256 = EXCLUDED.sha256,
    parser_version = EXCLUDED.parser_version,
    record_count = EXCLUDED.record_count,
    seconds = EXCLUDED.seconds,
    parsed_at = now()
"""


def parse_one(conn, doc_id: str, doc_type: str, local_path: str, sha: str, force: bool = False) -> dict:
    """Parse one document and save its records atomically.

    Returns a dict with keys: doc_id, skipped (bool), record_count (int), seconds (float).
    Skips if SHA-256 and parser version are unchanged (unless force=True).
    Replaces all existing line_records for this doc_id in a single transaction.
    """
    row = conn.execute(
        "SELECT sha256, parser_version FROM parse_runs WHERE doc_id = %s", (doc_id,)
    ).fetchone()
    done_sha, done_version = (row[0], row[1]) if row else (None, None)

    if not force and done_sha == sha and done_version == PARSER_VERSION:
        return {"doc_id": doc_id, "skipped": True, "record_count": 0, "seconds": 0.0}

    t0 = time.time()
    records = parse_pdf(PROJECT_ROOT / local_path, doc_type)
    seconds = time.time() - t0

    with conn.transaction():
        conn.execute("DELETE FROM line_records WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, [
                (doc_id, seq, r.record_type, r.line_label, r.heading_path,
                 r.text, r.page, Jsonb(r.boxes))
                for seq, r in enumerate(records, start=1)
            ])
        conn.execute(UPSERT_RUN_SQL, (doc_id, sha, PARSER_VERSION, len(records), seconds))

    kinds = Counter(r.record_type for r in records)
    return {
        "doc_id": doc_id,
        "skipped": False,
        "record_count": len(records),
        "seconds": seconds,
        "record_types": dict(kinds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse IRS PDFs into line_records.")
    parser.add_argument("--doc", action="append", help="only this doc_id (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-parse even if unchanged")
    args = parser.parse_args()

    with connect() as conn:
        docs = conn.execute(
            """
            SELECT d.doc_id, d.doc_type, d.local_path, d.sha256
            FROM documents d
            ORDER BY d.tax_year, d.doc_id
            """
        ).fetchall()
        conn.commit()

        started, parsed, skipped = time.time(), 0, 0
        for doc_id, doc_type, local_path, sha in docs:
            if args.doc and doc_id not in args.doc:
                continue

            result = parse_one(conn, doc_id, doc_type, local_path, sha, force=args.force)

            if result["skipped"]:
                skipped += 1
                continue

            kinds = result["record_types"]
            summary = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
            print(f"{doc_id:<16} {result['seconds']:5.1f}s  {summary}")
            parsed += 1

    print(f"\nParsed {parsed} document(s), skipped {skipped} unchanged, "
          f"in {time.time() - started:.0f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
