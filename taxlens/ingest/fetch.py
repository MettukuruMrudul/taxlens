"""Download the MVP corpus, hash each PDF, and record it in the documents table.

Run from the project folder:
    python -m taxlens.ingest.fetch

Safe to re-run. Files already on disk whose SHA-256 matches the database record
are skipped entirely (no download, no DB write). A manifest row is only rewritten
when the file's hash has changed.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

from taxlens.config import PROJECT_ROOT, RAW_DIR, USER_AGENT
from taxlens.db import connect
from taxlens.ingest.corpus import CorpusDoc, mvp_corpus

UPSERT_SQL = """
INSERT INTO documents
    (doc_id, form_code, doc_type, title, tax_year, source_url, local_path, sha256, size_bytes)
VALUES
    (%(doc_id)s, %(form_code)s, %(doc_type)s, %(title)s, %(tax_year)s,
     %(source_url)s, %(local_path)s, %(sha256)s, %(size_bytes)s)
ON CONFLICT (doc_id) DO UPDATE SET
    source_url = EXCLUDED.source_url,
    local_path = EXCLUDED.local_path,
    sha256     = EXCLUDED.sha256,
    size_bytes = EXCLUDED.size_bytes,
    fetched_at = now()
WHERE documents.sha256 IS DISTINCT FROM EXCLUDED.sha256
"""

# Delays (seconds) between successive retry attempts.  Two retries means three
# total attempts; the first delay is 2 s, the second is 4 s.
RETRY_DELAYS = [2, 4]


class NotOnIRS(Exception):
    """The IRS returned 404: this document doesn't exist for this year."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def db_sha256(conn, doc_id: str) -> str | None:
    """Return the SHA-256 stored in the documents table for doc_id, or None."""
    row = conn.execute(
        "SELECT sha256 FROM documents WHERE doc_id = %(doc_id)s",
        {"doc_id": doc_id},
    ).fetchone()
    return row[0] if row else None


def download(url: str, dest: Path, session: requests.Session) -> None:
    """Download *url* to *dest*, retrying temporary failures with backoff.

    Makes at most ``1 + len(RETRY_DELAYS)`` attempts (default: 3).
    Raises ``NotOnIRS`` immediately on HTTP 404 — that is a per-document
    failure, not a transient error.
    """
    last_exc: Exception | None = None
    delays = [None] + RETRY_DELAYS  # first attempt has no preceding delay
    for attempt_idx, delay in enumerate(delays):
        if delay is not None:
            retry_num = attempt_idx  # 1-based retry count
            print(f"    retry {retry_num}/{len(RETRY_DELAYS)} in {delay}s ({last_exc})")
            time.sleep(delay)
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                raise NotOnIRS(url)
            resp.raise_for_status()
            if not resp.content.startswith(b"%PDF"):
                raise ValueError(f"response is not a PDF: {url}")
            # Write to a temporary file first so a crash never leaves a half-written PDF.
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return
        except NotOnIRS:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
    # All attempts exhausted
    raise last_exc  # type: ignore[misc]


def manifest_row(doc: CorpusDoc, path: Path) -> dict:
    return {
        "doc_id": doc.doc_id,
        "form_code": doc.form_code,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "tax_year": doc.tax_year,
        "source_url": doc.url,
        "local_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_of(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the TaxLens MVP corpus.")
    parser.add_argument("--force", action="store_true", help="re-download files already on disk")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    docs = mvp_corpus()
    failed: list[tuple[str, str]] = []

    with connect() as conn:
        for i, doc in enumerate(docs, start=1):
            dest = RAW_DIR / doc.filename
            label = f"[{i:>2}/{len(docs)}] {doc.doc_id}"
            try:
                if not args.force and dest.exists():
                    # Idempotent skip: only skip when the on-disk hash matches
                    # the hash stored in the database, so a partial or corrupted
                    # download is re-fetched automatically.
                    disk_hash = sha256_of(dest)
                    stored_hash = db_sha256(conn, doc.doc_id)
                    if stored_hash is not None and disk_hash == stored_hash:
                        print(f"{label}: already on disk and DB hash matches — skipping")
                        continue  # skip both download AND DB upsert
                    else:
                        print(f"{label}: on disk but hash mismatch or not in DB — re-downloading")

                print(f"{label}: downloading")
                download(doc.url, dest, session)
                time.sleep(1)  # be polite to irs.gov
                conn.execute(UPSERT_SQL, manifest_row(doc, dest))
                conn.commit()
            except NotOnIRS as exc:
                # HTTP 404 is a per-document failure; log it and keep going.
                conn.rollback()
                reason = f"NotOnIRS: {exc}"
                failed.append((doc.doc_id, reason))
                print(f"{label}: FAILED — document not found on IRS.gov (404)")
            except Exception as exc:  # keep going; report everything at the end
                conn.rollback()
                failed.append((doc.doc_id, f"{type(exc).__name__}: {exc}"))
                print(f"{label}: FAILED ({exc})")

    print()
    print(f"Done: {len(docs) - len(failed)} of {len(docs)} documents in the manifest.")
    for doc_id, reason in failed:
        print(f"  missing {doc_id}: {reason}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
