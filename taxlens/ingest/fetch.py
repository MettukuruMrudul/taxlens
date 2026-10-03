"""Download the MVP corpus, hash each PDF, and record it in the documents table.

Run from the project folder:
    python -m taxlens.ingest.fetch

Safe to re-run. Files already on disk are not downloaded again, and a
manifest row is only rewritten when the file's hash has changed.
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


class NotOnIRS(Exception):
    """The IRS returned 404: this document doesn't exist for this year."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, session: requests.Session, retries: int = 3) -> None:
    """Download url to dest, retrying temporary failures with backoff."""
    for attempt in range(1, retries + 1):
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
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt}/{retries - 1} in {wait}s ({exc})")
            time.sleep(wait)


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
                if args.force or not dest.exists():
                    print(f"{label}: downloading")
                    download(doc.url, dest, session)
                    time.sleep(1)  # be polite to irs.gov
                else:
                    print(f"{label}: already on disk")
                conn.execute(UPSERT_SQL, manifest_row(doc, dest))
                conn.commit()
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
