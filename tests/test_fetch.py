"""Unit tests for taxlens/ingest/fetch.py — task 2.2 and 2.5.

All tests are pure-Python: no network calls, no database, no filesystem side-
effects beyond temporary directories provided by pytest's `tmp_path` fixture.
"""
import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from taxlens.ingest.fetch import (
    RETRY_DELAYS,
    NotOnIRS,
    db_sha256,
    download,
    sha256_of,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, content: bytes = b"%PDF-1.4 fake") -> MagicMock:
    """Return a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# sha256_of
# ---------------------------------------------------------------------------

def test_sha256_of_matches_known_hash(tmp_path):
    data = b"Hello, TaxLens!"
    f = tmp_path / "file.bin"
    f.write_bytes(data)
    assert sha256_of(f) == _sha256(data)


def test_sha256_of_large_file_matches_onshot(tmp_path):
    # Larger than the 1 MiB streaming chunk to exercise the loop
    data = b"x" * (2 * 1024 * 1024 + 7)
    f = tmp_path / "big.bin"
    f.write_bytes(data)
    assert sha256_of(f) == _sha256(data)


# ---------------------------------------------------------------------------
# db_sha256
# ---------------------------------------------------------------------------

def test_db_sha256_returns_stored_hash():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("abc123",)
    result = db_sha256(conn, "f1040-2023")
    assert result == "abc123"
    conn.execute.assert_called_once()


def test_db_sha256_returns_none_when_not_found():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None
    assert db_sha256(conn, "missing-doc") is None


# ---------------------------------------------------------------------------
# download — 404 → NotOnIRS (per-document failure, run continues)
# ---------------------------------------------------------------------------

def test_download_raises_not_on_irs_on_404(tmp_path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(status_code=404)

    with pytest.raises(NotOnIRS):
        download("https://example.com/f1040--2021.pdf", tmp_path / "out.pdf", session)

    # Only one attempt — 404 is not retried
    assert session.get.call_count == 1


def test_download_404_leaves_no_part_file(tmp_path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(status_code=404)

    dest = tmp_path / "out.pdf"
    with pytest.raises(NotOnIRS):
        download("https://example.com/f1040--2021.pdf", dest, session)

    assert not dest.with_suffix(".part").exists()
    assert not dest.exists()


# ---------------------------------------------------------------------------
# download — retry logic: exactly RETRY_DELAYS waits then raises
# ---------------------------------------------------------------------------

def test_download_retries_up_to_max_attempts_then_raises(tmp_path):
    """Transient failures exhaust all attempts (1 + len(RETRY_DELAYS) = 3)."""
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("timeout")

    with patch("taxlens.ingest.fetch.time.sleep") as mock_sleep, \
         pytest.raises(requests.ConnectionError):
        download("https://example.com/f1040--2021.pdf", tmp_path / "out.pdf", session)

    total_attempts = 1 + len(RETRY_DELAYS)  # 3
    assert session.get.call_count == total_attempts

    # Verify the sleep delays match RETRY_DELAYS exactly
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_calls == RETRY_DELAYS


def test_retry_delays_are_2_and_4():
    """Contract: exactly two retries with waits of 2 s and 4 s."""
    assert RETRY_DELAYS == [2, 4]


def test_download_succeeds_on_second_attempt(tmp_path):
    """First attempt fails, second succeeds — no exception raised."""
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        requests.ConnectionError("first failure"),
        _make_response(200, b"%PDF-1.4 ok"),
    ]

    dest = tmp_path / "out.pdf"
    with patch("taxlens.ingest.fetch.time.sleep"):
        download("https://example.com/f1040--2021.pdf", dest, session)

    assert dest.exists()
    assert dest.read_bytes() == b"%PDF-1.4 ok"


# ---------------------------------------------------------------------------
# download — atomic write (.part → rename) and PDF-magic validation
# ---------------------------------------------------------------------------

def test_download_success_creates_dest_not_part(tmp_path):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _make_response(200, b"%PDF-1.4 fake content")

    dest = tmp_path / "f1040--2021.pdf"
    download("https://example.com/f1040--2021.pdf", dest, session)

    assert dest.exists()
    assert not dest.with_suffix(".part").exists()


def test_download_non_pdf_raises_value_error(tmp_path):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [
        _make_response(200, b"not a pdf"),
        _make_response(200, b"still not"),
        _make_response(200, b"nope"),
    ]

    with patch("taxlens.ingest.fetch.time.sleep"), \
         pytest.raises(ValueError, match="not a PDF"):
        download("https://example.com/f1040--2021.pdf", tmp_path / "out.pdf", session)


# ---------------------------------------------------------------------------
# Idempotent skip logic (tested at the unit level via db_sha256 + sha256_of)
# ---------------------------------------------------------------------------

def test_idempotent_skip_when_hashes_match(tmp_path):
    """When disk hash == DB hash the caller should skip download and DB upsert."""
    data = b"%PDF-1.4 real content"
    dest = tmp_path / "f1040--2021.pdf"
    dest.write_bytes(data)

    disk_hash = sha256_of(dest)

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (disk_hash,)

    stored = db_sha256(conn, "f1040-2021")

    # Caller's skip condition
    should_skip = stored is not None and sha256_of(dest) == stored
    assert should_skip is True


def test_no_skip_when_disk_hash_differs_from_db(tmp_path):
    """Corrupted on-disk file (different hash) must NOT be skipped."""
    dest = tmp_path / "f1040--2021.pdf"
    dest.write_bytes(b"%PDF-1.4 corrupted")

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("deadbeef" * 8,)  # wrong hash

    stored = db_sha256(conn, "f1040-2021")
    should_skip = stored is not None and sha256_of(dest) == stored
    assert should_skip is False


def test_no_skip_when_not_in_db(tmp_path):
    """File on disk but not yet in DB must NOT be skipped."""
    dest = tmp_path / "f1040--2021.pdf"
    dest.write_bytes(b"%PDF-1.4 real")

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    stored = db_sha256(conn, "f1040-2021")
    should_skip = stored is not None and sha256_of(dest) == stored
    assert should_skip is False
