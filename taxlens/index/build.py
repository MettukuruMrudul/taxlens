"""Build the year-aware index and the year-specific figures table.

Run from the project folder (after run_parser):
    python -m taxlens.index.build

Rebuilds index_units from line_records, and refreshes year_figures. Run it
again whenever documents are re-parsed.
"""
import datetime
import sys
from collections import defaultdict

import redis

from psycopg.types.json import Jsonb

from taxlens.config import REDIS_URL
from taxlens.db import connect
from taxlens.index.figures import extract_standard_deduction
from taxlens.index.intervals import Unit, build_units, is_boilerplate, size_reduction

def _emit_index_version(version: str) -> None:
    """Write the new index version to Redis so the cache can be invalidated."""
    try:
        r = redis.from_url(REDIS_URL)
        r.set("taxlens:index_version", version)
    except Exception as e:
        print(f"Warning: could not write index version to Redis: {e}")


LOAD_SQL = """
SELECT r.record_id, d.form_code, d.tax_year, r.record_type, r.line_label, r.heading_path, r.text
FROM line_records r JOIN documents d USING (doc_id)
ORDER BY d.form_code, d.tax_year, r.seq
"""


def load_records(conn) -> list[dict]:
    return [
        {"ref": rid, "form_code": fc, "tax_year": y, "record_type": rt,
         "line_label": ll, "heading_path": hp, "text": tx}
        for rid, fc, y, rt, ll, hp, tx in conn.execute(LOAD_SQL)
    ]


def load_units(conn) -> list[Unit]:
    rows = conn.execute(
        "SELECT form_code, record_type, line_label, heading_path, text, valid_from, valid_to, sources "
        "FROM index_units ORDER BY unit_id")
    return [Unit(fc, rt, ll, hp, tx, vf, vt, {int(k): v for k, v in src.items()})
            for fc, rt, ll, hp, tx, vf, vt, src in rows]


def main() -> int | str:
    with connect() as conn:
        records = load_records(conn)
        conn.commit()
        if not records:
            print("No line_records yet. Run: python -m taxlens.ingest.run_parser")
            return 1
        n_all = len(records)
        records = [r for r in records if not is_boilerplate(r["text"])]
        print(f"Skipped {n_all - len(records)} page headers/footers (e.g. 'Form 1040 (2021) Page 2').")

        units = build_units(records)
        figures = extract_standard_deduction([r for r in records if r["form_code"] == "f1040"])

        with conn.transaction():
            conn.execute("TRUNCATE index_units RESTART IDENTITY")
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO index_units (form_code, record_type, line_label, heading_path, text,"
                    " valid_from, valid_to, sources) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [(u.form_code, u.record_type, u.line_label, u.heading_path, u.text,
                      u.valid_from, u.valid_to, Jsonb({str(y): ref for y, ref in u.sources.items()}))
                     for u in units])
                conn.execute("DELETE FROM year_figures WHERE parameter = 'standard_deduction'")
                cur.executemany(
                    "INSERT INTO year_figures (parameter, filing_status, tax_year, value,"
                    " source_record_id, source_text) VALUES (%s, %s, %s, %s, %s, %s)",
                    [(f.parameter, f.filing_status, f.tax_year, f.value, f.source_ref, f.source_text)
                     for f in figures])

    index_version = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    _emit_index_version(index_version)

    by_form_r, by_form_u = defaultdict(int), defaultdict(int)
    for r in records:
        by_form_r[r["form_code"]] += 1
    for u in units:
        by_form_u[u.form_code] += 1
    print(f"{'document':<10} {'records':>8} {'units':>7} {'saved':>7}")
    for fc in sorted(by_form_r):
        print(f"{fc:<10} {by_form_r[fc]:>8} {by_form_u[fc]:>7} {size_reduction(by_form_r[fc], by_form_u[fc]):>7.1%}")
    print(f"{'TOTAL':<10} {len(records):>8} {len(units):>7} {size_reduction(len(records), len(units)):>7.1%}")
    print(f"\nyear_figures: {len(figures)} standard deduction amounts "
          f"for years {sorted({f.tax_year for f in figures})}")

    reduction = size_reduction(len(records), len(units))
    if reduction < 0.30:
        print("WARNING: index reduction < 30% — check for duplicate records")

    print(f"\nindex_version: {index_version}")
    return index_version


if __name__ == "__main__":
    result = main()
    # CLI compatibility: exit 0 on success (version string), exit 1 on error (int 1)
    sys.exit(0 if isinstance(result, str) else result)
