"""Compute bi-encoder embeddings for every index unit (task 5.4).

    pip install -r requirements-ml.txt
    # in .env: EMBED_MODEL=BAAI/bge-base-en-v1.5   (768 dimensions, matches the schema)
    python -m taxlens.index.embed

Run after ``python -m taxlens.index.build`` (a rebuild clears the vectors).
The HNSW index (migration 004) updates automatically as rows are written.
The API loads these vectors at start-up and fuses dense and BM25 rankings
with RRF.
"""
from __future__ import annotations

import sys
import time

from taxlens import config
from taxlens.db import connect
from taxlens.retrieve.dense import get_encoder, to_pgvector
from taxlens.retrieve.search import index_text

BATCH = 256


def main() -> int:
    if not config.EMBED_MODEL:
        print("Set EMBED_MODEL in .env first (e.g. BAAI/bge-base-en-v1.5).")
        return 2
    encoder = get_encoder()
    if encoder is None:
        print("Could not load the embedding model. Did you run: pip install -r requirements-ml.txt ?")
        return 1
    if encoder.dim != 768:
        print(f"{config.EMBED_MODEL} produces {encoder.dim}-d vectors but index_units.embedding is "
              "vector(768). Pick a 768-d model or add a migration that changes the column.")
        return 1

    with connect() as conn:
        rows = conn.execute("SELECT unit_id, heading_path, line_label, text FROM index_units "
                            "ORDER BY unit_id").fetchall()
        if not rows:
            print("index_units is empty. Run: python -m taxlens.index.build")
            return 1
        start = time.perf_counter()
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            vecs = encoder.encode([index_text(hp, ll, tx) for _, hp, ll, tx in batch])
            with conn.cursor() as cur:
                cur.executemany("UPDATE index_units SET embedding = %s::vector WHERE unit_id = %s",
                                [(to_pgvector(v), uid) for (uid, *_), v in zip(batch, vecs)])
            conn.commit()
            print(f"  embedded {min(i + BATCH, len(rows))}/{len(rows)}")
        secs = time.perf_counter() - start
    print(f"Embedded {len(rows)} units with {config.EMBED_MODEL} in {secs:.1f}s "
          f"({len(rows) / max(secs, 1e-9):.0f} units/s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
