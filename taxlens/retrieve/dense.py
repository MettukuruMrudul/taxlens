"""Optional neural retrieval models: bi-encoder and cross-encoder.

Both are off unless configured in .env (EMBED_MODEL, RERANK_MODEL) and
``pip install -r requirements-ml.txt`` has been run. Everything else in
TaxLens works without them; BM25 + year filter + line lookup is the
always-on baseline.

* ``Encoder`` wraps a sentence-transformers bi-encoder. Vectors are
  L2-normalised, so a dot product is the cosine similarity.
* ``load_reranker`` returns ``reranker(query, texts) -> scores`` for
  YearAwareRetriever.
* ``pgvector_search`` is the database-side dense search with iterative
  over-fetch: pgvector's HNSW index can't pre-filter by tax year, so ask
  for 4k neighbours, filter by year, and double the request until k
  year-valid units are found (or 500 have been examined).
"""
from __future__ import annotations

from functools import lru_cache

from taxlens import config

INITIAL_FETCH_FACTOR = 4
MAX_EF = 500


class Encoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 500)

    def encode_query(self, text: str):
        # bge-style models expect an instruction prefix on queries only
        prefix = "Represent this sentence for searching relevant passages: " \
            if "bge" in self.model_name.lower() else ""
        return self.encode([prefix + text])[0]


@lru_cache(maxsize=1)
def get_encoder() -> Encoder | None:
    if not config.EMBED_MODEL:
        return None
    try:
        return Encoder(config.EMBED_MODEL)
    except Exception as e:
        print(f"Embedding model unavailable ({e}); dense retrieval off")
        return None


@lru_cache(maxsize=1)
def load_reranker():
    if not config.RERANK_MODEL:
        return None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(config.RERANK_MODEL)
    except Exception as e:
        print(f"Re-ranker unavailable ({e}); re-ranking off")
        return None

    def rerank(query: str, texts: list[str]):
        return model.predict([(query, t) for t in texts])
    return rerank


def parse_vector(value) -> list[float]:
    """pgvector values arrive as '[0.1,0.2,...]' strings without the pgvector adapter."""
    if value is None:
        return []
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",") if x]
    return list(value)


def load_embeddings(conn):
    """Unit embeddings as an (n_units, dim) array aligned with load_units()
    order (unit_id), or None if any unit lacks an embedding."""
    import numpy as np
    rows = conn.execute("SELECT embedding::text FROM index_units ORDER BY unit_id").fetchall()
    if not rows or any(r[0] is None for r in rows):
        return None
    return np.asarray([parse_vector(r[0]) for r in rows], dtype=np.float32)


def to_pgvector(vec) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def pgvector_search(conn, query_vector, year: int, k: int = 20) -> list[tuple[int, float]]:
    """(unit_id, cosine similarity) for the k nearest units valid in `year`."""
    q = to_pgvector(query_vector)
    ef = INITIAL_FETCH_FACTOR * k
    while True:
        conn.execute(f"SET LOCAL hnsw.ef_search = {min(ef, 1000)}")
        rows = conn.execute(
            "SELECT unit_id, valid_from, valid_to, 1 - (embedding <=> %s::vector) AS sim "
            "FROM index_units WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> %s::vector LIMIT %s", (q, q, ef)).fetchall()
        filtered = [(uid, float(sim)) for uid, vf, vt, sim in rows if vf <= year <= vt]
        if len(filtered) >= k or ef >= MAX_EF or len(rows) < ef:
            return filtered[:k]
        ef = min(ef * 2, MAX_EF)
