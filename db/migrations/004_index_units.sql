-- TaxLens schema, step 4: add vector embedding column and HNSW index to
-- index_units for semantic retrieval.

-- Add the embedding column if it doesn't already exist.
ALTER TABLE index_units
    ADD COLUMN IF NOT EXISTS embedding vector(768);

-- HNSW index for fast approximate nearest-neighbour search using cosine
-- distance.  m=16 controls the graph connectivity (higher → better recall,
-- more memory); ef_construction=200 controls build-time search quality.
CREATE INDEX IF NOT EXISTS index_units_hnsw_idx
    ON index_units
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
