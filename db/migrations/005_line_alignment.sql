-- TaxLens schema, step 5: cross-year line alignment table.

-- Tracks which line records across adjacent tax years represent the same
-- logical form line, enabling stable Canonical_Line_IDs even when the IRS
-- renumbers lines between years.
-- Populated by: python -m taxlens.index.align
CREATE TABLE IF NOT EXISTS line_alignment (
    alignment_id       BIGSERIAL PRIMARY KEY,
    canonical_line_id  TEXT  NOT NULL,                        -- stable ID shared by all matched lines in a group
    form_code          TEXT  NOT NULL,                        -- e.g. '1040', '1040s1'
    record_id_a        BIGINT NOT NULL REFERENCES line_records (record_id) ON DELETE CASCADE,
    record_id_b        BIGINT NOT NULL REFERENCES line_records (record_id) ON DELETE CASCADE,
    year_a             INT   NOT NULL,
    year_b             INT   NOT NULL,
    similarity_score   REAL  NOT NULL,                        -- weighted combination of cosine, edit, position, section
    match_type         TEXT  NOT NULL
                       CHECK (match_type IN ('matched', 'unmatched')),
    UNIQUE (record_id_a, record_id_b)                         -- prevent duplicate alignment pairs
);

CREATE INDEX IF NOT EXISTS line_alignment_canonical_idx ON line_alignment (canonical_line_id);
