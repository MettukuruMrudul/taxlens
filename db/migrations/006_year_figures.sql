-- TaxLens migration 006: ensure year_figures table exists with all required columns.
--
-- The year_figures table was introduced in 003_index_and_eval.sql.  This migration
-- is a safe no-op for databases that already have the complete schema and adds any
-- missing columns (specifically source_text) for databases created before it was
-- added.
--
-- Table definition (for reference):
--   CREATE TABLE year_figures (
--       parameter        TEXT   NOT NULL,
--       filing_status    TEXT   NOT NULL,
--       tax_year         INT    NOT NULL,
--       value            BIGINT NOT NULL,
--       source_record_id BIGINT REFERENCES line_records (record_id) ON DELETE SET NULL,
--       source_text      TEXT   NOT NULL,
--       PRIMARY KEY (parameter, filing_status, tax_year)
--   );

-- Ensure the table exists (no-op if already present from 003_index_and_eval.sql).
CREATE TABLE IF NOT EXISTS year_figures (
    parameter        TEXT   NOT NULL,
    filing_status    TEXT   NOT NULL,
    tax_year         INT    NOT NULL,
    value            BIGINT NOT NULL,
    source_record_id BIGINT REFERENCES line_records (record_id) ON DELETE SET NULL,
    source_text      TEXT   NOT NULL DEFAULT '',
    PRIMARY KEY (parameter, filing_status, tax_year)
);

-- Ensure source_text column exists with a safe default for legacy databases that
-- were created without it.  This is a no-op when the column is already present.
ALTER TABLE year_figures
    ADD COLUMN IF NOT EXISTS source_text TEXT NOT NULL DEFAULT '';
