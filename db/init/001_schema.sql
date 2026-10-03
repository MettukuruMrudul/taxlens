-- TaxLens schema, step 1: the document manifest.
-- Runs automatically the first time the Postgres container starts.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per IRS PDF per tax year.
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,                 -- e.g. 'f1040-2021'
    form_code   TEXT        NOT NULL,             -- e.g. 'f1040', 'i1040gi', 'p17'
    doc_type    TEXT        NOT NULL
                CHECK (doc_type IN ('form', 'instructions', 'publication')),
    title       TEXT        NOT NULL,
    tax_year    INT         NOT NULL CHECK (tax_year BETWEEN 2000 AND 2100),
    source_url  TEXT        NOT NULL,
    local_path  TEXT        NOT NULL,             -- relative to the project folder
    sha256      CHAR(64)    NOT NULL,             -- detects revised PDFs
    size_bytes  BIGINT      NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (form_code, tax_year)
);

CREATE INDEX IF NOT EXISTS documents_tax_year_idx ON documents (tax_year);
