-- TaxLens schema, step 2: parsed line-level records.

-- One row per parsed unit: a numbered form line, other form text, or a
-- chunk of an instructions/publication section.
CREATE TABLE IF NOT EXISTS line_records (
    record_id    BIGSERIAL PRIMARY KEY,
    doc_id       TEXT  NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    seq          INT   NOT NULL,                  -- reading order within the document
    record_type  TEXT  NOT NULL
                 CHECK (record_type IN ('form_line', 'form_text', 'section')),
    line_label   TEXT,                            -- e.g. '1a', '12' (form lines only)
    heading_path TEXT,                            -- e.g. 'Income > Line 1a > Wages'
    text         TEXT  NOT NULL,
    page         INT   NOT NULL,                  -- 1-based page where the record starts
    boxes        JSONB NOT NULL,                  -- [{page, x0, top, x1, bottom}, ...]
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS line_records_doc_label_idx ON line_records (doc_id, line_label);

-- Which version of the parser produced each document's records, and from
-- which exact file. Lets the parser skip documents that haven't changed.
CREATE TABLE IF NOT EXISTS parse_runs (
    doc_id         TEXT PRIMARY KEY REFERENCES documents (doc_id) ON DELETE CASCADE,
    sha256         CHAR(64)    NOT NULL,
    parser_version TEXT        NOT NULL,
    record_count   INT         NOT NULL,
    seconds        REAL        NOT NULL,
    parsed_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
