-- TaxLens schema, step 3: the year-aware index, year-specific figures, and
-- evaluation results.

-- Each distinct piece of text, stored once, with the tax years it applies to.
-- Rebuilt from line_records by: python -m taxlens.index.build
CREATE TABLE IF NOT EXISTS index_units (
    unit_id      BIGSERIAL PRIMARY KEY,
    form_code    TEXT  NOT NULL,
    record_type  TEXT  NOT NULL,
    line_label   TEXT,
    heading_path TEXT,
    text         TEXT  NOT NULL,
    valid_from   INT   NOT NULL,
    valid_to     INT   NOT NULL CHECK (valid_to >= valid_from),
    sources      JSONB NOT NULL          -- {"2023": record_id, ...}: what to cite per year
);
CREATE INDEX IF NOT EXISTS index_units_years_idx ON index_units (valid_from, valid_to);

-- Dollar amounts that change every year, each traceable to its IRS text.
CREATE TABLE IF NOT EXISTS year_figures (
    parameter        TEXT   NOT NULL,     -- e.g. 'standard_deduction'
    filing_status    TEXT   NOT NULL,     -- e.g. 'single_or_mfs'
    tax_year         INT    NOT NULL,
    value            BIGINT NOT NULL,
    source_record_id BIGINT REFERENCES line_records (record_id) ON DELETE SET NULL,
    source_text      TEXT   NOT NULL,
    PRIMARY KEY (parameter, filing_status, tax_year)
);

-- One row per evaluation run, and one row per question in that run.
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      BIGSERIAL PRIMARY KEY,
    system      TEXT        NOT NULL,
    benchmark   TEXT        NOT NULL,
    n_questions INT         NOT NULL,
    metrics     JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS eval_items (
    run_id          BIGINT  NOT NULL REFERENCES eval_runs (run_id) ON DELETE CASCADE,
    question_id     TEXT    NOT NULL,
    detected_year   INT,
    hit_at_1        BOOLEAN NOT NULL,
    recall_at_5     BOOLEAN NOT NULL,
    wrong_year_at_1 BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, question_id)
);
