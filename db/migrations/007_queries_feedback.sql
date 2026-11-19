-- TaxLens schema, step 7: query logging and user feedback.
-- Supports Requirements 12.10 (feedback per answer) and 12.11 (conversation history).

-- One row per question submitted by a user, capturing pipeline metadata for
-- history display, caching telemetry, and performance monitoring.
CREATE TABLE IF NOT EXISTS queries_log (
    query_id   BIGSERIAL    PRIMARY KEY,
    user_id    TEXT         NOT NULL,
    question   TEXT         NOT NULL,
    tax_year   INT,
    intent_type TEXT,
    cache_hit  BOOLEAN      NOT NULL DEFAULT FALSE,
    abstained  BOOLEAN      NOT NULL DEFAULT FALSE,
    latency_ms INT,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Index to support GET /v1/history: paginated query history per user ordered
-- by most recent first.
CREATE INDEX IF NOT EXISTS queries_log_user_idx
    ON queries_log (user_id, created_at DESC);

-- One row per thumbs-up / thumbs-down submitted via POST /v1/feedback.
-- query_id is nullable so feedback rows survive if the parent query row is
-- deleted; rating is constrained to the API contract values (+1 / -1).
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id BIGSERIAL    PRIMARY KEY,
    query_id    BIGINT       REFERENCES queries_log (query_id) ON DELETE SET NULL,
    user_id     TEXT         NOT NULL,
    rating      SMALLINT     NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
