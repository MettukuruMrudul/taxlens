-- Migration 008: ensure eval_runs and eval_items have all required columns.
--
-- eval_runs and eval_items were created in 003_index_and_eval.sql.
-- This migration adds columns that are needed for answer-grading and
-- code-version tracking but were not present in the original schema.
-- All statements use IF NOT EXISTS so the migration is idempotent.

-- Track which code commit produced the run.
ALTER TABLE eval_runs
    ADD COLUMN IF NOT EXISTS git_sha TEXT;

-- Store the generated answer and both grading verdicts per item.
ALTER TABLE eval_items
    ADD COLUMN IF NOT EXISTS answer      TEXT,
    ADD COLUMN IF NOT EXISTS grade       TEXT,
    ADD COLUMN IF NOT EXISTS judge_grade TEXT;
