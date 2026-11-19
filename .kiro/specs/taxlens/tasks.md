# Implementation Plan: TaxLens

## Overview

Build a tax-year-aware RAG question-answering system over official IRS documents. The pipeline flows: ETL (fetch → parse → index → figures → align) → Serving (year detect → intent → retrieve → generate → verify) → API (FastAPI + Redis) → Frontend (Next.js). Every component enforces tax year as a first-class constraint. MVP scope covers weeks 1–8 (C1–C10, C12, C14, C15 core); advanced components (C11, C13, C16, fine-tuning, C8/C9 full models) follow in weeks 9–16.

---

## Tasks

- [x] 1. Database migrations — new tables
  - [x] 1.1 Write migration `004_index_units.sql` for `index_units` table
    - Add `index_units` table with `unit_id`, `form_code`, `record_type`, `line_label`, `heading_path`, `text`, `valid_from`, `valid_to`, `sources JSONB`, `embedding vector(768)`
    - Add `index_units_years_idx` on `(valid_from, valid_to)`
    - Add HNSW index with `m=16, ef_construction=200`
    - Update `taxlens/migrate.py` to apply this migration
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 1.2 Write migration `005_line_alignment.sql` for `line_alignment` table
    - Add `line_alignment` table with `alignment_id`, `canonical_line_id`, `form_code`, `record_id_a`, `record_id_b`, `year_a`, `year_b`, `similarity_score`, `match_type`
    - Add `line_alignment_canonical_idx` on `canonical_line_id`
    - _Requirements: 11.3_

  - [x] 1.3 Write migration `006_year_figures.sql` for `year_figures` table
    - Add `year_figures` table with `(parameter, filing_status, tax_year)` PRIMARY KEY, `value BIGINT`, `source_record_id`, `source_text`
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 1.4 Write migration `007_queries_feedback.sql` for `queries_log` and `feedback` tables
    - Add `queries_log` with `query_id`, `user_id`, `question`, `tax_year`, `intent_type`, `cache_hit`, `abstained`, `latency_ms`, `created_at`
    - Add `feedback` with `feedback_id`, `query_id`, `user_id`, `rating SMALLINT`, `comment`, `created_at`
    - Add `queries_log_user_idx` on `(user_id, created_at DESC)`
    - _Requirements: 12.10, 12.11_

  - [x] 1.5 Write migration `008_eval_tables.sql` for `eval_runs` and `eval_items` tables
    - Add `eval_runs` with `run_id`, `system`, `benchmark`, `n_questions`, `metrics JSONB`, `created_at`
    - Add `eval_items` with `(run_id, question_id)` PK, `detected_year`, `hit_at_1`, `recall_at_5`, `wrong_year_at_1`
    - _Requirements: 10.5_

- [x] 2. C1 — Fetcher hardening
  - [x] 2.1 Harden `taxlens/ingest/fetch.py`: streaming SHA-256, atomic write, PDF-magic validation
    - Implement `sha256_of(path)` reading in 1 MiB streaming chunks
    - Implement write-then-rename (`file.part` → `file`) atomic pattern
    - Validate `%PDF` magic bytes immediately after download; delete `.part` and raise `ValueError` if absent
    - _Requirements: 1.3, 1.5, 1.7_

  - [x] 2.2 Harden `taxlens/ingest/fetch.py`: idempotent skip and retry logic
    - Before any HTTP call, check SHA-256 of existing on-disk file against `documents` table; skip if matching
    - Implement exponential back-off retry: up to 3 attempts, waits of 2 s and 4 s
    - Handle HTTP 404 as per-document failure: log, record in `failed` list, continue run
    - _Requirements: 1.2, 1.4, 1.8_

  - [x] 2.3 Harden `taxlens/ingest/corpus.py`: MVP corpus definition and IRS-prior URL derivation
    - Define MVP_Corpus as Form 1040, Schedules 1–3, A, B, Form 1040 general instructions, Schedule A instructions, Publication 17 for tax years 2021–2024 (36 documents)
    - Derive all download URLs exclusively from `IRS_PRIOR_BASE = "https://www.irs.gov/pub/irs-prior"` — no fallback to current-forms folder
    - Implement upsert SQL for `documents` table updating only when `sha256` changes
    - _Requirements: 1.1, 1.6, 1.9, 1.10_

  - [ ]* 2.4 Write property tests for fetcher (P3, P4)
    - **Property 3: Non-PDF rejection leaves no file on disk** — use `st.binary().filter(lambda b: not b.startswith(b"%PDF"))`
    - **Validates: Requirements 1.3**
    - **Property 4: Streaming SHA-256 matches one-shot SHA-256** — use `st.binary()`
    - **Validates: Requirements 1.7**

  - [ ]* 2.5 Write unit tests for fetcher
    - Test: HTTP 404 → document added to failed list, run continues
    - Test: retry count reaches 3 before failure is recorded
    - Test: atomic write creates `.part` file then renames; no `.part` on success
    - Test: URL derivation never contains "irs-current" substring
    - _Requirements: 1.1, 1.4, 1.5, 1.8, 1.10_

- [x] 3. C2 — Parser hardening
  - [x] 3.1 Implement two-column form parser in `taxlens/ingest/parse.py`
    - Identify line-number column: leftmost x-coordinate of tokens matching `\d{1,2}[a-z]?` immediately followed by capitalized text, appearing in ≥ 2 rows
    - Split each row at `num_x − 3` into margin and main area
    - Produce `form_line` records (numbered lines with `line_label`) and `form_text` records (unnumbered blocks)
    - Strip leader tokens (dots, ellipses, bullets, arrows, box labels) and visual decoration (TIP/CAUTION icons, page-footer text)
    - Exclude words with bounding boxes outside page boundaries
    - _Requirements: 2.1, 2.2, 2.7, 2.9, 2.10_

  - [x] 3.2 Implement recursive XY-cut prose parser in `taxlens/ingest/parse.py`
    - Detect vertical gutters (≥ 8 pt wide) to split columns; detect horizontal gaps (≥ 4.5 pt tall) to split blocks
    - Read blocks left-column-first, top-to-bottom within each column
    - Detect bold-sans headings; maintain heading path stack; tag each section record with `heading_path`
    - Chunk body text at sentence boundaries, max 1,200 characters per `section` record
    - _Requirements: 2.3, 2.8_

  - [x] 3.3 Implement parser skip logic and atomic DB writes in `taxlens/ingest/parse.py`
    - Skip a document when both file SHA-256 and `PARSER_VERSION` match the `parse_runs` record
    - Delete then re-insert all records for a document inside a single `conn.transaction()`
    - Store `doc_id`, `seq` (1-based), `record_type`, `line_label`, `heading_path`, `text`, `page`, `boxes` for every record
    - Enforce `UNIQUE (doc_id, seq)` — produce zero duplicates
    - _Requirements: 2.4, 2.5, 2.6, 2.12_

  - [ ]* 3.4 Write property tests for parser (P11, P12, P13, P14)
    - **Property 11: Section records ≤ 1,200 characters** — run against real PDFs from `data/raw/`
    - **Validates: Requirements 2.3**
    - **Property 12: Bounding boxes are well-formed** — `x0 < x1`, `top < bottom`, `page >= 1`
    - **Validates: Requirements 2.2, 2.4, 7.4**
    - **Property 13: Seq is a contiguous 1-based range per document**
    - **Validates: Requirements 2.4, 2.12**
    - **Property 14: Out-of-bounds words excluded** — use synthetic page dicts
    - **Validates: Requirements 2.9**

  - [ ]* 3.5 Write unit tests for parser
    - Test: form parser produces `form_line` records with correct `line_label` for known 1040 pages
    - Test: prose parser produces `section` records with correct `heading_path`
    - Test: decoration tokens (TIP, CAUTION, page numbers) absent from all records
    - Test: re-parse with unchanged SHA-256 + version skips document (no DB writes)
    - _Requirements: 2.1, 2.5, 2.7, 2.10_

- [x] 4. Checkpoint — ETL ingestion foundation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. C3 — Validity-interval index builder
  - [x] 5.1 Implement `normalize()` and `build_units()` in `taxlens/index/intervals.py`
    - `normalize(text)`: NFKC → smart-quote substitution (curly → straight) → whitespace collapse
    - `build_units()`: group records by `(form_code, record_type, line_label, normalize(text))`; sort by `tax_year`; find contiguous runs; produce one `Unit` per run with `valid_from`, `valid_to`, `sources: dict[int, record_id]`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 5.2 Write property tests for intervals (P5–P9)
    - **Property 5: Normalization is idempotent** — `st.text()`
    - **Validates: Requirements 3.1**
    - **Property 6: Consecutive identical text merges into one unit** — sorted contiguous year lists
    - **Validates: Requirements 3.2**
    - **Property 7: Text change closes and reopens a unit** — adjacent unit pairs with different text
    - **Validates: Requirements 3.3**
    - **Property 8: Non-consecutive years produce separate units** — records with a gap year
    - **Validates: Requirements 3.4**
    - **Property 9: Sources dict covers every year in the interval**
    - **Validates: Requirements 3.5**

  - [-] 5.3 Implement atomic index rebuild in `taxlens/index/build.py`
    - `TRUNCATE index_units RESTART IDENTITY` + `INSERT` in a single transaction
    - Verify index reduction: count of `index_units` rows ≤ 70% of total line_records count (≥ 30% reduction)
    - Emit `index_version` string (ETL timestamp) and write it to Redis key `taxlens:index_version`
    - _Requirements: 3.7, 3.8, 13.6, 13.8_

  - [-] 5.4 Implement bi-encoder embedding step in `taxlens/index/build.py`
    - Call sentence-transformers API to embed each unit's text; store result in `embedding vector(768)` column
    - Build HNSW index with `m=16, ef_construction=200` after all embeddings are inserted
    - Log embedding count and timing
    - _Requirements: 6.1, 6.4_

  - [ ]* 5.5 Write unit tests for index builder
    - Test: atomic rebuild — truncate + insert in one transaction; concurrent reader sees complete old or new index
    - Test: 36-doc MVP corpus produces ≥ 30% fewer index_units than line_records
    - _Requirements: 3.7, 3.8_

- [ ] 6. C4 — Figures extractor
  - [-] 6.1 Implement `taxlens/index/figures.py`: standard deduction extraction
    - Scan Form 1040 `form_text` records for "Standard Deduction for"
    - Apply `FILING_STATUS_PATTERNS` regex dict to extract amounts for `single_or_mfs`, `mfj_or_qss`, `head_of_household` per tax year
    - Store each `Figure` with `source_ref` (record_id) and `source_text` (matched substring)
    - Upsert into `year_figures` table; enforce `(parameter, filing_status, tax_year)` uniqueness
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 6.2 Write property tests for figures extractor (P15, P16)
    - **Property 15: Figure uniqueness per (parameter, filing_status, tax_year)** — synthetic form record lists
    - **Validates: Requirements 4.2**
    - **Property 16: Every Figure has a non-null source** — real Form 1040 records
    - **Validates: Requirements 4.3**

  - [ ]* 6.3 Write unit tests for figures extractor
    - Test: correct standard deduction amounts extracted for each of 2021–2024
    - Test: `source_record_id` and `source_text` populated for every row
    - _Requirements: 4.1, 4.3, 4.5_

- [x] 7. C5 — Year detector hardening
  - [x] 7.1 Implement all 6 detection paths in `taxlens/retrieve/yeardetect.py`
    - Path 1: `explicit` parameter (UI year selector) → `source = "explicit"`, confidence 1.0
    - Path 2: filing-year phrases (e.g., "filing in Y", "Y filing season") → `year = Y - 1`, `source = "filing_year"`, confidence 0.9
    - Path 3: tax-year phrases (e.g., "tax year Y", "for Y", "my Y return") → `source = "tax_year"`, confidence 0.9
    - Path 4: single unambiguous year anywhere in text → `source = "tax_year"`, confidence 0.9
    - Path 5: relative phrases ("last year", "this year") resolved against `today` → `source = "relative"`, confidence 0.75
    - Path 6: no year signal → `year = None`, `source = "none"`, confidence 0.0
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 7.2 Write property tests for year detector (P17, P18, P19)
    - **Property 17: Filing-year phrases produce Tax_Year = stated_year − 1** — `st.integers(2020, 2040)` × filing-year templates
    - **Validates: Requirements 5.2**
    - **Property 18: Explicit tax-year phrases return that exact year** — `st.integers(2020, 2040)` × tax-year templates
    - **Validates: Requirements 5.3, 5.4**
    - **Property 19: Relative year phrases resolve against the provided date** — `st.dates()`
    - **Validates: Requirements 5.5**

  - [ ]* 7.3 Write unit tests for year detector
    - Test all 7 source types with concrete question strings (one test per path)
    - Test: `explicit` overrides any year in question text
    - Test: `year = None` returned when question has no year signal
    - _Requirements: 5.1–5.7_

- [ ] 8. C6 — Hybrid retrieval and re-ranking
  - [-] 8.1 Implement year-aware filtering and HNSW over-fetch in `taxlens/retrieve/search.py`
    - Apply SQL filter `valid_from <= year AND year <= valid_to` before ranking
    - Implement iterative over-fetch: start `ef = 4 * k`; if filtered results < k, double ef up to `max_ef = 500`; return `filtered[:k]`
    - _Requirements: 3.6, 6.2, 6.7_

  - [ ] 8.2 Implement BM25 + bi-encoder RRF hybrid retrieval in `taxlens/retrieve/search.py`
    - Compute BM25 scores over year-filtered candidates using `taxlens/retrieve/bm25.py`
    - Compute dense scores: `cosine(bi_encoder.encode(query), embeddings[candidates])`
    - Compute RRF: `score[i] = 1/(60 + rank_bm25[i]) + 1/(60 + rank_dense[i])`
    - _Requirements: 6.1, 6.2_

  - [~] 8.3 Implement line-label structured lookup in `taxlens/retrieve/search.py`
    - Detect `\bline\s+(\d{1,2}[a-z]?)\b` in query
    - Direct DB lookup for `form_line` unit with matching `line_label` and requested year
    - Inject matched unit at rank 0, above all BM25/dense results
    - _Requirements: 6.3_

  - [~] 8.4 Integrate ModernBERT cross-encoder re-ranking in `taxlens/retrieve/search.py`
    - Re-rank top-20 hybrid results: `scores = cross_encoder.predict([(query, hit.text) for hit in top20])`
    - Return top-k sorted by cross-encoder score
    - _Requirements: 6.4_

  - [ ]* 8.5 Write property tests for retriever (P10, P20, P21)
    - **Property 10: Year-aware retrieval never returns off-year units** — `st.integers(2021, 2024)`, random unit list
    - **Validates: Requirements 3.6, 6.2**
    - **Property 20: RRF scores are correctly computed** — `st.lists(st.integers())` two ranked lists
    - **Validates: Requirements 6.1**
    - **Property 21: Line-label structured lookup returns matching line first** — queries with injected "line N"
    - **Validates: Requirements 6.3**

  - [ ]* 8.6 Write unit tests for retriever
    - Test: over-fetch doubles ef until k results collected or max_ef reached
    - Test: year filter removes all units outside requested year
    - Test: BM25 baseline scores match known queries against small corpus
    - _Requirements: 6.1, 6.2, 6.7_

- [~] 9. Checkpoint — ETL + retrieval core complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. C7 — Answer generator
  - [~] 10.1 Create `taxlens/answer/generate.py` with structured JSON output
    - Create `taxlens/answer/` package with `__init__.py`
    - Implement `generate_answer(question, hits, figures, tax_year)` → JSON `{"sentences": [...], "disclaimer": "..."}`
    - Prompt structure: SYSTEM with tax year context + figures JSON + passages by record_id; USER with question
    - Enforce disclaimer: "TaxLens provides information from IRS publications for educational purposes only and is not tax advice."
    - _Requirements: 7.1, 7.3, 7.7_

  - [~] 10.2 Implement attribution retry loop in `taxlens/answer/generate.py`
    - Retry any sentence not supported by cited passage up to `MAX_ATTRIBUTION_RETRIES = 3`
    - Omit sentences that fail attribution after all retries
    - If all sentences fail attribution, trigger abstention
    - _Requirements: 7.2_

  - [~] 10.3 Implement streaming generator in `taxlens/answer/generate.py`
    - `generate_answer_stream()` yields sentence objects one at a time as they are produced
    - Each yielded sentence has `text` and `record_ids`
    - Connect to `taxlens/retrieve/search.py` to resolve `record_ids` → `boxes` from `line_records`
    - _Requirements: 7.4, 12.1_

  - [ ]* 10.4 Write property tests for answer generator (P24, P25, P26)
    - **Property 24: Every answer contains the required disclaimer**
    - **Validates: Requirements 7.7**
    - **Property 25: Answer JSON structure is valid for all inputs** — schema `{"sentences": [...], "disclaimer": string}`
    - **Validates: Requirements 7.1**
    - **Property 26: All cited record_ids have bounding boxes in line_records**
    - **Validates: Requirements 7.4**

  - [ ]* 10.5 Write unit tests for answer generator
    - Test: disclaimer text is present and unchanged in every response
    - Test: attribution retry calls LLM up to `MAX_ATTRIBUTION_RETRIES` then omits sentence
    - Test: figure values are sourced from `year_figures` table, not free-text passages
    - _Requirements: 7.2, 7.3, 7.7_

- [ ] 11. C8 — Verifier and abstention classifier
  - [~] 11.1 Create `taxlens/verify/verifier.py` with claim splitter and NLI check
    - Create `taxlens/verify/` package with `__init__.py`
    - Implement atomic claim splitter: prompt LLM to extract distinct factual claims from a sentence as a JSON list
    - Implement NLI entailment check: for each claim, score `(claim, each_passage)` with ModernBERT cross-encoder; claim is supported if any passage scores > `τ_nli = 0.7`
    - Flag unsupported claims; compute `unsupported_claim_rate`
    - _Requirements: 8.1, 8.2_

  - [~] 11.2 Create `taxlens/verify/abstain.py` with LightGBM abstention classifier
    - Feature vector: `max_retrieval_score`, `mean_retrieval_score`, `year_detection_confidence`, `intent_type_encoded` (one-hot), `n_supported_claims`, `n_passages_for_year`
    - Train LightGBM binary classifier on eval data; `abstain = predict_proba(features)[1] > τ_abstain`
    - Tune `τ_abstain` to achieve AUROC ≥ 0.94 on eval set
    - Produce selective-accuracy curve (accuracy vs. fraction answered as `τ_abstain` varies)
    - _Requirements: 8.3, 8.5, 8.7_

  - [ ]* 11.3 Write property tests for verifier (P27, P34)
    - **Property 27: Abstention response contains no generated sentences**
    - **Validates: Requirements 8.4**
    - **Property 34: Selective-accuracy curve is non-increasing**
    - **Validates: Requirements 8.7**

  - [ ]* 11.4 Write unit tests for verifier
    - Test: `unsupported_claim_rate` ≤ 1.7% ceiling enforced in output
    - Test: abstention message returned when `abstain = True`; no sentences in response
    - _Requirements: 8.4, 8.6_

- [ ] 12. C9 — Intent classifier
  - [~] 12.1 Create `taxlens/intent/classify.py` with three-model comparison
    - Create `taxlens/intent/` package with `__init__.py`
    - Implement `IntentResult` dataclass: `intent: str`, `confidence: float`
    - Implement Model A: TF-IDF + logistic regression
    - Implement Model B: XGBoost
    - Implement Model C: fine-tuned DistilBERT
    - Evaluate all three on labeled benchmark; deploy highest macro-F1 model
    - _Requirements: 9.1, 9.2, 9.3_

  - [~] 12.2 Wire intent routing logic in `taxlens/intent/classify.py`
    - `out_of_scope` → return scope-explanation message; do not invoke Answer Generator
    - `year = None` + any non-OOS intent → return year-clarification prompt; do not invoke Answer Generator
    - All other intents → proceed to retrieval
    - _Requirements: 9.4, 9.5_

  - [ ]* 12.3 Write property tests for intent classifier (P28, P29, P30)
    - **Property 28: Intent classifier output is always one of the seven valid types**
    - **Validates: Requirements 9.1**
    - **Property 29: Out-of-scope questions never invoke the Answer Generator**
    - **Validates: Requirements 9.4**
    - **Property 30: Year=None + non-OOS intent triggers year-clarification response**
    - **Validates: Requirements 9.5**

  - [ ]* 12.4 Write unit tests for intent classifier
    - Test: macro-F1 ≥ 0.85 on seed benchmark for deployed model
    - Test: `out_of_scope` question returns scope message without `sentences` field
    - _Requirements: 9.1, 9.2, 9.4_

- [~] 13. Checkpoint — Serving pipeline complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. C10 — Evaluation harness
  - [~] 14.1 Harden `taxlens/eval/retrieval_eval.py`: full metric suite with confidence intervals
    - Compute `recall@5`, `hit@1`, `wrong_year@1`, `year_detection_accuracy`, `answer_correctness`, `citation_precision`, `unsupported_claim_rate`
    - Compute 95% bootstrap confidence intervals for all metrics (2,000 bootstrap samples)
    - Persist aggregate metrics to `eval_runs` and per-question outcomes to `eval_items`
    - _Requirements: 10.2, 10.3, 10.5_

  - [~] 14.2 Implement McNemar test and LLM judge calibration in `taxlens/eval/retrieval_eval.py`
    - `mcnemar_exact(system_a_hits, system_b_hits)` — exact two-sided p-value for pairwise binary metric comparison
    - LLM judge calibration: compute Cohen's κ against human grades; target κ ≥ 0.81
    - _Requirements: 10.4, 10.6_

  - [~] 14.3 Implement CI gate and failure cluster analysis in `taxlens/eval/run_retrieval.py`
    - `GATE_RULES`: fail if `answer_correctness` drops > 2 pp vs previous `eval_runs` row, or `wrong_year_at_1` > 3%
    - Failure cluster analysis: group incorrect answers by `intent_type`, `tax_year`, and document
    - Report unseen-year generalization: compute correctness separately for TY2025
    - _Requirements: 10.7, 10.8, 10.9_

  - [ ]* 14.4 Write unit tests for evaluation harness
    - Test: CI gate fails build when `answer_correctness` drops > 2 pp
    - Test: CI gate fails build when `wrong_year_at_1` > 3%
    - Test: `eval_runs` row written with JSONB metrics after every run
    - _Requirements: 10.5, 10.7_

- [ ] 15. C11 — Line aligner
  - [~] 15.1 Create `taxlens/index/align.py` with weighted similarity score and Hungarian matching
    - Compute `sim = 0.4 * cosine(embed(A), embed(B)) + 0.3 * (1 - norm_edit_dist(A, B)) + 0.2 * (1 - |A.rel_pos - B.rel_pos|) + 0.1 * (same_section ? 1 : 0)`
    - Use `scipy.optimize.linear_sum_assignment(-sim_matrix)` for globally optimal one-to-one matching
    - Assign `canonical_line_id` to each matched group; record unmatched lines when `sim < τ_align = 0.55`
    - Upsert results into `line_alignment` table
    - _Requirements: 11.1, 11.2, 11.3, 11.5_

  - [ ]* 15.2 Write property tests for line aligner (P32, P33)
    - **Property 32: Line aligner similarity scores are in [0.0, 1.0]**
    - **Validates: Requirements 11.1**
    - **Property 33: Lines in the same alignment group share the same Canonical_Line_ID**
    - **Validates: Requirements 11.3**

  - [ ]* 15.3 Write unit tests for line aligner
    - Test: Hungarian algorithm produces globally optimal matching on small known matrix
    - Test: pairs below `τ_align = 0.55` are recorded as `match_type = "unmatched"`
    - Test: `canonical_line_id` is consistent across all rows for a matched group
    - _Requirements: 11.2, 11.3, 11.5_

- [ ] 16. C12 — FastAPI backend
  - [~] 16.1 Create `taxlens/api/` package scaffolding with JWT auth and rate limiting
    - Create `taxlens/api/__init__.py`, `taxlens/api/main.py`, `taxlens/api/auth.py`, `taxlens/api/models.py`
    - Implement `get_current_user` FastAPI dependency: `jwt.decode(token, SECRET_KEY, algorithms=["HS256"])`
    - Apply `Depends(get_current_user)` to all non-health endpoints
    - Implement `slowapi` rate limiter: 20 requests/minute on `/v1/ask` and `/v1/ask/stream`
    - `GET /health` returns `{"status": "ok"}` with no auth
    - _Requirements: 12.4, 12.5, 12.6_

  - [~] 16.2 Implement `POST /v1/ask` synchronous endpoint
    - Accept `{"question": str, "tax_year": int}`; validate tax year is in corpus range or raise 422
    - Check Redis cache key `taxlens:answer:{sha256(normalized_q|year|index_version)}`; return compressed cached answer on hit
    - On miss: run year detect → intent classify → retrieve → generate → verify pipeline
    - Log query to `queries_log` with `latency_ms`, `cache_hit`, `abstained`
    - Return JSON answer or abstention message
    - _Requirements: 12.4, 12.7, 12.8_

  - [~] 16.3 Implement `POST /v1/ask/stream` SSE endpoint
    - Implement `event_generator()` async generator yielding `data: {sentence_json}\n\n`; send `data: [DONE]\n\n` at end
    - Return `StreamingResponse(event_generator(), media_type="text/event-stream")`
    - Achieve TTFT ≤ 1 s: yield first sentence as soon as generator produces it, before full answer completes
    - Cache complete answer after streaming finishes; update `queries_log`
    - _Requirements: 12.1, 12.7, 12.9_

  - [~] 16.4 Implement `GET /v1/documents/{id}`, `GET /v1/history`, `POST /v1/feedback` endpoints
    - `GET /v1/documents/{id}`: return document metadata + signed PDF URL from `documents` table; 422 if not found
    - `GET /v1/history`: paginated `queries_log` rows for authenticated user, ordered by `created_at DESC`
    - `POST /v1/feedback`: insert into `feedback` table with `rating` (+1/-1) and optional `comment`; link to `query_id`
    - _Requirements: 12.4, 12.10, 12.11_

  - [~] 16.5 Implement Redis cache with compression and version invalidation
    - `cache_key(question, tax_year, index_version)`: normalize question, SHA-256 of `"{q}|{year}|{version}"`
    - zlib-compress (level 6) before SET; decompress on GET; TTL = 24 hours
    - On ETL index swap, write new timestamp to `taxlens:index_version` — invalidates all old cached answers
    - Graceful degradation: if Redis unavailable, bypass cache and log `cache_unavailable` counter
    - _Requirements: 12.7, 13.8_

  - [ ]* 16.6 Write property tests for API (P31)
    - **Property 31: Cache returns identical responses for identical inputs** — verify second request served from cache with `cache_hit = True`
    - **Validates: Requirements 12.7**

  - [ ]* 16.7 Write unit tests for FastAPI backend
    - Test: JWT auth rejects missing/invalid/expired tokens with 401
    - Test: rate limiter returns 429 after 20 requests in one minute
    - Test: `POST /v1/ask` with unknown tax year returns 422
    - Test: `GET /health` returns 200 without auth
    - Test: `POST /v1/feedback` returns 200 with valid body
    - _Requirements: 12.4, 12.5, 12.6_

- [~] 17. Checkpoint — API complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. C13 — Next.js frontend
  - [~] 18.1 Scaffold `frontend/` Next.js project with TypeScript and core layout
    - Initialize Next.js app in `frontend/` with TypeScript, Tailwind CSS, and `pdfjs-dist`
    - Create `ChatWindow` component: renders conversation history; appends SSE tokens to current assistant message
    - Create `YearSelector` component: dropdown for corpus years 2021–2024; stores selection in `sessionStorage["selectedYear"]`; displays selected year prominently
    - _Requirements: 12.1, 12.2_

  - [~] 18.2 Implement SSE streaming client and citation chips
    - Implement `streamAnswer(question, year)` using `fetch` + `ReadableStream` decoder
    - Parse SSE chunks; append each sentence to `ChatWindow` state as it arrives; stop on `[DONE]`
    - `CitationChip`: render inline numbered chips `[1]`, `[2]` per sentence; wire click handler to open PDF viewer
    - _Requirements: 12.1, 12.3_

  - [~] 18.3 Implement `PDFViewer` with bounding-box highlight
    - Render source PDF using `pdfjs-dist`; open at the correct `page` from `boxes` data
    - Draw canvas overlay highlight using `{x0, top, x1, bottom}` coordinates from citation's `boxes` array
    - _Requirements: 12.3, 7.4_

  - [~] 18.4 Implement `FeedbackBar` and `HistoryPanel`
    - `FeedbackBar`: thumbs-up / thumbs-down buttons + optional text comment; POST to `/v1/feedback` with `rating` and `query_id`
    - `HistoryPanel`: collapsible sidebar; fetch from `GET /v1/history`; list past questions chronologically; click to restore session
    - _Requirements: 12.10, 12.11_

  - [ ]* 18.5 Write Playwright E2E tests for frontend
    - Test: ask a question with year selector set → answer renders with citation chips
    - Test: click citation chip → PDF viewer opens at correct page with highlight
    - Test: thumbs-down → `POST /v1/feedback` returns 200; comment field optional
    - Test: history panel lists past questions for authenticated user
    - _Requirements: 12.1, 12.3, 12.10, 12.11, 13.2_

- [ ] 19. C14 — Parse coverage audit hardening
  - [~] 19.1 Harden `taxlens/eval/parse_audit.py`: stratified sampling, PNG rendering, CSV export
    - Stratified sample: 12 form pages, 8 instruction pages, 10 publication pages, spread across all documents and tax years
    - Render each Gold_Page as PNG with colored bounding boxes: green (`form_line`), grey (`form_text`), blue (`section`)
    - Produce CSV checklist: `doc_id`, `doc_type`, `page`, `tag`, `record_type`, `text`, `verdict`, `note`
    - Compute coverage = `ok / (ok + wrong + missed)` overall and per document type from human verdicts
    - Detect and report gaps in form-line numbering sequences (e.g., line 11 missing when line 12 present)
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [ ]* 19.2 Write unit tests for parse audit
    - Test: coverage formula `ok / (ok + wrong + missed)` is computed correctly
    - Test: numbering gap detection flags `{missing: [11]}` when line 12 present but 11 absent
    - _Requirements: 14.4, 14.5_

- [ ] 20. C16 — OpenTelemetry observability
  - [~] 20.1 Add OpenTelemetry spans to all pipeline stages
    - Add `taxlens.request` root span in `taxlens/api/main.py` wrapping each request lifecycle
    - Add child spans: `taxlens.cache_lookup`, `taxlens.year_detect`, `taxlens.intent_classify`, `taxlens.retrieval` (with children `taxlens.bm25_search`, `taxlens.dense_search`, `taxlens.rerank`), `taxlens.answer_generate`, `taxlens.verify`
    - Set span attributes: `tax_year`, `intent_type`, `retrieval_k`, `cache_hit`, `abstained`, `answer_sentence_count`
    - Configure OTLP exporter for compatibility with standard observability backends
    - _Requirements: 13.7_

  - [ ]* 20.2 Write unit tests for observability
    - Test: all expected span names are created for a complete request cycle (mock OTLP exporter)
    - Test: `cache_hit` attribute is `True` when Redis cache is hit
    - _Requirements: 13.7_

- [ ] 21. C15 — Docker Compose and CI/CD
  - [~] 21.1 Harden `docker-compose.yml` for local development
    - Verify all services defined: PostgreSQL (pgvector), Redis, FastAPI backend; add `frontend` service (Next.js)
    - Confirm health checks on all services; add `depends_on` ordering
    - Document `docker compose up` workflow in README
    - _Requirements: 13.4_

  - [~] 21.2 Create CI pipeline configuration (`.github/workflows/ci.yml` or equivalent)
    - Step 1: `pytest --cov=taxlens --cov-report=xml --cov-fail-under=80` — fail if coverage < 80%
    - Step 2: `python -m taxlens.eval.run_retrieval` CI gate — fail if `answer_correctness` drops > 2 pp or `wrong_year_at_1` > 3%
    - Step 3: run Playwright E2E tests against Docker Compose stack
    - Step 4: lint (flake8 / ruff + mypy)
    - Block merge on any failing step
    - _Requirements: 13.1, 13.2, 13.3_

  - [~] 21.3 Add cross-year hard negative training data generation in `taxlens/index/build.py`
    - For each `(question, positive_unit)` pair, find units from adjacent years (year ± 1) with same `form_code + line_label` but different normalized text → cross-year hard negatives
    - Include BM25 top-k misses from same year as in-year hard negatives
    - Write ~40,000 training triples to disk as JSONL for bi-encoder fine-tuning
    - _Requirements: 6.6_

  - [ ]* 21.4 Write property tests for hard negative generation (P22)
    - **Property 22: Cross-year hard negatives are from adjacent years** — `abs(hard_neg.tax_year - pos.tax_year) == 1`
    - **Validates: Requirements 6.6**

  - [ ]* 21.5 Write integration tests
    - Test: full ETL flow — 36 MVP docs fetched → parsed → indexed → figures extracted (Docker Compose)
    - Test: retrieval on `eval/benchmark/seed_v0.jsonl` — Recall@5 ≥ 0.85 on 35 questions
    - Test: atomic index swap — mid-rebuild query returns fully old or fully new index
    - Test: Redis cache — second identical request has `cache_hit = True`
    - Test: wrong-year rate — `YearAwareRetriever`: `wrong_year@1 = 0%` on seed benchmark
    - Test: all 6 FastAPI endpoints return expected status codes
    - _Requirements: 3.8, 6.2, 12.7, 13.4_

- [~] 22. Final checkpoint — All systems integrated
  - Ensure all tests pass at the target thresholds, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP path
- MVP order (Weeks 1–8): tasks 1–10, 14, 16, 17, 21 core migrations + Docker + CI
- Advanced order (Weeks 9–16): tasks 11, 12, 13 (full serving stack), 15, 18, 19, 20
- All property tests reference specific design document properties (P1–P34) for traceability
- Checkpoints at tasks 4, 9, 13, 17, 22 validate incremental progress
- `taxlens/answer/`, `taxlens/verify/`, `taxlens/intent/`, `taxlens/api/` are new packages to be created
- `frontend/` is a new Next.js project to be created in the workspace root

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3", "3.1", "3.2", "3.3"] },
    { "id": 2, "tasks": ["2.4", "2.5", "3.4", "3.5", "5.1", "7.1"] },
    { "id": 3, "tasks": ["5.2", "5.3", "5.4", "6.1", "7.2", "7.3"] },
    { "id": 4, "tasks": ["5.5", "6.2", "6.3", "8.1", "8.2", "8.3"] },
    { "id": 5, "tasks": ["8.4", "8.5", "8.6", "10.1", "14.1", "15.1"] },
    { "id": 6, "tasks": ["10.2", "10.3", "10.4", "10.5", "11.1", "12.1", "14.2", "14.3", "15.2", "15.3"] },
    { "id": 7, "tasks": ["11.2", "11.3", "11.4", "12.2", "12.3", "12.4", "16.1", "19.1", "21.3"] },
    { "id": 8, "tasks": ["16.2", "16.3", "16.4", "16.5", "21.4"] },
    { "id": 9, "tasks": ["16.6", "16.7", "18.1", "20.1", "21.1", "21.2"] },
    { "id": 10, "tasks": ["18.2", "18.3", "21.5"] },
    { "id": 11, "tasks": ["18.4", "19.2", "20.2"] },
    { "id": 12, "tasks": ["18.5"] }
  ]
}
```
