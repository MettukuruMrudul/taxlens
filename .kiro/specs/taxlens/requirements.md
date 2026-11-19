# Requirements Document

## Introduction

TaxLens is a tax-year-aware question-answering system over official IRS documents. It answers questions about a specific tax year using IRS forms, schedules, instructions, and publications (initially tax years 2021–2024, expanding to 2019–2025). Every sentence in a generated answer cites the exact IRS line or passage it came from. When the evidence does not support an answer, TaxLens abstains rather than guessing.

The core problem TaxLens solves is year confusion in naive RAG systems: IRS documents are nearly identical from year to year, differing mainly in specific dollar amounts and occasionally in line numbers or wording. A standard RAG retrieval returns whichever passage looks most similar regardless of year, producing fluent, confident answers that cite the wrong year's figures. TaxLens prevents this by treating tax year as a first-class constraint at every stage — ingestion, indexing, retrieval, generation, and verification.

## Glossary

- **TaxLens**: The complete tax-year-aware Q&A system defined in this document.
- **IRS**: Internal Revenue Service, the U.S. federal tax authority.
- **Tax_Year**: The calendar year to which a tax return applies (e.g., 2023), as distinct from the filing year (the year in which the return is submitted).
- **Filing_Year**: The calendar year in which a tax return is submitted to the IRS. For most filers this is one year after the Tax_Year.
- **Document**: A single IRS PDF for a single Tax_Year — a form, schedule, instructions document, or publication.
- **Line_Record**: One parsed unit from a Document: a numbered form line, a block of form text, or a prose section from instructions or a publication.
- **Index_Unit**: A deduplicated text unit stored with a validity interval (`valid_from`, `valid_to`) covering the consecutive Tax_Years whose Line_Records have identical normalized text.
- **Validity_Interval**: The `[valid_from, valid_to]` range of Tax_Years for which an Index_Unit applies.
- **Year_Figure**: A specific dollar amount or threshold that changes each Tax_Year (e.g., standard deduction), stored in the `year_figures` table keyed by parameter, filing status, and Tax_Year.
- **Canonical_Line_ID**: A stable identifier for a form line that tracks that line across Tax_Years even when the IRS renumbers it.
- **Corpus**: The full set of Documents loaded into TaxLens.
- **MVP_Corpus**: Form 1040, its schedules (1, 2, 3, A, B), the Form 1040 instructions, Schedule A instructions, and Publication 17, for Tax_Years 2021–2024 (36 documents).
- **Full_Corpus**: 230+ IRS forms, instructions, and publications for Tax_Years 2019–2025 (~9,000 pages). TY2025 is held out from training and tuning.
- **BM25**: Okapi BM25 keyword ranking algorithm, used as the lexical retrieval component.
- **Bi_Encoder**: A dual-encoder dense retrieval model that embeds queries and passages independently for approximate nearest-neighbor search.
- **Cross_Encoder**: A cross-attention re-ranking model that scores a (query, passage) pair jointly.
- **Hybrid_Retrieval**: Combination of BM25 and Bi_Encoder results via reciprocal rank fusion.
- **NLI**: Natural Language Inference — classifying whether a hypothesis is entailed by, contradicts, or is neutral with respect to a premise.
- **Abstention**: The system's decision to decline to answer a question when evidence is insufficient, contradictory, or out of scope.
- **ETL**: Extract-Transform-Load pipeline. In TaxLens this means fetching IRS PDFs, parsing them into Line_Records, and loading them into the database.
- **Parser**: The component that converts a raw IRS PDF into a list of Line_Records with bounding boxes.
- **Gold_Page**: A PDF page whose parsed content is manually verified for coverage auditing.
- **Benchmark**: A fixed set of evaluation questions with gold answers and gold Line_Record IDs.
- **Eval_Run**: One execution of the evaluation harness against the Benchmark, producing per-question outcomes and aggregate metrics stored in the database.
- **Wrong_Year_Answer**: A generated answer whose retrieved passage does not exist in the asked Tax_Year's Documents.
- **Recall_At_5**: Whether at least one of the top-5 retrieved passages contains the gold answer for the asked Tax_Year.
- **Hit_At_1**: Whether the top-1 retrieved passage contains the gold answer for the asked Tax_Year.
- **AUROC**: Area under the Receiver Operating Characteristic curve, used to measure abstention quality.
- **LTR**: Learning to Rank, a supervised ranking framework. LightGBM-LambdaRank is the LTR baseline.
- **pgvector**: A PostgreSQL extension that adds a vector column type and approximate nearest-neighbor operators.
- **HNSW**: Hierarchical Navigable Small World graph — the approximate nearest-neighbor index algorithm used by pgvector.
- **Redis**: In-memory key-value store used to cache repeated (question, Tax_Year, index version) lookups.
- **Prefect**: Python workflow orchestration library used for the ETL pipeline.
- **vLLM**: High-throughput LLM inference server for serving open-weight models.
- **LoRA**: Low-Rank Adaptation, a parameter-efficient fine-tuning method for large language models.
- **ModernBERT**: The encoder model used as the Cross_Encoder re-ranker backbone.
- **OpenTelemetry**: Vendor-neutral observability framework used for distributed tracing.
- **JWT**: JSON Web Token, used for API authentication.
- **TTFT**: Time-to-first-token, the latency from request to first streaming token.
- **CI**: Continuous integration pipeline; also used to denote confidence interval in evaluation contexts.
- **McNemar_Test**: A paired statistical test for comparing two binary classifiers on the same examples.
- **Cohens_Kappa**: Inter-annotator agreement coefficient (κ), used to measure LLM judge vs. human grader agreement.

---

## Requirements

### Requirement 1: ETL — Corpus Ingestion

**User Story:** As a system administrator, I want IRS PDFs for all corpus years to be fetched, hashed, and recorded idempotently, so that the document manifest always reflects the exact files on disk without duplicating work.

#### Acceptance Criteria

1. WHEN the ETL pipeline runs, THE Fetcher SHALL download each Document from the IRS prior-year archive URL derived from the Document's form code and Tax_Year.
2. WHEN a PDF file already exists on disk and its SHA-256 matches the manifest record, THE Fetcher SHALL skip re-downloading that Document.
3. WHEN a downloaded file's content does not begin with the PDF magic bytes `%PDF`, THE Fetcher SHALL reject the file and raise an error without writing it to disk.
4. WHEN the IRS server returns HTTP 404 for a requested Document, THE Fetcher SHALL record the failure and continue with the remaining Documents rather than aborting the entire run.
5. THE Fetcher SHALL write each downloaded PDF to a temporary file first, then rename it atomically, so that a crash during download never leaves a partial file in the data directory.
6. THE Fetcher SHALL record each Document's `doc_id`, `form_code`, `doc_type`, `title`, `tax_year`, `source_url`, `local_path`, `sha256`, and `size_bytes` in the `documents` table via an upsert.
7. THE Fetcher SHALL compute SHA-256 by reading the file in streaming 1 MiB chunks to avoid loading large PDFs entirely into memory.
8. WHEN multiple fetch attempts fail with transient network errors, THE Fetcher SHALL retry up to 3 times with exponential back-off (2, 4 seconds) before recording the failure.
9. THE Corpus SHALL define the MVP_Corpus as Form 1040, Schedules 1–3, A, B, the Form 1040 general instructions, Schedule A instructions, and Publication 17, for Tax_Years 2021–2024.
10. THE Corpus SHALL use only the IRS prior-year archive (`https://www.irs.gov/pub/irs-prior`) as the download source, never the "current forms" folder, to prevent silently fetching the wrong year's PDF.

---

### Requirement 2: PDF Parsing into Line_Records

**User Story:** As a data engineer, I want each IRS PDF converted into typed Line_Records with bounding-box coordinates, so that every piece of parsed text can be cited back to its exact location in the source document.

#### Acceptance Criteria

1. WHEN a form PDF is parsed, THE Parser SHALL produce one `form_line` Line_Record per numbered form line (e.g., lines 1a, 12, 30), each tagged with its line label.
2. WHEN a form PDF is parsed, THE Parser SHALL produce `form_text` Line_Records for unnumbered text blocks on the form (headers, margin notes, captions), each with bounding-box coordinates.
3. WHEN an instructions or publication PDF is parsed, THE Parser SHALL produce `section` Line_Records by segmenting the prose into chunks of at most 1,200 characters at sentence boundaries, each tagged with its heading path (e.g., `Income > Line 1a > Wages`).
4. THE Parser SHALL store for each Line_Record: `doc_id`, `seq` (1-based reading order), `record_type`, `line_label` (NULL for non-form-line records), `heading_path`, `text`, `page`, and `boxes` as a JSON array of `{page, x0, top, x1, bottom}` objects.
5. WHEN a document is re-parsed and the file SHA-256 and parser version are unchanged, THE Parser SHALL skip that document to avoid redundant work.
6. WHEN a document is re-parsed, THE Parser SHALL replace all previous Line_Records for that document within a single database transaction, so the database never contains a partially updated document.
7. THE Parser SHALL parse forms using a two-column layout strategy: identify the line-number column by the leftmost x-coordinate of tokens matching the pattern `\d{1,2}[a-z]?` that are immediately followed by capitalized text.
8. THE Parser SHALL parse instructions and publications using recursive XY-cut layout analysis to detect multi-column layouts and yield blocks in reading order (left column top-to-bottom before right column).
9. IF a word's bounding box falls outside the page boundaries, THEN THE Parser SHALL exclude that word from all records.
10. THE Parser SHALL strip visual decoration tokens (TIP/CAUTION icon lettering, page-footer text) from all Line_Records.
11. THE Parser SHALL achieve a line coverage rate of at least 99.6% on the Gold_Pages set (30 pages for MVP_Corpus, 50 pages for Full_Corpus), verified by the parse coverage audit.
12. THE Parser SHALL produce zero duplicate Line_Records for the same `(doc_id, seq)` pair.

---

### Requirement 3: Validity-Interval Index

**User Story:** As a retrieval engineer, I want identical text shared across Tax_Years stored once with a validity interval, so that the retriever can never return a passage from the wrong year and the index is smaller.

#### Acceptance Criteria

1. THE Index_Builder SHALL normalize each Line_Record's text (NFKC Unicode normalization, smart-quote substitution, whitespace collapse) before comparing it across years.
2. WHEN two Line_Records from the same form and record type have identical normalized text in consecutive Tax_Years, THE Index_Builder SHALL merge them into a single Index_Unit with `valid_from` set to the first year and `valid_to` set to the last year.
3. WHEN the text of a form line changes between consecutive Tax_Years (e.g., a dollar amount changes), THE Index_Builder SHALL close the current Index_Unit at the previous year and open a new Index_Unit starting at the changed year.
4. WHEN the same normalized text reappears after a gap year (non-consecutive), THE Index_Builder SHALL create a separate Index_Unit for each contiguous run, not a single unit spanning the gap.
5. THE Index_Unit SHALL store a `sources` JSON object mapping each Tax_Year in its interval to the `record_id` of that year's Line_Record, so each year can be cited to its own document.
6. WHEN retrieving for a specific Tax_Year, THE Retriever SHALL filter to Index_Units where `valid_from <= requested_year <= valid_to` before ranking, so passages from other years cannot be returned.
7. THE Validity_Interval index SHALL reduce the total number of stored index entries by at least 30% compared to storing one copy per Tax_Year per record (measured on the MVP_Corpus).
8. THE Index_Builder SHALL rebuild the `index_units` table atomically (TRUNCATE + INSERT in one transaction) so queries always see a consistent index state.
9. THE Index_Builder SHALL reduce the Wrong_Year_Answer rate from the naive RAG baseline (approximately 31%) to at most 1.8% on the Benchmark.

---

### Requirement 4: Year-Specific Figures Table

**User Story:** As a user, I want answers about dollar amounts like the standard deduction to use the exact IRS figures for my Tax_Year, so that I never receive a figure from a different year.

#### Acceptance Criteria

1. THE Figures_Extractor SHALL extract standard deduction amounts from the margin text of Form 1040 (the text block adjacent to line 12) for each Tax_Year in the corpus.
2. THE Figures_Extractor SHALL store each extracted amount in the `year_figures` table keyed by `(parameter, filing_status, tax_year)`, where `filing_status` is one of `single_or_mfs`, `mfj_or_qss`, or `head_of_household`.
3. WHEN the `year_figures` table is populated, THE Figures_Extractor SHALL record `source_record_id` (the `record_id` of the Line_Record the figure was extracted from) and `source_text` (the raw matched text) for every row.
4. THE Answer_Generator SHALL retrieve year-specific figures exclusively from the `year_figures` table keyed by Tax_Year, and SHALL NOT use figure values retrieved from free-text passages.
5. THE Figures_Extractor SHALL achieve at least 99% accuracy on figure-lookup questions in the Benchmark (measured as the fraction of figure-lookup questions answered with the correct amount for the asked Tax_Year).

---

### Requirement 5: Tax-Year Detection

**User Story:** As a user, I want TaxLens to automatically determine which tax year my question is about, so that I do not have to always explicitly state the year.

#### Acceptance Criteria

1. WHEN the user selects a Tax_Year via the UI year selector, THE Year_Detector SHALL use that selection as the Tax_Year and ignore any year mentioned in the question text.
2. WHEN the question text contains a filing-year phrase (e.g., "filing in 2024", "filed in 2024"), THE Year_Detector SHALL interpret the year as the Filing_Year and return Tax_Year = Filing_Year − 1.
3. WHEN the question text contains an explicit tax-year phrase (e.g., "tax year 2023", "for 2023", "my 2023 return"), THE Year_Detector SHALL return that year as the Tax_Year.
4. WHEN the question text contains a single unambiguous year not explicitly qualified as a filing year, THE Year_Detector SHALL interpret it as the Tax_Year.
5. WHEN the question text contains a relative time phrase such as "last year" or "this year", THE Year_Detector SHALL resolve the Tax_Year relative to the current calendar date.
6. WHEN no year can be determined from the question text, THE Year_Detector SHALL return `year = None` with source `none`, and the system SHALL prompt the user to specify the Tax_Year.
7. THE Year_Detector SHALL achieve at least 98% accuracy on the year-detection portion of the Benchmark (across all phrasings including filing-year constructions).

---

### Requirement 6: Hybrid Retrieval and Re-Ranking

**User Story:** As a user, I want TaxLens to find the most relevant IRS passages for my question, prioritising exact year match and named form lines, so that answers are grounded in the correct source text.

#### Acceptance Criteria

1. THE Retriever SHALL combine BM25 keyword scores and Bi_Encoder dense retrieval scores using reciprocal rank fusion to produce a hybrid ranked list.
2. WHEN the detected Tax_Year is known, THE Retriever SHALL filter the candidate set to Index_Units whose Validity_Interval contains that Tax_Year before applying BM25 and dense scoring.
3. WHEN the question names a specific form line (e.g., "line 30"), THE Retriever SHALL perform a structured lookup that returns the `form_line` Index_Unit with the matching line label for the requested Tax_Year and rank it above BM25 results.
4. THE Cross_Encoder SHALL re-rank the top candidates returned by hybrid retrieval using a ModernBERT-based cross-attention model before the results are passed to the Answer_Generator.
5. THE Retriever SHALL achieve Recall_At_5 of at least 0.91 on the Benchmark (improved from a naive BM25 baseline of 0.65).
6. THE Bi_Encoder SHALL be fine-tuned using contrastive loss on approximately 40,000 synthetic training questions that include cross-year hard negatives (passages from adjacent Tax_Years with similar but incorrect text).
7. THE Retriever SHALL avoid the pgvector HNSW year-filter pitfall by using over-fetch or iterative scans when filtering by Validity_Interval, so that filtered recall does not degrade compared to unfiltered recall.

---

### Requirement 7: Answer Generation with Citations

**User Story:** As a user, I want TaxLens to produce a complete natural-language answer grounded in IRS source text, where every sentence is traceable to the exact IRS line it came from, so that I can verify each claim.

#### Acceptance Criteria

1. THE Answer_Generator SHALL produce structured JSON output containing an array of sentence objects, each with `text` and `record_ids` (the Line_Record IDs that support that sentence).
2. WHEN a generated sentence cannot be attributed to any retrieved passage, THE Answer_Generator SHALL retry that sentence up to a configurable number of times; if attribution still fails, THE Answer_Generator SHALL omit the sentence from the final answer.
3. THE Answer_Generator SHALL retrieve year-specific dollar amounts exclusively from the `year_figures` table via the Tax_Year key, never from free-text passages alone.
4. THE Answer_Generator SHALL present users with the ability to click any citation to view the source passage highlighted within the PDF at the exact page and bounding-box coordinates.
5. THE Answer_Generator SHALL achieve at least 90% answer correctness on the Benchmark as graded by human evaluators using the defined rubric (correct / partially correct / wrong / wrong year).
6. THE Answer_Generator SHALL achieve at least 95% citation precision (fraction of cited passages that genuinely support the claimed statement).
7. THE Answer_Generator SHALL include a disclaimer in every response stating that TaxLens provides information from IRS publications for educational purposes only and is not tax advice.

---

### Requirement 8: Claim Verification and Abstention

**User Story:** As a user, I want TaxLens to refuse to answer when its evidence is insufficient or contradictory, rather than generating a plausible but incorrect answer, so that I can trust that a given answer is actually supported.

#### Acceptance Criteria

1. THE Verifier SHALL split each generated answer into atomic claims before verification.
2. WHEN an atomic claim is not entailed by any retrieved passage for the asked Tax_Year, THE Verifier SHALL flag that claim as unsupported.
3. THE Abstention_Classifier SHALL use retrieval scores, NLI entailment scores, year-detection confidence, and intent classification as input features to decide whether to return an answer or abstain.
4. WHEN the Abstention_Classifier decides to abstain, THE System SHALL return a response indicating that the question cannot be answered from the available IRS sources for the asked Tax_Year, rather than returning a generated answer.
5. THE Abstention_Classifier SHALL achieve an AUROC of at least 0.94 on the abstention task evaluated on the Benchmark.
6. THE System SHALL keep the unsupported-claim rate at or below 1.7% of all claims in non-abstained answers.
7. THE Verifier SHALL produce a selective-accuracy curve (accuracy vs. fraction of questions answered) to support analysis of the abstention threshold.

---

### Requirement 9: Query Understanding and Intent Classification

**User Story:** As a system designer, I want each incoming question classified into one of seven intent types, so that the retrieval and generation components can adapt their strategy to the question's nature.

#### Acceptance Criteria

1. THE Intent_Classifier SHALL classify every question into exactly one of the following seven intent types: `figure_lookup`, `rule_or_eligibility`, `form_navigation`, `procedure_or_computation`, `definition`, `cross_year_change`, or `out_of_scope`.
2. THE Intent_Classifier SHALL achieve a macro-F1 score of at least 0.85 across all seven intent types on the Benchmark.
3. THE Intent_Classifier SHALL be evaluated by comparing three approaches: TF-IDF + logistic regression, XGBoost, and a fine-tuned small transformer, and the best-performing model SHALL be deployed.
4. WHEN the Intent_Classifier returns `out_of_scope`, THE System SHALL respond with a message explaining that the question is outside the coverage of the TaxLens corpus and SHALL NOT invoke the Answer_Generator.
5. WHEN the detected Tax_Year is None and the Intent_Classifier returns any type other than `out_of_scope`, THE System SHALL ask the user to specify the Tax_Year before proceeding.

---

### Requirement 10: Evaluation Harness

**User Story:** As a researcher, I want a rigorous automated evaluation harness with a fixed benchmark, human-grading rubric, and statistical tests, so that I can measure whether system changes improve or degrade performance.

#### Acceptance Criteria

1. THE Benchmark SHALL contain at least 200 questions for the MVP phase and at least 500 questions for the full system, covering all seven intent types, all corpus Tax_Years including TY2025, and approximately 10% unanswerable questions.
2. THE Benchmark SHALL store gold answers and gold Line_Record IDs for each question, enabling both retrieval and answer-level evaluation.
3. THE Eval_Harness SHALL measure Recall_At_5, Hit_At_1, Wrong_Year_At_1, year-detection accuracy, answer correctness, citation precision, and unsupported-claim rate, each with 95% bootstrap confidence intervals.
4. THE Eval_Harness SHALL perform McNemar exact tests for pairwise comparison of binary metrics (e.g., hit@1) between system versions.
5. THE Eval_Harness SHALL store every Eval_Run's aggregate metrics in the `eval_runs` table and every per-question outcome in the `eval_items` table.
6. THE LLM_Judge SHALL be calibrated against human grades with a target Cohen's κ of at least 0.81 (91% agreement) before being used to automate grading.
7. THE CI_Gate SHALL fail the build if answer correctness drops by more than 2 percentage points or Wrong_Year_At_1 exceeds 3% compared to the previous Eval_Run.
8. THE Eval_Harness SHALL report unseen-year generalization by computing correctness separately for TY2025, with a target gap of at most 3.6 percentage points versus the training years.
9. THE Eval_Harness SHALL produce failure cluster analysis by grouping incorrect answers by intent type, Tax_Year, and document, to guide targeted improvements.

---

### Requirement 11: Cross-Year Line Alignment

**User Story:** As a data engineer, I want form lines tracked across Tax_Years via stable Canonical_Line_IDs even when the IRS renumbers lines between years, so that citations remain accurate when form layouts change.

#### Acceptance Criteria

1. THE Line_Aligner SHALL compute a similarity score for each candidate line pair across adjacent Tax_Years using a weighted combination of embedding cosine similarity, string edit similarity, relative position on the page, and section membership.
2. THE Line_Aligner SHALL use the Hungarian algorithm to find the globally optimal one-to-one matching between lines in year A and lines in year B that maximises total similarity.
3. THE Line_Aligner SHALL assign a Canonical_Line_ID to each group of matched lines across years, stored in the `line_alignment` table with per-pair scores and match types.
4. THE Line_Aligner SHALL achieve at least 97% accuracy on a gold set of approximately 300 manually verified cross-year line pairs.
5. WHEN no sufficiently similar counterpart is found for a line in the adjacent year, THE Line_Aligner SHALL record the line as unmatched rather than forcing a low-confidence alignment.

---

### Requirement 12: Web Application and API

**User Story:** As a user, I want a browser-based chat interface with a tax-year selector, citation viewer, and streaming responses, so that I can ask tax questions and immediately verify each answer against its IRS source.

#### Acceptance Criteria

1. THE Web_App SHALL provide a chat interface built with Next.js that streams answer tokens to the browser via server-sent events.
2. THE Web_App SHALL include a tax-year selector that sets the Tax_Year for all subsequent questions in the session, with the selected year clearly displayed.
3. WHEN a user clicks a citation, THE Web_App SHALL open the source IRS PDF with the cited passage highlighted at the exact page and bounding-box coordinates.
4. THE API SHALL expose the following endpoints: `POST /v1/ask` (synchronous), `POST /v1/ask/stream` (streaming), `GET /v1/documents/{id}`, `GET /v1/history`, `POST /v1/feedback`, and `GET /health`.
5. THE API SHALL authenticate all non-health requests using JWT tokens.
6. THE API SHALL enforce rate limiting on the `/v1/ask` and `/v1/ask/stream` endpoints to protect backend resources.
7. THE System SHALL cache responses in Redis keyed on the normalized question, Tax_Year, and index version, returning cached results without invoking the Answer_Generator.
8. THE API SHALL achieve a p99 latency of at most 4 seconds for uncached requests from question receipt to complete response.
9. THE API SHALL achieve a TTFT of at most 1 second for streaming requests.
10. THE Web_App SHALL record user feedback (thumbs up / thumbs down and optional comment) per answer via the `/v1/feedback` endpoint.
11. THE Web_App SHALL display a conversation history for authenticated users, retrievable via `GET /v1/history`.

---

### Requirement 13: Engineering Quality and Operations

**User Story:** As an engineer, I want automated tests, CI checks, containerised deployment, and distributed tracing, so that I can develop and operate TaxLens reliably.

#### Acceptance Criteria

1. THE Test_Suite SHALL achieve at least 80% line coverage of the backend Python code, measured by pytest-cov.
2. THE Test_Suite SHALL include Playwright end-to-end tests for the Next.js web application covering the citation viewer, tax-year selector, and feedback flow.
3. THE CI_Pipeline SHALL run all tests, the retrieval benchmark CI_Gate check, and lint on every pull request, blocking merge if any check fails.
4. THE System SHALL be deployable via Docker Compose for local development, with PostgreSQL (pgvector), Redis, and the FastAPI backend as separate services.
5. THE System SHALL be deployable to AWS using container services, with separate compute for the API, the vLLM inference server (if used), and the Prefect ETL workers.
6. THE ETL_Pipeline SHALL run on a Prefect schedule to ingest updated IRS documents, using atomic index swaps so queries are never served from a partially rebuilt index.
7. THE System SHALL instrument all API calls, retrieval steps, and generation calls with OpenTelemetry spans, exportable to a compatible observability backend.
8. WHEN a new index is built by the ETL_Pipeline, THE System SHALL swap the serving index atomically so in-flight queries always complete against the old or new index, never a partial state.

---

### Requirement 14: Parser Coverage Audit

**User Story:** As a data engineer, I want a structured audit workflow for measuring parser coverage on a stratified sample of Gold_Pages, so that I can detect and fix parsing gaps before they propagate to retrieval and generation.

#### Acceptance Criteria

1. THE Parse_Audit_Tool SHALL select a stratified random sample of Gold_Pages (12 form pages, 8 instruction pages, 10 publication pages for MVP) spread across all documents and Tax_Years.
2. THE Parse_Audit_Tool SHALL render each Gold_Page as a PNG with every Line_Record drawn as a coloured bounding box (green for `form_line`, grey for `form_text`, blue for `section`).
3. THE Parse_Audit_Tool SHALL produce a CSV checklist with one row per parsed unit on each Gold_Page, with columns for `doc_id`, `doc_type`, `page`, `tag`, `record_type`, `text`, `verdict`, and `note`.
4. WHEN the audit CSV contains human verdicts, THE Parse_Audit_Tool SHALL compute coverage as `ok / (ok + wrong + missed)` overall and per document type.
5. THE Parse_Audit_Tool SHALL detect and report gaps in form-line numbering sequences (e.g., line 12 present but line 11 missing) for every form document in the corpus.

---

### Requirement 15: User Study

**User Story:** As a researcher, I want a within-subjects controlled user study comparing TaxLens to manual IRS.gov search, so that I can measure real-world task success and time savings.

#### Acceptance Criteria

1. THE User_Study SHALL use a within-subjects design with n=16 participants, counterbalanced to control for order effects.
2. THE User_Study SHALL measure task success rate and time to completion for a set of standardised tax questions using both TaxLens and manual IRS.gov search.
3. THE User_Study SHALL target a task success rate of at least 88% with TaxLens versus 59% with manual search.
4. THE User_Study SHALL target a median task completion time at least 52% lower with TaxLens versus manual search.
5. THE User_Study SHALL use Wilcoxon signed-rank tests for time comparisons and McNemar tests for success-rate comparisons between the two conditions.
