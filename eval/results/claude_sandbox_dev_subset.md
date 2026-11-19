# Retrieval results: seed_v0 (Claude's sandbox run, development subset)

Run on 2026-10-03 in Claude's sandbox, NOT on the full corpus.

- Corpus: Form 1040 + Instructions for Form 1040, tax years 2021-2024 (8 documents),
  2,991 parsed records, 1,888 validity-interval units.
- Benchmark: seed_v0, 44 year-sensitive questions drafted by Claude, gold answers
  checked against the parsed IRS text but NOT yet verified by a person.

| System | hit@1 | recall@5 | wrong-year@1 | year detection |
|---|---|---|---|---|
| Naive RAG (BM25, every year's copy) | 15.9% [7%–27%] | 68.2% [55%–82%] | 43.2% [27%–57%] | — |
| Year-aware (intervals, no line lookup) | 36.4% [23%–52%] | 88.6% [80%–98%] | 0.0% [0%–0%] | 100.0% [100%–100%] |
| TaxLens v1 (year-aware + line lookup) | 45.5% [32%–59%] | 95.5% [89%–100%] | 0.0% [0%–0%] | 100.0% [100%–100%] |

McNemar exact test, naive vs TaxLens v1: hit@1 p = 0.0010.

Index size: 36.9% fewer entries than one copy per year (Form 1040: 60.8%,
instructions: 32.7%). Instruction chunks change boundaries from year to year,
which limits sharing; paragraph-level chunking should raise this.

## Caveats (read before quoting any of these numbers)

1. Retrieval metrics only. Answer correctness needs the generator and human grading.
2. Year-aware wrong-year@1 is 0% partly by construction: once the year is detected
   correctly, the filter cannot return other years' text. Its real risk is year
   detection errors.
3. Year detection is 100% here because the same person (Claude) wrote both the
   detector and the question phrasings. Expect lower on questions written by others.
4. 44 questions is small; the confidence intervals are wide.
5. Run `python -m taxlens.eval.run_retrieval` for numbers on the full 36-document corpus.
