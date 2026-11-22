# Answer evaluation: seed_v0.jsonl

Run 2026-11-22 17:18. System: `taxlens-answers:none`. Questions: 44. Corpus years: 2021-2024.

| Metric | Value (95% CI) |
|---|---|
| answer_correct | 97.7% [93%–100%] |
| abstained | 0.0% [0%–0%] |
| wrong_year | 0.0% [0%–0%] |
| year_detection | 100.0% [100%–100%] |
| selective_accuracy | 97.7% [93%–100%] |
| unsupported_claim_rate | 0.0% |
| citation_year_precision | 100.0% |

## Failure clusters (incorrect or abstained)

- **intent**: rule_or_eligibility: 1
- **tax_year**: 2021: 1
- **topic**: charity_standard_deduction: 1
- **status**: answered: 1

## Incorrect answers

| id | year | status | answer |
|---|---|---|---|
| q037 | 2021 | answered | Publication 17, Your Federal Income Tax (2021) (What's New) says: “Standard deduction increased. The standard deduction for taxpayers who don't itemize their de |

All gold answers verified against the IRS source PDFs (python -m taxlens.eval.verify_gold).
