# Answer evaluation: heldout_v0.jsonl

Run 2026-11-22 17:07. System: `taxlens-answers:openai`. Questions: 26. Corpus years: 2021-2024.

| Metric | Value (95% CI) |
|---|---|
| answer_correct | 61.5% [42%–81%] |
| abstained | 11.5% [0%–23%] |
| wrong_year | 0.0% [0%–0%] |
| year_detection | 100.0% [100%–100%] |
| selective_accuracy | 69.6% [52%–87%] |
| unsupported_claim_rate | 0.0% |
| citation_year_precision | 100.0% |

## Failure clusters (incorrect or abstained)

- **intent**: figure_lookup: 4, out_of_scope: 3, form_navigation: 1, rule_or_eligibility: 1, definition: 1
- **tax_year**: 2022: 5, 2023: 3, 2021: 1, 2024: 1
- **topic**: salt_deduction_cap: 1, filing_due_date: 1, eic_max_no_children: 1, who_must_file_single_65: 1, eic_investment_income_limit: 1, additional_std_deduction_single: 1, eic_max_3plus_children: 1, 401k_contribution_limit: 1, health_fsa_limit: 1, business_mileage_rate: 1
- **status**: answered: 7, out_of_scope: 3

## Incorrect answers

| id | year | status | answer |
|---|---|---|---|
| h004 | 2023 | answered | Schedule A (Form 1040) (2023), line 5 says: “State and local taxes.” Instructions for Schedule A (Form 1040) (2023) (Actual Expenses) says: “Refund of general s |
| h006 | 2022 | answered | Form 1040 (2022) says: “Go to www.irs.gov/Form1040 for instructions and the latest information. Form 1040 (2022)” Form 1040 (2022) says: “For Disclosure, Privac |
| h010 | 2021 | answered | Instructions for Form 1040 (2021) (Your credit is–) says: “If the amount you are looking up from the worksheet is at least $27,350 but less than $27,380, and yo |
| h012 | 2022 | answered | A single dependent who was age 65 or older or blind must file a return if their unearned income was more than $2,900 ($4,650 if 65 or older and blind). |
| h013 | 2022 | answered | Instructions for Form 1040 (2022) (Your credit is–) says: “6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 6,935 |
| h015 | 2022 | answered | For tax year 2022, the standard deduction for single or married filing separately is $12,950. |
| h016 | 2022 | answered | Instructions for Form 1040 (2022) (Your credit is–) says: “** If the amount you are looking up from the worksheet is at least $59,150 but less than $59,187, and |
| h018 | 2023 | out_of_scope | — |
| h019 | 2023 | out_of_scope | — |
| h024 | 2024 | out_of_scope | — |

All gold answers verified against the IRS source PDFs (python -m taxlens.eval.verify_gold).
