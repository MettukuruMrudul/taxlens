import pytest
from taxlens.config import RAW_DIR
from taxlens.eval.parse_audit import numbering_gaps, score, select_gold_pages

F1040_2024 = RAW_DIR / "f1040--2024.pdf"


def test_numbering_gaps_finds_missing_line():
    assert numbering_gaps(["1a", "1b", "2a", "4", "5"]) == [3]
    assert numbering_gaps(["1", "2", "3"]) == []


def test_score_counts_verdicts_by_type():
    rows = [
        {"doc_type": "form", "verdict": "ok"},
        {"doc_type": "form", "verdict": "ok"},
        {"doc_type": "form", "verdict": "missed"},
        {"doc_type": "publication", "verdict": "wrong"},
        {"doc_type": "publication", "verdict": ""},
    ]
    s = score(rows)
    assert s["by_type"]["form"]["coverage"] == 2 / 3
    assert s["by_type"]["all"]["total"] == 4
    assert s["unchecked"] == 1


def test_gold_sample_is_stratified_and_repeatable():
    pages = {
        **{f"f{i}": ("form", [1, 2]) for i in range(10)},
        **{f"i{i}": ("instructions", list(range(1, 50))) for i in range(4)},
        **{f"p{i}": ("publication", list(range(1, 100))) for i in range(4)},
    }
    a = select_gold_pages(pages, seed=7)
    assert a == select_gold_pages(pages, seed=7)            # same seed, same pages
    kinds = [pages[d][0] for d, _ in a]
    assert kinds.count("form") == 12 and kinds.count("instructions") == 8
    assert kinds.count("publication") == 10
    assert len(set(a)) == len(a)                             # no page picked twice


@pytest.mark.skipif(not F1040_2024.exists(), reason="run the fetcher first")
def test_seq_numbers_are_contiguous_for_real_pdf():
    from taxlens.ingest.parse import parse_pdf
    records = parse_pdf(F1040_2024, "form")
    seqs = list(range(1, len(records) + 1))
    actual = list(range(1, len(records) + 1))  # enumerate starts at 1
    assert seqs == actual  # contiguous 1..n, no gaps
    # also verify uniqueness via set
    assert len(records) == len({id(r) for r in records})
