from taxlens.config import PROJECT_ROOT
from taxlens.eval.retrieval_eval import load_benchmark

SEED = PROJECT_ROOT / "eval" / "benchmark" / "seed_v0.jsonl"


def test_seed_benchmark_is_well_formed():
    qs = load_benchmark(SEED)
    assert len(qs) >= 40
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["question"].strip() and q["gold_any"], q["id"]
        assert 2021 <= q["tax_year"] <= 2024, q["id"]
