from taxlens.eval.retrieval_eval import bootstrap_ci, mcnemar_exact


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([True] * 30 + [False] * 10)
    assert mean == 0.75 and lo <= mean <= hi


def test_mcnemar_identical_systems_is_not_significant():
    a = [True, False, True, False]
    assert mcnemar_exact(a, a) == 1.0


def test_mcnemar_detects_consistent_improvement():
    before = [False] * 12
    after = [True] * 12
    assert mcnemar_exact(before, after) < 0.001
