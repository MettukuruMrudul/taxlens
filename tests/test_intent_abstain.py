from taxlens.intent.classify import INTENTS, IntentClassifier
from taxlens.verify.abstain import (AbstainFeatures, RuleAbstainer, choose_threshold,
                                    selective_accuracy_curve, squash)

CLF = IntentClassifier.load()


def test_intent_is_always_one_of_the_seven_types():
    for q in ["", "standard deduction 2023", "asdf qwerty", "line 30?", "How did line 30 change?"]:
        assert CLF.classify(q).intent in INTENTS


def test_obvious_intents():
    assert CLF.classify("best pizza in Boston?").intent == "out_of_scope"
    assert CLF.classify("How do I file California state taxes?").intent == "out_of_scope"
    assert CLF.classify("What is the standard deduction for single filers in 2023?").intent == "figure_lookup"
    assert CLF.classify("How did line 30 change from 2021 to 2022?").intent == "cross_year_change"


def test_tax_vocabulary_without_outside_topic_is_never_out_of_scope():
    assert CLF.classify("What is line 30 for this year?").intent != "out_of_scope"


def _features(**kw) -> AbstainFeatures:
    base = dict(max_retrieval_score=0.8, mean_retrieval_score=0.6, year_confidence=0.9,
                intent_confidence=0.9, supported_fraction=1.0, unsupported_claim_rate=0.0,
                n_passages_for_year=500, has_figure=0.0, intent="form_navigation")
    base.update(kw)
    return AbstainFeatures(**base)


def test_rule_abstainer():
    r = RuleAbstainer()
    assert not r.decide(_features()).abstain
    assert r.decide(_features(supported_fraction=0.0)).abstain
    assert r.decide(_features(n_passages_for_year=0)).abstain
    assert not r.decide(_features(has_figure=1.0, max_retrieval_score=0.0)).abstain


def test_feature_vector_length_is_stable():
    assert len(_features().vector()) == 8 + len(INTENTS)


def test_selective_accuracy_curve_and_threshold():
    p, y = [0.9, 0.8, 0.3, 0.1], [True, True, False, False]
    curve = selective_accuracy_curve(p, y)
    assert curve[0] == (0.25, 1.0) and curve[-1] == (1.0, 0.5)
    assert choose_threshold(p, y, 1.0) == 0.8
    assert 0.0 <= squash(5.0) < 1.0 and squash(-3) == 0.0
