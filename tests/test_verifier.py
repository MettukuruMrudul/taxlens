from taxlens.verify.verifier import Verifier, lexical_support, numbers_in, split_claims

PASSAGES = {
    1: "Standard Deduction for Single or Married filing separately, $13,850 Married filing jointly, $27,700",
    2: "Line 30 Reserved for future use",
    3: "You are considered age 65 at the end of 2022 if you were born before January 2, 1958.",
}


def test_numbers_ignore_form_and_line_identifiers():
    assert numbers_in("Form 1040, line 12, Schedule 8812, $12,550 and 7.65%") == {"12550", "7.65%"}


def test_correct_amount_is_supported():
    r = Verifier().verify([{"text": "For 2023, the standard deduction for a single filer is $13,850.",
                            "record_ids": [1]}], PASSAGES, {"2023"})
    assert r.sentences[0].supported
    assert r.unsupported_claim_rate == 0.0


def test_wrong_amount_is_rejected_even_if_wording_matches():
    r = Verifier().verify([{"text": "For 2023, the standard deduction for a single filer is $13,800.",
                            "record_ids": [1]}], PASSAGES, {"2023"})
    claim = r.sentences[0].claims[0]
    assert not claim.supported and "13800" in claim.missing_numbers


def test_sentence_citing_the_wrong_passage_is_rejected():
    r = Verifier().verify([{"text": "Line 30 is the recovery rebate credit.", "record_ids": [2]}], PASSAGES)
    assert not r.sentences[0].supported


def test_uncited_sentence_is_unsupported():
    r = Verifier().verify([{"text": "Line 30 is reserved for future use.", "record_ids": []}], PASSAGES)
    assert not r.sentences[0].supported


def test_dates_must_match_whole():
    ok = Verifier().check_sentence("Born before January 2, 1958 counts as 65 at the end of 2022.", [3], PASSAGES)
    bad = Verifier().check_sentence("Born before January 3, 1958 counts as 65 at the end of 2022.", [3], PASSAGES)
    assert ok.supported and not bad.supported


def test_claim_splitter_and_lexical_support():
    assert len(split_claims("You must file Schedule B if interest exceeds $1,500; enter the total on line 2b.")) == 2
    assert lexical_support("reserved for future use", PASSAGES[2]) == 1.0
