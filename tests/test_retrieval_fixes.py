"""Regression tests for the failure groups found by the first answer benchmark run."""
from taxlens.answer.generate import figure_answer
from taxlens.index.intervals import build_units
from taxlens.intent.classify import IntentClassifier
from taxlens.retrieve.search import YearAwareRetriever, expand_query, preferred_forms

CLF = IntentClassifier.load()


def units(rows):
    return build_units([{"form_code": fc, "tax_year": y, "record_type": rt, "line_label": ll,
                         "heading_path": hp, "text": tx, "ref": i}
                        for i, (y, fc, rt, ll, hp, tx) in enumerate(rows)])


def test_single_year_question_is_never_a_comparison():
    for q in ["What does the 2022 Form 1040 ask about crypto?",
              "For the age box on the 2022 Form 1040, what birth date counts as 65 or older?"]:
        assert CLF.classify(q).intent != "cross_year_change", q


def test_real_comparisons_are_still_detected():
    for q in ["How did line 30 change from 2021 to 2022?",
              "Is line 30 different in 2022 compared to 2021?"]:
        assert CLF.classify(q).intent == "cross_year_change", q


def test_form_line_outranks_long_instructions_for_where_questions():
    r = YearAwareRetriever(units([
        (2021, "f1040", "form_line", "26", None, "2021 estimated tax payments and amount applied from 2020 return"),
        (2021, "i1040gi", "section", None, "Line 26 > 2021 Estimated Tax Payments",
         "Enter any estimated federal income tax payments you made for 2021. Include any overpayment "
         "that you applied to your 2021 estimated tax from your 2020 return. Estimated tax payments "
         "made for 2021 are reported here."),
    ]))
    assert r.search("Where do I enter estimated tax payments I made for 2021?", year=2021)[0].form_code == "f1040"


def test_thesaurus_connects_crypto_to_irs_wording():
    r = YearAwareRetriever(units([
        (2021, "f1040", "form_text", None, None,
         "At any time during 2021, did you receive, sell, exchange, or otherwise dispose of any financial "
         "interest in any virtual currency?"),
        (2021, "p17", "section", None, "Form W-2",
         "Your employer is required to provide or send Form W-2 to you. Form 1040 asks about wages."),
    ]))
    assert "virtual currency" in r.search("What does the 2021 Form 1040 ask about crypto?", year=2021)[0].text
    assert "digital asset" in expand_query("bitcoin")


def test_asking_for_instructions_turns_off_the_form_preference():
    assert preferred_forms("What do the 2022 instructions for Form 1040 say about crypto?") == set()
    assert preferred_forms("Which line of Schedule A is for medical expenses?") == {"f1040sa"}


def test_choosing_the_standard_deduction_is_not_a_figure_question():
    figs = [{"parameter": "standard_deduction", "filing_status": "single_or_mfs", "tax_year": 2021,
             "value": 12550, "source_record_id": 1, "source_text": "Single $12,550"}]
    assert figure_answer("Could I deduct charitable contributions in 2021 while taking the standard "
                         "deduction?", 2021, figs) is None
    assert figure_answer("What is the standard deduction for a single filer for 2021?", 2021, figs)


def test_page_headers_are_boilerplate_and_content_is_not():
    from taxlens.index.intervals import is_boilerplate
    assert is_boilerplate("Form 1040 (2021) Page 2")
    assert is_boilerplate("Cat. No. 11320B Form 1040 (2021)")
    assert not is_boilerplate("Line 30 Reserved for future use")
    assert not is_boilerplate("For the latest information about developments related to Pub. 17, see page 2")


def test_form_question_promotes_the_topic_match_not_the_page_header():
    r = YearAwareRetriever(units([
        (2021, "f1040", "form_text", None, None, "Form 1040 (2021) Page 2"),
        (2021, "f1040", "form_text", None, None,
         "Age/Blindness You: Were born before January 2, 1957 Are blind Spouse: Was born before January 2, 1957"),
        (2021, "p17", "section", None, "Spouse 65 or Older or Blind",
         "You can take the higher standard deduction if your spouse is age 65 or older or blind and you "
         "file a joint return. You are considered 65 on the day before your 65th birthday."),
    ]))
    top = r.search("For the age box on the 2021 Form 1040, what birth date counts as 65 or older?", year=2021)[0]
    assert top.text.startswith("Age/Blindness")
