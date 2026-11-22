from taxlens.answer.generate import (DISCLAIMER, MAX_ATTRIBUTION_RETRIES, AnswerGenerator, Passage,
                                     detect_filing_status, figure_answer, iter_lines, parse_line)
from taxlens.llm import LLMTimeout, parse_json

P = [Passage(101, "Line 30 Reserved for future use", (2022, 2024), "30", None, "Form 1040 (2023)",
             score=5.0, pinned=True)]
FIGS = [{"parameter": "standard_deduction", "filing_status": s, "tax_year": 2023, "value": v,
         "source_record_id": 7, "source_text": f"{s} ${v:,}"}
        for s, v in [("single_or_mfs", 13850), ("mfj_or_qss", 27700), ("head_of_household", 20800)]]


class FakeLLM:
    """Streams a canned reply in small chunks; returns canned repair replies."""
    name = "fake"

    def __init__(self, stream_text="", repairs=(), fail=None):
        self.stream_text, self.repairs, self.fail = stream_text, list(repairs), fail
        self.repair_calls = 0

    def stream(self, system, user, **kw):
        if self.fail:
            raise self.fail
        for i in range(0, len(self.stream_text), 5):
            yield self.stream_text[i:i + 5]

    def complete(self, system, user, **kw):
        self.repair_calls += 1
        return self.repairs.pop(0) if self.repairs else '{"text": "", "cites": []}'


def test_figure_answers_come_from_the_table_not_the_llm():
    llm = FakeLLM('{"text": "It is $99,999.", "cites": ["P1"]}\n')
    a = AnswerGenerator(llm).generate("standard deduction for head of household 2023?", 2023, [], FIGS)
    assert a.mode == "figure"
    assert a.text() == "For tax year 2023, the standard deduction for head of household is $20,800."
    assert a.sentences[0].record_ids == [7]


def test_figure_answer_lists_all_statuses_when_status_unknown():
    s = figure_answer("What's the standard deduction for 2023?", 2023, FIGS)[0]
    assert "$13,850" in s.text and "$27,700" in s.text and "$20,800" in s.text


def test_filing_status_detection():
    assert detect_filing_status("we are married filing jointly") == "mfj_or_qss"
    assert detect_filing_status("as a single filer") == "single_or_mfs"
    assert detect_filing_status("single or head of household?") is None


def test_supported_llm_sentences_stream_through():
    llm = FakeLLM('{"text": "On the 2023 Form 1040, line 30 is reserved for future use.", "cites": ["P1"]}\n')
    a = AnswerGenerator(llm).generate("What is line 30 for in 2023?", 2023, P)
    assert a.mode == "llm" and len(a.sentences) == 1 and a.dropped == 0


def test_unsupported_sentence_is_repaired_then_dropped():
    llm = FakeLLM('{"text": "Line 30 gives a $1,400 rebate.", "cites": ["P1"]}\n',
                  repairs=['{"text": "Line 30 gives a $1,400 credit.", "cites": ["P1"]}'])
    a = AnswerGenerator(llm).generate("What is line 30 for in 2023?", 2023, P)
    assert a.sentences == [] and a.dropped == 1
    assert 1 <= llm.repair_calls <= MAX_ATTRIBUTION_RETRIES - 1


def test_bad_llm_output_falls_back_to_quoting():
    a = AnswerGenerator(FakeLLM("I am not JSON")).generate("What is line 30 for in 2023?", 2023, P)
    assert a.mode == "extractive_fallback" and "Reserved for future use" in a.text()


def test_llm_timeout_falls_back_to_quoting_with_a_note():
    a = AnswerGenerator(FakeLLM(fail=LLMTimeout("slow"))).generate("What is line 30 for in 2023?", 2023, P)
    assert a.mode == "extractive_fallback" and "Reserved for future use" in a.text()
    assert any("timeout" in n for n in a.notes)


def test_extractive_mode_without_llm_is_fully_supported():
    a = AnswerGenerator(None).generate("What is line 30 for in 2023?", 2023, P)
    assert a.mode == "extractive" and a.n_unsupported == 0 and a.disclaimer == DISCLAIMER


def test_jsonl_parsing_helpers():
    labels = {"P1": 101}
    assert parse_line('{"text": "x y", "cites": ["P1", "P9"]}', labels).record_ids == [101]
    assert parse_line('{"text": "", "cites": []}', labels) is None
    assert list(iter_lines(["a\nb", "c\n", "d"])) == ["a", "bc", "d"]
    assert parse_json('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
