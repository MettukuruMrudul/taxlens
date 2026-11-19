import json
import zlib

from taxlens.api.cache import AnswerCache, cache_key
from taxlens.api.render import parse_boxes
from taxlens.eval.gate import check_gate
from taxlens.eval.run_answers import grade, other_year_gold
from taxlens.index.align import Line, align, canonical_ids
from taxlens.index.hard_negatives import make_triples
from taxlens.index.intervals import build_units


class FakeRedis:
    def __init__(self, broken=False):
        self.data, self.broken = {}, broken

    def get(self, k):
        if self.broken:
            raise ConnectionError("down")
        return self.data.get(k)

    def set(self, k, v, ex=None):
        if self.broken:
            raise ConnectionError("down")
        self.data[k] = v


# ---------------------------------------------------------------- cache

def test_cache_key_normalises_whitespace_and_case():
    assert cache_key("What  is Line 30?", 2023, "v1") == cache_key(" what is line 30? ", 2023, "v1")
    assert cache_key("q", 2023, "v1") != cache_key("q", 2023, "v2")
    assert cache_key("q", None, "v1") != cache_key("q", 2023, "v1")


def test_cache_round_trip_is_compressed():
    r = FakeRedis()
    c = AnswerCache(r)
    c.set("q", 2023, {"status": "answered"})
    assert json.loads(zlib.decompress(next(iter(r.data.values())))) == {"status": "answered"}
    assert c.get("q", 2023) == {"status": "answered"} and c.hits == 1


def test_new_index_version_invalidates_old_answers():
    r = FakeRedis()
    c = AnswerCache(r)
    c.set("q", 2023, {"x": 1})
    r.data["taxlens:index_version"] = b"20270101T000000Z"
    assert c.get("q", 2023) is None


def test_cache_degrades_gracefully_when_redis_is_down():
    c = AnswerCache(FakeRedis(broken=True))
    c.set("q", 2023, {"x": 1})
    assert c.get("q", 2023) is None and c.unavailable >= 2


# ---------------------------------------------------------------- render, gate, grading

def test_parse_boxes_skips_invalid_parts():
    assert parse_boxes("1,2,3,4;bad;5,6,1,2;1,1,2,2") == ((1.0, 2.0, 3.0, 4.0), (1.0, 1.0, 2.0, 2.0))


def test_gate_rules():
    prev = {"answer_correct": [0.90, 0.8, 0.95]}
    assert check_gate({"answer_correct": [0.89, 0, 0], "wrong_year": [0.0, 0, 0]}, prev) == []
    assert len(check_gate({"answer_correct": [0.85, 0, 0]}, prev)) == 1
    assert len(check_gate({"wrong_year": [0.05, 0, 0]}, None)) == 1
    assert len(check_gate({"unsupported_claim_rate": 0.2}, None)) == 1


def test_wrong_year_answers_are_detected():
    qs = [{"id": "a", "topic": "sd", "tax_year": 2022, "gold_any": ["$12,950"]},
          {"id": "b", "topic": "sd", "tax_year": 2021, "gold_any": ["$12,550"]}]
    pools = other_year_gold(qs)
    res = {"status": "answered", "sentences": [{"text": "It is $12,550.", "citations": [1]}],
           "citations": [{"n": 1, "tax_year": 2021}], "tax_year": 2022}
    g = grade(qs[0], res, pools["a"])
    assert not g["correct"] and g["wrong_year"] and g["citation_year_precision"] == 0.0


# ---------------------------------------------------------------- alignment, hard negatives

def test_alignment_matches_identical_lines_and_rejects_rewrites():
    a = [Line(1, 2021, "2a", "Tax-exempt interest", 0.0), Line(2, 2021, "30", "Recovery rebate credit", 1.0)]
    b = [Line(3, 2022, "2a", "Tax-exempt interest", 0.0), Line(4, 2022, "30", "Reserved for future use", 1.0)]
    pairs = {(p.a.record_id, p.b.record_id): p for p in align(a, b)}
    assert pairs[(1, 3)].matched and not pairs[(2, 4)].matched
    assert all(0.0 <= p.score <= 1.0 for p in pairs.values())
    ids = canonical_ids("f1040", [2021, 2022], {2021: list(pairs.values())}, {2021: a, 2022: b})
    assert ids[1] == ids[3] and ids[2] != ids[4]


def test_hard_negatives_come_from_adjacent_years():
    rows = [(2021, "30", "Recovery rebate credit. See instructions"), (2022, "30", "Reserved for future use")]
    recs = [{"form_code": "f1040", "tax_year": y, "record_type": "form_line", "line_label": ll,
             "heading_path": None, "text": t, "ref": i} for i, (y, ll, t) in enumerate(rows)]
    triples = make_triples(build_units(recs))
    assert triples
    for t in triples:
        assert t["positive"] != t["cross_year_neg"]
        assert str(t["year"]) in t["query"]
