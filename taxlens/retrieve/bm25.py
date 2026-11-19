"""A small, dependency-free BM25 keyword index."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
my of on or so than that the their them then there these they this to was what when where which
who will with you your""".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())
    return [t.replace(",", "") for t in toks if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(d) for d in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(texts)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len))
                out[i] += idf * norm
        return out
