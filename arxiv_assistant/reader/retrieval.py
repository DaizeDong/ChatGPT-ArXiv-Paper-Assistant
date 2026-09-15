"""Okapi BM25 over the archive, implemented in-repo.

WHY IN-REPO: the archive is a few hundred days of JSON on local disk. A dependency
(rank_bm25) or an embedding index would add install surface, a model download, or a
network call to answer a question about files we already have open. BM25 is ~80 lines
and is exactly the right tool for "which of these few thousand short documents mention
what I asked about".

CJK HANDLING (read this before changing the tokenizer)
------------------------------------------------------
The corpus is mixed English and Chinese: hotspot reports carry Chinese headlines and
summaries alongside English ones. A naive ``re.findall(r"\\w+", text)`` is a trap here,
because Python's ``\\w`` is Unicode-aware and swallows an entire Chinese sentence into
ONE token. That token can only ever match a query that repeats the sentence verbatim,
so Chinese queries silently retrieve nothing -- a green run with an empty answer, which
is the failure mode this repo exists to stop.

So the tokenizer splits the text into two kinds of run and treats them differently:

* Non-CJK runs  -> case-folded word tokens (``\\w+`` minus underscore minus CJK).
* CJK runs      -> character BIGRAMS: "多智能体" yields "多智", "智能", "能体".
  A run of a single CJK character yields that character as a unigram, otherwise a
  one-character query term would be unmatchable.

Bigrams are the standard cheap substitute for a Chinese word segmenter: they need no
dictionary, they survive segmentation ambiguity, and because BOTH the document and the
query go through the same function, a query bigram matches a document bigram wherever
the same two adjacent characters occur. The cost is some over-matching across word
boundaries, which BM25's IDF term largely prices in (a bigram that spans a boundary is
common, so it is worth little).

Both documents and queries MUST be tokenized with :func:`tokenize`. Do not tokenize a
query any other way.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

# Han ideographs (BMP + ext A + compatibility) and Japanese kana. Korean Hangul is
# space-delimited in practice, so it goes down the word-token path.
_CJK_CLASS = "㐀-䶿一-鿿豈-﫿぀-ヿ"

# Alternation order matters: the CJK branch is tried first at every position, so a
# CJK run can never be absorbed into a word token. The word branch explicitly excludes
# CJK and underscore for the same reason.
_TOKEN_RUN = re.compile(f"[{_CJK_CLASS}]+|[^\\W_{_CJK_CLASS}]+")
_IS_CJK = re.compile(f"^[{_CJK_CLASS}]+$")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> List[str]:
    """Case-folded word tokens plus CJK character bigrams. See the module docstring."""
    tokens: List[str] = []
    for run in _TOKEN_RUN.findall(str(text or "")):
        if _IS_CJK.match(run):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run.casefold())
    return tokens


@dataclass(frozen=True)
class Document:
    """One retrievable unit of the archive.

    ``doc_id`` is stable and human-readable (``paper:2508.01234`` / ``hotspot:<TOPIC_ID>``)
    so a caller can dedupe or cite it. ``url`` is what the synthesis verifier checks
    citations against; it may be empty when the source carried no link, and an empty
    url simply means the document cannot be cited.
    """

    doc_id: str
    kind: str  # "paper" | "hotspot"
    date: str  # YYYY-MM-DD, the archive day it was found under
    title: str
    text: str
    url: str = ""


@dataclass(frozen=True)
class ScoredDocument:
    """A retrieval hit. ``matched_terms`` is what actually fired, for explainability:
    a hit whose only matched term is a stopword-ish bigram is visibly weak in --no-llm
    output instead of looking like a real answer."""

    rank: int
    score: float
    document: Document
    matched_terms: Tuple[str, ...] = ()


@dataclass
class BM25Index:
    """Okapi BM25 with the standard ``ln(1 + (N - df + 0.5) / (df + 0.5))`` IDF.

    That IDF variant is used rather than the textbook ``ln((N - df + 0.5) / (df + 0.5))``
    because the textbook form goes NEGATIVE for terms present in more than half the
    corpus, which on a small archive can make a document score below zero for containing
    the query term. Non-negative IDF keeps "scored 0" meaning "matched nothing".
    """

    documents: Sequence[Document]
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _freqs: List[Counter] = field(default_factory=list, init=False, repr=False)
    _lengths: List[int] = field(default_factory=list, init=False, repr=False)
    _idf: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _avgdl: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.documents = list(self.documents)
        df: Counter = Counter()
        for doc in self.documents:
            tokens = tokenize(f"{doc.title}\n{doc.text}")
            counts = Counter(tokens)
            self._freqs.append(counts)
            self._lengths.append(len(tokens))
            df.update(counts.keys())
        total = len(self.documents)
        self._avgdl = (sum(self._lengths) / total) if total else 0.0
        self._idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    @property
    def is_empty(self) -> bool:
        """True when there is nothing indexed. Callers MUST branch on this before
        reporting "no results": an empty index means we never had anything to search,
        which is a different statement from "searched and found nothing"."""
        return not self.documents

    def __len__(self) -> int:
        return len(self.documents)

    def score(self, query_tokens: Iterable[str], doc_index: int) -> Tuple[float, List[str]]:
        """BM25 score of one document, plus the query terms that actually matched."""
        counts = self._freqs[doc_index]
        length = self._lengths[doc_index]
        norm = self.k1 * (1.0 - self.b + self.b * (length / self._avgdl if self._avgdl else 0.0))
        total = 0.0
        matched: List[str] = []
        for term in query_tokens:
            tf = counts.get(term, 0)
            if not tf:
                continue
            total += self._idf.get(term, 0.0) * (tf * (self.k1 + 1.0)) / (tf + norm)
            matched.append(term)
        return total, matched

    def search(self, query: str, *, top_k: int = 10) -> List[ScoredDocument]:
        """Top-k hits with a positive score, best first.

        Ties break on ``doc_id`` so the ranking is deterministic across runs -- a test
        that asserts on ordering must not depend on dict iteration order.
        """
        if self.is_empty:
            return []
        query_tokens = list(dict.fromkeys(tokenize(query)))  # dedupe, keep order
        if not query_tokens:
            return []
        scored: List[Tuple[float, str, int, List[str]]] = []
        for index, doc in enumerate(self.documents):
            value, matched = self.score(query_tokens, index)
            if value > 0.0:
                scored.append((value, doc.doc_id, index, matched))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [
            ScoredDocument(
                rank=position + 1,
                score=round(value, 4),
                document=self.documents[index],
                matched_terms=tuple(matched),
            )
            for position, (value, _doc_id, index, matched) in enumerate(scored[: max(0, top_k)])
        ]
