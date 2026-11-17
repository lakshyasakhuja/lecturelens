"""Text analysis shared by lexical search, snippets and jargon extraction.

One analyser, used everywhere a string becomes terms, so the index and the query can never
disagree about what a token is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

_TOKEN = re.compile(r"[a-z0-9]+")

# A conventional English stopword list (function words only). Kept short on purpose: words
# like "not", "why" and "how" carry meaning in student questions, so they stay searchable.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his i if in into is it
    its itself just me more most my myself of off on once only or other our ours ourselves
    out over own s same she should so some such t than that the their theirs them themselves
    then there these they this those through to too under until up very was we were what
    when where which while who whom will with would you your yours yourself yourselves
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens: ``"L2-norm (x^2)"`` -> ``l2 norm x 2``."""
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _stemmer():
    import Stemmer

    return Stemmer.Stemmer("english")


@lru_cache(maxsize=65_536)
def stem(token: str) -> str:
    """Snowball (Porter2) stem, so ``validation sets`` matches ``validating a set``."""
    return _stemmer().stemWord(token)


def analyze(text: str, *, stemming: bool = True) -> list[str]:
    """The full analysis chain: tokenise, drop stopwords and stray single letters, stem."""
    terms = []
    for token in tokenize(text):
        if token in STOPWORDS or (len(token) == 1 and not token.isdigit()):
            continue
        terms.append(stem(token) if stemming else token)
    return terms


_WORD = re.compile(r"[A-Za-z0-9]+")


def match_spans(text: str, query_terms: Iterable[str]) -> list[tuple[int, int]]:
    """Character spans of the words in ``text`` whose stem is one of ``query_terms``.

    ``query_terms`` must already be analysed. Matching on stems means a query for
    "regularise" highlights "regularisation" in the snippet, which is what BM25 matched.
    """
    wanted = set(query_terms)
    if not wanted:
        return []
    spans = []
    for match in _WORD.finditer(text):
        token = match.group().lower()
        if token in STOPWORDS:
            continue
        if stem(token) in wanted:
            spans.append(match.span())
    return spans
