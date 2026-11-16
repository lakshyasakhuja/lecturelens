"""Online search: three rankers, one fusion, and an explanation for every result.

A query is ranked three ways and the rankings are fused:

* ``bm25``   lexical match against each moment's slide text and transcript
* ``dense``  cosine similarity between the query and the moment's text embedding
* ``image``  cosine similarity between the query and the keyframe, in SigLIP 2's space

**Why Reciprocal Rank Fusion.** BM25 scores are unbounded, cosine similarities live in
[-1, 1], and SigLIP similarities cluster in a narrow band of their own. Adding them needs
per-ranker calibration that would have to be re-tuned whenever a model changes. RRF uses
only each ranker's *order*::

    RRF(d) = sum over rankers of  weight / (k + rank(d))

so there is nothing to calibrate, and one hyperparameter, ``k``. A document a ranker did
not retrieve contributes nothing from that ranker.

**Ties share a rank.** Windows cut from one long slide share a keyframe, so they get
identical image scores. Ranking them 1, 2, 3, ... would let one slide occupy the whole
image ranking and push every other slide out. Instead they all take the rank of their
keyframe, and the text rankers decide between them.

**Results are landing points.** Two hits in the same lecture that start within
``dedupe_window_s`` of each other would drop the student at the same place, so only the
better one is kept.

Everything here runs on a CPU. The only models involved are the two query encoders; the
index side is matrix products over a few thousand rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lecturelens.embed import EmbedConfig, ImageEmbedder, TextEmbedder
from lecturelens.index import Index
from lecturelens.moments import moment_text
from lecturelens.text import analyze, match_spans
from lecturelens.utils import format_timestamp

# Which index data each ranker uses.
LEXICAL_FIELDS = {
    "bm25": ("slide_text", "transcript"),
    "bm25_speech": ("transcript",),
    "bm25_slide": ("slide_text",),
}
DENSE_VECTORS = {"dense": "text", "dense_speech": "speech", "dense_slide": "slide"}
RANKERS = (*LEXICAL_FIELDS, *DENSE_VECTORS, "image")
DEFAULT_RANKERS = ("bm25", "dense", "image")

# Badges. A source (speech or slide) is named as a match when it contains at least this
# share of the query's terms, and at least this fraction of what the other source contains.
MIN_COVERAGE = 1 / 3
COVERAGE_RATIO = 0.6


@dataclass(frozen=True)
class SearchConfig:
    rankers: tuple[str, ...] = DEFAULT_RANKERS
    weights: Mapping[str, float] = field(default_factory=dict)  # missing rankers weigh 1.0
    rrf_k: int = 60
    depth: int = 100  # how many candidates each ranker contributes to the fusion
    top_k: int = 5
    dedupe_window_s: float = 20.0
    rerank_depth: int = 20  # how many fused candidates the optional reranker rescores
    badge_depth: int = 10  # a ranker "matched" a result if it ranked it this high
    # A result is badged "visual" only if SigLIP itself rates the query and the keyframe as
    # a match with at least this probability. A rank cannot say that: some keyframe is
    # always ranked first, even for a query that describes nothing visual. SigLIP is
    # cautious with long descriptions (a correct match often scores 0.2 to 0.3), hence a
    # threshold well under one half. Tune it on the dev queries.
    visual_min_prob: float = 0.15
    # Optional abstention for the image ranker: keyframes below this match probability are
    # not ranked at all, the way BM25 does not rank a moment that shares no term with the
    # query. 0 disables it, so every keyframe is ranked (plain RRF, the default).
    image_min_prob: float = 0.0

    def __post_init__(self) -> None:
        unknown = set(self.rankers) - set(RANKERS)
        if unknown:
            raise ValueError(f"unknown ranker(s) {sorted(unknown)}; choose from {RANKERS}")
        if not self.rankers:
            raise ValueError("at least one ranker is required")


@dataclass(frozen=True)
class QueryEncoding:
    """A query in every form the rankers need. Encode once, rank many ways."""

    query: str
    terms: tuple[str, ...]
    dense: np.ndarray | None = None
    image: np.ndarray | None = None


@dataclass(frozen=True)
class Result:
    rank: int
    row: int  # row of the moment in the index
    moment_id: str
    lecture_id: str
    lecture_title: str
    t_start: float
    t_end: float
    timestamp: str  # t_start as a player would show it
    video_link: str
    thumb: Path | None
    score: float
    signals: dict[str, int]  # ranker -> the rank it gave this moment
    badges: tuple[str, ...]  # any of "said", "on slide", "visual"
    snippet: str
    highlights: tuple[tuple[int, int], ...]  # character spans within ``snippet``
    slide_text: str
    transcript: str


# --------------------------------------------------------------------------------------
# Fusion and deduplication: pure functions
# --------------------------------------------------------------------------------------


def ranks_from_order(order: Sequence[int]) -> dict[int, int]:
    """``[doc, doc, ...]`` best first -> ``{doc: rank}`` with ranks starting at 1."""
    return {int(doc): position for position, doc in enumerate(order, start=1)}


def rrf(
    rankings: Mapping[str, Mapping[int, int]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    ``rankings`` maps a ranker's name to ``{doc: rank}`` (rank 1 is best). Returns
    ``(doc, score)`` pairs, best first. Ties in the fused score are broken by the best
    individual rank, then by document id, so the output is deterministic.
    """
    weights = weights or {}
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for name, ranks in rankings.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0:
            continue
        for doc, rank in ranks.items():
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
            best[doc] = min(best.get(doc, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best[item[0]], item[0]))


def dedupe(
    order: Iterable[int],
    lecture_ids: Sequence[str],
    starts: Sequence[float],
    window_s: float = 20.0,
    limit: int | None = None,
) -> list[int]:
    """Drop results that land within ``window_s`` of a better result in the same lecture."""
    kept: list[int] = []
    for doc in order:
        duplicate = any(
            lecture_ids[doc] == lecture_ids[other] and abs(starts[doc] - starts[other]) <= window_s
            for other in kept
        )
        if not duplicate:
            kept.append(doc)
            if limit is not None and len(kept) >= limit:
                break
    return kept


def make_snippet(
    text: str, terms: Iterable[str], width: int = 240
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """A ``width``-character excerpt of ``text`` around the densest cluster of matches.

    Returns the excerpt and the character spans within it to highlight. With no lexical
    match (a purely semantic hit) the excerpt is the opening of the text.
    """
    text = " ".join(text.split())
    if not text:
        return "", ()
    spans = match_spans(text, terms)
    if len(text) <= width:
        return text, tuple(spans)

    start = 0
    if spans:
        # Slide a window over the match starts and keep the one covering the most matches.
        best_count = 0
        for i, (anchor, _) in enumerate(spans):
            count = sum(1 for a, b in spans[i:] if b <= anchor + width)
            if count > best_count:
                best_count, start = count, anchor
        start = max(0, start - width // 6)  # a little leading context
    end = min(len(text), start + width)
    if end - start < width:
        start = max(0, end - width)
    # Snap to word boundaries so the excerpt never begins or ends mid-word.
    if start > 0:
        start = text.find(" ", start) + 1 or start
    if end < len(text):
        end = text.rfind(" ", start, end) if text.rfind(" ", start, end) > start else end

    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(text) else ""
    offset = len(prefix) - start
    kept = tuple((a + offset, b + offset) for a, b in spans if a >= start and b <= end)
    return prefix + text[start:end] + suffix, kept


# --------------------------------------------------------------------------------------
# The searcher
# --------------------------------------------------------------------------------------


class Searcher:
    """Loads an index once and answers queries against it."""

    def __init__(
        self,
        index: Index,
        config: SearchConfig | None = None,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
        reranker=None,
        device: str = "cpu",
    ) -> None:
        self.index = index
        self.config = config or SearchConfig()
        # The query must be encoded by the same models that encoded the index.
        models = index.manifest.get("models", {})
        defaults = EmbedConfig()
        self.embed_config = EmbedConfig(
            text_model=models.get("text", defaults.text_model),
            image_model=models.get("image", defaults.image_model),
            query_instruction=index.manifest.get("query_instruction", defaults.query_instruction),
            device=device,
        )
        self._text_embedder = text_embedder
        self._image_embedder = image_embedder
        self.reranker = reranker
        calibration = index.manifest.get("image_calibration") or {}
        self._image_scale = calibration.get("scale")
        self._image_bias = calibration.get("bias")

        moments = index.moments
        self._lecture_ids = moments["lecture_id"].tolist()
        self._starts = moments["t_start"].to_numpy(dtype=np.float64)
        self._image_idx = moments["image_idx"].to_numpy(dtype=np.int64)
        self._has_vector = {
            name: np.linalg.norm(index.vectors[vec], axis=1) > 0
            if len(moments)
            else np.zeros(0, bool)
            for name, vec in DENSE_VECTORS.items()
        }
        self._bm25: dict[str, object] = {}

    # ---- encoders ----------------------------------------------------------------------

    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = TextEmbedder(self.embed_config)
        return self._text_embedder

    @property
    def image_embedder(self) -> ImageEmbedder:
        if self._image_embedder is None:
            self._image_embedder = ImageEmbedder(self.embed_config)
        return self._image_embedder

    def warm_up(self, rankers: Sequence[str] | None = None) -> None:
        """Load whatever the given rankers need, so the first real query is not slow."""
        self.encode("warm up", rankers)
        for name in rankers or self.config.rankers:
            if name in LEXICAL_FIELDS:
                self._lexical(name)

    def encode(self, query: str, rankers: Sequence[str] | None = None) -> QueryEncoding:
        """Encode a query for the given rankers (default: the configured ones)."""
        rankers = tuple(rankers or self.config.rankers)
        dense = image = None
        if any(name in DENSE_VECTORS for name in rankers):
            dense = self.text_embedder.encode_queries([query])[0]
        if "image" in rankers:
            image = self.image_embedder.encode_texts([query])[0]
        return QueryEncoding(query=query, terms=tuple(analyze(query)), dense=dense, image=image)

    # ---- individual rankers ----------------------------------------------------------------

    def _lexical(self, name: str):
        """BM25 over the ranker's fields, built on first use (about 0.5 s per 2,000 moments)."""
        if name not in self._bm25:
            import bm25s

            logging.getLogger("bm25s").setLevel(logging.WARNING)  # it logs every build at DEBUG
            fields = LEXICAL_FIELDS[name]
            moments = self.index.moments
            corpus = [
                analyze(" ".join(str(moments[f].iat[row]) for f in fields))
                for row in range(len(moments))
            ]
            retriever = bm25s.BM25()
            retriever.index(corpus, show_progress=False)
            self._bm25[name] = retriever
        return self._bm25[name]

    def scores(
        self, ranker: str, encoding: QueryEncoding, image_min_prob: float | None = None
    ) -> np.ndarray:
        """One score per moment; ``-inf`` marks moments the ranker cannot retrieve at all.

        ``image_min_prob`` overrides the configured abstention floor of the image ranker.
        """
        n = len(self.index)
        if ranker in LEXICAL_FIELDS:
            if not encoding.terms or n == 0:
                return np.full(n, -np.inf)
            scores = np.asarray(
                self._lexical(ranker).get_scores(list(encoding.terms)), dtype=np.float64
            )
            return np.where(scores > 0, scores, -np.inf)  # no shared term: not a match
        if ranker in DENSE_VECTORS:
            if encoding.dense is None:
                raise ValueError(f"query was not encoded for {ranker!r}")
            scores = self.index.vectors[DENSE_VECTORS[ranker]] @ encoding.dense
            return np.where(self._has_vector[ranker], scores, -np.inf)
        if ranker == "image":
            if encoding.image is None:
                raise ValueError("query was not encoded for 'image'")
            similarity = (self.index.image @ encoding.image).astype(np.float64)
            floor = self.config.image_min_prob if image_min_prob is None else image_min_prob
            if floor > 0 and self._image_scale is not None and self._image_bias is not None:
                logit = self._image_scale * similarity + self._image_bias
                similarity = np.where(logit >= np.log(floor / (1 - floor)), similarity, -np.inf)
            return similarity[self._image_idx]
        raise ValueError(f"unknown ranker {ranker!r}")

    def rank(
        self,
        ranker: str,
        encoding: QueryEncoding,
        mask: np.ndarray | None = None,
        depth: int | None = None,
        image_min_prob: float | None = None,
    ) -> dict[int, int]:
        """``{row: rank}`` for the ranker's top ``depth`` moments. Equal scores share a rank."""
        depth = depth or self.config.depth
        scores = self.scores(ranker, encoding, image_min_prob)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores, kind="stable")
        ranks: dict[int, int] = {}
        rank, previous = 0, None
        for row in order:
            score = scores[row]
            if not np.isfinite(score):
                break
            if previous is None or score < previous:
                rank += 1
                previous = score
                if rank > depth:
                    break
            ranks[int(row)] = rank
        return ranks

    # ---- the full search -------------------------------------------------------------------

    def lecture_mask(self, lectures: Iterable[str] | None) -> np.ndarray | None:
        if not lectures:
            return None
        wanted = set(lectures)
        return np.fromiter((lid in wanted for lid in self._lecture_ids), dtype=bool)

    def fuse(
        self,
        encoding: QueryEncoding,
        rankers: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
        mask: np.ndarray | None = None,
    ) -> tuple[list[int], dict[int, float], dict[str, dict[int, int]]]:
        """Rank with each ranker and fuse. Returns ``(order, fused scores, per-ranker ranks)``."""
        rankers = tuple(rankers or self.config.rankers)
        per_ranker = {name: self.rank(name, encoding, mask) for name in rankers}
        fused = rrf(per_ranker, k=self.config.rrf_k, weights=weights or self.config.weights)
        return [doc for doc, _ in fused], dict(fused), per_ranker

    def search(
        self,
        query: str,
        top_k: int | None = None,
        lectures: Iterable[str] | None = None,
        rankers: Sequence[str] | None = None,
        weights: Mapping[str, float] | None = None,
        rerank: bool = False,
    ) -> list[Result]:
        """The top moments for a query, deduplicated and explained."""
        query = query.strip()
        if not query or len(self.index) == 0:
            return []
        top_k = top_k or self.config.top_k
        rankers = tuple(rankers or self.config.rankers)
        encoding = self.encode(query, rankers)
        order, scores, per_ranker = self.fuse(
            encoding, rankers, weights, self.lecture_mask(lectures)
        )
        if rerank and order:
            order, scores = self.rerank(query, order, scores)
        kept = dedupe(order, self._lecture_ids, self._starts, self.config.dedupe_window_s, top_k)
        return [
            self._explain(position, row, scores[row], encoding, per_ranker)
            for position, row in enumerate(kept, start=1)
        ]

    def rerank(
        self, query: str, order: list[int], scores: dict[int, float]
    ) -> tuple[list[int], dict[int, float]]:
        """Rescore the head of the fused list with the cross-encoder; the tail is cut off."""
        if self.reranker is None:
            from lecturelens.rerank import Reranker

            self.reranker = Reranker(device=self.embed_config.device)
        head = order[: self.config.rerank_depth]
        moments = self.index.moments
        documents = [
            moment_text(moments["slide_text"].iat[row], moments["transcript"].iat[row])
            for row in head
        ]
        relevance = self.reranker.score(query, documents)
        reordered = [row for _, row in sorted(zip(-np.asarray(relevance), head, strict=True))]
        return reordered, {row: float(s) for row, s in zip(head, relevance, strict=True)}

    # ---- explanation ---------------------------------------------------------------------

    def _explain(
        self,
        position: int,
        row: int,
        score: float,
        encoding: QueryEncoding,
        per_ranker: Mapping[str, Mapping[int, int]],
    ) -> Result:
        moments = self.index.moments
        slide_text = str(moments["slide_text"].iat[row])
        transcript = str(moments["transcript"].iat[row])
        lecture_id = self._lecture_ids[row]
        signals = {name: ranks[row] for name, ranks in per_ranker.items() if row in ranks}
        badges = self._badges(row, encoding, signals, slide_text, transcript)

        # The snippet is what was said. The slide is shown as a picture next to it, and OCR of
        # a slide full of equations does not read well; slide text is the fallback for a
        # moment with no speech.
        snippet, highlights = make_snippet(transcript or slide_text, encoding.terms)
        lecture = self.index.lectures.get(lecture_id)
        return Result(
            rank=position,
            row=row,
            moment_id=str(moments["moment_id"].iat[row]),
            lecture_id=lecture_id,
            lecture_title=lecture.title if lecture else lecture_id,
            t_start=float(moments["t_start"].iat[row]),
            t_end=float(moments["t_end"].iat[row]),
            timestamp=format_timestamp(float(moments["t_start"].iat[row])),
            video_link=str(moments["video_link"].iat[row]),
            thumb=self.index.thumb_path(row),
            score=float(score),
            signals=signals,
            badges=badges,
            snippet=snippet,
            highlights=highlights,
            slide_text=slide_text,
            transcript=transcript,
        )

    def _badges(
        self,
        row: int,
        encoding: QueryEncoding,
        signals: Mapping[str, int],
        slide_text: str,
        transcript: str,
    ) -> tuple[str, ...]:
        """Which signals matched: what was said, what was on the slide, what it looked like.

        Lexical evidence comes first, measured as *coverage*: the share of the query's terms
        found in the transcript, and in the slide text. One shared word out of six is not a
        match, so a source needs at least a third of the terms; and when one source covers
        far more of the question than the other, only that one is named. A semantic match
        with too few shared words is attributed to whichever source is closer to the query
        in embedding space.
        """
        depth = self.config.badge_depth
        terms = set(encoding.terms)
        speech_cover = len(terms & set(analyze(transcript))) / len(terms) if terms else 0.0
        slide_cover = len(terms & set(analyze(slide_text))) / len(terms) if terms else 0.0
        best = max(speech_cover, slide_cover)
        said = speech_cover >= MIN_COVERAGE and speech_cover >= COVERAGE_RATIO * best
        on_slide = slide_cover >= MIN_COVERAGE and slide_cover >= COVERAGE_RATIO * best

        dense_hit = any(signals.get(name, depth + 1) <= depth for name in DENSE_VECTORS)
        if not (said or on_slide) and dense_hit and encoding.dense is not None:
            speech_sim = float(self.index.speech[row] @ encoding.dense)
            slide_sim = float(self.index.slide[row] @ encoding.dense)
            if speech_sim >= slide_sim:
                said = True
            else:
                on_slide = True

        badges = []
        if said:
            badges.append("said")
        if on_slide:
            badges.append("on slide")
        if signals.get("image", depth + 1) <= depth and self._looks_like(row, encoding):
            badges.append("visual")
        return tuple(badges)

    def match_probability(self, row: int, encoding: QueryEncoding) -> float | None:
        """SigLIP's probability that the query describes this moment's keyframe.

        ``None`` when the query was not encoded for the image ranker, or the index does not
        record the model's calibration.
        """
        if encoding.image is None or self._image_scale is None or self._image_bias is None:
            return None
        similarity = float(self.index.image[self._image_idx[row]] @ encoding.image)
        return float(1.0 / (1.0 + np.exp(-(self._image_scale * similarity + self._image_bias))))

    def _looks_like(self, row: int, encoding: QueryEncoding) -> bool:
        probability = self.match_probability(row, encoding)
        return probability is None or probability >= self.config.visual_min_prob
