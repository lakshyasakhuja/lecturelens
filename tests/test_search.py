"""The searcher end to end, on a hand-built index with deterministic stand-in encoders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fakes import DIM, FakeImageEmbedder, FakeTextEmbedder, make_index

from lecturelens.index import Index, InvalidIndex
from lecturelens.search import SearchConfig, Searcher


@pytest.fixture()
def searcher():
    return Searcher(
        make_index(), text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder()
    )


def ids(results):
    return [r.moment_id for r in results]


# ---- individual rankers ------------------------------------------------------------------


def test_bm25_matches_stems_in_slide_text_and_transcript(searcher):
    encoding = searcher.encode("validating a set", rankers=["bm25"])
    assert list(searcher.rank("bm25", encoding)) == [1]  # "The validation set" is slide text
    encoding = searcher.encode("chain rule", rankers=["bm25"])
    assert list(searcher.rank("bm25", encoding)) == [6]  # spoken, not on the slide


def test_bm25_field_variants_see_only_their_field(searcher):
    encoding = searcher.encode("chain rule", rankers=["bm25"])
    assert list(searcher.rank("bm25_speech", encoding)) == [6]
    assert searcher.rank("bm25_slide", encoding) == {}
    encoding = searcher.encode("backpropagation", rankers=["bm25"])
    assert searcher.rank("bm25_speech", encoding) == {}
    assert list(searcher.rank("bm25_slide", encoding)) == [6]


def test_bm25_returns_nothing_without_a_shared_term(searcher):
    assert searcher.rank("bm25", searcher.encode("quantum chromodynamics", ["bm25"])) == {}
    assert searcher.rank("bm25", searcher.encode("the of and", ["bm25"])) == {}  # all stopwords


def test_dense_ranker_never_returns_a_moment_without_text(searcher):
    encoding = searcher.encode("three-layer network", rankers=["dense_slide"])
    ranks = searcher.rank("dense_slide", encoding)
    assert 7 not in ranks  # row 7 has no slide text, so no slide vector
    assert min(ranks, key=ranks.get) == 5


def test_image_ranker_gives_windows_of_one_slide_the_same_rank(searcher):
    encoding = searcher.encode("bullet points", rankers=["image"])
    ranks = searcher.rank("image", encoding)
    assert ranks[3] == ranks[4] == 1  # two windows, one keyframe
    assert all(rank >= 2 for row, rank in ranks.items() if row not in (3, 4))


def test_ranking_depth_counts_distinct_ranks(searcher):
    encoding = searcher.encode("bullet points", rankers=["image"])
    assert set(searcher.rank("image", encoding, depth=1)) == {3, 4}


def test_lecture_mask_restricts_every_ranker(searcher):
    mask = searcher.lecture_mask(["lec02"])
    for ranker in ("bm25", "dense", "image"):
        encoding = searcher.encode("weighted sum of inputs curve", rankers=[ranker])
        rows = searcher.rank(ranker, encoding, mask=mask)
        assert rows and all(
            searcher.index.moments["lecture_id"].iat[row] == "lec02" for row in rows
        )
    assert searcher.lecture_mask(None) is None
    assert searcher.lecture_mask([]) is None


def test_scoring_a_ranker_the_query_was_not_encoded_for_is_an_error(searcher):
    encoding = searcher.encode("anything", rankers=["bm25"])
    with pytest.raises(ValueError, match="not encoded"):
        searcher.scores("dense", encoding)
    with pytest.raises(ValueError, match="not encoded"):
        searcher.scores("image", encoding)


# ---- full search -------------------------------------------------------------------------


def test_keyword_query_finds_the_moment_and_explains_it(searcher):
    results = searcher.search("chain rule")
    top = results[0]
    assert top.moment_id == "lec02:0006"
    assert top.rank == 1
    assert top.lecture_title == "Neural networks"
    assert top.timestamp == "1:20"
    assert top.video_link == "https://v.example/lec02?t=80"
    assert top.signals["bm25"] == 1 and top.signals["dense"] == 1
    assert "said" in top.badges and "on slide" not in top.badges
    assert [top.snippet[a:b] for a, b in top.highlights] == ["chain", "rule"]


def test_slide_only_match_is_badged_on_slide_but_the_snippet_is_what_was_said(searcher):
    top = searcher.search("backpropagation")[0]
    assert top.moment_id == "lec02:0006"
    assert top.badges == ("on slide",)
    # The word is only on the slide, so nothing in the spoken snippet is highlighted.
    assert top.snippet == "the chain rule applied from the output back to the input"
    assert top.highlights == ()
    assert top.slide_text == "Backpropagation"


def test_a_moment_with_no_speech_falls_back_to_its_slide_text():
    index = make_index()
    index.moments.loc[6, "transcript"] = ""
    silent = Searcher(index, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder())
    top = silent.search("backpropagation", rankers=["bm25"])[0]
    assert top.snippet == "Backpropagation"
    assert [top.snippet[a:b] for a, b in top.highlights] == ["Backpropagation"]


def test_visual_query_is_badged_visual(searcher):
    top = searcher.search("diagram of circles connected by lines", rankers=["image"])[0]
    assert top.moment_id == "lec02:0005"
    assert "visual" in top.badges
    assert top.signals == {"image": 1}


def test_badges_follow_how_much_of_the_question_each_source_covers(searcher):
    # Row 1: slide "The validation set"; speech "we keep some data aside that the model is
    # never fitted on". The question's words are all in the speech and none on the slide.
    (said_only,) = [r for r in searcher.search("data the model is never fitted on") if r.row == 1]
    assert said_only.badges[:1] == ("said",) and "on slide" not in said_only.badges

    # All on the slide, none spoken.
    (slide_only,) = [r for r in searcher.search("validation set", rankers=["bm25"]) if r.row == 1]
    assert slide_only.badges == ("on slide",)

    # Split evenly between the two: both are named.
    (both,) = [
        r for r in searcher.search("validation set data aside", rankers=["bm25"]) if r.row == 1
    ]
    assert both.badges == ("said", "on slide")

    # One word in six is not a match: no lexical badge from a single stray term. The dense
    # ranker did not run here, so there is nothing to attribute it to either.
    stray = [
        r
        for r in searcher.search(
            "gradient clipping momentum warmup schedules data", rankers=["bm25"]
        )
        if r.row == 1
    ]
    assert stray and stray[0].badges == ()


def test_visual_badge_needs_a_real_match_not_just_a_top_rank(searcher):
    # "backpropagation" describes nothing visual. Some keyframe is still ranked first by the
    # image ranker (here all of them tie), but none is a probable match, so no badge.
    top = searcher.search("backpropagation")[0]
    assert top.signals["image"] == 1
    assert "visual" not in top.badges
    encoding = searcher.encode("backpropagation")
    assert searcher.match_probability(top.row, encoding) < 0.05

    encoding = searcher.encode("diagram circles connected lines")
    assert searcher.match_probability(5, encoding) > 0.95

    # Without calibration in the manifest there is no probability, only the rank.
    uncalibrated = make_index()
    uncalibrated.manifest = {"models": {}}
    fallback = Searcher(
        uncalibrated, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder()
    )
    assert fallback.match_probability(5, encoding) is None
    assert "visual" in fallback.search("backpropagation")[0].badges


def test_image_ranker_can_abstain_on_improbable_matches():
    def image_ranks(query, **config):
        searcher = Searcher(
            make_index(),
            SearchConfig(**config),
            text_embedder=FakeTextEmbedder(),
            image_embedder=FakeImageEmbedder(),
        )
        return searcher.rank("image", searcher.encode(query, ["image"]))

    # By default every keyframe is ranked, however poor the match.
    assert len(image_ranks("backpropagation")) == 8
    # With a floor, a query that matches no keyframe gets no votes from the image ranker...
    assert image_ranks("backpropagation", image_min_prob=0.1) == {}
    # ...and one that does match keeps only the keyframes that clear the floor.
    assert set(image_ranks("diagram circles connected lines", image_min_prob=0.1)) == {5}


def test_results_are_deduplicated_to_distinct_landing_points(searcher):
    # Rows 3 and 4 are windows of the same slide but start 30 s apart: both may appear.
    both = ids(searcher.search("regularisation", rankers=["bm25"]))
    assert sorted(both) == ["lec01:0003", "lec01:0004"]
    # With a wider window the second is the same landing point as the first.
    wide = Searcher(
        make_index(),
        SearchConfig(dedupe_window_s=40),
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )
    assert ids(wide.search("regularisation", rankers=["bm25"])) == both[:1]


def test_top_k_and_lecture_filter(searcher):
    assert len(searcher.search("the model training data", top_k=2)) == 2
    results = searcher.search("training", lectures=["lec02"], top_k=8)
    assert results and all(r.lecture_id == "lec02" for r in results)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_blank_query_and_empty_index(searcher):
    assert searcher.search("   ") == []
    empty = make_index()
    empty.moments = empty.moments.iloc[0:0]
    empty.vectors = {k: v[0:0] for k, v in empty.vectors.items()}
    assert Searcher(empty, text_embedder=FakeTextEmbedder()).search("anything") == []


def test_only_the_encoders_a_search_needs_are_called():
    text, image = FakeTextEmbedder(), FakeImageEmbedder()
    searcher = Searcher(make_index(), text_embedder=text, image_embedder=image)
    searcher.search("chain rule", rankers=["bm25"])
    assert (text.calls, image.calls) == (0, 0)
    searcher.search("chain rule", rankers=["bm25", "dense"])
    assert (text.calls, image.calls) == (1, 0)
    searcher.search("chain rule")
    assert (text.calls, image.calls) == (2, 1)


def test_weights_can_silence_a_ranker(searcher):
    # The image ranker alone prefers the foggy photo; the text rankers prefer the fog analogy
    # moment too, so use a query where they disagree.
    query = "equations for the validation set"
    by_text = searcher.search(query, weights={"image": 0.0})[0].moment_id
    by_image = searcher.search(query, weights={"bm25": 0.0, "dense": 0.0})[0].moment_id
    assert by_text == "lec01:0001"
    assert by_image == "lec02:0006"


def test_reranker_reorders_the_fused_head(searcher):
    class ReverseReranker:
        """Scores documents in reverse of the order given, to prove the order is used."""

        def score(self, query, documents):
            self.documents = list(documents)
            return np.arange(len(documents), dtype=float)

    searcher.reranker = ReverseReranker()
    plain = ids(searcher.search("training model data", top_k=3))
    reranked = ids(searcher.search("training model data", top_k=3, rerank=True))
    assert reranked[0] != plain[0]
    assert searcher.reranker.documents[0].startswith("Slide: ")


def test_unknown_ranker_is_rejected():
    with pytest.raises(ValueError, match="unknown ranker"):
        SearchConfig(rankers=("bm25", "telepathy"))
    with pytest.raises(ValueError, match="at least one"):
        SearchConfig(rankers=())


# ---- index persistence -------------------------------------------------------------------


def test_index_round_trips_through_disk(tmp_path):
    original = make_index(tmp_path / "index")
    loaded = Index.load(tmp_path / "index")

    pd.testing.assert_frame_equal(loaded.moments, original.moments)
    for name in ("text", "speech", "slide", "image"):
        assert loaded.vectors[name].dtype == np.float32
        np.testing.assert_allclose(loaded.vectors[name], original.vectors[name], atol=1e-3)
    assert loaded.lectures == original.lectures
    assert loaded.manifest["n_moments"] == 8
    assert loaded.manifest["n_keyframes"] == 7
    assert loaded.manifest["dims"]["text"] == DIM

    # Search gives the same answer from the reloaded, float16-stored vectors.
    searcher = Searcher(
        loaded, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder()
    )
    assert searcher.search("chain rule")[0].moment_id == "lec02:0006"


def test_loading_a_missing_or_incompatible_index_fails_clearly(tmp_path):
    with pytest.raises(InvalidIndex, match="no index at"):
        Index.load(tmp_path / "nowhere")
    make_index(tmp_path / "index")
    manifest = tmp_path / "index" / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"format": 1', '"format": 99'))
    with pytest.raises(InvalidIndex, match="format 99"):
        Index.load(tmp_path / "index")


def test_validate_catches_inconsistent_pieces():
    index = make_index()
    index.vectors["speech"] = index.vectors["speech"][:-1]
    with pytest.raises(InvalidIndex, match="speech has 7 rows"):
        index.validate()

    index = make_index()
    index.moments.loc[0, "image_idx"] = 99
    with pytest.raises(InvalidIndex, match="image_idx"):
        index.validate()

    index = make_index()
    index.lectures.pop("lec02")
    with pytest.raises(InvalidIndex, match="not in the catalogue"):
        index.validate()


# ---- pictures --------------------------------------------------------------------------------


def picture_searcher(ocr=True):
    from fakes import FakePictureOcr

    index = make_index()
    return Searcher(
        index,
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        ocr=FakePictureOcr() if ocr else None,
    )


def test_a_picture_finds_the_slide_by_its_look_and_its_words():
    from fakes import make_picture, open_picture

    searcher = picture_searcher()
    picture = open_picture(make_picture("plot two curves rising falling", "Overfitting"))
    results = searcher.search_picture(picture, top_k=3)
    assert results[0].moment_id == "lec01:0002"
    assert results[0].signals == {"image": 1, "bm25_slide": 1}
    assert set(results[0].badges) == {"on slide", "visual"}
    assert "said" not in {badge for r in results for badge in r.badges}, "nobody said anything"


def test_a_picture_with_no_readable_words_is_matched_on_its_look():
    from fakes import make_picture, open_picture

    searcher = picture_searcher()
    results = searcher.search_picture(open_picture(make_picture("diagram circles connected lines")))
    assert results[0].moment_id == "lec02:0005"
    assert set(results[0].signals) == {"image"}


def test_the_words_alone_can_find_a_slide_whose_look_is_unclear():
    from fakes import make_picture, open_picture

    searcher = picture_searcher()
    picture = open_picture(make_picture("something blurry", "Backpropagation"))
    assert searcher.search_picture(picture)[0].moment_id == "lec02:0006"


def test_without_ocr_a_picture_is_matched_on_its_look_only(monkeypatch):
    from fakes import make_picture, open_picture

    searcher = picture_searcher(ocr=False)
    monkeypatch.setattr(type(searcher), "can_read_pictures", property(lambda self: False))
    encoding = searcher.encode_picture(open_picture(make_picture("equations", "Backpropagation")))
    assert encoding.kind == "picture" and encoding.terms == () and encoding.query == ""


def test_the_text_to_image_floor_does_not_apply_to_pictures():
    """The calibration turns text-to-image similarity into a probability; not picture-to-picture."""
    from fakes import make_picture, open_picture

    index = make_index()
    searcher = Searcher(
        index,
        SearchConfig(image_min_prob=0.99),
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
    )
    encoding = searcher.encode_picture(open_picture(make_picture("bullet")), read_text=False)
    assert np.isfinite(searcher.scores("image", encoding)).any()


def test_windows_of_one_slide_become_one_picture_result_at_its_start():
    """Rows 3 and 4 are two windows of one long slide: one result, where the slide came up."""
    from fakes import make_picture, open_picture

    searcher = picture_searcher()
    results = searcher.search_picture(open_picture(make_picture("bullet points", "Regularisation")))
    rows = [r.row for r in results]
    assert rows[0] == 3 and 4 not in rows
