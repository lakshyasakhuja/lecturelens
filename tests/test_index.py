"""The index on its own: combining several into one."""

from __future__ import annotations

import numpy as np
import pytest
from fakes import FakeImageEmbedder, FakeTextEmbedder, make_index

from slideseek.catalog import Lecture
from slideseek.index import Index, InvalidIndex
from slideseek.search import Searcher


def renamed(index: Index, prefix: str) -> Index:
    """The same moments under other lecture ids, as somebody's own library would hold."""
    moments = index.moments.copy()
    moments["lecture_id"] = prefix + moments["lecture_id"]
    moments["moment_id"] = prefix + moments["moment_id"]
    lectures = {
        prefix + lecture_id: Lecture(prefix + lecture_id, f"Own {lecture.title}")
        for lecture_id, lecture in index.lectures.items()
    }
    return Index(moments, dict(index.vectors), lectures, dict(index.manifest))


def test_combining_keeps_every_moment_pointing_at_its_own_keyframe():
    public, own = make_index(), renamed(make_index(), "own-")
    combined = Index.combine(public, own)
    assert len(combined) == 2 * len(public)
    assert len(combined.image) == 2 * len(public.image)
    offset = len(public.image)
    assert combined.moments["image_idx"].tolist() == [
        *public.moments["image_idx"],
        *(public.moments["image_idx"] + offset),
    ]
    for row in range(len(combined)):
        source, local = (public, row) if row < len(public) else (own, row - len(public))
        assert np.array_equal(
            combined.image[combined.moments["image_idx"].iat[row]],
            source.image[source.moments["image_idx"].iat[local]],
        )
    assert set(combined.lectures) == {"lec01", "lec02", "own-lec01", "own-lec02"}
    assert combined.manifest["n_moments"] == len(combined)


def test_a_combined_index_searches_both_and_can_be_limited_to_one():
    combined = Index.combine(make_index(), renamed(make_index(), "own-"))
    searcher = Searcher(
        combined, text_embedder=FakeTextEmbedder(), image_embedder=FakeImageEmbedder()
    )
    found = {r.lecture_id for r in searcher.search("dropout randomly removes units", top_k=10)}
    assert {"lec01", "own-lec01"} <= found
    only_own = searcher.search("dropout", top_k=10, lectures=["own-lec01", "own-lec02"])
    assert only_own and {r.lecture_id for r in only_own} <= {"own-lec01", "own-lec02"}


def test_thumbnails_are_found_in_each_lectures_own_directory(tmp_path):
    public = make_index(tmp_path / "public")
    own = renamed(make_index(), "own-")
    own.moments["thumb"] = "thumbs/own-lec01/000059000.jpg"
    own.root = tmp_path / "own"
    picture = own.root / "thumbs" / "own-lec01" / "000059000.jpg"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"\xff\xd8")

    combined = Index.combine(public, own)
    assert combined.thumb_path(len(public)) == picture
    assert combined.thumb_path(0) is None  # the public fixture has no thumbnails


def test_indexes_that_cannot_be_combined_are_refused():
    with pytest.raises(InvalidIndex, match="two indexes"):
        Index.combine(make_index(), make_index())
    other = renamed(make_index(), "own-")
    other.manifest = {**other.manifest, "models": {"text": "another/model"}}
    with pytest.raises(InvalidIndex, match="text models"):
        Index.combine(make_index(), other)
    with pytest.raises(ValueError):
        Index.combine()
