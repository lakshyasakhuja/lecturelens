from __future__ import annotations

import pytest

from slideseek.catalog import (
    CatalogError,
    Lecture,
    discover_videos,
    load_catalog,
    resolve_lectures,
    slugify,
)
from slideseek.utils import format_timestamp, parse_timestamp

HEADER = "lecture_id,title,course,instructors,source_url,licence,duration_s,video_link_template\n"


def write_catalog(tmp_path, rows: str):
    path = tmp_path / "lectures.csv"
    path.write_text(HEADER + rows, encoding="utf-8")
    return path


def test_load_catalog_and_link(tmp_path):
    path = write_catalog(
        tmp_path,
        'lec01,"Intro, part 1",Course A,"Ada Example, Bo Sample",https://example.org/1,'
        "CC BY-NC-SA 4.0,3605.5,https://video.example/watch?v=abc&t={seconds}s\n",
    )
    lecture = load_catalog(path)["lec01"]
    assert lecture.title == "Intro, part 1"
    assert lecture.instructors == "Ada Example, Bo Sample"
    assert lecture.duration_s == 3605.5
    assert lecture.link_at(1421.9) == "https://video.example/watch?v=abc&t=1421s"
    assert lecture.link_at(-5) == "https://video.example/watch?v=abc&t=0s"


def test_shipped_catalog_is_valid_and_fully_attributed():
    catalog = load_catalog("config/lectures.csv")
    assert catalog, "the catalogue should list the lectures of the public demo"
    for lecture in catalog.values():
        # Everything a CC BY-NC-SA attribution needs: who, what, where from, which licence.
        assert lecture.title and lecture.course and lecture.instructors
        assert lecture.source_url.startswith("https://ocw.mit.edu/")
        assert lecture.licence == "CC BY-NC-SA 4.0"
        assert lecture.duration_s > 0
        assert "{seconds}" in lecture.video_link_template


@pytest.mark.parametrize(
    "rows, message",
    [
        ("lec 01,T,,,,,,\n", "may only contain"),
        ("lec01,T,,,,,,\nlec01,U,,,,,,\n", "duplicate"),
        ("lec01,T,,,,,abc,\n", "not a number"),
        ("lec01,T,,,,,,https://video.example/watch\n", "must contain"),
    ],
)
def test_catalog_validation(tmp_path, rows, message):
    with pytest.raises(CatalogError, match=message):
        load_catalog(write_catalog(tmp_path, rows))


def test_catalog_requires_id_and_title_columns(tmp_path):
    path = tmp_path / "lectures.csv"
    path.write_text("id,name\nlec01,T\n", encoding="utf-8")
    with pytest.raises(CatalogError, match="missing required column"):
        load_catalog(path)


def test_discover_and_resolve_videos(tmp_path):
    (tmp_path / "lec01.mp4").write_bytes(b"")
    (tmp_path / "Week 3 - Bayes.MKV").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    (tmp_path / ".hidden.mp4").write_bytes(b"")

    videos = discover_videos(tmp_path)
    assert set(videos) == {"lec01", "Week_3_-_Bayes"}

    catalog = {"lec01": Lecture("lec01", "Catalogued", video_link_template="https://x/{seconds}")}
    lectures = resolve_lectures(videos, catalog)
    assert lectures["lec01"].title == "Catalogued"

    local = lectures["Week_3_-_Bayes"]
    assert local.title == "Week 3 - Bayes"
    assert local.link_at(90).startswith("file://")
    assert local.link_at(90).endswith("#t=90")


def test_discover_rejects_colliding_ids(tmp_path):
    (tmp_path / "lec 1.mp4").write_bytes(b"")
    (tmp_path / "lec_1.mkv").write_bytes(b"")
    with pytest.raises(CatalogError, match="both map to"):
        discover_videos(tmp_path)


def test_slugify():
    assert slugify("Week 3: Bayes' rule") == "Week_3_Bayes_rule"
    assert slugify("???") == "lecture"


@pytest.mark.parametrize(
    "seconds, text", [(0, "0:00"), (59.9, "0:59"), (1421, "23:41"), (3723, "1:02:03")]
)
def test_format_timestamp(seconds, text):
    assert format_timestamp(seconds) == text


@pytest.mark.parametrize(
    "text, seconds", [("83", 83), ("83.5", 83.5), ("1:23", 83), ("1:02:03", 3723), (17, 17)]
)
def test_parse_timestamp(text, seconds):
    assert parse_timestamp(text) == seconds


def test_parse_timestamp_rejects_garbage():
    with pytest.raises(ValueError):
        parse_timestamp("1:2:3:4")
