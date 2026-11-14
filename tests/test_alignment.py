"""Windows and word alignment: which words and which slide end up in which moment."""

from __future__ import annotations

import itertools

import pytest

from lecturelens.asr import Word
from lecturelens.catalog import Lecture
from lecturelens.moments import (
    MomentConfig,
    build_moments,
    moment_text,
    plan_windows,
    transcript_between,
)
from lecturelens.slides import SlideSegment

LECTURE = Lecture("lec01", "Test lecture", video_link_template="https://v.example/?t={seconds}")


def seg(t_start, t_end, kind="slide"):
    return SlideSegment(t_start, t_end, keyframe_t=t_end - 1, kind=kind)


def word(text, start, end):
    return Word(text, start, end)


def test_short_segments_become_one_window_each():
    segments = [seg(0, 20), seg(20, 110), seg(110, 125)]
    windows = plan_windows(segments)
    assert [(w.t_start, w.t_end) for w in windows] == [(0, 20), (20, 110), (110, 125)]
    assert [w.segment for w in windows] == [0, 1, 2]
    assert [w.keyframe_t for w in windows] == [19, 109, 124]


@pytest.mark.parametrize("duration", [90.5, 100, 137, 200, 600, 1234.5])
def test_long_segments_are_tiled_by_overlapping_windows(duration):
    config = MomentConfig()
    windows = plan_windows([seg(50, 50 + duration)], config)

    assert len(windows) >= 2
    assert windows[0].t_start == 50
    assert windows[-1].t_end == pytest.approx(50 + duration)
    for window in windows:
        assert window.t_end - window.t_start == pytest.approx(config.window_s)
    for prev, nxt in itertools.pairwise(windows):
        overlap = prev.t_end - nxt.t_start
        assert overlap >= config.overlap_s - 1e-6  # neighbours share at least the minimum
        assert nxt.t_start > prev.t_start  # and still make progress


def test_windowing_uses_no_more_windows_than_needed():
    # 100 s with 45 s windows and >= 15 s overlap needs exactly three windows.
    windows = plan_windows([seg(0, 100)])
    assert [(w.t_start, w.t_end) for w in windows] == [(0, 45), (27.5, 72.5), (55, 100)]


def test_windows_of_a_slide_share_its_keyframe():
    windows = plan_windows([seg(0, 300)])
    assert {w.keyframe_t for w in windows} == {299}


def test_windows_of_a_dynamic_segment_get_their_own_keyframes():
    windows = plan_windows([seg(0, 300, kind="dynamic")])
    keyframes = [w.keyframe_t for w in windows]
    assert len(set(keyframes)) == len(windows)
    for window in windows:
        assert window.t_start <= window.keyframe_t < window.t_end
        assert window.keyframe_t == int(window.keyframe_t)  # on the 1 fps frame grid


def test_segment_at_exactly_the_limit_is_not_windowed():
    assert len(plan_windows([seg(0, 90)])) == 1
    # Just over the limit. Two 45 s windows overlapping by 15 s span only 75 s, so it takes three.
    assert len(plan_windows([seg(0, 90.1)])) == 3


def test_invalid_moment_config():
    with pytest.raises(ValueError):
        MomentConfig(window_s=30, overlap_s=30)
    with pytest.raises(ValueError):
        MomentConfig(window_s=120, max_segment_s=90)


def test_words_are_assigned_by_midpoint():
    words = [word("alpha", 0.0, 0.4), word("straddler", 9.7, 10.4), word("omega", 10.5, 11.0)]
    # The straddling word's midpoint is 10.05, so it belongs to the second span.
    assert transcript_between(words, 0, 10) == "alpha"
    assert transcript_between(words, 10, 20) == "straddler omega"


def test_every_word_lands_in_exactly_one_adjacent_segment():
    words = [word(f"w{i}", i * 0.37, i * 0.37 + 0.3) for i in range(200)]
    segments = [seg(0, 13), seg(13, 41.5), seg(41.5, 80)]
    moments = build_moments(LECTURE, plan_windows(segments), words, {})
    assigned = [w for m in moments for w in m.transcript.split()]
    assert assigned == [w.text for w in words]


def test_build_moments_attaches_speech_slide_text_and_link():
    segments = [seg(0, 10), seg(10, 25)]
    words = [
        word("hello", 1, 1.5),
        word("class.", 1.5, 2),
        word("Next", 11, 11.4),
        word("slide.", 11.4, 12),
    ]
    slide_texts = {9.0: "Welcome\nCourse outline", 24.0: "Bayes' rule"}

    moments = build_moments(LECTURE, plan_windows(segments), words, slide_texts)

    assert [m.moment_id for m in moments] == ["lec01:0000", "lec01:0001"]
    first, second = moments
    assert (first.t_start, first.t_end) == (0, 10)
    assert first.transcript == "hello class."
    assert first.slide_text == "Welcome\nCourse outline"
    assert first.video_link == "https://v.example/?t=0"
    assert second.transcript == "Next slide."
    assert second.slide_text == "Bayes' rule"
    assert second.video_link == "https://v.example/?t=10"
    assert second.keyframe_t == 24


def test_overlapping_windows_repeat_the_words_they_share():
    words = [word(f"w{i}", i, i + 0.5) for i in range(100)]
    moments = build_moments(LECTURE, plan_windows([seg(0, 100)]), words, {99.0: "One long slide"})
    assert len(moments) == 3
    assert all(m.slide_text == "One long slide" for m in moments)
    assert "w30" in moments[0].transcript and "w30" in moments[1].transcript
    assert moments[0].transcript.split()[0] == "w0"
    assert moments[-1].transcript.split()[-1] == "w99"


def test_moment_without_speech_or_slide_text_is_kept_empty():
    moments = build_moments(LECTURE, plan_windows([seg(0, 10)]), [], {})
    assert moments[0].transcript == ""
    assert moments[0].slide_text == ""


def test_moment_text_labels_sources_and_omits_missing_ones():
    assert moment_text("Bayes' rule\nP(A|B)", "so we  flip it") == (
        "Slide: Bayes' rule P(A|B)\nSpeech: so we flip it"
    )
    assert moment_text("", "only speech") == "Speech: only speech"
    assert moment_text("only slide", "  ") == "Slide: only slide"
    assert moment_text("", "") == ""
