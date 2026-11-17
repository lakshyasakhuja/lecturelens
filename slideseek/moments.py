"""Moments: the unit of retrieval.

A moment is a stretch of one lecture with three things attached: the words spoken during
it, the text on the slide that was showing, and a picture of that slide. Search returns
moments; evaluation scores moments.

Building them takes two steps:

1. ``plan_windows`` turns slide segments into time windows. A slide shown for up to
   ``max_segment_s`` is one window. A slide that stays up longer is cut into overlapping
   windows, because a five-minute explanation embedded as one chunk is too diluted to match
   a specific question, and a result that lands the student five minutes early is not much
   of a result. This step uses time only, so the pipeline knows which keyframes to extract
   before any speech recognition has run.

2. ``build_moments`` attaches the transcript words and slide text to each window.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from slideseek.asr import Word
from slideseek.catalog import Lecture
from slideseek.slides import SlideSegment


@dataclass(frozen=True)
class MomentConfig:
    max_segment_s: float = 90.0  # longer segments are windowed
    window_s: float = 45.0
    overlap_s: float = 15.0  # minimum overlap between neighbouring windows

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_s < self.window_s <= self.max_segment_s:
            raise ValueError(
                "need 0 <= overlap_s < window_s <= max_segment_s, got "
                f"{self.overlap_s}, {self.window_s}, {self.max_segment_s}"
            )


@dataclass(frozen=True)
class Window:
    """A time span that will become one moment."""

    t_start: float
    t_end: float
    keyframe_t: float
    segment: int  # index of the slide segment it came from
    kind: str  # "slide" or "dynamic", inherited from the segment


@dataclass(frozen=True)
class Moment:
    moment_id: str
    lecture_id: str
    t_start: float
    t_end: float
    slide_text: str
    transcript: str
    keyframe_t: float
    video_link: str
    kind: str
    segment: int

    def to_dict(self) -> dict:
        return asdict(self)


def plan_windows(
    segments: Sequence[SlideSegment], config: MomentConfig | None = None
) -> list[Window]:
    """Turn slide segments into the time windows that become moments.

    Long segments get windows of exactly ``window_s``, spaced evenly so that neighbours
    overlap by at least ``overlap_s`` and the first and last windows are flush with the
    segment. Spacing them evenly, rather than stepping by a fixed stride and leaving a
    remainder, means there is never a runt window of a few seconds at the end.

    Keyframes: every window of a slide shares the slide's keyframe, which shows the
    complete slide. A dynamic segment has no single representative picture, so each of its
    windows takes the frame at its own midpoint.
    """
    config = config or MomentConfig()
    windows: list[Window] = []
    for index, seg in enumerate(segments):
        if seg.duration <= config.max_segment_s:
            spans = [(seg.t_start, seg.t_end)]
        else:
            stride = config.window_s - config.overlap_s
            count = math.ceil((seg.duration - config.overlap_s) / stride)
            step = (seg.duration - config.window_s) / (count - 1)
            spans = [
                (seg.t_start + k * step, seg.t_start + k * step + config.window_s)
                for k in range(count)
            ]
        for t_start, t_end in spans:
            if seg.kind == "dynamic":
                keyframe_t = float(math.floor(0.5 * (t_start + t_end)))
            else:
                keyframe_t = seg.keyframe_t
            windows.append(
                Window(
                    t_start=round(t_start, 3),
                    t_end=round(t_end, 3),
                    keyframe_t=keyframe_t,
                    segment=index,
                    kind=seg.kind,
                )
            )
    return windows


def transcript_between(words: Sequence[Word], t_start: float, t_end: float) -> str:
    """The words whose midpoint falls in ``[t_start, t_end)``, joined into text.

    Assigning by midpoint puts every word in exactly one of two adjacent, non-overlapping
    spans, even when the word straddles the boundary.
    """
    return " ".join(w.text for w in words if t_start <= w.mid < t_end)


def build_moments(
    lecture: Lecture,
    windows: Sequence[Window],
    words: Sequence[Word],
    slide_texts: Mapping[float, str],
) -> list[Moment]:
    """Attach speech and slide text to each window.

    ``slide_texts`` maps a keyframe time to the OCR text of that keyframe. ``words`` must be
    sorted by time, as speech recognition returns them.
    """
    moments = []
    for index, window in enumerate(windows):
        moments.append(
            Moment(
                moment_id=f"{lecture.lecture_id}:{index:04d}",
                lecture_id=lecture.lecture_id,
                t_start=window.t_start,
                t_end=window.t_end,
                slide_text=slide_texts.get(window.keyframe_t, ""),
                transcript=transcript_between(words, window.t_start, window.t_end),
                keyframe_t=window.keyframe_t,
                video_link=lecture.link_at(window.t_start),
                kind=window.kind,
                segment=window.segment,
            )
        )
    return moments


def moment_text(slide_text: str, transcript: str) -> str:
    """The text that is embedded and indexed for a moment.

    Labelling the two sources lets the embedding model tell a slide's terse bullet points
    from the lecturer's sentences. A missing source is left out entirely rather than
    labelled and empty.
    """
    parts = []
    slide_text = " ".join(slide_text.split())
    transcript = " ".join(transcript.split())
    if slide_text:
        parts.append(f"Slide: {slide_text}")
    if transcript:
        parts.append(f"Speech: {transcript}")
    return "\n".join(parts)
