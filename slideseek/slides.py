"""Slide segmentation: turn a lecture video into ``[t_start, t_end]`` segments, one per slide.

The input is a stack of small greyscale frames sampled at 1 fps (see ``ingest``). The output
is a list of segments that tile the video, each with the timestamp of its **keyframe**.

How it works
------------
1. **Preprocess.** Mask regions that move for unrelated reasons (webcam overlay, progress
   bar) and, for camera-filmed projectors, blur slightly so sensor noise does not register.

2. **Distance.** ``d[t]`` is the distance between frame ``t`` and frame ``t - 1``: mean
   absolute pixel difference by default, ``1 - SSIM`` as an alternative.

3. **Hysteresis against a reference frame.** Two thresholds, ``t_low < t_high``:

   * A frame is *quiet* when ``d[t] < t_low``, i.e. it looks like the frame before it.
   * A segment becomes *stable* once ``min_stable_s`` consecutive frames are quiet.
   * While stable, the **reference** is the most recent quiet frame. The segment ends when
     the *change score* between a frame and the reference exceeds ``t_high``.

   Comparing against the reference rather than only the previous frame matters twice over.
   A cross-fade spread over several frames, each step too small to trip ``t_high`` on its
   own, still accumulates against the reference and is caught. And changes that settle
   below ``t_high`` (a bullet point appearing, ink on a tablet, a cursor) are absorbed: the
   reference moves forward to the new quiet frame, so they never add up to a false cut.

   **Added versus replaced.** Mean absolute difference alone cannot tell two bullet points
   appearing from one sparse slide replacing another: both move about the same number of
   pixels. What differs is the *sign*. A build only adds ink, so every changed pixel moves
   the same way (darker on a light theme, lighter on a dark one). A new slide removes the
   old text and draws new text, so pixels move both ways. The change score therefore splits
   the difference into a balanced part (content replaced) and a one-sided part (content
   added, or a global brightness shift), and counts the one-sided part at only
   ``one_sided_weight`` of its size::

       total    = mean |frame - ref|
       net      = |mean (frame - ref)|          # the one-sided part
       score    = (total - net) + one_sided_weight * net

   With ``one_sided_weight = 1`` this is plain mean absolute difference. A large one-sided
   change (a full-frame figure appearing) still crosses ``t_high`` and starts a segment.

4. **Keyframe = the last quiet frame of the segment.** Animated slides are only complete at
   the end, so the last frame is the one that carries all the text.

5. **Dynamic stretches.** If the picture keeps changing for ``dynamic_min_s`` or longer
   without settling (an embedded video, a live demo, scrolling), that stretch becomes its
   own segment of kind ``"dynamic"`` rather than being glued onto the next slide.

6. **Merge.** Neighbouring slide segments whose keyframes are the same picture are merged.
   This happens when something brief interrupts a slide (a pop-up, a two-second flip to
   another window) and the lecturer returns to it. Two checks must agree: the perceptual
   hashes are within ``merge_bits`` bits, and the direct frame distance is below ``t_high``.
   The hash alone is not enough, because two different text slides on the same template can
   share their low-frequency structure.

7. **Absorb very short segments.** A slide that animates (a filter sliding across an image,
   a diagram assembling itself) changes every few seconds, and each step that replaces
   content is a segment. A segment shorter than ``min_segment_s`` is not worth a search
   result of its own: it holds a sentence or two of speech. It is folded into whichever
   neighbouring slide segment its keyframe resembles more, and the merged segment keeps the
   later keyframe. A short segment with no slide neighbour is left alone.

8. **Adaptive thresholds.** ``t_low`` and ``t_high`` are floors. If a recording is noisier
   than the floor allows (a filmed projector), the whole band is shifted up to sit above
   the lecture's own noise level, estimated robustly as ``median + 4 x MAD`` of ``d``.
   The shift is additive because noise adds to a change score rather than scaling it. It is
   capped: past ``max_noise_lift x t_low`` the "noise" is really motion, nothing counts as
   quiet, and the stretch is reported as dynamic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

Mask = tuple[float, float, float, float]

# Per-metric threshold floors, as (t_high, t_low), on pixel values scaled to [0, 1].
# For orientation, at 160x90 a text slide replacing another scores about 0.01 to 0.02, a
# sparse two-line slide change about 0.005, three bullet points appearing about 0.002, and
# a cursor about 0.0001. These are starting points: `eval/eval_transitions.py` tunes them on
# one annotated lecture and reports on another.
_DEFAULT_THRESHOLDS = {
    "mad": (0.0035, 0.0015),
    "ssim": (0.10, 0.02),
}


@dataclass(frozen=True)
class SlideConfig:
    """Settings for slide-change detection. All times are in seconds."""

    fps: float = 1.0
    size: tuple[int, int] = (160, 90)  # (width, height) of the detection frames
    metric: str = "mad"  # "mad" (mean absolute difference) or "ssim" (1 - SSIM)
    # Weight of the one-sided part of a change (content added rather than replaced) in the
    # change score. 1.0 is plain mean absolute difference. Only used by the "mad" metric.
    one_sided_weight: float = 0.25
    t_high: float | None = None  # None: use the metric's default floor
    t_low: float | None = None
    adaptive: bool = True  # lift thresholds above the lecture's own noise floor
    # The lift is capped at this multiple of t_low. A "noise floor" higher than that is not
    # noise but motion, and the recording is treated as continuously changing instead.
    max_noise_lift: float = 5.0
    min_stable_s: float = 3.0
    dynamic_min_s: float = 12.0
    merge_bits: int = 8  # max Hamming distance between 64-bit perceptual hashes
    # Slide segments shorter than this are folded into the neighbour they most resemble.
    # 0 keeps every segment.
    min_segment_s: float = 10.0
    camera: bool = False  # camera pointed at a projector, rather than a screen capture
    blur: float | None = None  # Gaussian sigma in detection pixels; None: 1.0 if camera else 0
    masks: tuple[Mask, ...] = ()

    def __post_init__(self) -> None:
        if self.metric not in _DEFAULT_THRESHOLDS:
            raise ValueError(
                f"unknown metric {self.metric!r}; use one of {set(_DEFAULT_THRESHOLDS)}"
            )
        if not 0 <= self.one_sided_weight <= 1:
            raise ValueError(f"one_sided_weight must be in [0, 1], got {self.one_sided_weight}")
        t_high, t_low = self.thresholds
        if not 0 < t_low < t_high:
            raise ValueError(f"thresholds must satisfy 0 < t_low < t_high, got {t_low}, {t_high}")
        for mask in self.masks:
            x0, y0, x1, y1 = mask
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"mask {mask} must be [x0, y0, x1, y1] as fractions, x0<x1, y0<y1")

    @property
    def thresholds(self) -> tuple[float, float]:
        """``(t_high, t_low)`` floors, with metric defaults filled in."""
        default_high, default_low = _DEFAULT_THRESHOLDS[self.metric]
        return (
            default_high if self.t_high is None else self.t_high,
            default_low if self.t_low is None else self.t_low,
        )

    @property
    def blur_sigma(self) -> float:
        if self.blur is not None:
            return self.blur
        return 1.0 if self.camera else 0.0


@dataclass(frozen=True)
class SlideSegment:
    """A stretch of video showing one slide (or one run of continuously changing content)."""

    t_start: float
    t_end: float
    keyframe_t: float
    kind: str = "slide"  # "slide" or "dynamic"

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "keyframe_t": self.keyframe_t,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlideSegment:
        return cls(
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            keyframe_t=float(data["keyframe_t"]),
            kind=str(data.get("kind", "slide")),
        )


@dataclass
class Detection:
    """Segments plus the signal they were derived from, kept for tuning and debugging."""

    segments: list[SlideSegment]
    distances: np.ndarray  # d[t]: distance between frame t and t-1; d[0] is 0
    t_high: float  # effective thresholds after noise adaptation
    t_low: float
    noise: float = 0.0
    merged: int = 0  # how many boundaries the merge step removed
    config: SlideConfig = field(default_factory=SlideConfig)

    @property
    def transitions(self) -> list[float]:
        """Times at which a new segment begins (the first segment's start is not a transition)."""
        return [seg.t_start for seg in self.segments[1:]]


# --------------------------------------------------------------------------------------
# Preprocessing and distances
# --------------------------------------------------------------------------------------


def _valid_mask(height: int, width: int, masks: Iterable[Mask]) -> np.ndarray:
    """Boolean array that is False inside every masked rectangle."""
    valid = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in masks:
        valid[
            int(np.floor(y0 * height)) : int(np.ceil(y1 * height)),
            int(np.floor(x0 * width)) : int(np.ceil(x1 * width)),
        ] = False
    if not valid.any():
        raise ValueError("masks cover the entire frame; nothing left to compare")
    return valid


def preprocess(frames: np.ndarray, config: SlideConfig) -> tuple[np.ndarray, np.ndarray]:
    """Scale to ``float32`` in [0, 1], blur if configured, and zero out masked regions.

    Returns ``(frames, valid)`` where ``valid`` marks the pixels that take part in distances.
    """
    if frames.ndim != 3:
        raise ValueError(f"expected frames of shape (n, height, width), got {frames.shape}")
    out = frames.astype(np.float32)
    if frames.dtype == np.uint8:
        out /= 255.0
    sigma = config.blur_sigma
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out, sigma=(0, sigma, sigma))
    valid = _valid_mask(out.shape[1], out.shape[2], config.masks)
    out[:, ~valid] = 0.0
    return out, valid


def _ssim(a: np.ndarray, b: np.ndarray, valid: np.ndarray, window: int = 7) -> float:
    """Mean structural similarity over the valid pixels, with a uniform window."""
    from scipy.ndimage import uniform_filter

    c1, c2 = 0.01**2, 0.03**2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a = uniform_filter(a, window)
    mu_b = uniform_filter(b, window)
    var_a = uniform_filter(a * a, window) - mu_a * mu_a
    var_b = uniform_filter(b * b, window) - mu_b * mu_b
    cov = uniform_filter(a * b, window) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    )
    return float(ssim_map[valid].mean())


Distance = Callable[[np.ndarray, np.ndarray], float]


def make_distance(metric: str, valid: np.ndarray) -> Distance:
    """Plain distance between two preprocessed frames; 0 means identical.

    Used to decide whether a frame is *quiet* and whether two keyframes show the same
    picture. Any change counts here, whichever way the pixels moved.
    """
    if metric == "mad":
        n_valid = float(valid.sum())

        def mad(a: np.ndarray, b: np.ndarray) -> float:
            # Masked pixels are already zero in both frames, so they add nothing to the sum.
            return float(np.abs(a - b).sum(dtype=np.float64) / n_valid)

        return mad
    if metric == "ssim":
        return lambda a, b: 1.0 - _ssim(a, b, valid)
    raise ValueError(f"unknown metric {metric!r}")


def make_change_score(metric: str, valid: np.ndarray, one_sided_weight: float = 1.0) -> Distance:
    """Score for "has the picture left the reference?", called as ``score(frame, ref)``.

    For ``"mad"`` this discounts the one-sided part of the change, so content being *added*
    to a slide counts for less than content being *replaced* (see the module docstring).
    For ``"ssim"`` it is the plain distance.
    """
    if metric != "mad" or one_sided_weight >= 1.0:
        return make_distance(metric, valid)
    n_valid = float(valid.sum())

    def score(frame: np.ndarray, ref: np.ndarray) -> float:
        diff = frame - ref
        total = float(np.abs(diff).sum(dtype=np.float64))
        # sum(diff) = gain - loss and sum(|diff|) = gain + loss, so |sum(diff)| is exactly
        # the part of the change not matched by an opposite change elsewhere in the frame.
        net = abs(float(diff.sum(dtype=np.float64)))
        return ((total - net) + one_sided_weight * net) / n_valid

    return score


def consecutive_distances(frames: np.ndarray, valid: np.ndarray, metric: str = "mad") -> np.ndarray:
    """``d[t]`` = distance between preprocessed frame ``t`` and ``t - 1``; ``d[0]`` is 0."""
    n = len(frames)
    d = np.zeros(n, dtype=np.float64)
    if metric == "mad":
        n_valid = float(valid.sum())
        chunk = 256  # bounds the size of the temporary difference array
        for start in range(1, n, chunk):
            stop = min(n, start + chunk)
            diff = np.abs(frames[start:stop] - frames[start - 1 : stop - 1])
            d[start:stop] = diff.sum(axis=(1, 2), dtype=np.float64) / n_valid
    else:
        dist = make_distance(metric, valid)
        for t in range(1, n):
            d[t] = dist(frames[t], frames[t - 1])
    return d


def noise_floor(d: np.ndarray) -> tuple[float, float]:
    """Robust level and spread of ``d``: the median, and the MAD scaled to a standard deviation."""
    if len(d) <= 1:
        return 0.0, 0.0
    body = d[1:]
    level = float(np.median(body))
    spread = float(1.4826 * np.median(np.abs(body - level)))
    return level, spread


def effective_thresholds(d: np.ndarray, config: SlideConfig) -> tuple[float, float, float]:
    """``(t_high, t_low, noise)`` after shifting the floors above the lecture's noise level.

    ``noise`` is ``median + 4 x MAD`` of the frame-to-frame distances. When it exceeds
    ``t_low``, both thresholds move up by the same amount, so a change has to stand as far
    above the noise as it would have stood above zero in a clean recording.
    """
    t_high, t_low = config.thresholds
    level, spread = noise_floor(d)
    noise = level + 4.0 * spread
    if config.adaptive and noise > t_low:
        lift = min(noise, config.max_noise_lift * t_low) - t_low
        t_low += lift
        t_high += lift
    return t_high, t_low, noise


# --------------------------------------------------------------------------------------
# Perceptual hash
# --------------------------------------------------------------------------------------


def phash(frame: np.ndarray, hash_size: int = 8, factor: int = 4) -> int:
    """64-bit DCT perceptual hash of a greyscale frame.

    The frame is shrunk to 32x32, transformed with a 2-D DCT, and the lowest 8x8 frequencies
    are thresholded at their median. Similar pictures give hashes a few bits apart.
    """
    from PIL import Image
    from scipy.fft import dctn

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    side = hash_size * factor
    small = np.asarray(
        Image.fromarray(arr).resize((side, side), Image.Resampling.LANCZOS), dtype=np.float64
    )
    low = dctn(small, norm="ortho")[:hash_size, :hash_size]
    # The DC term is the mean brightness; leave it out of the median so that a uniformly
    # brighter frame hashes the same.
    median = np.median(low.ravel()[1:])
    bits = (low > median).ravel()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


@dataclass
class _Span:
    """A segment in frame indices: ``[start, end)`` with its keyframe index."""

    start: int
    end: int
    keyframe: int
    kind: str


def _segment(
    frames: np.ndarray,
    d: np.ndarray,
    change: Distance,
    t_high: float,
    t_low: float,
    min_stable: int,
    dynamic_min: int,
) -> list[_Span]:
    """The hysteresis state machine. Works in frame indices; see the module docstring."""
    n = len(frames)
    spans: list[_Span] = []
    start = 0  # first frame of the segment being built
    ref: int | None = None  # latest quiet frame once stable; None while still settling
    quiet = 0  # length of the current run of quiet frames while settling

    for i in range(1, n):
        is_quiet = d[i] < t_low
        if ref is None:
            quiet = quiet + 1 if is_quiet else 0
            if quiet >= min_stable:
                stable_from = i - quiet  # first frame of the quiet run
                if stable_from - start >= dynamic_min:
                    # The picture churned for a long time before settling: that stretch is
                    # content in its own right, not a transition into this slide.
                    spans.append(_Span(start, stable_from, stable_from - 1, "dynamic"))
                    start = stable_from
                ref = i
        elif is_quiet:
            ref = i
        elif change(frames[i], frames[ref]) > t_high:
            # The picture has left the reference. The segment ends after its last quiet frame.
            spans.append(_Span(start, ref + 1, ref, "slide"))
            start = ref + 1
            ref = None
            quiet = 0

    if ref is not None:
        spans.append(_Span(start, n, ref, "slide"))
    elif start < n:
        tail = n - start
        if spans and tail < max(min_stable, 1):
            spans[-1].end = n  # a few unsettled frames at the very end: not worth a segment
        else:
            spans.append(_Span(start, n, n - 1, "dynamic"))
    return spans


def _merge(
    spans: Sequence[_Span],
    frames: np.ndarray,
    dist: Distance,
    t_high: float,
    merge_bits: int,
) -> list[_Span]:
    """Merge neighbouring slide segments that show the same picture."""
    merged: list[_Span] = []
    hashes: dict[int, int] = {}

    def frame_hash(index: int) -> int:
        if index not in hashes:
            hashes[index] = phash(frames[index])
        return hashes[index]

    for span in spans:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == "slide"
            and span.kind == "slide"
            and hamming(frame_hash(prev.keyframe), frame_hash(span.keyframe)) <= merge_bits
            and dist(frames[prev.keyframe], frames[span.keyframe]) < t_high
        ):
            # Same slide again: extend, and keep the later keyframe (the more complete one).
            merged[-1] = _Span(prev.start, span.end, span.keyframe, "slide")
        else:
            merged.append(_Span(span.start, span.end, span.keyframe, span.kind))
    return merged


def _absorb_short(
    spans: Sequence[_Span], frames: np.ndarray, dist: Distance, min_length: int
) -> list[_Span]:
    """Fold slide segments shorter than ``min_length`` frames into a neighbouring slide.

    Works left to right. A short segment joins the previous or the next slide segment,
    whichever keyframe is closer to its own; the merged segment keeps the later keyframe,
    which for an animation is the more complete picture. Merging can make the next short
    segment's neighbour longer, so a run of animation steps collapses into one segment.
    """
    if min_length <= 1:
        return list(spans)
    out: list[_Span] = []
    pending = [_Span(s.start, s.end, s.keyframe, s.kind) for s in spans]
    i = 0
    while i < len(pending):
        span = pending[i]
        short = span.kind == "slide" and (span.end - span.start) < min_length
        prev = out[-1] if out and out[-1].kind == "slide" else None
        nxt = pending[i + 1] if i + 1 < len(pending) and pending[i + 1].kind == "slide" else None
        if not short or (prev is None and nxt is None):
            out.append(span)
            i += 1
            continue
        to_prev = dist(frames[span.keyframe], frames[prev.keyframe]) if prev else float("inf")
        to_next = dist(frames[span.keyframe], frames[nxt.keyframe]) if nxt else float("inf")
        if to_prev <= to_next:
            out[-1] = _Span(prev.start, span.end, span.keyframe, "slide")
            i += 1
        else:
            pending[i + 1] = _Span(span.start, nxt.end, nxt.keyframe, "slide")
            i += 1
    return out


def detect_slides(frames: np.ndarray, config: SlideConfig | None = None) -> Detection:
    """Segment a stack of detection frames into slides.

    ``frames`` has shape ``(n, height, width)``, sampled at ``config.fps``. The returned
    segments tile ``[0, n / fps]`` with no gaps and no overlaps.
    """
    config = config or SlideConfig()
    if len(frames) == 0:
        raise ValueError("no frames to segment")
    prepared, valid = preprocess(frames, config)
    d = consecutive_distances(prepared, valid, config.metric)
    return _detect_prepared(prepared, valid, d, config)


def _detect_prepared(
    prepared: np.ndarray, valid: np.ndarray, d: np.ndarray, config: SlideConfig
) -> Detection:
    """Segmentation from already-preprocessed frames, so threshold sweeps can reuse them."""
    dist = make_distance(config.metric, valid)
    change = make_change_score(config.metric, valid, config.one_sided_weight)
    t_high, t_low, noise = effective_thresholds(d, config)
    fps = config.fps
    spans = _segment(
        prepared,
        d,
        change,
        t_high=t_high,
        t_low=t_low,
        min_stable=max(1, round(config.min_stable_s * fps)),
        dynamic_min=max(1, round(config.dynamic_min_s * fps)),
    )
    merged = _merge(spans, prepared, dist, t_high, config.merge_bits)
    merged = _absorb_short(merged, prepared, dist, round(config.min_segment_s * fps))
    segments = [
        SlideSegment(
            t_start=span.start / fps,
            t_end=span.end / fps,
            keyframe_t=span.keyframe / fps,
            kind=span.kind,
        )
        for span in merged
    ]
    return Detection(
        segments=segments,
        distances=d,
        t_high=t_high,
        t_low=t_low,
        noise=noise,
        merged=len(spans) - len(merged),
        config=config,
    )


class SlideDetector:
    """Preprocess once, then segment under many settings.

    Decoding and differencing dominate the cost; the state machine itself is cheap. Threshold
    tuning (``eval/eval_transitions.py``) uses this to sweep a grid without redoing either.
    Settings that change the preprocessed frames (masks, blur, metric, size) are fixed at
    construction; thresholds and durations may vary per call.
    """

    def __init__(self, frames: np.ndarray, config: SlideConfig | None = None) -> None:
        self.config = config or SlideConfig()
        self._prepared, self._valid = preprocess(frames, self.config)
        self._d = consecutive_distances(self._prepared, self._valid, self.config.metric)

    @property
    def distances(self) -> np.ndarray:
        return self._d

    def detect(self, **overrides: Any) -> Detection:
        frozen = {"masks", "blur", "camera", "metric", "size", "fps"} & overrides.keys()
        if frozen:
            raise ValueError(f"cannot override {sorted(frozen)} after preprocessing")
        return _detect_prepared(
            self._prepared, self._valid, self._d, replace(self.config, **overrides)
        )


# --------------------------------------------------------------------------------------
# Configuration file
# --------------------------------------------------------------------------------------


def load_slide_config(
    path: str | Path | None, lecture_id: str, base: SlideConfig | None = None
) -> SlideConfig:
    """Build the ``SlideConfig`` for one lecture from ``config/masks.yaml``.

    ``defaults`` in the file applies to every lecture; the lecture's own entry overrides it.
    A missing file or a lecture with no entry simply yields ``base``.
    """
    config = base or SlideConfig()
    if path is None or not Path(path).is_file():
        return config

    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    settings: dict[str, Any] = dict(data.get("defaults") or {})
    settings.update((data.get("lectures") or {}).get(lecture_id) or {})

    known = {f.name for f in fields(SlideConfig)}
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"{path}: unknown slide setting(s) for {lecture_id}: {sorted(unknown)}")
    if "masks" in settings:
        settings["masks"] = tuple(tuple(float(v) for v in mask) for mask in settings["masks"] or ())
    if "size" in settings:
        settings["size"] = tuple(int(v) for v in settings["size"])
    return replace(config, **settings)
