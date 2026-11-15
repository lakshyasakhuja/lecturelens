"""Slide segmentation against synthetic footage with exact ground truth.

Each test isolates one real-world nuisance the detector must survive.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import synthetic as syn

from lecturelens.slides import (
    SlideConfig,
    SlideDetector,
    detect_slides,
    hamming,
    load_slide_config,
    noise_floor,
    phash,
)


def boundaries(detection) -> list[int]:
    return [round(t) for t in detection.transitions]


def test_hard_cuts_are_found_exactly(lecture_frames):
    frames, cuts = lecture_frames
    detection = detect_slides(syn.to_uint8(frames))
    assert boundaries(detection) == cuts
    assert all(seg.kind == "slide" for seg in detection.segments)


def test_segments_tile_the_video(lecture_frames):
    frames, _ = lecture_frames
    segments = detect_slides(syn.to_uint8(frames)).segments
    assert segments[0].t_start == 0
    assert segments[-1].t_end == len(frames)
    for prev, nxt in itertools.pairwise(segments):
        assert prev.t_end == nxt.t_start
    for seg in segments:
        assert seg.t_start <= seg.keyframe_t < seg.t_end


def test_keyframe_is_the_last_frame_before_the_cut(lecture_frames):
    frames, cuts = lecture_frames
    segments = detect_slides(syn.to_uint8(frames)).segments
    assert [seg.keyframe_t for seg in segments[:-1]] == [c - 1 for c in cuts]
    assert segments[-1].keyframe_t == len(frames) - 1


def test_cursor_and_compression_noise_cause_no_false_cuts(lecture_frames):
    frames, cuts = lecture_frames
    noisy = syn.add_cursor(syn.add_sensor_noise(frames, sigma=0.002, seed=1), seed=2)
    assert boundaries(detect_slides(syn.to_uint8(noisy))) == cuts


def test_bullet_reveals_stay_on_one_slide_and_keyframe_has_them_all():
    rng = np.random.default_rng(7)
    base = syn.slide_image(11, lines=1)
    parts, current = [], base.copy()
    for row in range(1, 6):  # a new bullet every 8 seconds
        parts.append(syn.hold(current, 8))
        current = current.copy()
        syn.add_text_line(current, row, rng)
    parts.append(syn.hold(current, 8))
    final = current
    following = syn.slide_image(12, picture=True)
    frames = syn.timeline([*parts, syn.hold(following, 20)])

    detection = detect_slides(syn.to_uint8(frames))
    assert boundaries(detection) == [48]
    first = detection.segments[0]
    # The keyframe must show the complete slide, i.e. come after the last reveal at t=40.
    assert first.keyframe_t >= 40
    keyframe = frames[round(first.keyframe_t)]
    assert np.allclose(keyframe, final)


def _build_then_next(seed: int) -> np.ndarray:
    """A sparse slide gains three bullets at t=20; a different slide follows at t=40."""
    rng = np.random.default_rng(seed)
    before = syn.slide_image(seed, lines=2)
    after = before.copy()
    for row in (2, 3, 4):
        syn.add_text_line(after, row, rng)
    following = syn.slide_image(seed + 500, picture=True)
    return syn.timeline([syn.hold(before, 20), syn.hold(after, 20), syn.hold(following, 20)])


def _sparse_change(seed: int) -> np.ndarray:
    """A two-line slide is replaced by a different two-line slide at t=20."""
    a = syn.slide_image(seed, lines=2)
    b = syn.slide_image(seed + 500, lines=2)
    return syn.timeline([syn.hold(a, 20), syn.hold(b, 20)])


def _rates(config: SlideConfig, seeds=range(40)) -> tuple[float, float]:
    """(share of builds kept on one slide, share of sparse slide changes detected)."""
    absorbed = np.mean(
        [
            boundaries(detect_slides(syn.to_uint8(_build_then_next(s)), config)) == [40]
            for s in seeds
        ]
    )
    detected = np.mean(
        [boundaries(detect_slides(syn.to_uint8(_sparse_change(s)), config)) == [20] for s in seeds]
    )
    return float(absorbed), float(detected)


def test_sign_aware_score_separates_builds_from_sparse_slide_changes():
    """The reason the change score looks at the sign of the difference.

    Three bullets appearing and one sparse slide replacing another move a similar number of
    pixels, so no threshold on plain mean absolute difference gets both right. A build is
    one-sided (ink is only added); a replacement is two-sided.
    """
    absorbed, detected = _rates(SlideConfig())
    assert absorbed == 1.0
    assert detected >= 0.85

    # Plain mean absolute difference, at the same threshold: every build is split.
    plain_absorbed, plain_detected = _rates(SlideConfig(one_sided_weight=1.0))
    assert plain_absorbed <= 0.1
    assert plain_detected >= 0.75

    # Raise the threshold until builds are absorbed, and the sparse changes are lost instead.
    strict_absorbed, strict_detected = _rates(
        SlideConfig(one_sided_weight=1.0, t_high=0.012, t_low=0.0015)
    )
    assert strict_absorbed >= 0.9
    assert strict_detected <= 0.25


def test_a_large_one_sided_change_still_starts_a_new_segment():
    text_only = syn.slide_image(31)
    with_figure = syn.slide_image(31, picture=True)  # same text, a large figure appears
    frames = syn.timeline([syn.hold(text_only, 20), syn.hold(with_figure, 20)])
    assert boundaries(detect_slides(syn.to_uint8(frames))) == [20]


def test_brief_interruption_is_merged_back_into_the_slide():
    a, b = syn.slide_image(1), syn.slide_image(2, picture=True)
    frames = syn.timeline([syn.hold(a, 30), syn.hold(b, 2), syn.hold(a, 30)])
    detection = detect_slides(syn.to_uint8(frames))
    assert len(detection.segments) == 1
    assert detection.merged == 1
    assert detection.segments[0].t_end == 62


def test_returning_to_a_slide_after_a_real_detour_keeps_three_segments():
    a, b = syn.slide_image(1), syn.slide_image(2, picture=True)
    frames = syn.timeline([syn.hold(a, 30), syn.hold(b, 10), syn.hold(a, 30)])
    assert boundaries(detect_slides(syn.to_uint8(frames))) == [30, 40]


def test_slow_crossfade_is_caught_by_the_reference_frame():
    a, b = syn.slide_image(3), syn.slide_image(4)
    fade = syn.crossfade(a, b, steps=6)
    frames = syn.timeline([syn.hold(a, 20), fade, syn.hold(b, 20)])
    detection = detect_slides(syn.to_uint8(frames))

    # Every individual step of the fade is below t_high: only the accumulated change against
    # the reference frame can reveal it. This is the case the reference exists for.
    step_distances = detection.distances[20:27]
    assert step_distances.max() < detection.t_high
    assert len(detection.segments) == 2
    assert abs(detection.transitions[0] - 20) <= 1
    assert detection.segments[0].keyframe_t == 19  # the last clean frame of slide A


def test_long_churn_becomes_a_dynamic_segment():
    a, b = syn.slide_image(5), syn.slide_image(6)
    frames = syn.timeline([syn.hold(a, 20), syn.churn(20, seed=1), syn.hold(b, 20)])
    segments = detect_slides(syn.to_uint8(frames)).segments
    assert [seg.kind for seg in segments] == ["slide", "dynamic", "slide"]
    assert [round(seg.t_start) for seg in segments] == [0, 20, 40]


def test_short_churn_is_treated_as_a_transition():
    a, b = syn.slide_image(5), syn.slide_image(6)
    frames = syn.timeline([syn.hold(a, 20), syn.churn(4, seed=2), syn.hold(b, 20)])
    segments = detect_slides(syn.to_uint8(frames)).segments
    assert [seg.kind for seg in segments] == ["slide", "slide"]
    assert segments[1].t_start == 20
    assert segments[1].keyframe_t == 43


def test_video_that_never_settles_is_one_dynamic_segment():
    segments = detect_slides(syn.to_uint8(syn.churn(30, seed=3))).segments
    assert len(segments) == 1
    assert segments[0].kind == "dynamic"
    assert (segments[0].t_start, segments[0].t_end) == (0, 30)


def test_webcam_overlay_needs_a_mask(lecture_frames):
    frames, cuts = lecture_frames
    with_webcam, mask = syn.add_webcam(frames, seed=4)

    masked = detect_slides(syn.to_uint8(with_webcam), SlideConfig(masks=(mask,)))
    assert boundaries(masked) == cuts

    # Unmasked and with fixed thresholds, the overlay keeps every frame "changing".
    unmasked = detect_slides(syn.to_uint8(with_webcam), SlideConfig(adaptive=False))
    assert boundaries(unmasked) != cuts


def test_filmed_projector_noise_is_handled_by_camera_mode(lecture_frames):
    frames, cuts = lecture_frames
    filmed = syn.add_sensor_noise(frames, sigma=0.02, seed=5)

    # Fixed thresholds: sensor noise alone exceeds t_low, so nothing ever counts as stable.
    naive = detect_slides(syn.to_uint8(filmed), SlideConfig(adaptive=False))
    assert [seg.kind for seg in naive.segments] == ["dynamic"]

    camera = detect_slides(syn.to_uint8(filmed), SlideConfig(camera=True))
    floor_high, floor_low = SlideConfig().thresholds
    assert camera.t_low > floor_low  # the band was lifted above the noise...
    assert camera.t_high - camera.t_low == pytest.approx(floor_high - floor_low)  # ...as a whole
    found = boundaries(camera)
    assert len(found) == len(cuts)
    assert all(abs(f - c) <= 1 for f, c in zip(found, cuts, strict=True))


def test_noise_lift_is_capped_so_motion_is_not_mistaken_for_noise():
    # A recording that never stops moving has a high "noise floor". Without the cap the
    # thresholds would rise to meet it and the whole video would look like one still slide.
    detection = detect_slides(syn.to_uint8(syn.churn(40, seed=8)))
    floor_low = SlideConfig().thresholds[1]
    assert detection.noise > 5 * floor_low
    assert detection.t_low == pytest.approx(5 * floor_low)
    assert [seg.kind for seg in detection.segments] == ["dynamic"]


def _animation() -> tuple[np.ndarray, list[np.ndarray]]:
    """Slide A, then four 5-second animation steps of slide B, then B's final state."""
    a = syn.slide_image(41)
    b = syn.slide_image(42, picture=True)
    steps = []
    for k in range(4):  # a bright marker sweeps across B's picture: content replaced each step
        frame = b.copy()
        frame[50:70, 100 + 12 * k : 110 + 12 * k] = 0.05
        steps.append(frame)
    frames = syn.timeline(
        [syn.hold(a, 30), *[syn.hold(step, 5) for step in steps], syn.hold(b, 30)]
    )
    return frames, steps


def test_animation_steps_are_grouped_into_segments_of_useful_length():
    frames, _ = _animation()
    every_step = detect_slides(syn.to_uint8(frames), SlideConfig(min_segment_s=0))
    # Each move of the marker replaces content, so it is a cut. Its final disappearance
    # only removes content (one-sided), so that one is absorbed by the change score.
    assert boundaries(every_step) == [30, 35, 40, 45]

    detection = detect_slides(syn.to_uint8(frames))
    # The 5 s steps resemble each other and slide B, never slide A. They pair up until a
    # group reaches the minimum length; the cut from A survives untouched.
    assert boundaries(detection) == [30, 40]
    assert all(seg.duration >= 10 for seg in detection.segments)
    assert detection.segments[-1].t_end == 80
    assert detection.segments[-1].keyframe_t == 79  # B's final state, not a mid-animation frame


def test_short_segment_joins_the_neighbour_it_resembles():
    a, b = syn.slide_image(43), syn.slide_image(44, picture=True)
    late_a = a.copy()
    late_a[60:75, 20:60] = 0.3  # slide A with one more element, shown briefly
    frames = syn.timeline([syn.hold(a, 30), syn.hold(late_a, 6), syn.hold(b, 30)])
    detection = detect_slides(syn.to_uint8(frames))
    assert boundaries(detection) == [36]  # the 6 s segment went with A, the cut to B stays
    assert detection.segments[0].keyframe_t == 35  # and A's keyframe is its fuller, later state


def test_short_segment_between_dynamic_stretches_is_left_alone():
    frames = syn.timeline(
        [syn.churn(20, seed=11), syn.hold(syn.slide_image(45), 6), syn.churn(20, seed=12)]
    )
    segments = detect_slides(syn.to_uint8(frames)).segments
    assert [seg.kind for seg in segments] == ["dynamic", "slide", "dynamic"]
    assert segments[1].duration == 6


def test_ssim_metric_finds_the_same_cuts(lecture_frames):
    frames, cuts = lecture_frames
    detection = detect_slides(syn.to_uint8(frames), SlideConfig(metric="ssim"))
    assert boundaries(detection) == cuts


def test_constant_video_is_a_single_slide():
    frames = syn.to_uint8(syn.hold(syn.slide_image(9), 15))
    segments = detect_slides(frames).segments
    assert len(segments) == 1
    assert (segments[0].t_start, segments[0].t_end, segments[0].kind) == (0, 15, "slide")


def test_single_frame_video():
    segments = detect_slides(syn.to_uint8(syn.hold(syn.slide_image(9), 1))).segments
    assert len(segments) == 1
    assert segments[0].t_end == 1


def test_fps_scales_times(lecture_frames):
    frames, cuts = lecture_frames
    doubled = np.repeat(frames, 2, axis=0)  # the same lecture sampled at 2 fps
    detection = detect_slides(syn.to_uint8(doubled), SlideConfig(fps=2.0))
    assert [round(t) for t in detection.transitions] == cuts


def test_detector_reuses_preprocessing_for_threshold_sweeps(lecture_frames):
    frames, cuts = lecture_frames
    detector = SlideDetector(syn.to_uint8(frames))
    assert boundaries(detector.detect()) == cuts
    # An absurdly high threshold finds nothing; the sweep must actually apply the override.
    assert boundaries(detector.detect(t_high=0.9, t_low=0.5)) == []
    with pytest.raises(ValueError, match="cannot override"):
        detector.detect(masks=((0, 0, 0.5, 0.5),))


def test_phash_separates_same_from_different():
    a = syn.slide_image(21, picture=True)
    b = syn.slide_image(22, picture=True)
    noisy_a = syn.add_sensor_noise(a[None], sigma=0.01, seed=3)[0]
    brighter_a = np.clip(a * 0.9 + 0.05, 0, 1)

    assert hamming(phash(a), phash(a)) == 0
    assert hamming(phash(a), phash(noisy_a)) <= 4
    assert hamming(phash(a), phash(brighter_a)) <= 4
    assert hamming(phash(a), phash(b)) > SlideConfig().merge_bits


def test_noise_floor_is_robust_to_outliers():
    d = np.concatenate([[0.0], np.full(200, 0.001), [0.5, 0.4, 0.6]])
    level, spread = noise_floor(d)
    assert level == pytest.approx(0.001)
    assert spread == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"metric": "psnr"}, "unknown metric"),
        ({"t_high": 0.001, "t_low": 0.01}, "t_low < t_high"),
        ({"one_sided_weight": 1.5}, "one_sided_weight"),
        ({"masks": ((0.5, 0.5, 0.2, 0.9),)}, "mask"),
    ],
)
def test_invalid_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SlideConfig(**kwargs)


def test_masks_covering_everything_is_an_error():
    frames = syn.to_uint8(syn.hold(syn.slide_image(1), 5))
    with pytest.raises(ValueError, match="entire frame"):
        detect_slides(frames, SlideConfig(masks=((0, 0, 1, 1),)))


def test_load_slide_config_merges_defaults_and_overrides(tmp_path):
    path = tmp_path / "masks.yaml"
    path.write_text(
        """
defaults:
  masks: [[0.0, 0.95, 1.0, 1.0]]
  min_stable_s: 4
lectures:
  lec02:
    camera: true
    masks: [[0.8, 0.7, 1.0, 1.0]]
""",
        encoding="utf-8",
    )
    default = load_slide_config(path, "lec01")
    assert default.masks == ((0.0, 0.95, 1.0, 1.0),)
    assert default.min_stable_s == 4
    assert default.camera is False

    override = load_slide_config(path, "lec02")
    assert override.masks == ((0.8, 0.7, 1.0, 1.0),)
    assert override.camera is True
    assert override.blur_sigma == 1.0
    assert override.min_stable_s == 4

    assert load_slide_config(tmp_path / "missing.yaml", "lec01") == SlideConfig()


def test_load_slide_config_rejects_unknown_settings(tmp_path):
    path = tmp_path / "masks.yaml"
    path.write_text("lectures:\n  lec01:\n    thresh: 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown slide setting"):
        load_slide_config(path, "lec01")
