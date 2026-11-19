"""Offline indexing: from lecture videos to a search index.

Each lecture goes through six stages, and each stage writes its result into the lecture's
work directory before the next one starts::

    work/<lecture_id>/
      segments.json     slide segments and the thresholds that produced them
      keyframes/*.jpg   one full-resolution still per keyframe
      ocr.json          slide text per keyframe
      audio.wav         16 kHz mono
      asr.jsonl         transcript with word timestamps
      moments.parquet   the lecture's moments
      vectors.npz       text, speech, slide and image embeddings
      stages.json       fingerprint of the inputs each stage last ran with

**Resumable.** A stage is skipped when its output exists and its fingerprint (a hash of the
settings and upstream fingerprints it depends on) is unchanged. A free GPU session that
dies forty minutes into a run loses one stage of one lecture, not the run. Changing a
setting re-runs exactly the stages downstream of it: a new slide threshold redoes
segmentation and everything after, a new embedding model redoes only the vectors, and
speech recognition, the slowest stage, is left alone.

**Incremental.** Adding a lecture to the library processes that lecture only; the others
are read back from their work directories when the index is assembled.

Models load lazily and once. If every lecture is cached, no model is loaded at all.

**Stages without a video.** Each stage is also a plain function of its inputs
(``find_slides``, ``read_slides``, ``moments_table``, ``embed_moments``, ``assemble``), so
a lecture can be finished from pieces made somewhere else: a browser that sampled the
frames, grabbed the stills and transcribed the speech sends those, and the server runs the
rest (``slideseek.library``). The ``Indexer`` calls the same functions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slideseek import asr, ingest, ocr
from slideseek.catalog import Lecture
from slideseek.embed import EmbedConfig, ImageEmbedder, TextEmbedder
from slideseek.index import Index
from slideseek.moments import MomentConfig, Window, build_moments, moment_text, plan_windows
from slideseek.slides import SlideConfig, SlideSegment, detect_slides, load_slide_config
from slideseek.utils import format_timestamp, read_json, write_json

log = logging.getLogger("slideseek")


@dataclass(frozen=True)
class PipelineConfig:
    slides: SlideConfig = field(default_factory=SlideConfig)
    moments: MomentConfig = field(default_factory=MomentConfig)
    asr: asr.AsrConfig = field(default_factory=asr.AsrConfig)
    ocr: ocr.OcrConfig = field(default_factory=ocr.OcrConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    # Hint Whisper with jargon read off the slides. Makes ASR depend on OCR.
    jargon_boost: bool = True
    masks_path: Path | None = None  # config/masks.yaml: per-lecture masks and overrides
    keyframe_max_width: int = 1920
    thumb_width: int = 480


# Bump a stage's number when its logic changes in a way that makes cached results stale.
# It is part of the stage's fingerprint, so existing work directories recompute that stage
# (and everything downstream) instead of silently keeping the old output.
STAGE_VERSION = {"segments": 2, "keyframes": 1, "ocr": 2, "asr": 2, "moments": 1, "vectors": 1}


def fingerprint(*parts: Any) -> str:
    """A short stable hash of JSON-serialisable inputs."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


class StageCache:
    """Remembers, per stage, the fingerprint of the inputs it last completed with."""

    def __init__(self, work: Path) -> None:
        self.path = work / "stages.json"
        self.state: dict[str, str] = read_json(self.path) if self.path.is_file() else {}

    def fresh(self, stage: str, key: str, *outputs: Path) -> bool:
        return self.state.get(stage) == key and all(path.exists() for path in outputs)

    def done(self, stage: str, key: str) -> None:
        self.state[stage] = key
        write_json(self.path, self.state)


def _keyframe_ms(t: float) -> int:
    return round(t * 1000)


# ---- the stages, as functions of their inputs ------------------------------------------------


def still_name(t: float) -> str:
    """The file name of the still taken at ``t`` seconds: milliseconds, nine digits."""
    return f"{_keyframe_ms(t):09d}.jpg"


def find_slides(
    frames: np.ndarray, config: SlideConfig, *, lecture_id: str, duration_s: float
) -> dict[str, Any]:
    """Slide segments in greyscale frames, as ``segments.json`` stores them."""
    detection = detect_slides(frames, config)
    return {
        "lecture_id": lecture_id,
        "duration_s": duration_s,
        "t_high": detection.t_high,
        "t_low": detection.t_low,
        "noise": detection.noise,
        "merged": detection.merged,
        "segments": [seg.to_dict() for seg in detection.segments],
    }


def load_segments(path: Path) -> list[SlideSegment]:
    return [SlideSegment.from_dict(d) for d in read_json(path)["segments"]]


def keyframe_times(windows: Sequence[Window]) -> list[float]:
    """The times a still is needed at: every window's keyframe, once each, in order."""
    return sorted({w.keyframe_t for w in windows})


def read_slides(engine, stills: Mapping[float, Path]) -> dict[str, dict[str, Any]]:
    """OCR of each still, keyed by its time in milliseconds, as ``ocr.json`` stores it."""
    return {str(_keyframe_ms(t)): engine.read(path).to_dict() for t, path in stills.items()}


def slide_texts(pages: Mapping[str, Any], times: Sequence[float]) -> dict[float, str]:
    """``{keyframe time: the text on that slide}`` from ``read_slides``'s pages."""
    return {t: ocr.SlideText.from_dict(pages[str(_keyframe_ms(t))]).text for t in times}


def moments_table(
    lecture: Lecture,
    windows: Sequence[Window],
    words: Sequence[asr.Word],
    texts: Mapping[float, str],
) -> pd.DataFrame:
    """The lecture's moments as ``moments.parquet`` stores them, with image rows and thumbs."""
    moments = build_moments(lecture, windows, words, texts)
    frame = pd.DataFrame([m.to_dict() for m in moments])
    position = {t: i for i, t in enumerate(keyframe_times(windows))}
    frame["image_idx"] = [position[m.keyframe_t] for m in moments]
    frame["thumb"] = [f"thumbs/{lecture.lecture_id}/{still_name(m.keyframe_t)}" for m in moments]
    return frame


def embed_moments(
    frame: pd.DataFrame, keyframes: Sequence[Path], text_embedder, image_embedder
) -> dict[str, np.ndarray]:
    """The four embedding matrices for one lecture's moments, as ``vectors.npz`` stores them."""
    slides = frame["slide_text"].tolist()
    transcripts = frame["transcript"].tolist()
    combined = [moment_text(s, t) for s, t in zip(slides, transcripts, strict=True)]
    return {
        "text": text_embedder.encode_documents(combined),
        "speech": text_embedder.encode_documents(transcripts),
        "slide": text_embedder.encode_documents([" ".join(s.split()) for s in slides]),
        "image": image_embedder.encode_images(list(keyframes)),
        "image_calibration": np.asarray(image_embedder.calibration, dtype=np.float64),
    }


def write_thumbnails(frame: pd.DataFrame, keyframe_dir: Path, thumb_dir: Path, width: int) -> None:
    """One thumbnail per keyframe the moments use, ``width`` wide; stale ones are removed."""
    from PIL import Image

    thumb_dir.mkdir(parents=True, exist_ok=True)
    wanted = {Path(relative).name for relative in frame["thumb"]}
    for stale in thumb_dir.glob("*.jpg"):
        if stale.name not in wanted:
            stale.unlink()
    for name in wanted:
        target = thumb_dir / name
        if not target.is_file():
            with Image.open(keyframe_dir / name) as still:
                still = still.convert("RGB")
                height = round(still.height * width / still.width)
                still.resize((width, height), Image.Resampling.LANCZOS).save(
                    target, quality=85, optimize=True
                )


def assemble(
    lectures: Mapping[str, Lecture],
    work_dirs: Mapping[str, Path],
    out_dir: str | Path,
    config: PipelineConfig,
    *,
    asr_model: str | None = None,
    timings: Mapping[str, Any] | None = None,
) -> Index:
    """Concatenate the lectures' work directories into one index, with thumbnails, and save it.

    ``asr_model`` names the speech recogniser in the manifest when it was not the one the
    config names (a browser's, say).
    """
    out_dir = Path(out_dir)
    frames, parts = [], {"text": [], "speech": [], "slide": [], "image": []}
    image_offset = 0
    total_duration = 0.0
    calibration = np.zeros(2)
    for lecture_id in lectures:
        work = Path(work_dirs[lecture_id])
        frame = pd.read_parquet(work / "moments.parquet")
        with np.load(work / "vectors.npz") as stored:
            for name in parts:
                parts[name].append(stored[name].astype(np.float32))
            n_images = len(stored["image"])
            calibration = stored["image_calibration"]
        frame["image_idx"] = frame["image_idx"] + image_offset
        image_offset += n_images
        frames.append(frame)
        total_duration += float(read_json(work / "segments.json")["duration_s"])
        write_thumbnails(
            frame, work / "keyframes", out_dir / "thumbs" / lecture_id, config.thumb_width
        )

    index = Index(
        moments=pd.concat(frames, ignore_index=True),
        vectors={name: np.concatenate(chunks) for name, chunks in parts.items()},
        lectures=dict(lectures),
        manifest={
            "models": {
                "text": config.embed.text_model,
                "image": config.embed.image_model,
                "asr": asr_model or config.asr.model,
                "ocr": f"{config.ocr.version} ({config.ocr.detector} det, "
                f"{config.ocr.language} rec)",
            },
            "query_instruction": config.embed.query_instruction,
            "image_calibration": {"scale": float(calibration[0]), "bias": float(calibration[1])},
            "jargon_boost": config.jargon_boost,
            "moments": asdict(config.moments),
            "total_duration_s": round(total_duration, 1),
            "stage_seconds": dict(timings or {}),
        },
    )
    index.save(out_dir)
    return index


class Indexer:
    """Runs the pipeline for a set of lectures and assembles the index."""

    def __init__(self, config: PipelineConfig | None = None, work_dir: str | Path = "work") -> None:
        self.config = config or PipelineConfig()
        self.work_dir = Path(work_dir)
        self._asr_model = None
        self._ocr_engine: ocr.OcrEngine | None = None
        self._text_embedder: TextEmbedder | None = None
        self._image_embedder: ImageEmbedder | None = None
        self.timings: dict[str, dict[str, float]] = {}

    # ---- lazily loaded models ----------------------------------------------------------

    @property
    def asr_model(self):
        if self._asr_model is None:
            log.info("loading speech model %s", self.config.asr.model)
            self._asr_model = asr.load_model(self.config.asr)
        return self._asr_model

    @property
    def ocr_engine(self) -> ocr.OcrEngine:
        if self._ocr_engine is None:
            self._ocr_engine = ocr.OcrEngine(self.config.ocr)
        return self._ocr_engine

    @property
    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            log.info("loading text encoder %s", self.config.embed.text_model)
            self._text_embedder = TextEmbedder(self.config.embed)
        return self._text_embedder

    @property
    def image_embedder(self) -> ImageEmbedder:
        if self._image_embedder is None:
            log.info("loading image encoder %s", self.config.embed.image_model)
            self._image_embedder = ImageEmbedder(self.config.embed)
        return self._image_embedder

    # ---- one lecture -------------------------------------------------------------------

    def process_lecture(self, lecture: Lecture, video: Path, force: bool = False) -> Path:
        """Run every stage for one lecture. Returns its work directory."""
        cfg = self.config
        work = self.work_dir / lecture.lecture_id
        work.mkdir(parents=True, exist_ok=True)
        cache = StageCache(work)
        if force:
            cache.state.clear()
        timing = self.timings.setdefault(lecture.lecture_id, {})

        def timed(stage: str, started: float) -> None:
            timing[stage] = round(time.perf_counter() - started, 2)

        info = ingest.probe(video)
        video_key = fingerprint(video.name, video.stat().st_size, round(info.duration_s, 2))
        log.info(
            "[%s] %s (%s)", lecture.lecture_id, lecture.title, format_timestamp(info.duration_s)
        )

        # 1. Slide segmentation -----------------------------------------------------------
        slide_cfg = load_slide_config(cfg.masks_path, lecture.lecture_id, cfg.slides)
        segments_path = work / "segments.json"
        segments_key = fingerprint(STAGE_VERSION["segments"], video_key, asdict(slide_cfg))
        if not cache.fresh("segments", segments_key, segments_path):
            started = time.perf_counter()
            frames = ingest.read_gray_frames(video, fps=slide_cfg.fps, size=slide_cfg.size)
            found = find_slides(
                frames, slide_cfg, lecture_id=lecture.lecture_id, duration_s=info.duration_s
            )
            write_json(segments_path, found)
            cache.done("segments", segments_key)
            timed("segments", started)
        segments = load_segments(segments_path)
        windows = plan_windows(segments, cfg.moments)
        log.info(
            "[%s] %d slide segments, %d moments", lecture.lecture_id, len(segments), len(windows)
        )

        # 2. Keyframe stills ---------------------------------------------------------------
        times = keyframe_times(windows)
        keyframe_dir = work / "keyframes"
        keyframes_key = fingerprint(
            STAGE_VERSION["keyframes"], segments_key, asdict(cfg.moments), cfg.keyframe_max_width
        )
        if not cache.fresh("keyframes", keyframes_key, keyframe_dir):
            started = time.perf_counter()
            if keyframe_dir.exists():  # segmentation changed: old stills no longer apply
                for stale in keyframe_dir.glob("*.jpg"):
                    stale.unlink()
            ingest.extract_stills(
                video,
                times,
                keyframe_dir,
                max_width=cfg.keyframe_max_width,
                duration_s=info.duration_s,
            )
            cache.done("keyframes", keyframes_key)
            timed("keyframes", started)
        stills = {t: keyframe_dir / still_name(t) for t in times}

        # 3. OCR -----------------------------------------------------------------------------
        ocr_path = work / "ocr.json"
        ocr_key = fingerprint(STAGE_VERSION["ocr"], keyframes_key, asdict(cfg.ocr))
        if not cache.fresh("ocr", ocr_key, ocr_path):
            started = time.perf_counter()
            write_json(ocr_path, read_slides(self.ocr_engine, stills))
            cache.done("ocr", ocr_key)
            timed("ocr", started)
        texts = slide_texts(read_json(ocr_path), times)

        # 4. Speech recognition ---------------------------------------------------------------
        hotwords = asr.build_hotwords(texts.values()) if cfg.jargon_boost else ""
        asr_path = work / "asr.jsonl"
        asr_key = fingerprint(STAGE_VERSION["asr"], video_key, asdict(cfg.asr), hotwords)
        if not cache.fresh("asr", asr_key, asr_path):
            started = time.perf_counter()
            if info.has_audio:
                audio = ingest.extract_audio(video, work / "audio.wav")
                segments_asr = asr.transcribe(
                    audio, cfg.asr, model=self.asr_model, hotwords=hotwords
                )
            else:
                log.warning(
                    "[%s] no audio track: moments will have no transcript", lecture.lecture_id
                )
                segments_asr = []
            asr.save_segments(asr_path, segments_asr)
            write_json(work / "asr.meta.json", {"hotwords": hotwords, "model": cfg.asr.model})
            cache.done("asr", asr_key)
            timed("asr", started)
        words = list(asr.iter_words(asr.load_segments(asr_path)))

        # 5. Moments ---------------------------------------------------------------------------
        moments_path = work / "moments.parquet"
        moments_key = fingerprint(STAGE_VERSION["moments"], ocr_key, asr_key, lecture.to_dict())
        if not cache.fresh("moments", moments_key, moments_path):
            moments_table(lecture, windows, words, texts).to_parquet(moments_path, index=False)
            cache.done("moments", moments_key)
        frame = pd.read_parquet(moments_path)

        # 6. Embeddings ------------------------------------------------------------------------
        vectors_path = work / "vectors.npz"
        vectors_key = fingerprint(STAGE_VERSION["vectors"], moments_key, asdict(cfg.embed))
        if not cache.fresh("vectors", vectors_key, vectors_path):
            started = time.perf_counter()
            vectors = self._embed(frame, [stills[t] for t in times])
            np.savez(vectors_path, **vectors)
            cache.done("vectors", vectors_key)
            timed("vectors", started)

        if timing:
            spent = ", ".join(f"{stage} {seconds:.0f}s" for stage, seconds in timing.items())
            log.info("[%s] done: %s", lecture.lecture_id, spent)
        else:
            log.info("[%s] up to date", lecture.lecture_id)
        return work

    def _embed(self, frame: pd.DataFrame, keyframes: list[Path]) -> dict[str, np.ndarray]:
        """The four embedding matrices for one lecture's moments."""
        return embed_moments(frame, keyframes, self.text_embedder, self.image_embedder)

    # ---- the whole library -------------------------------------------------------------

    def build(
        self,
        lectures: Mapping[str, Lecture],
        videos: Mapping[str, Path],
        out_dir: str | Path,
        force: bool = False,
    ) -> Index:
        """Process every lecture, then assemble and save the index."""
        if not lectures:
            raise ValueError("no lectures to index")
        started = time.perf_counter()
        for lecture_id, lecture in lectures.items():
            self.process_lecture(lecture, Path(videos[lecture_id]), force=force)
        index = self.assemble(lectures, out_dir)
        log.info(
            "index written to %s: %d moments, %d keyframes, %d lectures (%.0fs)",
            out_dir,
            len(index),
            len(index.image),
            len(lectures),
            time.perf_counter() - started,
        )
        return index

    def assemble(self, lectures: Mapping[str, Lecture], out_dir: str | Path) -> Index:
        """Concatenate the per-lecture results into one index and write thumbnails."""
        work_dirs = {lecture_id: self.work_dir / lecture_id for lecture_id in lectures}
        return assemble(lectures, work_dirs, out_dir, self.config, timings=self.timings)
