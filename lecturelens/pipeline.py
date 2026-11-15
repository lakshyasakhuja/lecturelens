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
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lecturelens import asr, ingest, ocr
from lecturelens.catalog import Lecture
from lecturelens.embed import EmbedConfig, ImageEmbedder, TextEmbedder
from lecturelens.index import Index
from lecturelens.moments import MomentConfig, build_moments, moment_text, plan_windows
from lecturelens.slides import SlideConfig, SlideSegment, detect_slides, load_slide_config
from lecturelens.utils import format_timestamp, read_json, write_json

log = logging.getLogger("lecturelens")


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
STAGE_VERSION = {"segments": 1, "keyframes": 1, "ocr": 2, "asr": 2, "moments": 1, "vectors": 1}


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
            detection = detect_slides(frames, slide_cfg)
            write_json(
                segments_path,
                {
                    "lecture_id": lecture.lecture_id,
                    "duration_s": info.duration_s,
                    "t_high": detection.t_high,
                    "t_low": detection.t_low,
                    "noise": detection.noise,
                    "merged": detection.merged,
                    "segments": [seg.to_dict() for seg in detection.segments],
                },
            )
            cache.done("segments", segments_key)
            timed("segments", started)
        segments = [SlideSegment.from_dict(d) for d in read_json(segments_path)["segments"]]
        windows = plan_windows(segments, cfg.moments)
        log.info(
            "[%s] %d slide segments, %d moments", lecture.lecture_id, len(segments), len(windows)
        )

        # 2. Keyframe stills ---------------------------------------------------------------
        keyframe_times = sorted({w.keyframe_t for w in windows})
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
                keyframe_times,
                keyframe_dir,
                max_width=cfg.keyframe_max_width,
                duration_s=info.duration_s,
            )
            cache.done("keyframes", keyframes_key)
            timed("keyframes", started)
        stills = {t: keyframe_dir / f"{_keyframe_ms(t):09d}.jpg" for t in keyframe_times}

        # 3. OCR -----------------------------------------------------------------------------
        ocr_path = work / "ocr.json"
        ocr_key = fingerprint(STAGE_VERSION["ocr"], keyframes_key, asdict(cfg.ocr))
        if not cache.fresh("ocr", ocr_key, ocr_path):
            started = time.perf_counter()
            pages = {
                str(_keyframe_ms(t)): self.ocr_engine.read(path).to_dict()
                for t, path in stills.items()
            }
            write_json(ocr_path, pages)
            cache.done("ocr", ocr_key)
            timed("ocr", started)
        pages = read_json(ocr_path)
        slide_texts = {
            t: ocr.SlideText.from_dict(pages[str(_keyframe_ms(t))]).text for t in keyframe_times
        }

        # 4. Speech recognition ---------------------------------------------------------------
        hotwords = asr.build_hotwords(slide_texts.values()) if cfg.jargon_boost else ""
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
            moments = build_moments(lecture, windows, words, slide_texts)
            frame = pd.DataFrame([m.to_dict() for m in moments])
            position = {t: i for i, t in enumerate(keyframe_times)}
            frame["image_idx"] = [position[m.keyframe_t] for m in moments]
            frame["thumb"] = [
                f"thumbs/{lecture.lecture_id}/{_keyframe_ms(m.keyframe_t):09d}.jpg" for m in moments
            ]
            frame.to_parquet(moments_path, index=False)
            cache.done("moments", moments_key)
        frame = pd.read_parquet(moments_path)

        # 6. Embeddings ------------------------------------------------------------------------
        vectors_path = work / "vectors.npz"
        vectors_key = fingerprint(STAGE_VERSION["vectors"], moments_key, asdict(cfg.embed))
        if not cache.fresh("vectors", vectors_key, vectors_path):
            started = time.perf_counter()
            vectors = self._embed(frame, [stills[t] for t in keyframe_times])
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
        slide_texts = frame["slide_text"].tolist()
        transcripts = frame["transcript"].tolist()
        combined = [moment_text(s, t) for s, t in zip(slide_texts, transcripts, strict=True)]
        text = self.text_embedder
        return {
            "text": text.encode_documents(combined),
            "speech": text.encode_documents(transcripts),
            "slide": text.encode_documents([" ".join(s.split()) for s in slide_texts]),
            "image": self.image_embedder.encode_images(keyframes),
            "image_calibration": np.asarray(self.image_embedder.calibration, dtype=np.float64),
        }

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
        from PIL import Image

        out_dir = Path(out_dir)
        frames, parts = [], {"text": [], "speech": [], "slide": [], "image": []}
        image_offset = 0
        total_duration = 0.0
        for lecture_id in lectures:
            work = self.work_dir / lecture_id
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

            thumb_dir = out_dir / "thumbs" / lecture_id
            thumb_dir.mkdir(parents=True, exist_ok=True)
            wanted = {Path(relative).name for relative in frame["thumb"]}
            for stale in thumb_dir.glob("*.jpg"):
                if stale.name not in wanted:
                    stale.unlink()
            for name in wanted:
                target = thumb_dir / name
                if not target.is_file():
                    with Image.open(work / "keyframes" / name) as still:
                        still = still.convert("RGB")
                        height = round(still.height * self.config.thumb_width / still.width)
                        still.resize(
                            (self.config.thumb_width, height), Image.Resampling.LANCZOS
                        ).save(target, quality=85, optimize=True)

        cfg = self.config
        index = Index(
            moments=pd.concat(frames, ignore_index=True),
            vectors={name: np.concatenate(chunks) for name, chunks in parts.items()},
            lectures=dict(lectures),
            manifest={
                "models": {
                    "text": cfg.embed.text_model,
                    "image": cfg.embed.image_model,
                    "asr": cfg.asr.model,
                    "ocr": f"{cfg.ocr.version} ({cfg.ocr.detector} det, {cfg.ocr.language} rec)",
                },
                "query_instruction": cfg.embed.query_instruction,
                "image_calibration": {
                    "scale": float(calibration[0]),
                    "bias": float(calibration[1]),
                },
                "jargon_boost": cfg.jargon_boost,
                "moments": asdict(cfg.moments),
                "total_duration_s": round(total_duration, 1),
                "stage_seconds": self.timings,
            },
        )
        index.save(out_dir)
        return index
