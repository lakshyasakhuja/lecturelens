"""The search index: what indexing produces and search consumes.

An index is a directory of plain files::

    manifest.json        format version, model names, counts, how it was built
    lectures.json        catalogue entries for the lectures it covers
    moments.parquet      one row per moment: times, slide text, transcript, links
    text.npy             (n_moments, d)    "Slide: ... Speech: ..." embeddings
    speech.npy           (n_moments, d)    transcript-only embeddings
    slide.npy            (n_moments, d)    slide-text-only embeddings
    image.npy            (n_keyframes, d') SigLIP 2 keyframe embeddings
    thumbs/<lecture>/<ms>.jpg              one thumbnail per keyframe

Row ``i`` of the three text matrices is moment ``i``. Windows of a long slide share one
keyframe, so image vectors are stored once per keyframe and each moment points at its row
through ``image_idx``.

There is no vector database. A library of a dozen lectures is a couple of thousand
moments, and an exhaustive dot product over that takes a few hundredths of a millisecond.
Lexical (BM25) structures are not stored either: they are rebuilt from ``moments.parquet``
when the index loads, in about half a second for 2,000 moments, which removes any chance
of the stored index and the tokeniser in the code drifting apart.

The three text matrices exist so that the ablations can isolate each source (transcript
only, slide only, both) and so that a result can say whether it matched what was said or
what was on the slide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from slideseek import __version__
from slideseek.catalog import Lecture
from slideseek.utils import read_json, write_json

INDEX_FORMAT = 1
TEXT_VECTORS = ("text", "speech", "slide")
MOMENT_COLUMNS = (
    "moment_id",
    "lecture_id",
    "t_start",
    "t_end",
    "slide_text",
    "transcript",
    "keyframe_t",
    "video_link",
    "kind",
    "segment",
    "image_idx",
    "thumb",
)


class InvalidIndex(RuntimeError):
    """The index directory is missing, incomplete, or inconsistent."""


@dataclass
class Index:
    moments: pd.DataFrame
    vectors: dict[str, np.ndarray]
    lectures: dict[str, Lecture]
    manifest: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None
    # Where a lecture's thumbnails are, when not under ``root``: an index combined from
    # several (``combine``) keeps each lecture's own directory.
    lecture_roots: dict[str, Path] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.moments)

    @property
    def text(self) -> np.ndarray:
        return self.vectors["text"]

    @property
    def speech(self) -> np.ndarray:
        return self.vectors["speech"]

    @property
    def slide(self) -> np.ndarray:
        return self.vectors["slide"]

    @property
    def image(self) -> np.ndarray:
        return self.vectors["image"]

    def thumb_path(self, row: int) -> Path | None:
        """Absolute path of a moment's thumbnail, if the index is on disk and has one."""
        relative = self.moments["thumb"].iat[row]
        root = self.lecture_roots.get(self.moments["lecture_id"].iat[row], self.root)
        if root is None or not relative:
            return None
        path = root / relative
        return path if path.is_file() else None

    @classmethod
    def combine(cls, *indexes: Index) -> Index:
        """One index holding every moment of ``indexes``, in order. Nothing is written.

        Image rows are shifted the way ``pipeline.assemble`` shifts them, so each moment
        still points at its own keyframe. Indexes that name their embedding models must name
        the same ones, and no lecture may be in two of them. The manifest is the first index's,
        with the counts and durations of all of them.
        """
        if not indexes:
            raise ValueError("nothing to combine")
        first = indexes[0]
        models = first.manifest.get("models", {})
        lectures: dict[str, Lecture] = {}
        lecture_roots: dict[str, Path] = {}
        frames, parts = [], {name: [] for name in (*TEXT_VECTORS, "image")}
        offset = 0
        for index in indexes:
            theirs = index.manifest.get("models", {})
            for model in ("text", "image"):
                if theirs.get(model) and models.get(model) and theirs[model] != models[model]:
                    raise InvalidIndex(
                        f"cannot combine indexes built with different {model} models: "
                        f"{models.get(model)!r} and {theirs.get(model)!r}"
                    )
            clash = set(lectures) & set(index.lectures)
            if clash:
                raise InvalidIndex(f"lecture(s) in two indexes: {sorted(clash)}")
            lectures.update(index.lectures)
            for lecture_id in index.lectures:
                root = index.lecture_roots.get(lecture_id, index.root)
                if root is not None:
                    lecture_roots[lecture_id] = root
            frame = index.moments.copy()
            frame["image_idx"] = frame["image_idx"] + offset
            offset += len(index.image)
            frames.append(frame)
            for name in parts:
                parts[name].append(index.vectors[name])
        manifest = {
            **first.manifest,
            "n_moments": sum(len(index) for index in indexes),
            "n_keyframes": offset,
            "n_lectures": len(lectures),
            "total_duration_s": round(
                sum(float(index.manifest.get("total_duration_s", 0)) for index in indexes), 1
            ),
        }
        combined = cls(
            moments=pd.concat(frames, ignore_index=True),
            vectors={name: np.concatenate(chunks) for name, chunks in parts.items()},
            lectures=lectures,
            manifest=manifest,
            root=first.root,
            lecture_roots=lecture_roots,
        )
        combined.validate()
        return combined

    def validate(self) -> None:
        """Check that the pieces agree with each other. Raises ``InvalidIndex`` if not."""
        n = len(self.moments)
        missing = [c for c in MOMENT_COLUMNS if c not in self.moments.columns]
        if missing:
            raise InvalidIndex(f"moments table is missing column(s): {missing}")
        for name in TEXT_VECTORS:
            if name not in self.vectors:
                raise InvalidIndex(f"missing vectors: {name}")
            if len(self.vectors[name]) != n:
                raise InvalidIndex(
                    f"{name} has {len(self.vectors[name])} rows but there are {n} moments"
                )
        if "image" not in self.vectors:
            raise InvalidIndex("missing vectors: image")
        if n:
            image_idx = self.moments["image_idx"].to_numpy()
            if image_idx.min() < 0 or image_idx.max() >= len(self.image):
                raise InvalidIndex("image_idx points outside the image matrix")
        unknown = set(self.moments["lecture_id"]) - set(self.lectures)
        if unknown:
            raise InvalidIndex(f"moments refer to lectures not in the catalogue: {sorted(unknown)}")
        if self.moments["moment_id"].duplicated().any():
            raise InvalidIndex("duplicate moment ids")

    def save(self, root: str | Path) -> Path:
        """Write the index. ``manifest.json`` goes last: its presence means the index is whole."""
        self.validate()
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").unlink(missing_ok=True)

        tmp = root / "moments.parquet.partial"
        self.moments[list(MOMENT_COLUMNS)].to_parquet(tmp, index=False)
        os.replace(tmp, root / "moments.parquet")

        for name, matrix in self.vectors.items():
            # Unit vectors survive float16 with a similarity error around 1e-4, at half the size.
            tmp = root / f"{name}.partial.npy"
            np.save(tmp, np.asarray(matrix, dtype=np.float16))
            os.replace(tmp, root / f"{name}.npy")

        write_json(root / "lectures.json", [lec.to_dict() for lec in self.lectures.values()])
        manifest = {
            **self.manifest,
            "format": INDEX_FORMAT,
            "slideseek_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_moments": len(self.moments),
            "n_keyframes": len(self.image),
            "n_lectures": len(self.lectures),
            "dims": {
                name: int(m.shape[1]) if m.ndim == 2 else 0 for name, m in self.vectors.items()
            },
        }
        write_json(root / "manifest.json", manifest)
        self.manifest = manifest
        self.root = root
        return root

    @classmethod
    def load(cls, root: str | Path) -> Index:
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise InvalidIndex(
                f"no index at {root} (manifest.json not found). Build one with `slideseek index`."
            )
        manifest = read_json(manifest_path)
        if manifest.get("format") != INDEX_FORMAT:
            raise InvalidIndex(
                f"index format {manifest.get('format')} is not supported by this version "
                f"(expected {INDEX_FORMAT}); rebuild the index"
            )
        moments = pd.read_parquet(root / "moments.parquet")
        vectors = {
            name: np.load(root / f"{name}.npy").astype(np.float32)
            for name in (*TEXT_VECTORS, "image")
        }
        lectures = {
            item["lecture_id"]: Lecture(**item) for item in read_json(root / "lectures.json")
        }
        index = cls(
            moments=moments, vectors=vectors, lectures=lectures, manifest=manifest, root=root
        )
        index.validate()
        return index
