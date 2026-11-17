"""A tiny hand-built index and deterministic stand-in encoders.

Search and evaluation logic can then be tested exactly, in milliseconds, with no model.
The fake encoders embed text as a bag of stems hashed into a small vector, so texts that
share words are close and texts that share none are orthogonal. That is all the structure
the tests need.
"""

from __future__ import annotations

import io
import zlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from slideseek.catalog import Lecture
from slideseek.index import Index
from slideseek.moments import moment_text
from slideseek.text import analyze

DIM = 4096  # large enough that two different stems practically never share a bucket


def embed_text(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for term in analyze(text):
        vector[zlib.crc32(term.encode()) % DIM] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class FakeTextEmbedder:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def encode_documents(self, texts, progress=False):
        return (
            np.stack([embed_text(t) for t in texts])
            if len(texts)
            else np.zeros((0, DIM), np.float32)
        )

    def encode_queries(self, queries):
        self.calls += 1
        return np.stack([embed_text(q) for q in queries])


class FakeImageEmbedder:
    """Images are described by a caption; the 'image' vector is the caption's embedding."""

    device = "cpu"

    def __init__(self):
        self.calls = 0

    def encode_texts(self, texts):
        self.calls += 1
        return np.stack([embed_text(t) for t in texts])

    def encode_pictures(self, pictures):
        """A stand-in picture says what it shows in its caption (see ``make_picture``)."""
        self.calls += 1
        return np.stack([embed_text(picture.info.get("caption", "")) for picture in pictures])


class FakePictureOcr:
    """Reads the words a stand-in picture carries (see ``make_picture``)."""

    def __init__(self):
        self.calls = 0

    def read(self, picture):
        self.calls += 1
        return SimpleNamespace(text=picture.info.get("words", ""))


def make_picture(caption: str, words: str = "", size=(64, 48)) -> bytes:
    """PNG bytes standing in for a photo of a slide.

    What the picture looks like (``caption``) and the text on it (``words``) travel in PNG
    text chunks, which survive an upload and decoding, so the fakes can read them back.
    """
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("caption", caption)
    info.add_text("words", words)
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


def open_picture(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data))


# (lecture, t_start, t_end, slide text, transcript, keyframe id, what the keyframe looks like)
# Rows 3 and 4 are two overlapping windows of one long slide, so they share a keyframe.
MOMENTS = [
    (
        "lec01",
        0,
        60,
        "Course overview",
        "welcome to the course on machine learning",
        0,
        "title page",
    ),
    (
        "lec01",
        60,
        150,
        "The validation set",
        "we keep some data aside that the model is never fitted on",
        1,
        "bullet list",
    ),
    (
        "lec01",
        150,
        240,
        "Overfitting",
        "the orange curve turns around and starts to climb",
        2,
        "plot two curves rising falling",
    ),
    (
        "lec01",
        240,
        285,
        "Regularisation",
        "weight decay discourages large parameter values",
        3,
        "bullet points",
    ),
    (
        "lec01",
        270,
        315,
        "Regularisation",
        "dropout randomly removes units during training",
        3,
        "bullet points",
    ),
    (
        "lec02",
        0,
        80,
        "A three-layer network",
        "each unit computes a weighted sum of its inputs",
        4,
        "diagram circles connected lines",
    ),
    (
        "lec02",
        80,
        170,
        "Backpropagation",
        "the chain rule applied from the output back to the input",
        5,
        "equations",
    ),
    ("lec02", 170, 260, "", "imagine walking downhill in thick fog", 6, "photo of a foggy valley"),
]


def make_index(root=None) -> Index:
    rows, captions = [], {}
    for n, (lecture_id, t0, t1, slide, speech, keyframe, caption) in enumerate(MOMENTS):
        captions[keyframe] = caption
        rows.append(
            {
                "moment_id": f"{lecture_id}:{n:04d}",
                "lecture_id": lecture_id,
                "t_start": float(t0),
                "t_end": float(t1),
                "slide_text": slide,
                "transcript": speech,
                "keyframe_t": float(t1 - 1),
                "video_link": f"https://v.example/{lecture_id}?t={t0}",
                "kind": "slide",
                "segment": keyframe,
                "image_idx": keyframe,
                "thumb": "",
            }
        )
    moments = pd.DataFrame(rows)
    vectors = {
        "text": np.stack([embed_text(moment_text(r["slide_text"], r["transcript"])) for r in rows]),
        "speech": np.stack([embed_text(r["transcript"]) for r in rows]),
        "slide": np.stack([embed_text(r["slide_text"]) for r in rows]),
        "image": np.stack([embed_text(captions[k]) for k in sorted(captions)]),
    }
    lectures = {
        "lec01": Lecture(
            "lec01", "Generalisation", video_link_template="https://v.example/lec01?t={seconds}"
        ),
        "lec02": Lecture(
            "lec02", "Neural networks", video_link_template="https://v.example/lec02?t={seconds}"
        ),
    }
    # With this calibration an identical caption (cosine 1) is a 99% match and an unrelated
    # one (cosine 0) a 1% match, which is roughly how sharply SigLIP separates them.
    manifest = {"models": {}, "image_calibration": {"scale": 10.0, "bias": -5.0}}
    index = Index(moments=moments, vectors=vectors, lectures=lectures, manifest=manifest)
    index.validate()
    if root is not None:
        index.save(root)
    return index


# --------------------------------------------------------------------------------------
# Stand-ins for the indexing models, so the pipeline's orchestration can be tested with
# real ffmpeg decoding and real slide detection but no network and no weights.
# --------------------------------------------------------------------------------------


class _Piece:
    def __init__(self, word, start, end):
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


class _Phrase:
    def __init__(self, start, words):
        self.start, self.end = start, words[-1].end
        self.words = words
        self.text = "".join(w.word for w in words)


class FakeWhisper:
    """Says "second <n> of the lecture" once every five seconds of audio."""

    def __init__(self):
        self.calls = 0
        self.hotwords = None

    def transcribe(self, audio, **options):
        self.calls += 1
        self.hotwords = options.get("hotwords")
        seconds = int(len(audio) / 16_000)
        phrases = []
        for t in range(0, seconds, 5):
            words = [
                _Piece(" second", t + 0.2, t + 0.6),
                _Piece(f" {t}", t + 0.6, t + 0.9),
                _Piece(" of", t + 0.9, t + 1.0),
                _Piece(" the", t + 1.0, t + 1.1),
                _Piece(" lecture", t + 1.1, t + 1.6),
            ]
            phrases.append(_Phrase(t + 0.2, words))
        return iter(phrases), None


class FakeOcr:
    """Reads every keyframe as a slide titled after its timestamp."""

    def __init__(self):
        self.calls = 0

    def read(self, image_path):
        from pathlib import Path

        from slideseek.ocr import OcrLine, SlideText

        self.calls += 1
        seconds = int(Path(image_path).stem) // 1000
        return SlideText((OcrLine(f"Gaussian Processes slide at {seconds}", 0.99, (0, 0, 10, 10)),))


class FakeIndexTextEmbedder(FakeTextEmbedder):
    def encode_documents(self, texts, progress=False):
        self.calls += 1
        return super().encode_documents(texts, progress)


class FakeIndexImageEmbedder(FakeImageEmbedder):
    calibration = (10.0, -5.0)

    def encode_images(self, paths):
        from pathlib import Path

        self.calls += 1
        if not len(paths):
            return np.zeros((0, DIM), np.float32)
        return np.stack([embed_text(f"keyframe{Path(p).stem}") for p in paths])
