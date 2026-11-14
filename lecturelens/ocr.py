"""Slide OCR: read the text on a keyframe, in reading order.

Models are PP-OCRv5 (text detection + English recognition), run through ONNX Runtime via
RapidOCR rather than through PaddlePaddle. The weights are the same; the runtime is one
wheel with no CUDA build of its own, so it installs next to PyTorch on a GPU notebook
without the two fighting over CUDA libraries.

OCR returns text boxes in detection order, which is not reading order. ``reading_order``
groups boxes into rows by vertical overlap and sorts each row left to right, so a slide
reads title first, then bullets top to bottom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OcrConfig:
    version: str = "PP-OCRv5"
    detector: str = "server"  # "mobile" is faster, "server" finds small text more reliably
    recognizer: str = "mobile"  # the English PP-OCRv5 recogniser ships as "mobile" only
    language: str = "en"
    min_confidence: float = 0.5
    use_gpu: bool = False


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

    @property
    def y_center(self) -> float:
        return 0.5 * (self.box[1] + self.box[3])

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "conf": round(self.confidence, 3),
            "box": [round(v, 1) for v in self.box],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OcrLine:
        return cls(data["text"], float(data["conf"]), tuple(float(v) for v in data["box"]))


@dataclass(frozen=True)
class SlideText:
    """The text of one keyframe, as lines in reading order."""

    lines: tuple[OcrLine, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {"lines": [line.to_dict() for line in self.lines]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlideText:
        return cls(tuple(OcrLine.from_dict(line) for line in data.get("lines", [])))


def reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Sort text boxes top to bottom, and left to right within a row.

    Two boxes share a row when the second one's vertical centre lies within half a line
    height of the row's. Sorting by ``y`` alone would interleave boxes that sit on the same
    visual line but differ by a pixel or two.
    """
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: item.y_center):
        if rows:
            row = rows[-1]
            row_center = sum(item.y_center for item in row) / len(row)
            row_height = sum(item.height for item in row) / len(row)
            if abs(line.y_center - row_center) <= 0.5 * min(row_height, line.height):
                row.append(line)
                continue
        rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda item: item.box[0])]


class OcrEngine:
    """PP-OCRv5 through RapidOCR. Loads its models on first use."""

    def __init__(self, config: OcrConfig | None = None) -> None:
        self.config = config or OcrConfig()
        self._engine = None

    def _load(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangDet, LangRec, ModelType, OCRVersion

            cfg = self.config
            version = OCRVersion(cfg.version)
            self._engine = RapidOCR(
                params={
                    "Global.log_level": "warning",
                    "Global.text_score": cfg.min_confidence,
                    # Slides are upright; the orientation classifier only costs time.
                    "Global.use_cls": False,
                    "Det.ocr_version": version,
                    "Det.lang_type": LangDet.CH,  # the PP-OCRv5 detector is script-agnostic
                    "Det.model_type": ModelType(cfg.detector),
                    "Rec.ocr_version": version,
                    "Rec.lang_type": LangRec(cfg.language),
                    "Rec.model_type": ModelType(cfg.recognizer),
                    "EngineConfig.onnxruntime.use_cuda": cfg.use_gpu,
                }
            )
        return self._engine

    def read(self, image_path: str | Path) -> SlideText:
        """OCR one image and return its lines in reading order."""
        result = self._load()(str(image_path))
        if result.txts is None or result.boxes is None:
            return SlideText()
        lines = []
        for text, score, box in zip(result.txts, result.scores, result.boxes, strict=True):
            text = text.strip()
            if not text or score < self.config.min_confidence:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(OcrLine(text, float(score), (min(xs), min(ys), max(xs), max(ys))))
        return SlideText(tuple(reading_order(lines)))
