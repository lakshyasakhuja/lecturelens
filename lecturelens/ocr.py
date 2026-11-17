"""Slide OCR: read the text on a keyframe, in reading order.

Models are PP-OCRv5 (text detection + English recognition), run through ONNX Runtime via
RapidOCR rather than through PaddlePaddle. The weights are the same; the runtime is one
wheel with no CUDA build of its own, so it installs next to PyTorch on a GPU notebook
without the two fighting over CUDA libraries.

OCR returns text boxes in detection order, which is not reading order, and the detector
sometimes splits one line of text into several boxes. ``reading_order`` groups boxes into
rows, sorts each row left to right, and joins the fragments of a line back together, so a
slide reads title first, then each bullet as one line, top to bottom.
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


def group_rows(lines: Sequence[OcrLine]) -> list[list[OcrLine]]:
    """Group text boxes into visual rows, top to bottom, each row sorted left to right.

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
    return [sorted(row, key=lambda item: item.box[0]) for row in rows]


def _join(left: OcrLine, right: OcrLine) -> OcrLine:
    """Join two fragments of one text line, removing a character read twice at the seam.

    The detector pads every box, so when it splits a line, neighbouring boxes overlap by
    about a character and the recogniser reads that character in both: ``"Noisy"`` then
    ``"y but cheap"``, or ``"Shuffle t"`` then ``"the data"``. If the boxes overlap and a
    one- or two-letter token at the seam repeats the edge of its neighbour, it is dropped.
    """
    left_tokens, right_tokens = left.text.split(), right.text.split()
    if left.box[2] > right.box[0] and left_tokens and right_tokens:
        tail, head = left_tokens[-1], right_tokens[0]
        if len(head) <= 2 and len(tail) > len(head) and tail.endswith(head):
            right_tokens = right_tokens[1:]
        elif len(tail) <= 2 and len(head) > len(tail) and head.startswith(tail):
            left_tokens = left_tokens[:-1]
    return OcrLine(
        text=" ".join(left_tokens + right_tokens),
        confidence=min(left.confidence, right.confidence),
        box=(
            min(left.box[0], right.box[0]),
            min(left.box[1], right.box[1]),
            max(left.box[2], right.box[2]),
            max(left.box[3], right.box[3]),
        ),
    )


def join_row(row: Sequence[OcrLine], max_gap: float = 1.0) -> list[OcrLine]:
    """Merge the fragments of a row that belong to one line of text.

    Neighbours are joined when the gap between them is at most ``max_gap`` line heights
    (an ordinary word space, or an overlap). A wider gap means separate columns, which stay
    separate lines.
    """
    merged: list[OcrLine] = []
    for line in row:
        if merged:
            previous = merged[-1]
            gap = line.box[0] - previous.box[2]
            if gap <= max_gap * min(previous.height, line.height):
                merged[-1] = _join(previous, line)
                continue
        merged.append(line)
    return merged


def reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Text lines in reading order: rows top to bottom, fragments of a line joined."""
    return [line for row in group_rows(lines) for line in join_row(row)]


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

    def read(self, image: str | Path | Any) -> SlideText:
        """OCR one image and return its lines in reading order.

        ``image`` is a path, a PIL image or an RGB array.
        """
        if isinstance(image, (str, Path)):
            source = str(image)
        else:
            import numpy as np

            # RapidOCR reads arrays as OpenCV does, in BGR order.
            source = np.ascontiguousarray(
                np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)[..., ::-1]
            )
        result = self._load()(source)
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
