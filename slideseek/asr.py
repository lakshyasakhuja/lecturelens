"""Speech recognition with word-level timestamps (faster-whisper).

Word timestamps are the point: they are what lets the transcript be cut at slide boundaries
and attached to the right moment.

Two settings depart from Whisper's defaults, both for long-form robustness:

* ``vad_filter=True`` removes silence before decoding. Whisper tends to hallucinate text in
  long silences, and lectures are full of them (the lecturer writing, a pause for questions).
* ``condition_on_previous_text=False``. Conditioning each 30-second window on the previous
  window's text makes spelling a little more consistent, but when the model does hallucinate
  or loop, the error feeds itself and can run for minutes. For search that is the worse
  failure: one repeated phrase pollutes every moment it spreads to.

**Jargon boost.** Whisper misspells technical terms it has no context for. The slides have
that context. ``build_hotwords`` picks distinctive terms out of the slide OCR, and
``transcribe`` passes them as hotwords, which faster-whisper prepends to the prompt of every
window. The effect on word error rate is measured by ``eval/eval_asr.py``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slideseek.text import STOPWORDS


@dataclass(frozen=True)
class AsrConfig:
    model: str = "large-v3-turbo"
    # "auto" picks CUDA when there is one, else the CPU. "mps" (Apple GPU) must be asked for:
    # CTranslate2 calls that backend experimental, though it ran about twice as fast as the
    # CPU on an M4 with near-identical output.
    device: str = "auto"
    compute_type: str = "auto"  # "auto": float16 on a GPU, int8 on CPU
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)


@dataclass(frozen=True)
class AsrSegment:
    """A phrase as Whisper segmented it, with its words."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [
                {"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.prob, 3)}
                for w in self.words
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AsrSegment:
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            words=tuple(
                Word(w["w"], float(w["s"]), float(w["e"]), float(w.get("p", 1.0)))
                for w in data["words"]
            ),
        )


def resolve_device(device: str = "auto", compute_type: str = "auto") -> tuple[str, str]:
    """Pick the device and numeric precision CTranslate2 should use."""
    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"
    return device, compute_type


def load_model(config: AsrConfig | None = None):
    """Load the Whisper model once; pass it to ``transcribe`` for every lecture."""
    from faster_whisper import WhisperModel

    config = config or AsrConfig()
    device, compute_type = resolve_device(config.device, config.compute_type)
    return WhisperModel(config.model, device=device, compute_type=compute_type)


def transcribe(
    audio_path: str | Path,
    config: AsrConfig | None = None,
    *,
    model=None,
    hotwords: str | None = None,
) -> list[AsrSegment]:
    """Transcribe an audio (or video) file into segments with word timestamps."""
    from slideseek.ingest import read_audio

    config = config or AsrConfig()
    model = model or load_model(config)
    raw_segments, _info = model.transcribe(
        read_audio(audio_path, sample_rate=16_000),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        word_timestamps=True,
        condition_on_previous_text=config.condition_on_previous_text,
        hotwords=hotwords or None,
    )
    segments = []
    for seg in raw_segments:  # a generator: decoding happens as we iterate
        words = merge_word_pieces(
            (w.word, float(w.start), float(w.end), float(w.probability)) for w in (seg.words or [])
        )
        if words:
            segments.append(AsrSegment(float(seg.start), float(seg.end), seg.text.strip(), words))
    return segments


def merge_word_pieces(pieces: Iterable[tuple[str, float, float, float]]) -> tuple[Word, ...]:
    """Join Whisper's word pieces into whitespace-delimited words.

    Whisper marks the start of a new word with a leading space. A piece without one
    continues the previous word: "held-out" arrives as ``" held"`` and ``"-out"``. Joining
    them here means a word is always something a reader would call a word, and transcripts
    rebuilt from words read ``held-out`` rather than ``held -out``.
    """
    words: list[Word] = []
    for raw, start, end, prob in pieces:
        text = raw.strip()
        if not text:
            continue
        if words and not raw[:1].isspace():
            last = words[-1]
            words[-1] = Word(last.text + text, last.start, end, min(last.prob, prob))
        else:
            words.append(Word(text, start, end, prob))
    return tuple(words)


def save_segments(path: str | Path, segments: Iterable[AsrSegment]) -> None:
    """Write one JSON object per line, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_segments(path: str | Path) -> list[AsrSegment]:
    with Path(path).open(encoding="utf-8") as handle:
        return [AsrSegment.from_dict(json.loads(line)) for line in handle if line.strip()]


def iter_words(segments: Iterable[AsrSegment]) -> Iterator[Word]:
    for segment in segments:
        yield from segment.words


def words_between(words: Sequence[Word], t0: float, t1: float) -> list[Word]:
    """Words whose midpoint falls in ``[t0, t1)``."""
    return [w for w in words if t0 <= w.mid < t1]


def jargon_terms(
    slide_texts: Iterable[str], max_terms: int = 50, max_chars: int = 600
) -> list[str]:
    """Choose distinctive terms from slide text, most useful first.

    A term qualifies if it looks like vocabulary Whisper might get wrong: it is long, or it
    is capitalised in the middle of a line (a name such as "Bellman"), or it mixes letters
    with digits or hyphens ("L2", "k-fold"). Terms are ranked by how many slides they appear
    on, since a word that recurs across slides is a subject of the lecture rather than OCR
    noise, with names and mixed tokens ahead of ordinary long words on a tie. The budget is
    capped because Whisper's prompt holds only 224 tokens.
    """
    import re

    slide_count: Counter[str] = Counter()
    surface: dict[str, str] = {}
    distinctive: set[str] = set()  # names and mixed tokens: the likeliest to be misspelt
    for text in slide_texts:
        seen: set[str] = set()
        for line in text.splitlines():
            for position, token in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9-]*[A-Za-z0-9]", line)):
                key = token.lower()
                mixed = any(ch.isdigit() for ch in token) or "-" in token
                if key in STOPWORDS or (len(key) < 3 and not mixed):
                    continue
                proper = position > 0 and token[0].isupper() and not token.isupper()
                if not (len(key) >= 7 or mixed or proper):
                    continue
                seen.add(key)
                if mixed or proper:
                    distinctive.add(key)
                # Prefer the capitalised spelling if we have seen one.
                if key not in surface or (token[0].isupper() and not surface[key][0].isupper()):
                    surface[key] = token
        slide_count.update(seen)

    ranked = sorted(slide_count, key=lambda k: (-slide_count[k], k not in distinctive, -len(k), k))
    chosen: list[str] = []
    used = 0
    for key in ranked[:max_terms]:
        term = surface[key]
        if used + len(term) + 2 > max_chars:
            break
        chosen.append(term)
        used += len(term) + 2
    return chosen


def build_hotwords(slide_texts: Iterable[str], max_terms: int = 50, max_chars: int = 600) -> str:
    """The hotword prompt for a lecture: its jargon, phrased as a sentence.

    The phrasing matters. Whisper imitates the *style* of its prompt as well as its
    vocabulary: given a bare comma-separated list, it stops ending sentences with full stops
    and strings the whole transcript together with commas. Wrapping the same terms as
    ``Glossary: a, b, c.`` keeps the vocabulary hint and leaves punctuation intact.
    """
    terms = jargon_terms(slide_texts, max_terms=max_terms, max_chars=max_chars)
    return f"Glossary: {', '.join(terms)}." if terms else ""
