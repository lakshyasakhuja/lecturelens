"""Small helpers shared across modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Render seconds as ``M:SS`` or ``H:MM:SS``, the way video players show time."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_timestamp(text: str | float | int) -> float:
    """Parse ``83``, ``83.5``, ``1:23`` or ``1:02:03`` into seconds.

    Annotation files are written by hand while watching a video, so they accept the same
    notation a player displays.
    """
    if isinstance(text, (int, float)):
        return float(text)
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"not a timestamp: {text!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON atomically, so an interrupted run never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
