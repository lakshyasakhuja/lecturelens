"""The lecture catalogue: which lectures exist, where they came from, and how to link into them.

Lecture metadata lives in ``config/lectures.csv`` so that provenance and licence travel with
every lecture. Videos are matched to rows by file name: ``<videos_dir>/<lecture_id>.mp4``.
A video with no row still works (useful for private, local-only lectures); it gets default
metadata and a ``file://`` link.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"})
CATALOG_COLUMNS = (
    "lecture_id",
    "title",
    "course",
    "instructors",
    "source_url",
    "licence",
    "duration_s",
    "video_link_template",
)

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class CatalogError(ValueError):
    """Raised when ``lectures.csv`` is malformed."""


@dataclass(frozen=True)
class Lecture:
    """One lecture recording and its provenance."""

    lecture_id: str
    title: str
    course: str = ""
    instructors: str = ""  # who to credit; Creative Commons licences require naming the authors
    source_url: str = ""
    licence: str = ""
    duration_s: float = 0.0
    # A URL containing ``{seconds}``, e.g. ``https://www.youtube.com/watch?v=ID&t={seconds}s``.
    video_link_template: str = ""

    def link_at(self, seconds: float) -> str:
        """URL that opens this lecture at ``seconds``. Empty when no template is known."""
        if not self.video_link_template:
            return ""
        return self.video_link_template.replace("{seconds}", str(max(0, int(seconds))))

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(name: str) -> str:
    """Turn a file stem into a valid lecture id (letters, digits, ``_`` and ``-``)."""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return slug or "lecture"


def load_catalog(path: str | Path) -> dict[str, Lecture]:
    """Read ``lectures.csv`` into ``{lecture_id: Lecture}``, validating as it goes."""
    path = Path(path)
    lectures: dict[str, Lecture] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("lecture_id", "title") if c not in (reader.fieldnames or [])]
        if missing:
            raise CatalogError(f"{path}: missing required column(s): {', '.join(missing)}")
        for line_no, row in enumerate(reader, start=2):
            lecture_id = (row.get("lecture_id") or "").strip()
            if not lecture_id:
                continue  # blank line
            if not _ID_PATTERN.match(lecture_id):
                raise CatalogError(
                    f"{path}:{line_no}: lecture_id {lecture_id!r} may only contain letters, "
                    "digits, '_' and '-'"
                )
            if lecture_id in lectures:
                raise CatalogError(f"{path}:{line_no}: duplicate lecture_id {lecture_id!r}")
            template = (row.get("video_link_template") or "").strip()
            if template and "{seconds}" not in template:
                raise CatalogError(
                    f"{path}:{line_no}: video_link_template must contain '{{seconds}}'"
                )
            try:
                duration = float(row.get("duration_s") or 0.0)
            except ValueError as err:
                raise CatalogError(f"{path}:{line_no}: duration_s is not a number") from err
            lectures[lecture_id] = Lecture(
                lecture_id=lecture_id,
                title=(row.get("title") or "").strip() or lecture_id,
                course=(row.get("course") or "").strip(),
                instructors=(row.get("instructors") or "").strip(),
                source_url=(row.get("source_url") or "").strip(),
                licence=(row.get("licence") or "").strip(),
                duration_s=duration,
                video_link_template=template,
            )
    return lectures


def discover_videos(videos_dir: str | Path) -> dict[str, Path]:
    """Map lecture ids to video files found directly inside ``videos_dir``."""
    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"video directory not found: {videos_dir}")
    found: dict[str, Path] = {}
    for path in sorted(videos_dir.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTENSIONS or path.name.startswith("."):
            continue
        lecture_id = slugify(path.stem)
        if lecture_id in found:
            raise CatalogError(
                f"{path.name} and {found[lecture_id].name} both map to lecture id {lecture_id!r}"
            )
        found[lecture_id] = path
    return found


def resolve_lectures(
    videos: dict[str, Path], catalog: dict[str, Lecture] | None = None
) -> dict[str, Lecture]:
    """Attach catalogue metadata to discovered videos.

    Videos without a catalogue row get a title from their file name and a ``file://`` link
    with a media-fragment timestamp, which browsers honour for local files.
    """
    catalog = catalog or {}
    resolved: dict[str, Lecture] = {}
    for lecture_id, path in videos.items():
        if lecture_id in catalog:
            resolved[lecture_id] = catalog[lecture_id]
        else:
            resolved[lecture_id] = Lecture(
                lecture_id=lecture_id,
                title=path.stem.replace("_", " "),
                video_link_template=path.resolve().as_uri() + "#t={seconds}",
            )
    return resolved
