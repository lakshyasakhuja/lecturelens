"""How well does a picture of a slide find the moment it was shown?

The queries are made from the lectures themselves, so no annotation is needed. For each
lecture, frames are taken at random times inside slide segments (never the keyframe that
was indexed), and each frame is turned into three kinds of query:

    frame   the frame as it is, re-encoded: a screenshot of the recording
    photo   a simulated phone photo of a screen: perspective, margin, colour, blur, JPEG
    crop    a random crop covering 55 to 75 per cent of the frame

A query is answered correctly when a result's keyframe is the keyframe of the segment the
frame came from. Three ways of searching are compared:

    image   SigLIP's image tower: the query picture against every indexed keyframe
    ocr     the text read off the picture, by BM25 over slide text
    both    the two fused by reciprocal rank fusion, which is what the app does

Run from the repository root:

    .venv/bin/python eval/eval_screenshot.py --index index --videos videos --per-lecture 25
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from slideseek.index import Index
from slideseek.ingest import extract_still
from slideseek.search import Searcher

VARIANTS = ("frame", "photo", "crop")
METHODS = ("image", "ocr", "both")


def sample_times(index: Index, lecture: str, count: int, rng: random.Random) -> list[tuple]:
    """``(time, keyframe)`` pairs inside slide segments, away from edges and the keyframe."""
    moments = index.moments
    rows = moments[(moments.lecture_id == lecture) & (moments.kind == "slide")]
    segments = {}
    for _, row in rows.iterrows():
        seg = segments.setdefault(
            row.segment, [row.t_start, row.t_end, row.keyframe_t, row.image_idx]
        )
        seg[0], seg[1] = min(seg[0], row.t_start), max(seg[1], row.t_end)
    usable = [s for s in segments.values() if s[1] - s[0] >= 8]
    picks = []
    for start, end, keyframe_t, image_idx in rng.sample(usable, min(count, len(usable))):
        for _ in range(20):
            t = rng.uniform(start + 2, end - 2)
            if abs(t - keyframe_t) >= 2:
                break
        picks.append((round(t, 2), int(image_idx)))
    return picks


def make_photo(image: Image.Image, rng: random.Random) -> Image.Image:
    """A phone photo of a screen showing this frame."""
    w, h = image.size
    margin = int(0.12 * w)
    canvas = Image.new(
        "RGB", (w + 2 * margin, h + 2 * margin), tuple(rng.randint(20, 90) for _ in range(3))
    )
    canvas.paste(image, (margin, margin))
    W, H = canvas.size
    jitter = 0.06
    corners = [(0, 0), (W, 0), (W, H), (0, H)]
    moved = [
        (x + rng.uniform(-jitter, jitter) * W, y + rng.uniform(-jitter, jitter) * H)
        for x, y in corners
    ]
    coeffs = _perspective_coeffs(moved, corners)
    photo = canvas.transform((W, H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    photo = photo.rotate(rng.uniform(-4, 4), resample=Image.Resampling.BICUBIC, expand=False)
    photo = ImageEnhance.Brightness(photo).enhance(rng.uniform(0.75, 1.2))
    photo = ImageEnhance.Contrast(photo).enhance(rng.uniform(0.7, 1.1))
    photo = ImageEnhance.Color(photo).enhance(rng.uniform(0.7, 1.3))
    return photo.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))


def _perspective_coeffs(source, target):
    matrix = []
    for (x, y), (u, v) in zip(target, source, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(source, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()


def make_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    w, h = image.size
    share = rng.uniform(0.55, 0.75)
    cw, ch = int(w * share**0.5), int(h * share**0.5)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return image.crop((x, y, x + cw, y + ch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", default="index")
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--work", default="work/screenshot-eval")
    parser.add_argument("--per-lecture", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/screenshot/metrics.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    index = Index.load(args.index)
    searcher = Searcher(index)
    work = Path(args.work)

    queries = []
    for lecture in sorted(index.lectures):
        video = Path(args.videos) / f"{lecture}.mp4"
        for t, image_idx in sample_times(index, lecture, args.per_lecture, rng):
            frame_path = work / lecture / f"{round(t * 1000):09d}.jpg"
            if not frame_path.is_file():
                extract_still(video, t, frame_path)
            frame = Image.open(frame_path).convert("RGB")
            for variant in VARIANTS:
                picture = {
                    "frame": frame,
                    "photo": make_photo(frame, rng),
                    "crop": make_crop(frame, rng),
                }[variant]
                path = work / lecture / f"{round(t * 1000):09d}-{variant}.jpg"
                picture.save(path, quality=70 if variant == "photo" else 85)
                queries.append(
                    {
                        "lecture": lecture,
                        "t": t,
                        "keyframe": image_idx,
                        "variant": variant,
                        "path": str(path),
                    }
                )
    print(f"{len(queries)} queries from {len(index.lectures)} lectures")

    hits = {(m, v): [] for m in METHODS for v in VARIANTS}
    timings = {"image": [], "ocr": []}
    for n, query in enumerate(queries, start=1):
        picture = Image.open(query["path"]).convert("RGB")
        start = time.perf_counter()
        encoding = searcher.encode_picture(picture, read_text=False)
        timings["image"].append(time.perf_counter() - start)
        start = time.perf_counter()
        terms = searcher.read_picture_text(picture)
        timings["ocr"].append(time.perf_counter() - start)
        encoding = encoding.with_terms(terms)
        for method, rankers in (
            ("image", ("image",)),
            ("ocr", ("bm25_slide",)),
            ("both", ("image", "bm25_slide")),
        ):
            if method == "ocr" and not encoding.terms:
                ranks = []
            else:
                order, scores, per_ranker = searcher.fuse(encoding, rankers)
                results = searcher.results(order, scores, encoding, per_ranker, top_k=5)
                ranks = [int(index.moments.image_idx.iat[r.row]) for r in results]
            hit = ranks.index(query["keyframe"]) + 1 if query["keyframe"] in ranks else None
            hits[(method, query["variant"])].append(hit)
        if n % 25 == 0:
            print(f"  {n}/{len(queries)}", flush=True)

    table = {}
    for (method, variant), found in hits.items():
        table.setdefault(method, {})[variant] = {
            "hit@1": round(sum(h == 1 for h in found) / len(found), 3),
            "hit@5": round(sum(h is not None for h in found) / len(found), 3),
            "n": len(found),
        }
    report = {
        "queries": len(queries),
        "lectures": sorted(index.lectures),
        "per_lecture": args.per_lecture,
        "seed": args.seed,
        "methods": table,
        "median_seconds": {k: round(float(np.median(v)), 3) for k, v in timings.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
