"""Does a lecture added in the browser search as well as the same lecture indexed in Python?

The browser's index of a lecture (made by the page: its frames, stills and transcript, then
the server's own stages, see slideseek/library.py) is compared with the Python pipeline's
index of the same recording, on the same queries:

    pictures   frames of the recording at random times inside slides, as a screenshot,
               a phone photo and a crop (eval_screenshot.py makes them the same way). Each
               index is right when its top results include its own keyframe for that time.
               Both search the one lecture alone, so neither has more slides to confuse.
    questions  typed questions. There is no ground truth for these yet (eval/queries.jsonl
               is still to be written), so what is measured is agreement: whether the two
               indexes put the same moment first, and how much of their top five they share.

Run from the repository root, with a lecture added through the page:

    .venv/bin/python eval/compare_browser_index.py \\
        --python index --browser libraries/<owner>/<lecture id>/index \\
        --lecture mit6-7960-lec04 --videos videos --out results/browser
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_screenshot import VARIANTS, make_crop, make_photo, sample_times

from slideseek.index import Index
from slideseek.ingest import extract_still
from slideseek.search import Searcher

# Questions about MIT 6.7960 lecture 4, "Architectures for Grids": the three shipped
# examples, then topics taken from its slide titles, written before searching either index.
QUESTIONS = [
    "why do we share weights across positions",
    "the picture of birds with arrows classifying each patch",
    "grid of feature visualisations for alexnet and vgg",
    "why build better architectures",
    "convolutional layers",
    "what is a convolution filter",
    "pooling and downsampling",
    "image pyramids",
    "the architecture zoo",
    "neural fields and positional encodings",
    "multilayer perceptron inductive biases",
    "receptive field grows with depth",
    "translation equivariance",
    "residual connections",
    "U-Net for segmentation",
    "how many parameters does a fully connected layer need for an image",
    "stride and padding",
    "feature maps in deeper layers",
    "fourier features for coordinates",
    "the lecture summary",
]
AGREE_S = 20.0  # two results land on the same moment if they start this close


def keyframe_at(index: Index, lecture: str, t: float) -> int | None:
    """The keyframe of the slide moment showing at ``t``, if a slide was showing."""
    moments = index.moments
    rows = moments[
        (moments.lecture_id == lecture)
        & (moments.kind == "slide")
        & (moments.t_start <= t)
        & (t < moments.t_end)
    ]
    return int(rows.image_idx.iat[0]) if len(rows) else None


def picture_hits(
    searcher: Searcher, picture: Image.Image, keyframe: int, lecture: str
) -> int | None:
    """The rank at which the app's picture search (image + slide text) finds ``keyframe``,
    searching ``lecture`` alone: the browser's index holds nothing else, so the Python one
    must not be searched with the other lectures' slides as well."""
    encoding = searcher.encode_picture(picture, read_text=False)
    encoding = encoding.with_terms(searcher.read_picture_text(picture))
    rankers = ("image", "bm25_slide") if encoding.terms else ("image",)
    mask = searcher.lecture_mask([lecture])
    order, scores, per_ranker = searcher.fuse(encoding, rankers, mask=mask)
    results = searcher.results(order, scores, encoding, per_ranker, top_k=5)
    ranks = [int(searcher.index.moments.image_idx.iat[r.row]) for r in results]
    return ranks.index(keyframe) + 1 if keyframe in ranks else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default="index", help="The Python pipeline's index.")
    parser.add_argument("--browser", required=True, help="The lecture's index from the library.")
    parser.add_argument("--lecture", default="mit6-7960-lec04")
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--pictures", type=int, default=40, help="Frames to make pictures of.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--work", default="work/browser-compare")
    parser.add_argument("--out", default="results/browser")
    args = parser.parse_args()

    python = Index.load(args.python)
    browser = Index.load(args.browser)
    (own,) = browser.lectures
    a = Searcher(python)
    b = Searcher(browser, text_embedder=a.text_embedder, image_embedder=a.image_embedder, ocr=a.ocr)
    only = [args.lecture]

    # Questions: do the two put the same moment first, and share their top five?
    rows, first, shared = [], 0, []
    for question in QUESTIONS:
        pa = [r.t_start for r in a.search(question, top_k=5, lectures=only)]
        pb = [r.t_start for r in b.search(question, top_k=5)]
        same = bool(pa and pb and abs(pa[0] - pb[0]) <= AGREE_S)
        first += same
        share = sum(any(abs(x - y) <= AGREE_S for y in pa) for x in pb) / max(1, len(pb))
        shared.append(share)
        rows.append({"query": question, "python": pa, "browser": pb, "same_first": same})

    # Pictures: the same frames, each index judged against its own keyframe for that time.
    rng = random.Random(args.seed)
    work = Path(args.work)
    video = Path(args.videos) / f"{args.lecture}.mp4"
    hits = {side: {v: [] for v in VARIANTS} for side in ("python", "browser")}
    for t, keyframe_python in sample_times(python, args.lecture, args.pictures, rng):
        keyframe_browser = keyframe_at(browser, own, t)
        if keyframe_browser is None:
            continue
        frame_path = work / f"{round(t * 1000):09d}.jpg"
        if not frame_path.is_file():
            extract_still(video, t, frame_path)
        frame = Image.open(frame_path).convert("RGB")
        pictures = {"frame": frame, "photo": make_photo(frame, rng), "crop": make_crop(frame, rng)}
        for variant, picture in pictures.items():
            hits["python"][variant].append(picture_hits(a, picture, keyframe_python, args.lecture))
            hits["browser"][variant].append(picture_hits(b, picture, keyframe_browser, own))

    def table(found: dict[str, list]) -> dict:
        return {
            variant: {
                "hit@1": round(sum(h == 1 for h in ranks) / len(ranks), 3),
                "hit@5": round(sum(h is not None for h in ranks) / len(ranks), 3),
                "n": len(ranks),
            }
            for variant, ranks in found.items()
            if ranks
        }

    lecture_rows = python.moments.lecture_id == args.lecture
    summary = {
        "lecture": args.lecture,
        "python": {
            "moments": int(lecture_rows.sum()),
            "keyframes": int(python.moments[lecture_rows].image_idx.nunique()),
            "asr": python.manifest.get("models", {}).get("asr"),
            "pictures": table(hits["python"]),
        },
        "browser": {
            "moments": len(browser),
            "keyframes": len(browser.image),
            "asr": browser.manifest.get("models", {}).get("asr"),
            "pictures": table(hits["browser"]),
        },
        "questions": {
            "n": len(QUESTIONS),
            "same_first": first,
            "top5_shared": round(float(np.mean(shared)), 3),
            "agree_within_s": AGREE_S,
            "rows": rows,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    brief = {key: value for key, value in summary.items() if key != "questions"}
    brief["questions"] = {k: v for k, v in summary["questions"].items() if k != "rows"}
    print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
