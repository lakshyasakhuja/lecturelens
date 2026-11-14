"""Generate small synthetic lectures with exact ground truth.

``lecturelens demo <dir>`` renders slides, narrates them with the system text-to-speech
voice, and assembles real ``.mp4`` files. Because every slide, word and timestamp is known
by construction, the output comes with ground truth for each stage of the pipeline:

* slide transition times        -> checks slide segmentation
* the narration script           -> checks speech recognition (word error rate)
* the text drawn on each slide   -> checks OCR
* queries with answer intervals  -> checks retrieval and the evaluation scripts

This is a **smoke test and a quickstart**, not a benchmark. Synthetic slides are clean and
the synthetic voice is clear, so numbers measured here say that the pipeline is wired
correctly, not how well it works on real lectures.

Text-to-speech uses ``say`` on macOS and ``espeak-ng`` on Linux.
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lecturelens.catalog import CATALOG_COLUMNS
from lecturelens.utils import write_json

WIDTH, HEIGHT = 1280, 720
SAMPLE_RATE = 22_050
LEAD_IN_S = 0.6  # silence before the narration of each slide
TRAIL_S = 0.9  # silence after it

INK = (28, 34, 48)
ACCENT = (26, 82, 148)
MUTED = (110, 118, 132)
BLUE = (38, 110, 190)
ORANGE = (224, 122, 24)
GREEN = (46, 148, 92)


@dataclass(frozen=True)
class DemoSlide:
    title: str
    narration: str
    bullets: tuple[str, ...] = ()
    figure: str | None = None  # "loss_curves", "network", "learning_rate" or "folds"
    build: bool = False  # reveal the bullets one at a time while the narration plays


@dataclass(frozen=True)
class DemoQuery:
    query: str
    type: str  # keyword | paraphrase | spoken | visual
    slide: int  # index of the slide that answers it
    split: str


@dataclass(frozen=True)
class DemoLecture:
    lecture_id: str
    title: str
    slides: tuple[DemoSlide, ...]
    queries: tuple[DemoQuery, ...]


LECTURES: tuple[DemoLecture, ...] = (
    DemoLecture(
        lecture_id="demo01",
        title="Generalisation and Overfitting",
        slides=(
            DemoSlide(
                title="Generalisation and Overfitting",
                bullets=("Introduction to Machine Learning", "Demo lecture one"),
                narration=(
                    "Welcome back. Today we look at why a model that does well on its training "
                    "data can still fail on new data, and what we can do about it."
                ),
            ),
            DemoSlide(
                title="Training error and test error",
                bullets=(
                    "Training error: measured on data the model has seen",
                    "Test error: measured on unseen data",
                    "Generalisation gap = test error minus training error",
                ),
                narration=(
                    "The training error tells us how well the model fits the examples it has "
                    "already seen. What we actually care about is the test error, which is "
                    "measured on examples the model has never encountered. The difference "
                    "between the two is called the generalisation gap."
                ),
            ),
            DemoSlide(
                title="The validation set",
                bullets=(
                    "Split the data: training, validation, test",
                    "Tune hyperparameters on the validation set",
                    "Touch the test set only once",
                ),
                build=True,
                narration=(
                    "So how do we choose hyperparameters without cheating? We keep some data "
                    "aside that the model is never fitted on. This held out portion lets us "
                    "compare candidate models honestly. If we tuned on the test set instead, "
                    "our final number would be far too optimistic."
                ),
            ),
            DemoSlide(
                title="Overfitting",
                figure="loss_curves",
                narration=(
                    "Here is the classic picture. As training continues, the blue curve keeps "
                    "going down, but the orange curve turns around and starts to climb. That "
                    "turning point is where the model begins to memorise noise."
                ),
            ),
            DemoSlide(
                title="Early stopping",
                bullets=(
                    "Monitor validation loss during training",
                    "Stop when it has not improved for several epochs",
                    "Keep the best checkpoint",
                ),
                narration=(
                    "A simple remedy is to stop training early. Think of a student preparing "
                    "for an exam by memorising last year's answers. They score perfectly on "
                    "the old paper and badly on the new one. Stopping early is like telling "
                    "that student to stop cramming once their practice scores stop improving."
                ),
            ),
            DemoSlide(
                title="Regularisation",
                bullets=(
                    "L2 penalty on the size of the weights",
                    "Dropout: randomly switch off units",
                    "Data augmentation",
                ),
                narration=(
                    "Another family of techniques is regularisation. Weight decay discourages "
                    "large parameter values. Dropout randomly removes units during training, "
                    "so the network cannot rely on any single feature."
                ),
            ),
            DemoSlide(
                title="K-fold cross-validation",
                figure="folds",
                narration=(
                    "When data is scarce, we can rotate which part is held out. We cut the "
                    "data into five equal blocks, train on four of them, evaluate on the "
                    "remaining one, and repeat until every block has had its turn."
                ),
            ),
        ),
        queries=(
            DemoQuery("generalisation gap", "keyword", 1, "dev"),
            DemoQuery("dropout", "keyword", 5, "test"),
            DemoQuery("why do we hold out data we never train on", "paraphrase", 2, "dev"),
            DemoQuery("when should I halt training to avoid memorising", "paraphrase", 4, "test"),
            DemoQuery("the student who memorises last year's exam answers", "spoken", 4, "dev"),
            DemoQuery("weight decay", "spoken", 5, "test"),
            DemoQuery(
                "the plot where one curve goes down and the other goes back up", "visual", 3, "dev"
            ),
            DemoQuery(
                "coloured blocks showing which part of the data is held out", "visual", 6, "test"
            ),
        ),
    ),
    DemoLecture(
        lecture_id="demo02",
        title="Neural Networks and Gradient Descent",
        slides=(
            DemoSlide(
                title="Neural Networks and Gradient Descent",
                bullets=("Introduction to Machine Learning", "Demo lecture two"),
                narration=(
                    "In this lecture we build a small neural network and work out how to train it."
                ),
            ),
            DemoSlide(
                title="A three-layer network",
                figure="network",
                narration=(
                    "A feed forward network stacks layers of units. Each unit computes a "
                    "weighted sum of its inputs and passes the result through a non-linear "
                    "activation function."
                ),
            ),
            DemoSlide(
                title="Gradient descent",
                bullets=(
                    "Compute the gradient of the loss",
                    "Step in the opposite direction",
                    "The learning rate sets the step size",
                ),
                narration=(
                    "To train the network we use gradient descent. Imagine walking downhill "
                    "in thick fog. You cannot see the valley, but you can feel which way the "
                    "ground slopes under your feet, so you take a small step that way and "
                    "repeat."
                ),
            ),
            DemoSlide(
                title="Choosing the learning rate",
                figure="learning_rate",
                narration=(
                    "If the learning rate is too large, the updates overshoot the minimum and "
                    "the loss bounces around or even diverges. If it is too small, training "
                    "crawls and takes forever to converge."
                ),
            ),
            DemoSlide(
                title="Backpropagation",
                bullets=(
                    "Apply the chain rule layer by layer",
                    "Reuse intermediate results",
                    "Cost: about two forward passes",
                ),
                narration=(
                    "Gradients are computed efficiently with backpropagation. It is simply the "
                    "chain rule applied from the output back towards the input, storing "
                    "intermediate values along the way so nothing is computed twice."
                ),
            ),
            DemoSlide(
                title="Stochastic gradient descent",
                bullets=(
                    "Use a mini-batch instead of the full dataset",
                    "Noisy but cheap updates",
                    "Shuffle the data every epoch",
                ),
                narration=(
                    "In practice we rarely use the whole dataset for each update. Instead we "
                    "estimate the gradient from a small random sample. That is far cheaper, "
                    "and the randomness can even help the optimiser escape poor solutions."
                ),
            ),
        ),
        queries=(
            DemoQuery("chain rule", "keyword", 4, "dev"),
            DemoQuery("mini-batch", "keyword", 5, "test"),
            DemoQuery("what goes wrong if the step size is too big", "paraphrase", 3, "dev"),
            DemoQuery(
                "how are derivatives computed without repeating work", "paraphrase", 4, "test"
            ),
            DemoQuery("the analogy about walking downhill in fog", "spoken", 2, "dev"),
            DemoQuery("escape poor solutions", "spoken", 5, "test"),
            DemoQuery("diagram of circles connected by lines in three columns", "visual", 1, "dev"),
            DemoQuery("a bowl-shaped curve with zigzag steps", "visual", 3, "test"),
        ),
    ),
)


# --------------------------------------------------------------------------------------
# Slide rendering
# --------------------------------------------------------------------------------------


def _font(size: int):
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def _figure_loss_curves(draw, box) -> list[str]:
    x0, y0, x1, y1 = box
    draw.line([(x0, y0), (x0, y1), (x1, y1)], fill=INK, width=3)
    xs = np.linspace(0, 1, 120)
    train = 0.85 * np.exp(-3.2 * xs) + 0.06
    valid = 0.80 * np.exp(-4.0 * xs) + 0.16 + 0.55 * np.clip(xs - 0.32, 0, None) ** 1.6
    for curve, colour in ((train, BLUE), (valid, ORANGE)):
        points = [(x0 + x * (x1 - x0), y1 - c * (y1 - y0)) for x, c in zip(xs, curve, strict=True)]
        draw.line(points, fill=colour, width=6, joint="curve")
    draw.text((x1 - 330, y0 + 20), "training loss", font=_font(30), fill=BLUE)
    draw.text((x1 - 330, y0 + 62), "validation loss", font=_font(30), fill=ORANGE)
    draw.text(((x0 + x1) // 2 - 50, y1 + 14), "epochs", font=_font(28), fill=MUTED)
    return ["training loss", "validation loss", "epochs"]


def _figure_network(draw, box) -> list[str]:
    x0, y0, x1, y1 = box
    layers = [3, 5, 2]
    columns = np.linspace(x0 + 120, x1 - 120, len(layers))
    centres = []
    for x, count in zip(columns, layers, strict=True):
        ys = np.linspace(y0 + 40, y1 - 60, count + 2)[1:-1]
        centres.append([(float(x), float(y)) for y in ys])
    for left, right in itertools.pairwise(centres):
        for a in left:
            for b in right:
                draw.line([a, b], fill=MUTED, width=2)
    for layer, colour in zip(centres, (BLUE, GREEN, ORANGE), strict=True):
        for cx, cy in layer:
            draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=colour, outline=INK, width=3)
    labels = ["input", "hidden", "output"]
    for x, label in zip(columns, labels, strict=True):
        draw.text((x - 45, y1 - 34), label, font=_font(30), fill=INK)
    return labels


def _figure_learning_rate(draw, box) -> list[str]:
    x0, y0, x1, y1 = box
    cx, width, depth = (x0 + x1) / 2, (x1 - x0) / 2 - 40, y1 - y0 - 60

    def bowl(u: float) -> tuple[float, float]:  # u in [-1, 1]
        return cx + u * width, y1 - 30 - (1 - u * u) * 0 - (u * u) * depth

    draw.line([bowl(u) for u in np.linspace(-1, 1, 100)], fill=INK, width=5, joint="curve")
    steps = [-0.92, 0.78, -0.62, 0.46, -0.30, 0.16, -0.06]
    points = [bowl(u) for u in steps]
    draw.line(points, fill=ORANGE, width=4)
    for px, py in points:
        draw.ellipse([px - 10, py - 10, px + 10, py + 10], fill=ORANGE, outline=INK, width=2)
    draw.text((x0 + 10, y0), "loss", font=_font(30), fill=MUTED)
    draw.text((cx - 60, y1 - 4), "weight", font=_font(28), fill=MUTED)
    return ["loss", "weight"]


def _figure_folds(draw, box) -> list[str]:
    x0, y0, x1, y1 = box
    rows, cols, gap = 5, 5, 12
    cell_w = (x1 - x0 - 150 - gap * (cols - 1)) / cols
    cell_h = (y1 - y0 - gap * (rows - 1)) / rows
    labels = []
    for r in range(rows):
        top = y0 + r * (cell_h + gap)
        label = f"fold {r + 1}"
        labels.append(label)
        draw.text((x0, top + cell_h / 2 - 16), label, font=_font(28), fill=INK)
        for c in range(cols):
            left = x0 + 150 + c * (cell_w + gap)
            colour = ORANGE if c == r else BLUE
            draw.rectangle([left, top, left + cell_w, top + cell_h], fill=colour)
    return labels


_FIGURES = {
    "loss_curves": _figure_loss_curves,
    "network": _figure_network,
    "learning_rate": _figure_learning_rate,
    "folds": _figure_folds,
}


def render_slide(slide: DemoSlide, visible_bullets: int | None = None):
    """Draw one slide. Returns ``(image, text_lines)`` where the lines are what OCR should read."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), (250, 250, 247))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 12], fill=ACCENT)
    draw.text((70, 56), slide.title, font=_font(58), fill=ACCENT)
    draw.line([(70, 140), (WIDTH - 70, 140)], fill=(205, 210, 218), width=3)
    lines = [slide.title]

    bullets = slide.bullets if visible_bullets is None else slide.bullets[:visible_bullets]
    y = 190
    for bullet in bullets:
        draw.ellipse([82, y + 18, 98, y + 34], fill=ACCENT)
        draw.text((122, y), bullet, font=_font(40), fill=INK)
        lines.append(bullet)
        y += 86
    if slide.figure:
        lines += _FIGURES[slide.figure](draw, (150, 190, WIDTH - 150, HEIGHT - 70))
    return image, lines


# --------------------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------------------


def _tts_command(text: str, out_wav: Path) -> list[str]:
    if shutil.which("say"):
        return [
            "say", "-o", str(out_wav), "--file-format=WAVE",
            f"--data-format=LEI16@{SAMPLE_RATE}", "-r", "172", text,
        ]  # fmt: skip
    for binary in ("espeak-ng", "espeak"):
        if shutil.which(binary):
            return [binary, "-w", str(out_wav), "-s", "150", text]
    raise RuntimeError(
        "no text-to-speech engine found: the demo needs `say` (macOS) or `espeak-ng` (Linux)"
    )


def synthesize_speech(text: str, out_wav: Path) -> np.ndarray:
    """Speak ``text`` and return mono ``int16`` samples at ``SAMPLE_RATE``."""
    subprocess.run(_tts_command(text, out_wav), check=True, capture_output=True)
    with wave.open(str(out_wav), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != SAMPLE_RATE:  # espeak has its own rate; resample by linear interpolation
        target = round(len(samples) * SAMPLE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target), np.arange(len(samples)), samples
        ).astype(np.int16)
    return samples


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def build_lecture(lecture: DemoLecture, out_dir: Path) -> dict:
    """Render one demo lecture to ``videos/<id>.mp4`` and return its ground truth."""
    work = out_dir / "build" / lecture.lecture_id
    work.mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    silence = lambda seconds: np.zeros(round(seconds * SAMPLE_RATE), dtype=np.int16)  # noqa: E731
    audio: list[np.ndarray] = []
    frames: list[tuple[Path, float]] = []  # (image, seconds on screen)
    truth_slides = []
    clock = 0.0

    for index, slide in enumerate(lecture.slides):
        speech = synthesize_speech(slide.narration, work / f"speech{index:02d}.wav")
        # Round each slide to the video frame grid so cut times are exact.
        duration = round((LEAD_IN_S + len(speech) / SAMPLE_RATE + TRAIL_S) * 25) / 25
        clip = np.concatenate([silence(LEAD_IN_S), speech])
        n_samples = round(duration * SAMPLE_RATE)
        audio.append(
            np.concatenate([clip, np.zeros(max(0, n_samples - len(clip)), np.int16)])[:n_samples]
        )

        steps = range(1, len(slide.bullets) + 1) if slide.build else [None]
        step_duration = round(duration / len(steps) * 25) / 25
        builds = []
        for step_no, visible in enumerate(steps):
            image, lines = render_slide(slide, visible)
            path = work / f"slide{index:02d}_{step_no}.png"
            image.save(path)
            is_last = step_no == len(steps) - 1
            shown = duration - step_duration * (len(steps) - 1) if is_last else step_duration
            frames.append((path, shown))
            builds.append(round(clock + step_no * step_duration, 3))

        truth_slides.append(
            {
                "index": index,
                "title": slide.title,
                "t_start": round(clock, 3),
                "t_end": round(clock + duration, 3),
                "speech_start": round(clock + LEAD_IN_S, 3),
                "speech_end": round(clock + LEAD_IN_S + len(speech) / SAMPLE_RATE, 3),
                "text": lines,
                "narration": slide.narration,
                "build_steps": builds[1:],  # times at which a bullet is revealed
            }
        )
        clock += duration

    audio_path = work / "audio.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.concatenate(audio).tobytes())

    concat = work / "frames.txt"
    with concat.open("w", encoding="utf-8") as handle:
        for path, seconds in frames:
            handle.write(f"file '{path.resolve()}'\nduration {seconds:.3f}\n")
        handle.write(
            f"file '{frames[-1][0].resolve()}'\n"
        )  # the concat demuxer needs a final frame

    video = out_dir / "videos" / f"{lecture.lecture_id}.mp4"
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-i", str(audio_path),
        "-vf", "fps=25,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "50",
        "-c:a", "aac", "-b:a", "96k",
        "-t", f"{clock:.3f}",
        str(video),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return {
        "lecture_id": lecture.lecture_id,
        "title": lecture.title,
        "duration_s": round(clock, 3),
        "video": str(video),
        "slides": truth_slides,
    }


def make_demo(out_dir: str | Path, lectures: tuple[DemoLecture, ...] = LECTURES) -> Path:
    """Build every demo lecture and write the files a real project would have by hand.

    Layout of ``out_dir``::

        videos/<id>.mp4
        lectures.csv                  catalogue with links into the local files
        truth/<id>.json               everything known about each lecture
        eval/queries.jsonl            queries with ground-truth intervals
        eval/transitions/<id>.json    slide transition times
        eval/asr_ref/<id>.txt         reference transcript, plus excerpts.jsonl
    """
    out_dir = Path(out_dir)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to build the demo lectures")
    truths = [build_lecture(lecture, out_dir) for lecture in lectures]

    with (out_dir / "lectures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for truth in truths:
            writer.writerow(
                {
                    "lecture_id": truth["lecture_id"],
                    "title": truth["title"],
                    "course": "LectureLens demo",
                    "source_url": "",
                    "licence": "synthetic (generated by lecturelens demo)",
                    "duration_s": truth["duration_s"],
                    "video_link_template": Path(truth["video"]).resolve().as_uri() + "#t={seconds}",
                }
            )

    queries, excerpts = [], []
    for lecture, truth in zip(lectures, truths, strict=True):
        slides = truth["slides"]
        write_json(out_dir / "truth" / f"{lecture.lecture_id}.json", truth)
        write_json(
            out_dir / "eval" / "transitions" / f"{lecture.lecture_id}.json",
            {
                "lecture_id": lecture.lecture_id,
                "transitions": [s["t_start"] for s in slides[1:]],
                # Bullet reveals are listed separately: they are not slide transitions.
                "builds": [t for s in slides for t in s["build_steps"]],
            },
        )
        ref_dir = out_dir / "eval" / "asr_ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / f"{lecture.lecture_id}.txt").write_text(
            "\n".join(s["narration"] for s in slides) + "\n", encoding="utf-8"
        )
        excerpts.append(
            {
                "lecture_id": lecture.lecture_id,
                "t0": 0.0,
                "t1": truth["duration_s"],
                "reference": f"{lecture.lecture_id}.txt",
            }
        )
        for number, query in enumerate(lecture.queries, start=1):
            slide = slides[query.slide]
            queries.append(
                {
                    "qid": f"{lecture.lecture_id}-q{number:02d}",
                    "query": query.query,
                    "type": query.type,
                    "lecture_id": lecture.lecture_id,
                    "gt": [[slide["t_start"], slide["t_end"]]],
                    "split": query.split,
                    "author": "demo",
                }
            )

    eval_dir = out_dir / "eval"
    with (eval_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query) + "\n")
    with (eval_dir / "asr_ref" / "excerpts.jsonl").open("w", encoding="utf-8") as handle:
        for excerpt in excerpts:
            handle.write(json.dumps(excerpt) + "\n")

    shutil.rmtree(out_dir / "build", ignore_errors=True)
    return out_dir
