# LectureLens

[![CI](https://github.com/lakshyasakhuja/lecturelens/actions/workflows/ci.yml/badge.svg)](https://github.com/lakshyasakhuja/lecturelens/actions/workflows/ci.yml)

Semantic search for lecture recordings. Ask a question in plain English, such as *"where did she explain why we need a validation set?"*, and jump straight to that moment in the video.

Each result is ranked using three signals: what was **said**, what text was **on the slide**, and what the slide **looked like**.

```console
$ lecturelens search "the student who memorises last year's exam answers"

1. Generalisation and Overfitting  at  1:00   said
   … is to stop training early. Think of a student preparing for an exam by
   memorising last year's answers. They score perfectly on the old paper and
   badly on the new one. Stopping early is like telling that student to stop …
   bm25 #1  dense #1  image #6
   file:///…/demo01.mp4#t=60
```

## Status

| Part | State |
|---|---|
| Indexing pipeline: slide segmentation, OCR, speech recognition, moments, embeddings | Built. Runs end to end with the real models. |
| Search: BM25 + dense + image, rank fusion, deduplication, explanations, optional reranker | Built. |
| CLI, resumable and incremental indexing, publishing to Hugging Face | Built. |
| Evaluation harness: retrieval ablations, slide transitions, word error rate | Built, and exercised end to end on synthetic lectures. |
| **Benchmark results on real lectures** | **Not yet.** Needs a course to be indexed and queries to be annotated. No numbers are claimed until then. |
| Web demo | Not yet. |

More than 200 unit tests run in CI on every push, with no model downloads. A further integration test (`pytest -m models`) runs the whole system with the real models on synthetic lectures.

---

## The problem

Lectures run one to two hours. Finding the three minutes you need before an exam means scrubbing through a recording or rewatching it at 2x.

Keyword search over a transcript helps only when you remember the exact words the lecturer used. It fails when you describe the idea in your own words, when the term only appeared on a slide, or when what you remember is a picture ("the plot where validation loss goes back up").

## Try it in five minutes

No lecture videos needed: `lecturelens demo` generates two short synthetic lectures (rendered slides, a text-to-speech voice) with ground truth for every stage.

Requirements: Python 3.10 to 3.12, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, and for the demo a text-to-speech command (`say` on macOS, `espeak-ng` on Linux). The first run downloads about 5 GB of model weights.

```bash
git clone https://github.com/lakshyasakhuja/lecturelens.git
```

```bash
cd lecturelens && uv sync --extra index --extra eval
```

```bash
uv run lecturelens demo demo
```

```bash
uv run lecturelens index demo/videos --catalog demo/lectures.csv --out demo/index --work demo/work
```

```bash
uv run lecturelens search "why do we hold out data we never train on" --index demo/index
```

On an Apple M4 laptop with no GPU, once the models are downloaded, indexing the three minutes of demo video takes about 90 seconds, and a query takes about 100 ms including both query encoders.

## Search your own lectures

Put one video per lecture in a folder and index it. Everything stays on your machine: nothing is uploaded, and results link back to the local files.

```bash
uv run lecturelens index ~/lectures/comp5318 --out ~/lectures/comp5318-index --work ~/lectures/comp5318-work
```

```bash
uv run lecturelens search "what is the kernel trick" --index ~/lectures/comp5318-index
```

University recordings and slides belong to the institution and its lecturers. Index them for personal study only; do not publish the index.

| Command | What it does |
|---|---|
| `lecturelens index <videos>` | Build or update an index. Cached per stage, so it resumes after an interruption and only processes new lectures. |
| `lecturelens search "<query>"` | Top moments with timestamps, snippets, badges and links. `--lecture` to filter, `--rerank` for the cross-encoder, `--json` for scripts. |
| `lecturelens info` | What an index contains and which models built it. |
| `lecturelens slides <video> --sheet out.jpg` | Run slide detection alone and write a contact sheet of keyframes, to check masks and thresholds before indexing. |
| `lecturelens push / pull <repo>` | Publish or fetch an index as a Hugging Face dataset, with an attribution card. |
| `lecturelens demo <dir>` | Generate the synthetic lectures. |

---

## How it works

The core design decision is to **index once offline on a GPU and search online on a CPU**. Heavy models run a single time per lecture; a query needs two small text encoders and a few matrix products.

### Offline indexing

```
 lecture.mp4
   |
   +--> ffmpeg: 160x90 grey frames at 1 fps, piped into memory
   |      --> slide-change detector --> slide segments [t0, t1] + keyframe time
   |             keyframe at native resolution --> PP-OCRv5  --> slide text
   |             keyframe at native resolution --> SigLIP 2  --> image embedding
   |
   +--> ffmpeg: audio, 16 kHz mono
          --> faster-whisper (large-v3-turbo), prompted with jargon from the slides
          --> words with timestamps

 align words to slide segments --> "moments"
 moment text --> Qwen3-Embedding-0.6B --> three text vectors (both, speech only, slide only)
```

A **moment** is the unit of retrieval: one slide, with the words spoken while it was on screen. A slide that stays up for more than 90 seconds is cut into 45-second windows, spaced evenly so neighbours overlap by at least 15 seconds and the windows sit flush with the slide at both ends.

### Online search

```
 query
   +--> BM25 over moment text              --> ranking A
   +--> Qwen3-Embedding (query) . text     --> ranking B
   +--> SigLIP 2 text tower . image vecs   --> ranking C
            |
     Reciprocal Rank Fusion (k = 60)
            |
     (optional) rerank top 20 with Qwen3-Reranker-0.6B
            |
     dedupe into distinct landing points --> top 5, each with an explanation
```

### Design notes

These are the decisions that are not obvious from the diagrams. Each one is covered by tests that isolate it.

**Slide segmentation is hysteresis against a reference frame.** A segment becomes stable after 3 quiet seconds. It ends when a frame differs from the *most recent quiet frame* by more than a threshold. Comparing against a reference rather than the previous frame means a cross-fade spread over several frames still accumulates and is caught, while a cursor, tablet ink or a bullet appearing is absorbed as the reference moves forward.

**Added content is not replaced content.** Mean absolute pixel difference cannot tell three bullet points appearing from one sparse slide replacing another: both move about the same number of pixels. The sign of the change can. A build only adds ink, so every changed pixel moves the same way; a new slide removes old text and draws new text, so pixels move both ways. The detector splits a change into a balanced part and a one-sided part and counts the one-sided part at a quarter weight. On synthetic slides no single threshold on plain difference gets both cases right; this does ([test](tests/test_slides.py)).

**The keyframe is the last quiet frame of a segment**, because an animated slide is only complete at the end. OCR and the image encoder read it at the video's native resolution, not from the 160x90 detection frame.

**Whisper is prompted with the slides' own vocabulary.** Distinctive terms from the OCR (names, long words, tokens like `L2` and `k-fold`) are passed as hotwords. They are phrased as a sentence, `Glossary: a, b, c.`, because Whisper imitates the style of its prompt: given a bare comma-separated list it stops writing full stops.

**OCR fragments are rejoined.** The detector sometimes splits a line of text into overlapping boxes, so the boundary character is read twice (`Noisy` + `y but cheap updates`). Fragments on one row are merged and the doubled character dropped.

**Rank fusion, not score fusion.** BM25 scores, cosine similarities and SigLIP similarities sit on different scales. Reciprocal Rank Fusion uses only ranks, so it needs no calibration and has one hyperparameter:

```
RRF(d) = sum over rankers of  weight / (60 + rank(d))
```

**Ties share a rank.** Windows of one long slide share a keyframe and so have identical image scores. Ranked 1, 2, 3, … they would fill the image ranking and push every other slide out. They all take their keyframe's rank instead, and the text rankers choose between them.

**Results are landing points.** Two hits in one lecture that start within 20 seconds of each other would drop the student in the same place, so only the better one is kept.

**A "visual" badge needs evidence.** Some keyframe is always ranked first by the image ranker, even for a query that describes nothing visual. SigLIP is trained with a sigmoid loss, so its similarity converts to a match probability; the badge is shown only when that probability clears a threshold.

**No vector database.** A library this size is a few thousand moments. On a synthetic index of 2,000 moments, the whole of search apart from the two query encoders (three rankers, fusion, deduplication, explanations) takes 1.3 ms at the median on a laptop CPU, and 8 ms at 20,000 moments, so a vector store would add setup and nothing else. That changes at hundreds of thousands of moments, or once the index takes concurrent writes. BM25 structures are not stored either: they are rebuilt from the moments table when the index loads, in about half a second for 2,000 moments, so the stored index can never disagree with the tokeniser in the code.

**Indexing is resumable and incremental.** Each stage writes to a per-lecture work directory and records a fingerprint of its inputs. An interrupted run resumes; a changed setting reruns only what is downstream of it (a new embedding model does not redo speech recognition); a new lecture is processed alone.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Environment | Python 3.12, `uv` | Fast, reproducible installs |
| Video and audio | `ffmpeg` | Standard and fast |
| Speech recognition | faster-whisper, `large-v3-turbo` | Decoder pruned from 32 layers to 4: much faster than large-v3 at close accuracy |
| Slide segmentation | Custom detector (see above) | Tuned for slides; PySceneDetect is the baseline it is measured against |
| OCR | PP-OCRv5 via RapidOCR (ONNX Runtime) | Same weights as PaddleOCR, in a runtime that installs next to PyTorch without a second CUDA stack |
| Text embeddings | Qwen3-Embedding-0.6B | Open (Apache 2.0), instruction-aware, larger sizes available |
| Image-text embeddings | SigLIP 2 (`google/siglip2-base-patch16-224`) | Good text-to-image retrieval, with calibrated match probabilities |
| Sparse retrieval | `bm25s`, Snowball stemming | Fast BM25 in numpy |
| Reranker (optional) | Qwen3-Reranker-0.6B | Same family as the embedder |
| Storage | Parquet and numpy arrays | See "No vector database" above |

The package is split along the offline/online line. `uv sync` alone installs the core (index loading, fusion, metrics) with no deep-learning framework; `--extra models` adds the query encoders, `--extra index` adds speech recognition and OCR, `--extra eval` adds the evaluation tools.

---

## Evaluation

The goal is not only a working demo but a measurement of **when each signal helps**. The harness is built; the benchmark needs real lectures and hand-written ground truth. The annotation formats and protocol are in [eval/README.md](eval/README.md).

**Query set.** 60 queries over 5 lectures, each with ground-truth time intervals, written before retrieval is tuned and phrased the way a student would ask. 20 are for tuning (dev), 40 for reporting (test).

| Query type | Example | Signal expected to win |
|---|---|---|
| Keyword | "Bellman equation" | BM25 |
| Paraphrase | "why do we hold out data we never train on" | Dense text |
| Spoken-only | a worked example the lecturer said aloud but never put on a slide | Transcript, not OCR |
| Visual | "the diagram with the three-layer network" | SigLIP 2 |

**Relevance.** A returned moment is relevant if its span overlaps a ground-truth interval within ±15 seconds: close enough that a student who lands there sees the explanation. Overlap is lenient towards long moments (one that merely ends where the answer begins still overlaps), so a stricter **landing** criterion is reported next to it: the result must start inside the answer or at most N seconds before it.

**Metrics.** Recall@1, Recall@5 and MRR@10, overall and per query type, each with a bootstrap confidence interval, because 40 queries is a small sample. The hybrid is compared with keyword transcript search by a paired bootstrap over the same queries.

**Ablations.** Every system runs through the same fusion and deduplication as the served system.

| # | System | What it isolates |
|---|---|---|
| 1 | BM25, transcript only | Baseline, close to ordinary keyword transcript search |
| 2 | BM25, transcript + slide OCR | Value of reading the slides |
| 3 | Dense, transcript only | Value of semantic matching |
| 4 | Dense, transcript + slide text | Both together |
| 5 | Image only (SigLIP 2) | Value of the visual signal on its own |
| 6 | Hybrid RRF (2 + 4 + 5) | The full system |
| 7 | Hybrid + reranker | Whether reranking earns its cost |

The working hypothesis is that BM25 wins keyword queries, dense wins paraphrases, the image signal helps visual queries, and the hybrid is best overall. That is a hypothesis to test. Whatever happens gets reported, including any query type where the hybrid loses.

**Dev and test discipline.** Fusion weights, `k` and the image ranker's abstention floor are grid-searched on dev only. Every evaluation of the test split is appended to `results/test_runs.log` with the commit it ran at.

**Component evaluations.**
- Slide transitions: precision and recall at ±2 seconds against hand annotations, with thresholds tuned on one lecture and reported on another, against PySceneDetect and against ablations of the change score.
- Speech recognition: pooled word error rate on hand-corrected excerpts, with and without the slide glossary prompt.
- Failure analysis: for each miss, what came back and where each ranker had placed the answer.

```bash
python eval/evaluate.py --split dev --tune
```

```bash
python eval/evaluate.py --split test --tuned results/dev/tuned.json --latency
```

### Results

**Not yet available.** This section will hold the ablation table, the per-type chart, latency, the component results and the failure analysis, from measured runs only.

<details>
<summary>Pipeline check on the synthetic demo lectures (not a benchmark)</summary>

These numbers come from the two generated lectures: clean rendered slides, a synthetic voice, 3 minutes of video, 13 moments. They show that every stage and every evaluation script works. They say nothing about quality on real lectures, where slides are dense, audio is noisy and the library is a hundred times larger.

| Check | Outcome |
|---|---|
| Slide transitions found (±1 s) | 11 of 11, no spurious cuts; the two bullet reveals correctly not reported |
| Same, plain mean absolute difference | 2 spurious cuts, both on the bullet reveals |
| Same, PySceneDetect `ContentDetector` defaults | 1 of 11 found |
| Word error rate, no prompt | 1.54% |
| Word error rate, slide glossary prompt | 0.66% (the gain here is spelling convention: the slides say "generalisation") |
| Query latency on an Apple M4 CPU, hybrid, encoders included | p50 98 ms, p95 103 ms |

Retrieval metrics are omitted on purpose: with 13 moments nearly everything is in the top 5.

</details>

---

## Data and licensing

- **Public demo.** To be built on [MIT OpenCourseWare](https://ocw.mit.edu) lectures. OCW's default licence is CC BY-NC-SA, though individual courses can differ, so the licence is checked per course and recorded per lecture in `config/lectures.csv`. Derived transcripts, slide text and thumbnails are shared under the same licence, for non-commercial use, with credit to OCW and the lecturers. `lecturelens push` generates the attribution card from the index itself.
- **Your own lectures.** Indexed locally, for personal study, and never uploaded, hosted or committed here. `.gitignore` excludes video, audio, work directories and indexes.
- **This repository** contains code only. No lecture content is included.

## Related work

- **Echo360** offers interactive transcript search within a recording. Its slide search does not read text that sits inside images. ([docs](https://support.echo360.com/hc/en-us/articles/11077509703437))
- **Panopto** has searched spoken words and on-screen text via OCR since 2014. ([announcement](https://campustechnology.com/articles/2014/03/25/panopto-adds-video-content-search-to-lecture-capture-platform.aspx))
- **Cutsio** is a commercial product offering semantic, cross-library lecture search with visual indexing. ([blog](https://cutsio.com/blog/lecture-capture-alternative-for-search-cutsio))

Lecture search already exists. What LectureLens adds is an **open, measured, multimodal system**, with a benchmark that shows which signal helps for which kind of question.

---

## Repository layout

```
lecturelens/
  ingest.py       ffmpeg: audio, detection frames, keyframe stills
  slides.py       slide segmentation
  ocr.py          PP-OCRv5, reading order, fragment joining
  asr.py          faster-whisper, word timestamps, jargon glossary
  moments.py      windowing and word alignment
  embed.py        Qwen3-Embedding and SigLIP 2
  index.py        the on-disk index format
  pipeline.py     staged, cached, incremental indexing
  search.py       rankers, RRF, dedupe, explanations
  rerank.py       optional cross-encoder
  metrics.py      relevance, Recall@k, MRR, transition matching, WER
  evaluation.py   ablation systems, tuning, reports
  publish.py      Hugging Face push / pull, dataset card
  demo.py         synthetic lectures with ground truth
  cli.py
eval/
  README.md               annotation formats and protocol
  evaluate.py             retrieval benchmark
  eval_transitions.py     slide transition precision / recall
  eval_asr.py             word error rate
config/
  lectures.csv            catalogue: title, course, source, licence, link template
  masks.yaml              per-lecture regions to ignore, and detector overrides
notebooks/
  build_index_kaggle.ipynb   build and publish the index on a free GPU
tests/                    unit tests, plus an end-to-end test with the real models
```

## Development

```bash
uv sync --extra eval
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

The unit tests replace the four models with deterministic stand-ins and synthesise their own video, so they need `ffmpeg` but no weights and finish in seconds. To run the real models end to end:

```bash
uv sync --extra index --extra eval && uv run pytest -m models
```

**macOS note.** If the repository lives in an iCloud-synced folder (Desktop or Documents), macOS marks dot-folders such as `.venv` as hidden, and Python 3.12 ignores the hidden file that makes an editable install importable. The symptom is `ModuleNotFoundError: No module named 'lecturelens'` from `uv run lecturelens`. Two ways round it: run the module form from the repository root, `uv run python -m lecturelens …`, which needs no install; or add `export UV_NO_EDITABLE=1` to your shell profile so that uv installs the package as ordinary files (it must be set for every uv command, because a later `uv run` without it switches back).

## Roadmap

- [x] Ingest, slide segmentation, OCR, speech recognition, moments
- [x] Embeddings, index format, resumable pipeline
- [x] BM25, dense and image retrieval with rank fusion; optional reranker
- [x] CLI and Hugging Face publishing
- [x] Evaluation harness, tuning, failure reports
- [x] Kaggle notebook for indexing on a GPU
- [ ] Choose an open-licensed course, index it, tune slide thresholds on real footage
- [ ] Annotate transitions, ASR excerpts and 60 queries; run the benchmark; fill in Results
- [ ] Web demo

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Qwen3 Embedding and Reranker](https://arxiv.org/abs/2506.05176)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [PaddleOCR (PP-OCRv5)](https://www.paddleocr.ai/latest/en/update/update.html), [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [bm25s](https://github.com/xhluca/bm25s)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack, Clarke and Büttcher, SIGIR 2009)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [SliTraNet: slide transition detection](https://arxiv.org/abs/2202.03540)
- [MIT OpenCourseWare terms](https://ocw.mit.edu/pages/get-started)
