# LectureLens

Semantic search for lecture recordings. Ask a question in plain English, such as *"where did she explain why we need a validation set?"*, and jump straight to that moment in the video.

Each result is ranked using three signals: what was **said**, what text was **on the slide**, and what the slide **looked like**.

> **Status: in development.** The design below is settled, the code is being built. No numbers are reported yet, and the results sections will only ever be filled with measured values.

---

## The problem

Lectures run one to two hours. Finding the three minutes you need before an exam means scrubbing through a recording or rewatching it at 2x.

Keyword search over a transcript helps only when you remember the exact words the lecturer used. It fails when you describe the idea in your own words, when the term only appeared on a slide, or when what you remember is a picture ("the plot where validation loss goes back up").

## What it does

- Searches a whole library of lectures at once, not one recording at a time
- Returns the top 5 moments, each with a timestamp, slide thumbnail, text snippet, and a badge showing which signal matched (`said`, `on slide`, `visual`)
- Links open the video at the matching timestamp, so no video is rehosted
- Ships as a public web demo over open-licensed lectures, and as a local CLI for indexing your own recordings privately

### Scope of v1

| In scope | Out of scope |
|---|---|
| Slide-based lectures (screen capture, or a camera on the projector) | Chalkboard and handwriting-heavy lectures |
| English | Other languages |
| A library of 5 to 15 lectures | Generated answers (retrieval only) |
| Web demo on a prebuilt index, plus a local CLI | Uploading and indexing videos in the public demo |

---

## How it works

The core design decision is to **index once offline on a GPU and search online on a CPU**. Heavy models run a single time per lecture; a query only needs two small text encoders and a few matrix products.

### Offline indexing

```
 lecture.mp4
   |
   +--> ffmpeg: audio, 16 kHz mono --> faster-whisper (large-v3-turbo)
   |                                    --> words with timestamps
   |
   +--> ffmpeg: frames at 1 fps, 640 px wide
          --> slide-change detector --> slide segments [t0, t1] + keyframe
                 keyframe --> PP-OCRv5  --> slide text
                 keyframe --> SigLIP 2  --> image embedding

 align transcript words to slide segments --> "moments"
 moment text = slide text + transcript
   --> Qwen3-Embedding-0.6B --> dense text vector
   --> BM25 index
```

A **moment** is the unit of retrieval: one slide segment, with the words spoken while it was on screen. Slides that stay up for more than 90 seconds are split into overlapping 45-second windows so a long explanation is not buried in one large chunk.

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
     dedupe near-identical moments --> top 5
```

### Design notes

**Slide segmentation.** Frames are compared at 1 fps on a downscaled greyscale image, with per-lecture masks for regions that move for unrelated reasons (webcam overlay, player progress bar). A new segment starts when the frame difference crosses a high threshold and the picture then stays stable for at least 3 seconds, which suppresses cursor movement, laser pointers and animation steps. The keyframe is the **last** stable frame of a segment, because an animated slide is only complete at the end. Neighbouring segments with near-identical perceptual hashes are merged.

**Rank fusion.** BM25 scores, cosine similarities and SigLIP logits sit on different scales. Reciprocal Rank Fusion uses only ranks, so it needs no score calibration and has a single hyperparameter:

```
RRF(d) = sum over rankers of 1 / (60 + rank_i(d))
```

**No vector database.** A library this size is roughly 2,000 moments. Brute-force cosine similarity over a numpy array is sub-millisecond, so a vector store would add setup and nothing else. That changes at hundreds of thousands of moments, or once the index takes concurrent writes.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Environment | Python 3.12, `uv` | Fast, reproducible installs |
| Video and audio | `ffmpeg` | Standard and fast |
| Speech recognition | faster-whisper, `large-v3-turbo` | Decoder pruned from 32 layers to 4: much faster than large-v3 at close accuracy |
| Slide segmentation | Custom frame-difference detector | Tuned for slides; PySceneDetect is the baseline it is measured against |
| OCR | PaddleOCR 3.x (PP-OCRv5) | Strong on dense slide text |
| Text embeddings | Qwen3-Embedding-0.6B | Open (Apache 2.0), instruction-aware, larger sizes available |
| Image-text embeddings | SigLIP 2 (`google/siglip2-base-patch16-224`) | Good text-to-image retrieval |
| Sparse retrieval | `bm25s` | Fast BM25 in pure Python and numpy |
| Reranker (optional) | Qwen3-Reranker-0.6B | Same family as the embedder |
| Storage | Parquet and numpy arrays | See "No vector database" above |

---

## Evaluation plan

The goal is not only a working demo but a measurement of **when each signal helps**.

**Query set.** 60 queries over 5 lectures, each with one or more ground-truth time intervals. They are written before retrieval is built, phrased the way a student would ask, and split into 20 dev queries (for tuning) and 40 test queries (reported once, at the end).

| Query type | Example | Signal expected to win |
|---|---|---|
| Keyword | "Bellman equation" | BM25 |
| Paraphrase | "why do we hold out data we never train on" | Dense text |
| Spoken-only | a worked example the lecturer said aloud but never put on a slide | Transcript, not OCR |
| Visual | "the diagram with the three-layer network" | SigLIP 2 |

**Relevance.** A returned moment counts as relevant if its time span overlaps a ground-truth interval within ±15 seconds: close enough that a student who lands there sees the explanation.

**Metrics.** Recall@1, Recall@5 and MRR@10, overall and per query type, plus CPU query latency (p50, p95) and indexing time per hour of video.

**Ablations.** Each system below is evaluated on the same test queries.

| # | System | What it isolates |
|---|---|---|
| 1 | BM25, transcript only | Baseline, close to ordinary keyword transcript search |
| 2 | BM25, transcript + slide OCR | Value of reading the slides |
| 3 | Dense, transcript only | Value of semantic matching |
| 4 | Dense, transcript + slide text | Both together |
| 5 | Image only (SigLIP 2) | Value of the visual signal on its own |
| 6 | Hybrid RRF (2 + 4 + 5) | The full system |
| 7 | Hybrid + reranker | Whether reranking earns its cost |

The working hypothesis is that BM25 wins keyword queries, dense wins paraphrases, the image signal helps visual queries, and the hybrid is best overall. That is a hypothesis to test, not a result. Whatever happens gets reported, including any query type where the hybrid loses.

**Component evaluations.**
- Slide transitions: precision and recall at ±2 seconds against hand-annotated lectures, compared with PySceneDetect. Thresholds are tuned on one lecture and reported on another.
- Speech recognition: word error rate on hand-corrected excerpts.
- Failure analysis: a set of real misses, each with the query, the expected moment, what came back, and the cause.

## Results

Not yet available.

---

## Data and licensing

- **Public demo.** Built on [MIT OpenCourseWare](https://ocw.mit.edu) lectures. OCW's default licence is CC BY-NC-SA, though individual courses can differ, so the licence is checked per course. Derived transcripts and index files are released under the same licence, for non-commercial use, with credit to OCW and the lecturers.
- **Your own lectures.** The CLI indexes recordings locally. University recordings and slides belong to the institution and its lecturers, so they are for personal study only and are never uploaded, hosted, or committed to this repository.

## Related work

- **Echo360** offers interactive transcript search within a recording. Its slide search does not read text that sits inside images. ([docs](https://support.echo360.com/hc/en-us/articles/11077509703437))
- **Panopto** has searched spoken words and on-screen text via OCR since 2014. ([announcement](https://campustechnology.com/articles/2014/03/25/panopto-adds-video-content-search-to-lecture-capture-platform.aspx))
- **Cutsio** is a commercial product offering semantic, cross-library lecture search with visual indexing. ([blog](https://cutsio.com/blog/lecture-capture-alternative-for-search-cutsio))

Lecture search already exists. What LectureLens adds is an **open, measured, multimodal system**, with a benchmark that shows which signal helps for which kind of question.

---

## Roadmap

- [ ] Ingest: audio and frame extraction
- [ ] Slide-change detection, keyframes, segment merging
- [ ] Speech recognition with word timestamps
- [ ] OCR on keyframes
- [ ] Moment construction and alignment
- [ ] Text and image embeddings, saved index
- [ ] BM25, dense and image retrieval with rank fusion
- [ ] CLI: `index`, `search`, `serve`
- [ ] Evaluation harness, ablations and failure analysis
- [ ] Web demo

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Qwen3 Embedding and Reranker](https://arxiv.org/abs/2506.05176)
- [SigLIP 2](https://arxiv.org/abs/2502.14786)
- [PaddleOCR (PP-OCRv5)](https://www.paddleocr.ai/latest/en/update/update.html)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [SliTraNet: slide transition detection](https://arxiv.org/abs/2202.03540)
- [MIT OpenCourseWare terms](https://ocw.mit.edu/pages/get-started)
