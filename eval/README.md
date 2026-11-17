# Evaluation

Three things are measured, each against ground truth written by hand:

| What | Script | Ground truth | Output |
|---|---|---|---|
| Retrieval: does the right moment come back? | `evaluate.py` | `queries.jsonl` | `results/<split>/` |
| Slide transitions: are slide changes found? | `eval_transitions.py` | `transitions/<lecture_id>.json` | `results/transitions.md` |
| Speech recognition: how many words are wrong? | `eval_asr.py` | `asr_ref/excerpts.jsonl` + text files | `results/asr.md` |

Every script runs from the repository root and takes `--help`.

To see all three run before any annotation exists, generate the synthetic demo lectures.
They come with ground truth in exactly these formats:

```bash
slideseek demo demo
slideseek index demo/videos --catalog demo/lectures.csv --out demo/index --work demo/work
python eval/evaluate.py --index demo/index --queries demo/eval/queries.jsonl --split all --out demo/results
python eval/eval_transitions.py --videos demo/videos --annotations demo/eval/transitions --out demo/results
python eval/eval_asr.py --work demo/work --refs demo/eval/asr_ref --out demo/results
```

Numbers from the demo show that the pipeline is wired correctly. They are not results.

---

## 1. Retrieval queries: `eval/queries.jsonl`

One JSON object per line:

```json
{"qid": "q017", "query": "why do we hold out data we never train on", "type": "paraphrase", "lecture_id": "lec03", "gt": [["21:52", "23:40"]], "split": "test", "author": "me"}
```

| Field | Meaning |
|---|---|
| `qid` | Unique id. |
| `query` | The question, phrased the way a student would type it. |
| `type` | `keyword`, `paraphrase`, `spoken` or `visual` (see below). |
| `lecture_id` | The lecture that contains the answer. |
| `gt` | One or more `[start, end]` intervals that answer the query. Seconds, or `m:ss` / `h:mm:ss` as the player shows. |
| `split` | `dev` or `test`. |
| `author` | Who wrote it. Queries by someone other than the system's author are reported separately. |

### Query types

Each type is designed so that a different signal should win. That is what makes the
per-type breakdown informative.

| Type | What it is | Example | Should favour |
|---|---|---|---|
| `keyword` | A term that appears verbatim. | "Bellman equation" | BM25 |
| `paraphrase` | The idea, in words the lecture did not use. | "why do we hold out data we never train on" | Dense text |
| `spoken` | Something said aloud that is on no slide: a worked example, an aside, an analogy. | "the analogy about walking downhill in fog" | Transcript over OCR |
| `visual` | A description of what a slide looks like. | "the diagram with the three-layer network" | SigLIP 2 |

### Protocol

1. **Write queries before building or tuning retrieval**, from watching the lecture, not
   from reading the transcript. A query written while looking at the index text will share
   its vocabulary and flatter lexical search.
2. **Target: 60 queries over 5 lectures**, 12 per lecture, 15 per type.
3. **Mark the interval generously but honestly**: from where the explanation starts to
   where it ends. If the same thing is explained twice, list both intervals.
4. **Split 20 dev / 40 test**, balanced across types and lectures. Assign the split when
   the query is written, not after seeing how it scores.
5. **Tune only on dev.** `evaluate.py --split dev --tune` chooses fusion settings.
   `evaluate.py --split test` is for reporting; every test run is appended to
   `results/test_runs.log` with the commit it ran at.

### What counts as relevant

A returned moment is relevant if it is in the right lecture and its span overlaps a
ground-truth interval widened by 15 seconds on each side. The intent: the result lands the
student close enough to see the explanation, even when the moment boundary and the
annotated boundary differ slightly.

This definition is lenient towards long moments (one that only *ends* where the answer
begins still overlaps it), so `results/<split>/tolerance.md` also reports a stricter
criterion, **landing**: the result must start inside the answer or at most N seconds
before it. Both are shown at several tolerances so the headline number can be read in
context.

Metrics are Recall@1, Recall@5 and MRR@10, overall and per query type, each with a 95%
bootstrap confidence interval. The hybrid is compared with keyword transcript search by a
paired bootstrap over the same queries.

---

## 2. Slide transitions: `eval/transitions/<lecture_id>.json`

```json
{"lecture_id": "lec01", "transitions": ["0:42", "1:35", "3:07", 412.5]}
```

List the time of every slide change: the moment a new slide appears. Play the lecture at
2x and note the timestamp each time; about 40 minutes per lecture.

**Convention:** a bullet point or animation step appearing on the *same* slide is not a
transition. Switching between slides and a camera view, or to a different window, is.

A predicted transition within 2 seconds of an annotated one is a match; matching is one to
one. Thresholds are tuned on one lecture (`--tune-on lec01`) and reported on the others.

---

## 3. ASR references: `eval/asr_ref/`

`excerpts.jsonl` lists the excerpts; each points at a text file in the same directory:

```json
{"lecture_id": "lec02", "t0": "12:00", "t1": "17:00", "reference": "lec02_excerpt.txt"}
```

The reference file is the corrected transcript of that stretch, as plain text. The fastest
way to produce one is to take the recogniser's own output for the excerpt and correct it
while listening, but then correct *everything*, including words it got right by luck in a
different spelling.

Word error rate is computed after lower-casing and removing punctuation and hyphenation.
Spelling variants and number formats are not normalised: write the reference the way the
recogniser is expected to write. Two 5-minute excerpts from different lectures are enough
for a stable number.

`--compare-jargon` re-transcribes each excerpt with and without the glossary prompt built
from that lecture's slides, which measures what the jargon boost is worth.

## 4. Search by picture: no annotation needed

`eval/eval_screenshot.py` measures searching with a picture of a slide. It needs the index
and the lecture videos, and nothing written by hand: it takes frames at random times
inside slide segments, never the keyframe that was indexed, and makes three queries from
each (the frame itself, a simulated phone photo, and a crop). A query is found when a
result shows the slide the frame came from. It compares the image signal, the OCR signal
and the two fused, and writes `results/screenshot/metrics.json`, which the web app shows.

```bash
python eval/eval_screenshot.py --index index --videos videos --per-lecture 20
```
