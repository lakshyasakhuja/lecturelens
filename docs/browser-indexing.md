# Adding a lecture in the browser: measurements

Whether the browser can do the slow half of indexing (speech recognition, reading the
frames and stills) well enough that someone with no terminal can add their own lecture,
with the video never leaving their computer. Measured before building it.

## How it was measured

`slideseek serve --lab DIR` serves a test page at `/lab` on your own machine (it refuses
any address but 127.0.0.1). The page runs each browser step on a recording, timed, and saves
what it captures in `DIR/<run>/`:

| Step | Browser code | What the indexer does instead |
|---|---|---|
| 1 fps greyscale frames, 160×90 | `static/capture/video.js` (`sampleFrames`): seek a `<video>`, draw to a canvas | `ingest.read_gray_frames` (ffmpeg) |
| Slide segments | sent to the server, `slides.detect_slides` | the same |
| Stills of each keyframe | `grabStills` | `ingest.extract_still` |
| Sound, 16 kHz mono | `static/capture/audio.js` (`decodeAudioData`) | `ingest.read_audio` |
| Speech with word times | `static/capture/speech-worker.js`: Whisper on WebGPU, transformers.js 4.3.2 | faster-whisper, with VAD and a slide glossary |

The scripts it runs are fetched at pinned versions, checked by hash, by `slideseek vendor`.

Run on MIT 6.7960 lecture 4 (84 minutes, 640×360, 147 MB) in Chromium 156 on an Apple M4
(10 cores, 16 GB), November 2026, and compared with that lecture's Python work directory.

## Results

| | Browser | Compared with the Python pipeline |
|---|---|---|
| Frames | 5,036 in 70 s | Same count. Median difference 2.7 grey levels out of 255 |
| Slide segments | 105 (server, 4.5 s) | 103 from ffmpeg's frames. 100 of its 102 cuts found within 1 s |
| Stills | 125, 3.8 MB, 1.9 s | |
| Sound | 7.6 s for 84 minutes | |
| Speech model | `whisper-large-v3-turbo`, 4-bit with 16-bit activations: a 566 MB download, once, then cached | Same model family as the Python pipeline (`large-v3-turbo`) |
| Speech | 84 minutes in 19 minutes: 4.4× faster than real time | 14,025 words against 13,691 |
| Words that differ | 7.4% over the lecture. Median minute 6.2%. First 10 minutes 3.5% | Most of the difference is extra words: quiet audience questions and end-of-class chatter that faster-whisper's voice-activity filter drops, and the browser transcribes, sometimes badly |
| Word times | median 0.24 s from Python's, 90% within 0.52 s | Moments are 45 s windows, so this does not move a word to another moment except at the edge |

"Words that differ" is the word error rate of the browser's transcript measured against the
Python one: disagreement, not accuracy. Neither transcript is the truth.

Two other options were measured on the first 10 minutes, and neither is used:

- **The 16-bit encoder:** a 1.6 GB download, 2.5× real time, and no better (3.9% against
  3.5%).
- **The slide glossary as a prompt:** it works, but it made speech recognition 3× slower
  (1.5× real time) and changed nothing measurable in those 10 minutes. Its benefit has only
  been measured on the synthetic demo lectures.

## What this means

- Adding an 84-minute lecture on an M4 takes about 22 minutes with the tab open: 1 minute
  of frames and stills, 19 of speech, and a first-time model download. A slower GPU takes
  longer.
- The video stays on the visitor's computer. What would be sent is the tiny frames (72 MB,
  or 13.6 MB compressed with gzip), 3.8 MB of stills and a 0.7 MB transcript.
- Search finds the same moments in an index built from the browser's pieces as in the
  Python one: see [below](#searching-a-lecture-added-in-the-browser).
- Possible improvement: a voice-activity filter in the browser (Silero VAD runs in ONNX
  Runtime), to drop the garbled audience questions the way faster-whisper does.

## Keeping the speech model

transformers.js keeps what it downloads in the browser's Cache API. Building the page showed
that Chrome stores the 194 MB decoder there but refuses the 370 MB encoder
(`Failed to execute 'put' on 'Cache': Unexpected internal error`), so every lecture would
download the encoder again. The speech worker therefore gives transformers.js a cache of its
own over the origin's private file system (`static/capture/speech-worker.js`): each file is
streamed to disk, and a marker named after the model's revision is written once all of them
are there, which is how the page knows not to warn about the download. With the model kept
this way, loading it took about 2 seconds before listening started, against 30 seconds or
more to download it.

## Searching a lecture added in the browser

Lecture 4 was then added through the page itself, on `localhost` with a test account: the
browser sent its frames, 125 stills and transcript, and the server built the lecture's index
with its own stages (`slideseek/library.py`). `eval/compare_browser_index.py` asks that index
and the Python pipeline's index of the same recording the same queries, each searching
lecture 4 alone.

| | Python pipeline | Added in the browser |
|---|---|---|
| Moments, slide keyframes | 141, 123 | 143, 125 |
| Screenshot of the recording | 82.5% / 97.5% | 82.5% / 100% |
| Simulated phone photo | 60.0% / 97.5% | 57.5% / 97.5% |
| Crop of 55 to 75% of the slide | 67.5% / 97.5% | 67.5% / 100% |

<sub>Found first / found in the top five: 40 frames at random times inside slides, seed 7,
made into each kind of picture the way `eval/eval_screenshot.py` does, and each index judged
against its own keyframe for that time. One picture is 2.5 points.</sub>

For 17 of 20 typed questions the two put the same moment first, and they share 90% of their
top five (two results agree when they start within 20 s). In each of the three that differ,
the same two moments come first in the other order. The questions have no ground truth yet,
so this is agreement between the two, not accuracy. The numbers are in
`results/browser/metrics.json`.

Once the transcript arrived, the server built the 84-minute lecture's search in about
80 seconds on the M4 (it had read the slides as they came in).

## What adding lectures showed

- **Whisper's word times slip.** In the 84-minute lecture one word of 14,025 ended before it
  started; in a 12-minute clip the last word was placed 3.5 s past the end of the sound, in
  the padding of Whisper's last 30-second window. The server refused both transcripts at
  first. The worker now keeps each word inside the sound, the right way round, and the
  server mends the same two slips rather than refusing them: it swaps the two times, and
  moves a word up to 30 s past the end to the end.
- **Nothing heard is lost.** With the 12-minute clip, the server was stopped after the first
  five-minute block and the lecture deleted from its disk, which is what a Space restart does
  to an unfinished lecture (only finished ones are copied to the dataset). Dropping the same
  file again made a new lecture, took the block the browser had kept and listened from
  minute 5. The first five minutes' 816 words were in the transcript, and search found them.
- **The model stays.** Kept in the private file system, it loaded in about 3 seconds. Once,
  in the test browser, the model's folder was found empty after a long run, with its space
  still counted, and the model was downloaded again; the cause was not found, and restarting
  the server did not do it. The worker now logs a warning whenever it cannot keep a file.
  The test browser did not grant `navigator.storage.persist()`, which the page asks for, so
  a browser short of space may clear the model.
- **Listening speed** held with the page in view: about 4× real time for the clip, as for the
  full lecture.
