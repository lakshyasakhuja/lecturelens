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
- Still to be measured (step 7 of the plan): whether search finds the same moments in an
  index built from the browser's pieces, using `eval/eval_screenshot.py` and the example
  queries.
- Possible improvement: a voice-activity filter in the browser (Silero VAD runs in ONNX
  Runtime), to drop the garbled audience questions the way faster-whisper does.
