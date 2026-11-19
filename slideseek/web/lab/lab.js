/* The lab: run each browser step of adding a lecture, timed, and save what it captures.
 *
 * Served only by `slideseek serve --lab DIR`, on the machine it runs on. The server finds the
 * slides in the frames exactly as the indexer does (POST /api/lab/detect), and writes what
 * is sent to POST /api/lab/files/... into DIR, where eval scripts compare it with the
 * Python pipeline's work directory.
 */

import { decodeAudio, SAMPLE_RATE } from "/capture/audio.js";
import { FRAME_HEIGHT, FRAME_WIDTH, grabStills, openVideo, sampleFrames } from "/capture/video.js";

// Pinned revisions, so a measurement can be repeated with exactly the same weights.
const MODELS = {
  "turbo-q4f16": {
    model: "onnx-community/whisper-large-v3-turbo_timestamped",
    revision: "b3f77bf9a8c4d5ea3415827033d1ffea7955fd9a",
    dtype: { encoder_model: "q4f16", decoder_model_merged: "q4f16" },
  },
  "turbo-fp16": {
    model: "onnx-community/whisper-large-v3-turbo_timestamped",
    revision: "b3f77bf9a8c4d5ea3415827033d1ffea7955fd9a",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
  },
  small: {
    model: "onnx-community/whisper-small_timestamped",
    revision: "65caa70f294b46e1c33ff820aae6b16d048ab818",
    dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
  },
};

const $ = (id) => document.getElementById(id);
const state = { running: false, summary: null, error: null };

function step(text) {
  const item = document.createElement("li");
  item.textContent = text;
  $("steps").append(item);
  return (more) => {
    item.textContent = `${text}${more ? `: ${more}` : ""}`;
  };
}

const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`;

async function save(path, body, type = "application/octet-stream") {
  const response = await fetch(`/api/lab/files/${path}`, {
    method: "POST",
    headers: { "Content-Type": type },
    body,
  });
  if (!response.ok) throw new Error(`Saving ${path} failed: ${response.status}`);
}

/** Words into phrases, as Whisper's segments would hold them, for asr.jsonl. */
export function toSegments(words) {
  const segments = [];
  let current = [];
  const flush = () => {
    if (!current.length) return;
    segments.push({
      start: current[0].s,
      end: current[current.length - 1].e,
      text: current.map((word) => word.w).join(" "),
      words: current,
    });
    current = [];
  };
  for (const word of words) {
    const last = current[current.length - 1];
    if (last && (word.s - last.e > 1.0 || (/[.?!]$/.test(last.w) && current.length >= 3) || current.length >= 40)) {
      flush();
    }
    current.push(word);
  }
  flush();
  return segments;
}

async function gpuInfo() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    shaderF16: adapter.features.has("shader-f16"),
  };
}

function transcribeInWorker(audio, choice, prompt, report) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("/capture/speech-worker.js", location.href), { type: "module" });
    const words = [];
    const blocks = [];
    let loadSeconds = 0;
    const downloads = new Map();
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "download") {
        downloads.set(data.file, [data.loaded, data.total]);
        let loaded = 0;
        let total = 0;
        for (const [l, t] of downloads.values()) {
          loaded += l;
          total += t;
        }
        report(`downloading the model, ${(loaded / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`);
      } else if (data.type === "loaded") {
        loadSeconds = data.seconds;
        report(`model ready in ${data.seconds.toFixed(1)} s; listening`);
        worker.postMessage(
          { type: "transcribe", audio, blockSeconds: 300, padSeconds: 5, prompt },
          [audio.buffer],
        );
      } else if (data.type === "block") {
        words.push(...data.words);
        blocks.push(data.seconds);
        report(`block ${data.index + 1} of ${data.count} (${data.seconds.toFixed(1)} s)`);
      } else if (data.type === "done") {
        const downloaded = [...downloads.values()].reduce((sum, [, t]) => sum + t, 0);
        worker.terminate();
        resolve({ words, blocks, loadSeconds, speechSeconds: data.seconds, downloaded });
      } else if (data.type === "error") {
        worker.terminate();
        reject(new Error(data.message));
      }
    });
    worker.addEventListener("error", (event) => {
      worker.terminate();
      reject(new Error(event.message || "The speech worker failed to start."));
    });
    worker.postMessage({ type: "load", ...choice });
  });
}

async function recording() {
  const picked = $("file").files[0];
  if (picked) return picked;
  const url = $("url").value.trim();
  if (!url) throw new Error("Pick a recording, or give the path of one the server serves.");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not fetch ${url}: ${response.status}`);
  const blob = await response.blob();
  return new File([blob], url.split("/").pop(), { type: blob.type });
}

export async function run(options = {}) {
  const name = options.name ?? $("name").value.trim();
  const choiceKey = options.model ?? $("model").value;
  const limit = Number(options.limit ?? $("limit").value) || 0;
  const prompt = options.prompt ?? $("prompt").value.trim();
  const doFrames = options.frames ?? $("do-frames").checked;
  const doStills = options.stills ?? $("do-stills").checked;
  const doSpeech = options.speech ?? $("do-speech").checked;
  if (options.url) $("url").value = options.url;
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(name)) throw new Error("Save as: lowercase letters, digits, . _ -");

  state.running = true;
  state.error = null;
  $("start").disabled = true;
  $("steps").replaceChildren();
  const summary = { name, userAgent: navigator.userAgent, gpu: await gpuInfo(), timings: {} };
  state.summary = summary;
  try {
    let report = step("Reading the file");
    let started = performance.now();
    const file = await recording();
    summary.file = { name: file.name, bytes: file.size, type: file.type };
    const source = await openVideo(file);
    summary.duration = source.duration;
    summary.video = { width: source.width, height: source.height };
    report(`${(file.size / 1e6).toFixed(0)} MB, ${(source.duration / 60).toFixed(1)} min, ${source.width}×${source.height}`);

    let keyframes = [];
    if (doFrames) {
      report = step("Frames, 1 a second");
      started = performance.now();
      const { frames, count } = await sampleFrames(source, {
        onProgress: (done) => report(`${Math.round(done * 100)}%`),
      });
      summary.timings.frames = (performance.now() - started) / 1000;
      summary.frames = { count, width: FRAME_WIDTH, height: FRAME_HEIGHT };
      report(`${count} frames in ${seconds(performance.now() - started)}`);
      await save(`${name}/frames.bin`, frames);

      report = step("Finding the slides (on the server)");
      started = performance.now();
      const response = await fetch(`/api/lab/detect?count=${count}&duration=${source.duration}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: frames,
      });
      if (!response.ok) throw new Error(`Detection failed: ${response.status}`);
      const detected = await response.json();
      summary.timings.detect = (performance.now() - started) / 1000;
      summary.segments = detected.segments.length;
      keyframes = detected.keyframe_times;
      report(`${detected.segments.length} segments, ${keyframes.length} keyframes, ${seconds(performance.now() - started)}`);
    }

    if (doStills && keyframes.length) {
      report = step("Stills of each slide");
      started = performance.now();
      const stills = await grabStills(source, keyframes, {
        onProgress: (done) => report(`${Math.round(done * 100)}%`),
      });
      summary.timings.stills = (performance.now() - started) / 1000;
      let bytes = 0;
      for (const { t, blob } of stills) {
        bytes += blob.size;
        await save(`${name}/stills/${String(Math.round(t * 1000)).padStart(9, "0")}.jpg`, blob, "image/jpeg");
      }
      summary.stills = { count: stills.length, bytes };
      report(`${stills.length} stills, ${(bytes / 1e6).toFixed(1)} MB, ${seconds(summary.timings.stills * 1000)}`);
    }
    source.close();

    if (doSpeech) {
      report = step("Sound, 16 kHz mono");
      started = performance.now();
      let audio = await decodeAudio(file);
      summary.timings.audio = (performance.now() - started) / 1000;
      if (limit > 0) audio = audio.slice(0, Math.round(limit * SAMPLE_RATE));
      summary.audioSeconds = audio.length / SAMPLE_RATE;
      report(`${(summary.audioSeconds / 60).toFixed(1)} min in ${seconds(performance.now() - started)}`);

      report = step(`Speech (${choiceKey})`);
      const choice = MODELS[choiceKey];
      const heard = await transcribeInWorker(audio, choice, prompt, report);
      summary.model = { key: choiceKey, ...choice, downloadedBytes: heard.downloaded };
      summary.timings.modelLoad = heard.loadSeconds;
      summary.timings.speech = heard.speechSeconds;
      summary.timings.speechBlocks = heard.blocks;
      summary.words = heard.words.length;
      summary.realTimeFactor = summary.audioSeconds / heard.speechSeconds;
      summary.prompt = prompt;
      report(`${heard.words.length} words in ${heard.speechSeconds.toFixed(0)} s, ${summary.realTimeFactor.toFixed(1)}× real time`);
      const lines = toSegments(heard.words).map((segment) => JSON.stringify(segment)).join("\n");
      await save(`${name}/asr.jsonl`, `${lines}\n`, "application/x-ndjson");
    }

    await save(`${name}/summary.json`, JSON.stringify(summary, null, 2), "application/json");
    $("summary").textContent = JSON.stringify(summary, null, 2);
    return summary;
  } catch (error) {
    state.error = String(error?.message ?? error);
    step(`Stopped: ${state.error}`);
    throw error;
  } finally {
    state.running = false;
    $("start").disabled = false;
  }
}

$("form").addEventListener("submit", (event) => {
  event.preventDefault();
  run().catch(() => {});
});

window.lab = { run, state };
