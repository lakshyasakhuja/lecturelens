/* Adding a lecture from the visitor's own computer.
 *
 * The recording is read here, in the browser, and never sent anywhere. What is sent to
 * /api/library is what the server needs to finish it with the indexer's own code:
 *
 *   frames      160x90 grey frames, one a second, gzipped        (video.js: sampleFrames)
 *   stills      a JPEG at each time the server asks for           (video.js: grabStills)
 *   transcript  the words, with their times                       (speech-worker.js)
 *
 * Every piece is kept in IndexedDB until the server has it, and the transcript is kept a
 * block of five minutes at a time, so closing the tab, or the server restarting, costs at
 * most the block that was being transcribed.
 *
 * A lecture's file is remembered for playing it later: as a file handle where the browser
 * has them (Chrome and Edge), which it can be opened from again with the visitor's
 * permission, and otherwise by its name and size, for the visitor to pick it again.
 */

import { decodeAudio } from "./audio.js";
import { grabStills, openVideo, sampleFrames } from "./video.js";

// The speech model, at a pinned revision: the 4-bit large-v3-turbo measured in
// docs/browser-indexing.md, 4.4 times faster than real time on an M4.
export const SPEECH = {
  model: "onnx-community/whisper-large-v3-turbo_timestamped",
  revision: "b3f77bf9a8c4d5ea3415827033d1ffea7955fd9a",
  dtype: { encoder_model: "q4f16", decoder_model_merged: "q4f16" },
  device: "webgpu",
  label: "whisper-large-v3-turbo, 4-bit, in the browser",
  downloadBytes: 566_000_000,
};
const BLOCK_SECONDS = 300;
const VIDEO_TYPES = [".mp4", ".m4v", ".mov", ".webm", ".mkv"];

/* ------------------------------------------------------------ what this browser can do */

/** Can this browser add a lecture? Resolves to { ok, reason }. */
export async function support() {
  const missing = [
    ["OffscreenCanvas", typeof OffscreenCanvas],
    ["OfflineAudioContext", typeof OfflineAudioContext],
    ["CompressionStream", typeof CompressionStream],
    ["indexedDB", typeof indexedDB],
  ].filter(([, kind]) => kind === "undefined");
  if (missing.length || !navigator.gpu) {
    return {
      ok: false,
      reason:
        "This browser can’t listen to a lecture: it has no WebGPU. Use Chrome or Edge on a computer.",
    };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no adapter");
  } catch {
    return {
      ok: false,
      reason: "This browser has WebGPU, but no graphics processor it can use. Try Chrome or Edge on a computer.",
    };
  }
  return { ok: true, reason: "" };
}

/** Has the speech model been downloaded before? The worker leaves a marker once it is kept
 * whole in this site's private files (speech-worker.js). */
export async function speechModelCached() {
  try {
    const root = await navigator.storage.getDirectory();
    const folder = await root.getDirectoryHandle("slideseek-speech");
    await folder.getFileHandle(`${SPEECH.revision}.ready`);
    return true;
  } catch {
    return false;
  }
}

/** Let the visitor choose a recording. Resolves to { file, handle } or null. */
export async function chooseRecording(input) {
  if (typeof window.showOpenFilePicker === "function") {
    try {
      const [handle] = await window.showOpenFilePicker({
        id: "slideseek-lectures",
        types: [{ description: "Lecture recordings", accept: { "video/*": VIDEO_TYPES } }],
      });
      return { file: await handle.getFile(), handle };
    } catch (error) {
      if (error && error.name === "AbortError") return null;
      // Anything else (a policy, a frame): fall back to the ordinary picker below.
    }
  }
  return new Promise((resolve) => {
    const done = () => {
      input.removeEventListener("change", done);
      const file = input.files && input.files[0];
      input.value = "";
      resolve(file ? { file, handle: null } : null);
    };
    input.addEventListener("change", done);
    input.click();
  });
}

export function isRecording(file) {
  const name = (file.name || "").toLowerCase();
  return (file.type || "").startsWith("video/") || VIDEO_TYPES.some((ext) => name.endsWith(ext));
}

/** How long a recording is, if this browser can play it. */
export async function lengthOf(file) {
  const source = await openVideo(file);
  try {
    return { duration: source.duration, width: source.width, height: source.height };
  } finally {
    source.close();
  }
}

export function titleFrom(name) {
  const stem = name.replace(/\.[A-Za-z0-9]{2,4}$/, "");
  const words = stem.replace(/[_]+/g, " ").replace(/\s+/g, " ").trim();
  return (words || "My lecture").slice(0, 120);
}

/* ------------------------------------------------------------ IndexedDB */

const DB_NAME = "slideseek";
const STORE = "lectures";

function openStore() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "lecture_id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore(mode, work) {
  const db = await openStore();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const result = work(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(result && "result" in result ? result.result : undefined);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

export const remembered = {
  get: (lectureId) => withStore("readonly", (store) => store.get(lectureId)),
  all: () => withStore("readonly", (store) => store.getAll()),
  put: (record) => withStore("readwrite", (store) => store.put(record)),
  forget: (lectureId) => withStore("readwrite", (store) => store.delete(lectureId)),
  async update(lectureId, changes) {
    const record = (await remembered.get(lectureId)) || { lecture_id: lectureId };
    const next = { ...record, ...changes };
    await remembered.put(next);
    return next;
  },
};

/* ------------------------------------------------------------ the server */

async function call(method, url, body, type) {
  const headers = { Accept: "application/json" };
  let payload = body;
  if (body !== undefined && type === undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  } else if (type) {
    headers["Content-Type"] = type;
  }
  const response = await fetch(url, { method, headers, body: payload });
  const data = response.status === 204 ? {} : await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Slideseek could not be reached. Try again.";
    throw Object.assign(new Error(detail), { status: response.status });
  }
  return data;
}

const lectureUrl = (id) => `/api/library/lectures/${encodeURIComponent(id)}`;

export const library = {
  list: () => call("GET", "/api/library"),
  get: async (id) => (await call("GET", lectureUrl(id))).lecture,
  rename: async (id, title) => (await call("PATCH", lectureUrl(id), { title })).lecture,
  async remove(id) {
    await call("DELETE", lectureUrl(id));
    await remembered.forget(id).catch(() => {});
  },
};

async function gzip(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).blob();
}

/** Words into phrases of a sentence or so, as Whisper's own segments hold them. */
export function toSegments(words) {
  const segments = [];
  let current = [];
  const flush = () => {
    if (current.length) segments.push({ words: current });
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

/* ------------------------------------------------------------ listening */

function listen(audio, from, { onDownload, onLoaded, onBlock, signal }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./speech-worker.js", import.meta.url), { type: "module" });
    const stop = () => {
      worker.terminate();
      reject(new DOMException("Stopped.", "AbortError"));
    };
    signal?.addEventListener("abort", stop, { once: true });
    const finish = (fn, value) => {
      signal?.removeEventListener("abort", stop);
      worker.terminate();
      fn(value);
    };
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "download") onDownload?.(data);
      else if (data.type === "loaded") {
        onLoaded?.();
        worker.postMessage(
          { type: "transcribe", audio, blockSeconds: BLOCK_SECONDS, padSeconds: 5, from },
          [audio.buffer]
        );
      } else if (data.type === "block") {
        Promise.resolve(onBlock(data)).catch((error) => finish(reject, error));
      } else if (data.type === "done") finish(resolve);
      else if (data.type === "error") finish(reject, new Error(`Speech recognition stopped: ${data.message}`));
    });
    worker.addEventListener("error", (event) =>
      finish(reject, new Error(event.message || "Speech recognition could not start in this browser."))
    );
    const { model, revision, dtype, device } = SPEECH;
    worker.postMessage({ type: "load", model, revision, dtype, device });
  });
}

/* ------------------------------------------------------------ adding one lecture */

/**
 * Add a lecture, or carry on adding one. Resolves to the lecture once it is ready.
 *
 *   file      the recording (a File)
 *   handle    its FileSystemFileHandle, if there is one, to play it from later
 *   title     what to call it (a new lecture only)
 *   lecture   the lecture id, to carry on with one already made
 *   owner     the username, so one browser keeps different people's lectures apart
 *   report    called as report(step, details) with step one of "made" (the lecture has
 *             an id), "frames", "slides", "sound", "model", "speech", "building", "ready"
 */
export async function addLecture({ file, handle = null, title, lecture, owner, report = () => {}, signal }) {
  const check = () => signal?.throwIfAborted();
  // Half a gigabyte of model is worth keeping: ask the browser not to clear it on its own.
  navigator.storage?.persist?.().catch(() => {});
  let id = lecture;
  if (!id) {
    const { duration } = await lengthOf(file);
    const made = await call("POST", "/api/library/lectures", {
      title,
      duration_s: duration,
      file: { name: file.name, size: file.size, type: file.type },
    });
    id = made.lecture.lecture_id;
    report("made", { lecture_id: id });
    await remembered.put({
      lecture_id: id,
      owner,
      handle,
      file: { name: file.name, size: file.size, lastModified: file.lastModified },
      blocks: {},
      count: 0,
    });
  } else if (handle) {
    await remembered.update(id, { handle, owner });
  }
  let progress = await library.get(id);
  let record = (await remembered.get(id)) || (await remembered.update(id, { owner, blocks: {} }));

  // 1. The frames: sampled here, then the server finds the slide changes in them.
  if (!progress.keyframe_times.length) {
    let frames = record.frames;
    if (!frames) {
      report("frames", { fraction: 0 });
      const source = await openVideo(file);
      try {
        const sampled = await sampleFrames(source, {
          signal,
          onProgress: (fraction) => report("frames", { fraction }),
        });
        frames = await gzip(sampled.frames);
      } finally {
        source.close();
      }
      record = await remembered.update(id, { frames });
    }
    check();
    report("frames", { fraction: 1, sending: true });
    progress = (await call("PUT", `${lectureUrl(id)}/frames`, frames, "application/gzip")).lecture;
    record = await remembered.update(id, { frames: null });
    progress = await library.get(id);
  }

  // 2. A still of each slide, at the times the server asked for.
  const received = new Set(progress.received);
  const wanted = progress.keyframe_times.filter((t) => !received.has(Math.round(t * 1000)));
  if (wanted.length) {
    const total = progress.keyframe_times.length;
    let sent = total - wanted.length;
    report("slides", { sent, total });
    const source = await openVideo(file);
    try {
      for (const t of wanted) {
        check();
        const [still] = await grabStills(source, [t], { signal });
        await call("PUT", `${lectureUrl(id)}/stills/${String(Math.round(t * 1000)).padStart(9, "0")}.jpg`, still.blob, "image/jpeg");
        sent += 1;
        report("slides", { sent, total });
      }
    } finally {
      source.close();
    }
  }

  // 3. What was said: five minutes at a time, each block kept as soon as it is heard.
  if (!(await library.get(id)).transcript) {
    report("sound", {});
    const audio = await decodeAudio(file);
    check();
    const count = Math.max(1, Math.ceil(audio.length / (BLOCK_SECONDS * 16000)));
    const blocks = { ...(record.blocks || {}) };
    let from = 0;
    while (from < count && blocks[from]) from += 1;
    if (from < count) {
      let started = performance.now();
      let heard = 0;
      await listen(audio, from, {
        signal,
        onDownload: ({ loaded, total, file: name }) => report("model", { loaded, total, file: name }),
        onLoaded() {
          started = performance.now(); // time left counts listening, not the download
          report("speech", { done: from, count, left: null });
        },
        async onBlock({ index, words }) {
          blocks[index] = words;
          await remembered.update(id, { blocks, count });
          heard += 1;
          const perBlock = (performance.now() - started) / heard;
          report("speech", { done: index + 1, count, left: (count - index - 1) * perBlock / 1000 });
        },
      });
    }
    check();
    const words = [];
    for (let index = 0; index < count; index += 1) words.push(...(blocks[index] || []));
    await call("PUT", `${lectureUrl(id)}/transcript`, { model: SPEECH.label, segments: toSegments(words) });
    await remembered.update(id, { blocks: {}, count: 0 });
  }

  // 4. The server reads the slides and builds the search; wait for it.
  for (;;) {
    check();
    progress = await library.get(id);
    if (progress.stage === "ready") break;
    if (progress.stage === "failed") throw new Error(progress.error || "This lecture could not be finished.");
    report("building", {});
    await new Promise((resolve) => setTimeout(resolve, 2500));
  }
  report("ready", { lecture: progress });
  return progress;
}

/* ------------------------------------------------------------ playing it back */

const chosen = new Map(); // lecture id -> File, for this visit

/** Is `file` the one a lecture was made from? Same name and size. */
export function sameFile(file, meta) {
  return Boolean(file && meta && file.name === meta.name && file.size === meta.size);
}

/**
 * The recording of one of the visitor's own lectures, if it can be had without asking them
 * to choose it: picked earlier in this visit, or opened from the file handle kept for it
 * (which may ask the browser's permission). Resolves to a File or null.
 */
export async function recordingOf(lectureId, meta) {
  if (sameFile(chosen.get(lectureId), meta)) return chosen.get(lectureId);
  let record = null;
  try {
    record = await remembered.get(lectureId);
  } catch {
    record = null;
  }
  const handle = record && record.handle;
  if (!handle || typeof handle.getFile !== "function") return null;
  try {
    let permission = await handle.queryPermission({ mode: "read" });
    if (permission === "prompt") permission = await handle.requestPermission({ mode: "read" });
    if (permission !== "granted") return null;
    const file = await handle.getFile();
    if (!sameFile(file, meta)) return null;
    chosen.set(lectureId, file);
    return file;
  } catch {
    return null;
  }
}

/** Remember the file the visitor picked to play a lecture, if it is the right one. */
export function rememberChoice(lectureId, file, meta, handle = null) {
  if (!sameFile(file, meta)) return false;
  chosen.set(lectureId, file);
  if (handle) remembered.update(lectureId, { handle }).catch(() => {});
  return true;
}
