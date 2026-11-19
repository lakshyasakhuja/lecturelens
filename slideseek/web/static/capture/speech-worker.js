/* Speech recognition in the visitor's browser: Whisper on WebGPU, through transformers.js.
 *
 * A module worker, so the page stays responsive while it runs. It produces what
 * slideseek/asr.py produces with faster-whisper: words with start and end times, joined from
 * Whisper's word pieces the same way (asr.merge_word_pieces).
 *
 * The audio is transcribed in blocks of a few minutes, each padded with a few seconds of
 * its neighbours; a block keeps the words whose midpoint falls inside it. That gives the page
 * progress it can show and a point to resume from, and the padding means no word is lost
 * or doubled at a block boundary. Inside a block, transformers.js cuts the audio into
 * 30-second windows that overlap by 5 seconds and joins them.
 *
 * The model's files are kept in the origin's private file system (``speechCache``), not in
 * the Cache API that transformers.js uses by default: Chrome refuses to put the 370 MB
 * encoder there ("Unexpected internal error"), and every lecture would download it again.
 * Once every file is kept, a marker named after the model's revision says so, for the page.
 *
 * Messages in:
 *   { type: "load", model, revision, dtype, device }
 *   { type: "transcribe", audio: Float32Array, blockSeconds, padSeconds, prompt, from }
 * Messages out:
 *   { type: "download", file, loaded, total }   while the model downloads (once; then cached)
 *   { type: "loaded", seconds }
 *   { type: "block", index, count, words, seconds }
 *   { type: "done", seconds }
 *   { type: "error", message }
 */

import { env, pipeline } from "../vendor/browser/transformers.min.js";

const SAMPLE_RATE = 16000;

env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = new URL("../vendor/browser/", import.meta.url).href;

export const CACHE_DIR = "slideseek-speech";

async function cacheName(key) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(String(key)));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** What transformers.js needs of a cache, over files on disk: match(key) and put(key, response). */
export const speechCache = {
  failures: 0,
  async folder() {
    return (await navigator.storage.getDirectory()).getDirectoryHandle(CACHE_DIR, { create: true });
  },
  async match(key) {
    try {
      const handle = await (await this.folder()).getFileHandle(await cacheName(key));
      const file = await handle.getFile();
      if (!file.size) return undefined;
      return new Response(file, { headers: { "content-length": String(file.size) } });
    } catch {
      return undefined;
    }
  },
  async put(key, response) {
    const folder = await this.folder();
    const name = await cacheName(key);
    const partial = await folder.getFileHandle(`${name}.partial`, { create: true });
    try {
      if (typeof partial.createWritable === "function") {
        await response.body.pipeTo(await partial.createWritable());
      } else {
        const access = await partial.createSyncAccessHandle(); // Safari, in a worker
        try {
          access.truncate(0);
          access.write(new Uint8Array(await response.arrayBuffer()), { at: 0 });
          access.flush();
        } finally {
          access.close();
        }
      }
      await partial.move(name);
    } catch (error) {
      this.failures += 1;
      console.warn(`Slideseek could not keep the speech model's file ${key}:`, error);
      await folder.removeEntry(`${name}.partial`).catch(() => {});
      throw error;
    }
  },
};

env.useBrowserCache = false;
env.useCustomCache = true;
env.customCache = speechCache;

let transcriber = null;

function post(message, transfer) {
  self.postMessage(message, transfer ?? []);
}

/** Whisper's pieces, joined into whitespace-delimited words: asr.merge_word_pieces. */
export function mergeWordPieces(pieces) {
  const words = [];
  for (const { text: raw, start, end } of pieces) {
    const text = raw.trim();
    if (!text) continue;
    if (words.length && !/^\s/.test(raw)) {
      const last = words[words.length - 1];
      words[words.length - 1] = { w: last.w + text, s: last.s, e: end, p: last.p };
    } else {
      words.push({ w: text, s: start, e: end, p: 1 });
    }
  }
  return words;
}

async function load({ model, revision, dtype, device = "webgpu" }) {
  const started = performance.now();
  speechCache.failures = 0;
  transcriber = await pipeline("automatic-speech-recognition", model, {
    revision,
    device,
    dtype,
    progress_callback(event) {
      if (event.status === "progress") {
        post({ type: "download", file: event.file, loaded: event.loaded, total: event.total });
      }
    },
  });
  if (!speechCache.failures) {
    // Every file is on disk: say so, for the page to tell the visitor nothing will download.
    const folder = await speechCache.folder();
    const marker = await folder.getFileHandle(`${revision}.ready`, { create: true });
    const writable = await marker.createWritable?.();
    if (writable) {
      await writable.write(model);
      await writable.close();
    }
  }
  post({ type: "loaded", seconds: (performance.now() - started) / 1000 });
}

function tokenId(tokenizer, token) {
  const id =
    tokenizer.convert_tokens_to_ids?.(token) ??
    tokenizer.model?.tokens_to_ids?.get?.(token) ??
    tokenizer.get_vocab?.()?.[token];
  if (id == null) throw new Error(`The tokenizer has no ${token}`);
  return Number(Array.isArray(id) ? id[0] : id);
}

/** The decoder's first tokens: a glossary as Whisper's "previous text", then the task. */
function promptTokens(prompt) {
  if (!prompt) return null;
  const tokenizer = transcriber.tokenizer;
  const text = tokenizer.encode(` ${prompt.trim()}`, { add_special_tokens: false }).map(Number);
  // Whisper's prompt holds at most 224 tokens, the previous text included.
  return [
    tokenId(tokenizer, "<|startofprev|>"),
    ...text.slice(-223),
    tokenId(tokenizer, "<|startoftranscript|>"),
    tokenId(tokenizer, "<|en|>"),
    tokenId(tokenizer, "<|transcribe|>"),
  ];
}

async function transcribe({ audio, blockSeconds = 300, padSeconds = 5, prompt = "", from = 0 }) {
  const started = performance.now();
  const block = Math.round(blockSeconds * SAMPLE_RATE);
  const pad = Math.round(padSeconds * SAMPLE_RATE);
  const count = Math.max(1, Math.ceil(audio.length / block));
  const decoder_input_ids = promptTokens(prompt);
  for (let index = from; index < count; index += 1) {
    const blockStarted = performance.now();
    const owned = [index * block, Math.min(audio.length, (index + 1) * block)];
    const first = Math.max(0, owned[0] - pad);
    const slice = audio.subarray(first, Math.min(audio.length, owned[1] + pad));
    const offset = first / SAMPLE_RATE;
    const output = await transcriber(slice, {
      language: "en",
      task: "transcribe",
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: "word",
      ...(decoder_input_ids ? { decoder_input_ids } : {}),
    });
    const pieces = (output.chunks ?? []).map((chunk) => ({
      text: chunk.text,
      start: offset + (chunk.timestamp?.[0] ?? 0),
      end: offset + (chunk.timestamp?.[1] ?? chunk.timestamp?.[0] ?? 0),
    }));
    const lo = owned[0] / SAMPLE_RATE;
    const hi = owned[1] / SAMPLE_RATE;
    // Whisper's times can run past the end of the sound (its last window is padded to 30 s)
    // and, now and then, a word ends before it starts: each word is kept inside the sound,
    // the right way round.
    const length = audio.length / SAMPLE_RATE;
    const clamp = (t) => Math.round(Math.min(Math.max(t, 0), length) * 1000) / 1000;
    const words = mergeWordPieces(pieces)
      .filter((word) => {
        const mid = 0.5 * (word.s + word.e);
        return mid >= lo && (mid < hi || index === count - 1);
      })
      .map((word) => ({
        ...word,
        s: clamp(Math.min(word.s, word.e)),
        e: clamp(Math.max(word.s, word.e)),
      }));
    post({ type: "block", index, count, words, seconds: (performance.now() - blockStarted) / 1000 });
  }
  post({ type: "done", seconds: (performance.now() - started) / 1000 });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "transcribe") await transcribe(data);
  } catch (error) {
    post({ type: "error", message: String(error?.message ?? error) });
  }
});
