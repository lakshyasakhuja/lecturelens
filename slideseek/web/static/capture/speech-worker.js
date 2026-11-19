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
    const words = mergeWordPieces(pieces)
      .filter((word) => {
        const mid = 0.5 * (word.s + word.e);
        return mid >= lo && (mid < hi || index === count - 1);
      })
      .map((word) => ({
        ...word,
        s: Math.round(word.s * 1000) / 1000,
        e: Math.round(word.e * 1000) / 1000,
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
