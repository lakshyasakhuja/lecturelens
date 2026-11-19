/* Reading pictures out of a recording the visitor picked, without sending it anywhere.
 *
 * A <video> element decodes the file, seeking to each time asked for, and a canvas reads
 * the picture back. This is what the indexer does with ffmpeg (slideseek/ingest.py):
 *
 *   sampleFrames  small greyscale frames, one a second: ingest.read_gray_frames
 *   grabStills    a full-size JPEG at each keyframe:     ingest.extract_still
 *
 * Seeking a <video> is slower than decoding in one pass, but it works for every file the
 * browser can play, in every browser, and needs nothing beyond the page.
 */

export const FRAME_WIDTH = 160;
export const FRAME_HEIGHT = 90;

/** Open a recording for reading. Call close() when done. */
export async function openVideo(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;
  video.src = url;
  await new Promise((resolve, reject) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
    video.addEventListener(
      "error",
      () => reject(new Error("This browser cannot play that file.")),
      { once: true },
    );
  });
  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    URL.revokeObjectURL(url);
    throw new Error("The recording has no length this browser can read.");
  }
  return {
    video,
    duration: video.duration,
    width: video.videoWidth,
    height: video.videoHeight,
    close() {
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
    },
  };
}

/** Show the frame at time t. Resolves once it can be drawn. */
export function seek(video, t) {
  return new Promise((resolve, reject) => {
    const done = () => {
      video.removeEventListener("error", failed);
      resolve();
    };
    const failed = () => {
      video.removeEventListener("seeked", done);
      reject(new Error(`Could not read the picture at ${t.toFixed(1)} s.`));
    };
    video.addEventListener("seeked", done, { once: true });
    video.addEventListener("error", failed, { once: true });
    video.currentTime = t;
  });
}

/**
 * Luma of an RGBA picture, full range, like ffmpeg's format=gray: on MIT 6.7960 lecture 4 the
 * two differ by a median of 2.7 grey levels a frame (docs/browser-indexing.md).
 */
export function toGray(rgba, out, offset = 0) {
  for (let i = 0, j = offset; i < rgba.length; i += 4, j += 1) {
    out[j] = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
  }
}

/**
 * Greyscale frames of FRAME_WIDTH x FRAME_HEIGHT, sampled at `fps`: frame i is the picture
 * at i / fps seconds. Returns { frames: Uint8Array of count * width * height, count }.
 */
export async function sampleFrames(source, { fps = 1, onProgress, signal } = {}) {
  const { video, duration } = source;
  const count = Math.max(1, Math.floor(duration * fps));
  const canvas = new OffscreenCanvas(FRAME_WIDTH, FRAME_HEIGHT);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  const size = FRAME_WIDTH * FRAME_HEIGHT;
  const frames = new Uint8Array(count * size);
  for (let i = 0; i < count; i += 1) {
    signal?.throwIfAborted();
    await seek(video, Math.min(i / fps, duration - 0.05));
    context.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
    toGray(context.getImageData(0, 0, FRAME_WIDTH, FRAME_HEIGHT).data, frames, i * size);
    if (onProgress && (i % 30 === 0 || i === count - 1)) onProgress((i + 1) / count);
  }
  return { frames, count };
}

/**
 * A JPEG of the picture at each time, at most `maxWidth` wide. A time at or past the end
 * steps back just inside it, as the indexer does. Returns [{ t, blob }] in time order.
 */
export async function grabStills(source, times, { maxWidth = 1920, quality = 0.92, onProgress, signal } = {}) {
  const { video, duration, width, height } = source;
  const scale = Math.min(1, maxWidth / width);
  const w = Math.round(width * scale);
  const h = Math.round((height * scale) / 2) * 2;
  const canvas = new OffscreenCanvas(w, h);
  const context = canvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  const sorted = [...new Set(times)].sort((a, b) => a - b);
  const stills = [];
  for (const [n, t] of sorted.entries()) {
    signal?.throwIfAborted();
    await seek(video, Math.min(t, Math.max(0, duration - 0.25)));
    context.drawImage(video, 0, 0, w, h);
    stills.push({ t, blob: await canvas.convertToBlob({ type: "image/jpeg", quality }) });
    onProgress?.((n + 1) / sorted.length);
  }
  return stills;
}
