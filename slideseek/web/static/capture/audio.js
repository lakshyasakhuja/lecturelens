/* The sound of a recording, as speech recognition wants it: mono, 16 kHz, float32.
 *
 * The browser's own decoder reads the audio track of any file it can play, and an
 * OfflineAudioContext at 16 kHz resamples it. This is ingest.read_audio, without ffmpeg.
 * It holds the whole track in memory: an hour is about 230 MB, plus the file itself while
 * it is being decoded.
 */

export const SAMPLE_RATE = 16000;

export async function decodeAudio(file, { sampleRate = SAMPLE_RATE } = {}) {
  const bytes = await file.arrayBuffer();
  // Any length will do: decodeAudioData resamples to the context's rate whatever it is.
  const context = new OfflineAudioContext(1, sampleRate, sampleRate);
  let decoded;
  try {
    decoded = await context.decodeAudioData(bytes);
  } catch {
    throw new Error("This browser cannot read the sound in that file.");
  }
  const channels = decoded.numberOfChannels;
  const samples = new Float32Array(decoded.length);
  for (let c = 0; c < channels; c += 1) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < data.length; i += 1) samples[i] += data[i] / channels;
  }
  return samples;
}
