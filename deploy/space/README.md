---
title: LectureLens
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Search lecture recordings and jump to the moment
preload_from_hub:
  - Qwen/Qwen3-Embedding-0.6B
  - google/siglip2-base-patch16-224
  - Qwen/Qwen3-Reranker-0.6B
---

# LectureLens

Semantic search for lecture recordings: what was said, what was on the slide, and what the
slide looked like. Source code and documentation:
https://github.com/lakshyasakhuja/lecturelens

This Space runs the web app from that repository. Searching runs on the Space's CPU and
costs a visitor nothing. The optional "Rerank on a GPU" switch rescores the top results
with a cross-encoder on ZeroGPU, which uses a second or two of the visitor's free daily
GPU time. The search index is downloaded at start-up from the dataset repository named in
the `LECTURELENS_INDEX_REPO` variable.

The lectures indexed here come from MIT OpenCourseWare and are used under CC BY-NC-SA 4.0.
This Space hosts no lecture video; every result links to the recording at its source.
LectureLens is an independent student project and is not affiliated with or endorsed by MIT.
