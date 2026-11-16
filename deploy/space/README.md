---
title: LectureLens
emoji: 🔎
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Search lecture recordings and jump to the moment
---

# LectureLens

Semantic search for lecture recordings: what was said, what was on the slide, and what the
slide looked like. Source code and documentation:
https://github.com/lakshyasakhuja/lecturelens

This Space runs the web app from that repository on a CPU. The search index is downloaded
at start-up from the dataset repository named in the `LECTURELENS_INDEX_REPO` variable.

The lectures indexed here come from MIT OpenCourseWare and are used under CC BY-NC-SA 4.0.
This Space hosts no lecture video; every result links to the recording at its source.
LectureLens is an independent student project and is not affiliated with or endorsed by MIT.
