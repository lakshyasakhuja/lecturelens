/* Slideseek frontend.
 *
 * Plain JavaScript and no build step. Everything shown comes from three endpoints:
 * /api/info (what the index contains), /api/search (the moments for a question) and
 * /api/search/picture (the moments for a picture of a slide). Where the server offers it,
 * one more call reranks the results on a GPU; see "reranking" below.
 * Text from the server is only ever inserted as text nodes, never as HTML.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const form = $("search");
  const input = $("query");
  const lectureSelect = $("lecture");
  const rerankOption = $("rerank-option");
  const rerankToggle = $("rerank");
  const examples = $("examples");
  const resultsSection = $("results");
  const grid = $("results-grid");
  const resultsTitle = $("results-title");
  const status = $("status");
  const empty = $("empty");
  const template = $("result-template");
  const player = $("player");
  const playerFrame = $("player-frame");
  const read = $("read");
  const pictureInput = $("picture-input");
  const pictureButton = $("picture-button");
  const pictureChip = $("picture-chip");
  const picturePreview = $("picture-preview");
  const pictureClear = $("picture-clear");

  const BADGES = {
    said: { label: "Said", kind: "said" },
    "on slide": { label: "On slide", kind: "slide" },
    visual: { label: "Visual", kind: "visual" },
  };
  const RANKERS = { bm25: "keyword", dense: "meaning", image: "picture", bm25_slide: "slide text" };
  const PICTURE_VARIANTS = { photo: "phone photos", frame: "screenshots", crop: "crops" };
  const QUERY_KINDS = { keyword: "Keyword", paraphrase: "Paraphrase", spoken: "Spoken", visual: "Visual" };
  const SVG_NS = "http://www.w3.org/2000/svg";

  let info = null;
  let inFlight = null;
  let picture = null; // { file, url } while a picture is the query

  /* ------------------------------------------------------------ helpers */

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function icon(pathData, viewBox = "0 0 12 12") {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", viewBox);
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", pathData);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.5");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.append(path);
    return svg;
  }

  const chevron = () => icon("m4.5 2.5 3.5 3.5-3.5 3.5");
  const arrowOut = () => icon("M3.5 8.5 8.5 3.5M4.5 3.5h4v4");

  async function getJSON(url, signal) {
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : "The search could not be completed.";
      throw new Error(detail);
    }
    return body;
  }

  function duration(seconds) {
    const minutes = Math.round(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (!hours) return `${minutes} min`;
    return `${hours} h ${String(minutes % 60).padStart(2, "0")} min`;
  }

  function plural(count, word) {
    return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
  }

  /* Write `text` into `node`, wrapping the given character spans in highlights. */
  function highlight(node, text, spans) {
    node.textContent = "";
    let cursor = 0;
    for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0])) {
      if (start < cursor || end > text.length) continue;
      if (start > cursor) node.append(text.slice(cursor, start));
      node.append(element("span", "hit", text.slice(start, end)));
      cursor = end;
    }
    if (cursor < text.length) node.append(text.slice(cursor));
  }

  /* ------------------------------------------------------------ library info */

  function renderStats() {
    $("stat-lectures").textContent = info.n_lectures.toLocaleString();
    const hours = info.total_duration_s / 3600;
    $("stat-hours").textContent = hours >= 10 ? Math.round(hours).toString() : hours.toFixed(1);
    $("stat-moments").textContent = info.n_moments.toLocaleString();
    $("stat-slides").textContent = info.n_keyframes.toLocaleString();
  }

  function renderLectureFilter() {
    for (const lecture of info.lectures) {
      const option = element("option", "", lecture.title);
      option.value = lecture.lecture_id;
      lectureSelect.append(option);
    }
    if (info.lectures.length < 2) lectureSelect.closest(".select").hidden = true;
  }

  function renderExamples() {
    if (!info.examples.length) return;
    for (const example of info.examples) {
      const chip = element("button", "chip");
      chip.type = "button";
      const kind = QUERY_KINDS[example.type];
      if (kind) chip.append(element("span", "chip__kind", kind));
      chip.append(example.query);
      chip.addEventListener("click", () => {
        input.value = example.query;
        search(example.query, { push: true, scroll: true });
      });
      examples.append(chip);
    }
    examples.hidden = false;
  }

  /* One sentence crediting each course: who made it, and under which licence. */
  function attributionText() {
    const courses = new Map();
    for (const lecture of info.lectures) {
      if (!lecture.course) continue;
      const key = `${lecture.course}|${lecture.instructors}|${lecture.licence}`;
      courses.set(key, lecture);
    }
    return [...courses.values()].map((lecture) => {
      const by = lecture.instructors ? ` by ${lecture.instructors}` : "";
      const licence = lecture.licence ? `, licensed ${lecture.licence}` : "";
      return `Lectures from ${lecture.course}${by}${licence}.`;
    });
  }

  function renderLibrary() {
    const list = $("library-list");
    for (const lecture of info.lectures) {
      const row = element("li", "lecture");

      const words = element("div");
      words.append(element("h3", "lecture__title", lecture.title));
      const facts = [duration(lecture.duration_s), plural(lecture.n_moments, "moment"), plural(lecture.n_slides, "slide")];
      words.append(element("p", "lecture__meta", facts.join(" · ")));

      const actions = element("div", "lecture__actions");
      const only = element("button", "link", "Search this lecture");
      only.type = "button";
      only.append(chevron());
      only.addEventListener("click", () => {
        lectureSelect.value = lecture.lecture_id;
        input.focus();
        form.scrollIntoView({ behavior: "smooth", block: "center" });
        if (input.value.trim()) search(input.value, { push: true });
      });
      actions.append(only);
      if (lecture.source_url) {
        const source = element("a", "link link--quiet", "View at the source");
        source.href = lecture.source_url;
        source.target = "_blank";
        source.rel = "noopener";
        source.append(arrowOut());
        actions.append(source);
      }

      row.append(words, actions);
      list.append(row);
    }

    const credits = attributionText();
    $("library-lead").textContent =
      `${plural(info.n_lectures, "lecture")}, ${duration(info.total_duration_s)} of video, ` +
      `cut into ${plural(info.n_moments, "searchable moment")}.`;
    if (credits.length) {
      $("attribution").textContent =
        `${credits.join(" ")} The transcripts, slide text and thumbnails shown on this page are derived ` +
        "from those lectures and shared under the same licence, for non-commercial use.";
      $("footer-attribution").textContent = credits.join(" ");
    }
  }

  function renderBenchmark() {
    const benchmark = info.benchmark;
    if (!benchmark || !benchmark.systems || !benchmark.systems.length) return;
    const kinds = Object.keys(benchmark.systems[0].by_type);
    const percent = (value) => (value === null || value === undefined ? "–" : (100 * value).toFixed(1));

    const table = element("table", "table");
    const head = element("tr");
    for (const name of ["System", "Recall@1", "Recall@5", "MRR@10", ...kinds.map((k) => QUERY_KINDS[k] || k)]) {
      head.append(element("th", "", name));
    }
    const thead = element("thead");
    thead.append(head);
    const tbody = element("tbody");
    for (const system of benchmark.systems) {
      const row = element("tr", system.label.startsWith("Hybrid RRF") ? "is-headline" : "");
      const name = element("td");
      name.append(element("span", "table__num", String(system.number)), system.label);
      row.append(name);
      for (const value of [system.r1, system.r5, system.mrr10, ...kinds.map((k) => system.by_type[k])]) {
        row.append(element("td", "", percent(value)));
      }
      tbody.append(row);
    }
    table.append(thead, tbody);
    const wrap = element("div", "table-wrap");
    wrap.append(table);

    const parts = [
      `${benchmark.n_queries} ${benchmark.split} questions.`,
      `A result counts if it overlaps the answer within ±${benchmark.tolerance_s} s.`,
      "All figures are percentages; the columns on the right are Recall@5 by kind of question.",
    ];
    if (benchmark.latency) {
      parts.push(
        `Query latency on ${benchmark.latency.device}: ${Math.round(benchmark.latency.p50_ms)} ms at the median, ` +
          `${Math.round(benchmark.latency.p95_ms)} ms at the 95th percentile.`
      );
    }
    const body = $("benchmark-body");
    body.textContent = "";
    body.append(wrap, element("p", "benchmark__note", parts.join(" ")));
  }

  /* ------------------------------------------------------------ search */

  function showSkeletons() {
    grid.textContent = "";
    for (let i = 0; i < 3; i += 1) grid.append(element("div", "skeleton"));
  }

  function renderResult(result) {
    const card = template.content.firstElementChild.cloneNode(true);
    const media = card.querySelector(".result__media");
    const thumb = card.querySelector(".result__thumb");
    const playable = Boolean(result.player);

    if (result.thumb) {
      thumb.src = result.thumb;
      thumb.alt = `Slide shown at ${result.timestamp} in ${result.lecture_title}`;
      thumb.addEventListener("error", () => media.classList.add("is-empty"), { once: true });
    } else {
      media.classList.add("is-empty");
    }
    card.querySelector(".result__time").textContent = result.timestamp;
    media.disabled = !playable;
    media.setAttribute("aria-label", `Play ${result.lecture_title} from ${result.timestamp}`);

    card.querySelector(".result__lecture").textContent = result.lecture_title;
    const text = result.snippet || result.slide_text || "No speech or slide text in this moment.";
    highlight(card.querySelector(".result__snippet"), text, result.snippet ? result.highlights : []);

    const badges = card.querySelector(".badges");
    for (const name of result.badges) {
      const badge = BADGES[name];
      if (badge) badges.append(element("li", `badge badge--${badge.kind}`, badge.label));
    }
    if (!badges.children.length) badges.hidden = true;

    const open = card.querySelector(".result__open");
    card.querySelector(".result__open-label").textContent = `Play from ${result.timestamp}`;
    open.hidden = !playable;
    const external = card.querySelector(".result__external");
    if (result.video_link) external.href = result.video_link;
    else external.hidden = true;

    const ranked = Object.entries(result.signals)
      .filter(([name]) => RANKERS[name])
      .map(([name, rank]) => `#${rank} by ${RANKERS[name]}`);
    const signals = card.querySelector(".result__signals");
    if (ranked.length) signals.textContent = `Ranked ${ranked.join(", ")}`;
    else signals.hidden = true;

    if (playable) {
      const play = () => openPlayer(result);
      media.addEventListener("click", play);
      open.addEventListener("click", play);
    }
    return card;
  }

  function renderResults(data) {
    grid.textContent = "";
    empty.hidden = data.results.length > 0;
    for (const result of data.results) grid.append(renderResult(result));

    const where = lectureSelect.value ? ` in ${lectureSelect.selectedOptions[0].textContent}` : "";
    const reranked = data.reranked ? ` · reranked in ${(data.rerank_ms / 1000).toFixed(1)} s` : "";
    const took = data.took_ms >= 1000 ? `${(data.took_ms / 1000).toFixed(1)} s` : `${Math.round(data.took_ms)} ms`;
    status.textContent = `${plural(data.count, "moment")}${where} · ${took}${reranked}`;
    if (data.kind === "picture") {
      read.textContent = data.read
        ? `Matched by how it looks, and by the words read off it: “${data.read}”`
        : "Matched by how it looks. No words could be read off it.";
      read.hidden = false;
      return;
    }
    read.hidden = true;
    // A search that ran while the models were still loading says nothing about speed.
    if (data.warm) {
      $("stat-latency").textContent = `${Math.round(data.took_ms)} ms`;
      $("stat-latency-label").textContent = "this search, encoders included";
    }
  }

  /* ------------------------------------------------------------ reranking
   *
   * Offered only where the server says it can (a Hugging Face Space on ZeroGPU, which lends
   * a GPU for the length of one call). The results of the ordinary search are shown first
   * and replaced when the reranked order arrives; if it never does, they simply stay.
   *
   * The call goes through Gradio's own client rather than fetch. On a Space that client
   * asks the surrounding huggingface.co page who the visitor is, which is how the GPU time
   * is counted against the visitor's own free allowance.
   */
  let gradioClient = null;

  function connectGradio() {
    if (!gradioClient) {
      gradioClient = import("/vendor/gradio-client.js")
        .then(({ Client }) => Client.connect(location.origin))
        .catch((error) => {
          gradioClient = null;
          throw error;
        });
    }
    return gradioClient;
  }

  async function refine(query, lecture, count, signal) {
    const firstStage = status.textContent;
    status.textContent = `${firstStage} · reranking on a GPU`;
    try {
      const client = await connectGradio();
      const reply = await client.predict("/rerank", { query, k: count, lectures: lecture ? [lecture] : null });
      if (signal.aborted) return;
      renderResults(reply.data[0]);
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `${firstStage} · not reranked: no GPU was free, or today's free GPU time is used up`;
    }
  }

  async function search(rawQuery, { push = false, scroll = false } = {}) {
    const query = rawQuery.trim().replace(/\s+/g, " ");
    if (!query) return;
    clearPicture();

    if (inFlight) inFlight.abort();
    inFlight = new AbortController();
    const { signal } = inFlight;

    const lecture = lectureSelect.value;
    const params = new URLSearchParams({ q: query });
    if (lecture) params.set("lecture", lecture);
    if (push) history.pushState({ query, lecture }, "", `?${params}`);

    resultsSection.hidden = false;
    resultsTitle.textContent = `“${query}”`;
    status.textContent = info && !info.ready ? "Warming up the models. The first search takes a few seconds." : "Searching";
    empty.hidden = true;
    showSkeletons();
    form.classList.add("is-busy");
    if (scroll) resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
      const data = await getJSON(`/api/search?${params}`, signal);
      if (info) info.ready = true;
      renderResults(data);
      // Reordering one result is not worth a GPU.
      if (rerankToggle.checked && !rerankOption.hidden && data.count > 1) {
        await refine(query, lecture, data.count, signal);
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      grid.textContent = "";
      status.textContent = error.message || "Something went wrong. Try again.";
    } finally {
      if (!signal.aborted) form.classList.remove("is-busy");
    }
  }

  /* ------------------------------------------------------------ pictures
   *
   * A picture is sent as the body of one POST and never kept: the server searches with it
   * and forgets it. The page shows it back to the visitor from memory (a blob: URL).
   */

  function showPicture(file) {
    if (picture) URL.revokeObjectURL(picture.url);
    picture = { file, url: URL.createObjectURL(file) };
    picturePreview.src = picture.url;
    pictureChip.hidden = false;
    form.classList.add("has-picture");
    input.value = "";
  }

  function clearPicture() {
    if (!picture) return;
    URL.revokeObjectURL(picture.url);
    picture = null;
    picturePreview.removeAttribute("src");
    pictureChip.hidden = true;
    form.classList.remove("has-picture");
    pictureInput.value = "";
  }

  function pictureProblem(file) {
    if (!file || !file.type.startsWith("image/")) return "That file is not a picture.";
    const limit = info && info.pictures ? info.pictures.max_bytes : 8 * 1024 * 1024;
    if (file.size > limit) return `The picture is larger than ${Math.round(limit / 1024 / 1024)} MB.`;
    return null;
  }

  async function searchPicture(file, { scroll = true } = {}) {
    const problem = pictureProblem(file);
    resultsSection.hidden = false;
    if (problem) {
      resultsTitle.textContent = "Your picture";
      grid.textContent = "";
      read.hidden = true;
      status.textContent = problem;
      return;
    }
    if (inFlight) inFlight.abort();
    inFlight = new AbortController();
    const { signal } = inFlight;

    showPicture(file);
    // A picture cannot live in the address bar; leave no stale question there either.
    if (location.search) history.pushState({ picture: true }, "", location.pathname);
    const lecture = lectureSelect.value;
    const params = new URLSearchParams({ k: "6" });
    if (lecture) params.set("lecture", lecture);

    resultsTitle.textContent = "Your picture";
    status.textContent = info && !info.ready ? "Warming up the models. The first search takes a few seconds." : "Looking at your picture";
    read.hidden = true;
    empty.hidden = true;
    showSkeletons();
    form.classList.add("is-busy");
    if (scroll) resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
      const response = await fetch(`/api/search/picture?${params}`, {
        method: "POST",
        body: file,
        headers: { "Content-Type": file.type, Accept: "application/json" },
        signal,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The picture could not be searched.");
      if (info) info.ready = true;
      renderResults(data);
    } catch (error) {
      if (error.name === "AbortError") return;
      grid.textContent = "";
      status.textContent = error.message || "Something went wrong. Try again.";
    } finally {
      if (!signal.aborted) form.classList.remove("is-busy");
    }
  }

  function imageFrom(items) {
    for (const item of items || []) {
      const file = item.kind === "file" ? item.getAsFile() : item;
      if (file && file.type && file.type.startsWith("image/")) return file;
    }
    return null;
  }

  pictureButton.addEventListener("click", () => pictureInput.click());
  pictureInput.addEventListener("change", () => {
    const file = pictureInput.files && pictureInput.files[0];
    if (file) searchPicture(file);
  });
  pictureClear.addEventListener("click", () => {
    clearPicture();
    input.focus();
  });

  // Paste a screenshot anywhere on the page. Text pasted into a field is left alone.
  document.addEventListener("paste", (event) => {
    const file = imageFrom(event.clipboardData && event.clipboardData.items);
    if (!file) return;
    event.preventDefault();
    searchPicture(file);
  });

  // Drop a picture anywhere; the search box lights up while one is dragged over the page.
  let dragDepth = 0;
  const carriesFiles = (event) => event.dataTransfer && [...event.dataTransfer.types].includes("Files");
  document.addEventListener("dragenter", (event) => {
    if (!carriesFiles(event)) return;
    dragDepth += 1;
    form.classList.add("is-dropping");
  });
  document.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) form.classList.remove("is-dropping");
  });
  document.addEventListener("dragover", (event) => {
    if (carriesFiles(event)) event.preventDefault();
  });
  document.addEventListener("drop", (event) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    form.classList.remove("is-dropping");
    const file = imageFrom(event.dataTransfer.files);
    if (file) searchPicture(file);
    else {
      resultsSection.hidden = false;
      resultsTitle.textContent = "Your picture";
      status.textContent = "That file is not a picture.";
    }
  });

  $("picture-example").addEventListener("click", async () => {
    const response = await fetch("/samples/slide-photo.jpg");
    if (!response.ok) return;
    const blob = await response.blob();
    searchPicture(new File([blob], "slide-photo.jpg", { type: blob.type || "image/jpeg" }));
  });

  /* Measured figures for each kind of picture, if the server has them. */
  function renderPictureBenchmark() {
    const measured = info.pictures && info.pictures.benchmark;
    const both = measured && measured.methods && measured.methods.both;
    for (const node of document.querySelectorAll("[data-variant]")) {
      const figures = both && both[node.dataset.variant];
      const label = document.querySelector(`[data-variant-label="${node.dataset.variant}"]`);
      if (!figures) {
        node.hidden = true;
        label.hidden = true;
        continue;
      }
      node.textContent = `${Math.round(100 * figures["hit@1"])}%`;
      label.textContent =
        `of ${PICTURE_VARIANTS[node.dataset.variant]} found the slide first, ` +
        `${Math.round(100 * figures["hit@5"])}% in the top five`;
    }
    if (measured) {
      $("picture-note").textContent =
        `Measured on ${measured.queries.toLocaleString()} pictures made from frames of ${plural(measured.lectures.length, "lecture")}, ` +
        "never the frame that was indexed. “Found first” means the top result shows that slide.";
    }
  }

  /* ------------------------------------------------------------ player */

  function openPlayer(result) {
    playerFrame.textContent = "";
    const source = result.player;
    if (source.kind === "youtube") {
      const frame = element("iframe");
      const start = encodeURIComponent(source.start);
      frame.src = `https://www.youtube-nocookie.com/embed/${encodeURIComponent(source.video_id)}?start=${start}&autoplay=1&rel=0`;
      frame.title = `${result.lecture_title}, from ${result.timestamp}`;
      frame.allow = "autoplay; encrypted-media; picture-in-picture; fullscreen";
      frame.referrerPolicy = "strict-origin-when-cross-origin";
      playerFrame.append(frame);
    } else {
      const video = element("video");
      video.src = `${source.src}#t=${source.start}`;
      video.controls = true;
      video.autoplay = true;
      video.playsInline = true;
      playerFrame.append(video);
    }
    $("player-title").textContent = `${result.lecture_title} · ${result.timestamp}`;
    $("player-text").textContent = result.snippet || result.slide_text || "";
    const external = $("player-external");
    external.hidden = !result.video_link;
    if (result.video_link) external.href = result.video_link;
    player.showModal();
  }

  // Emptying the frame is what stops the video.
  player.addEventListener("close", () => {
    playerFrame.textContent = "";
  });
  player.addEventListener("click", (event) => {
    if (event.target === player) player.close();
  });

  /* ------------------------------------------------------------ page behaviour */

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!input.value.trim() && picture) {
      searchPicture(picture.file, { scroll: true });
      return;
    }
    search(input.value, { push: true, scroll: true });
    input.blur();
  });

  lectureSelect.addEventListener("change", () => {
    if (picture) searchPicture(picture.file, { scroll: false });
    else if (input.value.trim()) search(input.value, { push: true });
  });

  rerankToggle.addEventListener("change", () => {
    if (input.value.trim()) search(input.value);
  });

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !player.open) {
      event.preventDefault();
      input.focus();
      input.select();
    }
  });

  function restoreFromUrl({ fallback = false } = {}) {
    const params = new URLSearchParams(location.search);
    const query = params.get("q");
    const lecture = params.get("lecture") || "";
    lectureSelect.value = [...lectureSelect.options].some((o) => o.value === lecture) ? lecture : "";
    if (query) {
      input.value = query;
      search(query);
    } else if (fallback && info && info.examples.length) {
      // Show the page doing its job straight away, with the first example.
      input.value = info.examples[0].query;
      search(info.examples[0].query);
    } else {
      resultsSection.hidden = true;
    }
  }

  window.addEventListener("popstate", () => restoreFromUrl());

  /* Scroll behaviour: the navigation bar's hairline and theme, and sections fading in.
   *
   * Reveals are driven by plain position checks rather than an IntersectionObserver, and
   * everything is shown at once if the page loads in the background. Content must never
   * depend on an animation callback arriving in order to become visible.
   */
  const nav = document.querySelector(".nav");
  const darkSections = [...document.querySelectorAll(".section--dark")];
  let hidden = [...document.querySelectorAll(".reveal")];

  function revealInView() {
    if (!hidden.length) return;
    const limit = window.innerHeight * 0.94;
    hidden = hidden.filter((node) => {
      if (node.getBoundingClientRect().top < limit) {
        node.classList.add("is-visible");
        return false;
      }
      return true;
    });
  }

  function revealAll() {
    for (const node of hidden) node.classList.add("is-visible");
    hidden = [];
  }

  function onScroll() {
    nav.classList.toggle("is-scrolled", window.scrollY > 8);
    const edge = nav.offsetHeight;
    nav.classList.toggle(
      "nav--dark",
      darkSections.some((section) => {
        const box = section.getBoundingClientRect();
        return box.top <= edge && box.bottom > edge;
      })
    );
    revealInView();
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  if (document.visibilityState === "hidden" || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    revealAll();
  }
  onScroll();
  // A last resort for anything still waiting, for example if the layout shifted after load.
  setTimeout(revealInView, 1200);

  async function start() {
    try {
      info = await getJSON("/api/info");
    } catch (error) {
      status.textContent = "The library could not be loaded.";
      resultsSection.hidden = false;
      return;
    }
    rerankOption.hidden = !info.rerank;
    renderPictureBenchmark();
    renderStats();
    renderLectureFilter();
    renderExamples();
    renderLibrary();
    renderBenchmark();
    restoreFromUrl({ fallback: true });
  }

  start();
})();
