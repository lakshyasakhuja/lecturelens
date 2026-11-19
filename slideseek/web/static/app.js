/* Slideseek frontend.
 *
 * Plain JavaScript and no build step. Everything shown comes from three endpoints:
 * /api/info (what the index contains), /api/search (the moments for a question) and
 * /api/search/picture (the moments for a picture of a slide). Where the server offers it,
 * one more call reranks the results on a GPU; see "reranking" below.
 *
 * Searching needs an account, so the two search endpoints answer 401 until the visitor
 * signs in; see "accounts" below. Text from the server is only ever inserted as text
 * nodes, never as HTML.
 *
 * Where the server takes them, people can add lectures of their own; see "your own
 * lectures" below. The work of reading a recording is in /capture/adder.js, loaded only
 * when it is needed.
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
  const accountDialog = $("account");
  const accountButton = $("account-button");
  const gate = $("gate");
  const toast = $("toast");
  const scopeControl = $("scope");
  const adder = $("adder");
  const recordingInput = $("recording-input");

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
  let account = null; // the signed-in visitor, as /api/account describes them
  let ownLectures = []; // the visitor's own lectures, as /api/library lists them
  let scope = "all"; // which lectures a search covers: all, mine or public
  let adding = null; // the lecture being added from this page: { id, title, controller, summary }
  let choice = null; // the recording chosen in the sheet, not yet started: { file, handle, duration }

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
      throw Object.assign(new Error(detail), { status: response.status });
    }
    return body;
  }

  /* A JSON request that changes something. Resolves to { ok, status, data } either way. */
  async function sendJSON(method, url, body) {
    const response = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = response.status === 204 ? {} : await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, data };
  }

  let toastTimer = 0;
  function announce(message) {
    toast.textContent = message;
    toast.classList.add("is-shown");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("is-shown"), 3600);
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

  function megabytes(bytes) {
    const mb = bytes / 1_000_000;
    return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.max(1, Math.round(mb))} MB`;
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
    lectureSelect.closest(".select").hidden = lectureSelect.options.length < 3;
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

    let where = "";
    if (lectureSelect.value) where = ` in ${lectureSelect.selectedOptions[0].textContent}`;
    else if (scope === "mine") where = " in your lectures";
    else if (scope === "public") where = ` in the ${publicLabel()} lectures`;
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
      const reply = await client.predict("/rerank", { query, k: count, lectures: lecture ? [lecture] : null, scope });
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
    if (!account) {
      whenSignedIn(() => search(query, { push, scroll }), `Sign in to see the moments for “${query}”.`);
      return;
    }
    clearPicture();

    if (inFlight) inFlight.abort();
    inFlight = new AbortController();
    const { signal } = inFlight;

    const lecture = lectureSelect.value;
    const params = new URLSearchParams({ q: query });
    if (lecture) params.set("lecture", lecture);
    if (scope !== "all") params.set("scope", scope);
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
      if (error.status === 401) sessionEnded(() => search(query, { push: false, scroll }));
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
    if (!account) {
      whenSignedIn(() => searchPicture(file, { scroll }), "Sign in to search with your picture.");
      return;
    }
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
    if (scope !== "all") params.set("scope", scope);

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
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : "The picture could not be searched.";
        throw Object.assign(new Error(detail), { status: response.status });
      }
      if (info) info.ready = true;
      renderResults(data);
    } catch (error) {
      if (error.name === "AbortError") return;
      grid.textContent = "";
      status.textContent = error.message || "Something went wrong. Try again.";
      if (error.status === 401) sessionEnded(() => searchPicture(file, { scroll }));
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

  // Choosing a file right after signing in may be blocked by the browser; a second click works.
  pictureButton.addEventListener("click", () =>
    whenSignedIn(() => pictureInput.click(), "Sign in to search with a picture of a slide.")
  );
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
    if (accountDialog.open) return;
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
    if (accountDialog.open) return;
    const file = imageFrom(event.dataTransfer.files);
    if (file) {
      searchPicture(file);
      return;
    }
    // A recording is a lecture to add. The drop must be read now: it is gone after a wait.
    const files = [...event.dataTransfer.files];
    const recording = info && info.own_lectures ? files.find(isRecording) : null;
    if (recording) {
      const items = [...(event.dataTransfer.items || [])].filter((item) => item.kind === "file");
      const single = files.length === 1 && items.length === 1 && typeof items[0].getAsFileSystemHandle === "function";
      const handle = single ? items[0].getAsFileSystemHandle().catch(() => null) : Promise.resolve(null);
      handle.then((found) => openAdder({ file: recording, handle: found && found.kind === "file" ? found : null }));
      return;
    }
    resultsSection.hidden = false;
    resultsTitle.textContent = "Your picture";
    status.textContent = info && info.own_lectures ? "That file is not a picture or a recording." : "That file is not a picture.";
  });

  $("picture-example").addEventListener("click", async () => {
    if (!account) {
      whenSignedIn(() => $("picture-example").click(), "Sign in to try a photo of a slide.");
      return;
    }
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

  /* ------------------------------------------------------------ accounts
   *
   * The session is an HttpOnly cookie, which this script cannot read and never sees. It asks
   * /api/account who is signed in, and the server decides. The rules checked here as the
   * visitor types are the server's rules, repeated so the form can say what is missing
   * before it is sent; the server checks them again either way.
   */
  const views = {
    signin: { title: "Sign in to Slideseek", lead: "Sign in to search the lectures." },
    signup: { title: "Create your account", lead: "It’s free, and it is all you need to search every lecture." },
    profile: { title: "Your account", lead: "" },
  };
  const RULE_TESTS = {
    length: (text) => [...text].length >= 8,
    upper: (text) => /\p{Lu}/u.test(text),
    lower: (text) => /\p{Ll}/u.test(text),
    number: (text) => /\p{Nd}/u.test(text),
    special: (text) => /[\p{P}\p{S}]/u.test(text),
  };
  const USERNAME = /^[A-Za-z0-9][A-Za-z0-9._-]{2,29}$/;
  let pending = null; // what to do once the visitor is signed in
  let countries = null;

  function initials(name) {
    const words = name.trim().split(/\s+/).filter(Boolean);
    const first = (word) => (word ? [...word][0] : "");
    return (first(words[0]) + (words.length > 1 ? first(words[words.length - 1]) : "")).toUpperCase();
  }

  function setAccount(next) {
    account = next || null;
    accountButton.hidden = false;
    const avatar = $("account-avatar");
    const label = $("account-label");
    if (account) {
      avatar.textContent = initials(account.name);
      avatar.hidden = false;
      label.textContent = `Your account, ${account.name}`;
      label.classList.add("visually-hidden");
    } else {
      avatar.hidden = true;
      label.textContent = "Sign in";
      label.classList.remove("visually-hidden");
    }
    accountButton.classList.toggle("is-signed-in", Boolean(account));
    gate.hidden = Boolean(account);
    form.classList.toggle("is-locked", !account);
    if (!account && adding) adding.controller.abort();
    if (info) loadYours();
  }

  /* Run `action` now if somebody is signed in; otherwise ask them to sign in first. */
  function whenSignedIn(action, reason) {
    if (account) {
      action();
      return;
    }
    pending = action;
    openAccount("signin", reason);
  }

  function sessionEnded(retry) {
    setAccount(null);
    whenSignedIn(retry, "Your session has ended. Sign in again to carry on.");
  }

  function openAccount(view, reason) {
    showView(view, reason);
    if (!accountDialog.open) accountDialog.showModal();
    focusFirst(view);
  }

  function showView(view, reason) {
    for (const node of accountDialog.querySelectorAll("[data-view]")) node.hidden = node.dataset.view !== view;
    $("account-title").textContent = views[view].title;
    const lead = $("account-lead");
    lead.textContent = view === "profile" ? `Signed in as @${account.username}` : reason || views[view].lead;
    // An <svg> has no `hidden` property of its own; the attribute still works.
    $("account-mark").toggleAttribute("hidden", view === "profile");
    $("profile-avatar").hidden = view !== "profile";
    if (view === "signup") loadCountries();
    if (view === "profile") renderProfile();
    for (const formNode of accountDialog.querySelectorAll("form.auth__form")) clearErrors(formNode);
  }

  function focusFirst(view) {
    const container = accountDialog.querySelector(`[data-view="${view}"]`);
    const target = container && container.querySelector("input, select, button");
    if (target) requestAnimationFrame(() => target.focus());
  }

  function renderProfile() {
    $("profile-avatar").textContent = initials(account.name);
    $("profile-name").textContent = account.name;
    $("profile-username").textContent = `@${account.username}`;
    $("profile-email").textContent = account.email;
    $("profile-phone").textContent = account.phone;
    $("admin-link").hidden = !account.admin;
    $("yours-link").hidden = !(info && info.own_lectures);
    $("delete-form").hidden = true;
    $("delete-start").setAttribute("aria-expanded", "false");
    $("delete-password").value = "";
  }

  function showAlert(node, message) {
    node.textContent = message;
    node.hidden = false;
  }

  function clearErrors(formNode) {
    for (const node of formNode.querySelectorAll(".field__error, .auth__alert")) {
      node.textContent = "";
      node.hidden = true;
    }
    for (const node of formNode.querySelectorAll("[aria-invalid]")) node.removeAttribute("aria-invalid");
  }

  /* Show each field's message under it; returns whether there were any. */
  function showErrors(prefix, errors) {
    let first = null;
    for (const [field, message] of Object.entries(errors)) {
      const note = $(`${prefix}-${field}-error`);
      if (note) {
        note.textContent = message;
        note.hidden = false;
      }
      const inputs = field === "phone" ? [$(`${prefix}-country`), $(`${prefix}-phone`)] : [$(`${prefix}-${field}`)];
      for (const node of inputs.filter(Boolean)) node.setAttribute("aria-invalid", "true");
      if (!first) first = inputs.find(Boolean);
    }
    if (first) first.focus();
    return Object.keys(errors).length > 0;
  }

  /* Editing a field takes its message away; the next attempt to send checks it again. */
  function clearFieldError(node) {
    if (!node.id || !node.hasAttribute("aria-invalid")) return;
    const field = node.id.replace(/-country$/, "-phone");
    for (const id of [field, field.replace(/-phone$/, "-country")]) {
      const related = document.getElementById(id);
      if (related) related.removeAttribute("aria-invalid");
    }
    const note = document.getElementById(`${field}-error`);
    if (note) {
      note.textContent = "";
      note.hidden = true;
    }
  }

  accountDialog.addEventListener("input", (event) => clearFieldError(event.target));
  accountDialog.addEventListener("change", (event) => clearFieldError(event.target));

  function setBusy(formNode, busy) {
    formNode.setAttribute("aria-busy", busy ? "true" : "false");
    for (const button of formNode.querySelectorAll("button[type=submit]")) button.disabled = busy;
  }

  const unreachable = "Slideseek could not be reached. Check your connection and try again.";

  /* Hugging Face shows the app inside a frame on its own site, and some browsers (Safari)
   * keep cookies away from such frames. Signing in there would seem to work and then
   * fail, so inside a frame the page checks that the session stuck. */
  const framed = window.self !== window.top;

  async function sessionHeld() {
    if (!framed) return true;
    try {
      return Boolean((await getJSON("/api/account")).account);
    } catch (error) {
      return true;
    }
  }

  function cookieBlocked(node, created) {
    node.textContent = created
      ? "Your account is ready, but this browser does not allow signing in inside an embedded page. "
      : "This browser does not allow signing in inside an embedded page. ";
    const link = element("a", "link", "Open Slideseek in a new tab");
    link.href = location.href;
    link.target = "_blank";
    link.rel = "noopener";
    node.append(link);
    node.hidden = false;
  }

  function signedIn(next, greeting) {
    setAccount(next);
    $("signin-form").reset();
    $("signup-form").reset();
    updateRules("");
    const action = pending;
    pending = null;
    accountDialog.close();
    announce(greeting);
    if (action) action();
    else input.focus();
  }

  /* -- sign in */

  $("signin-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formNode = event.currentTarget;
    clearErrors(formNode);
    const login = $("signin-login").value.trim();
    const password = $("signin-password").value;
    const errors = {};
    if (!login) errors.login = "Enter your username or email address.";
    if (!password) errors.password = "Enter your password.";
    if (showErrors("signin", errors)) return;
    setBusy(formNode, true);
    try {
      const reply = await sendJSON("POST", "/api/account/signin", { login, password });
      if (!reply.ok) {
        showAlert($("signin-alert"), reply.data.detail || "Signing in did not work. Try again.");
        $("signin-password").select();
        return;
      }
      if (!(await sessionHeld())) {
        cookieBlocked($("signin-alert"), false);
        return;
      }
      signedIn(reply.data.account, `Welcome back, ${reply.data.account.name.split(" ")[0]}.`);
    } catch (error) {
      showAlert($("signin-alert"), unreachable);
    } finally {
      setBusy(formNode, false);
    }
  });

  /* -- sign up */

  function updateRules(password) {
    for (const item of $("password-rules").children) {
      const met = RULE_TESTS[item.dataset.rule](password);
      item.classList.toggle("is-met", met);
      item.querySelector("[data-rule-state]").textContent = met ? ", done" : ", not yet";
    }
  }

  $("signup-password").addEventListener("input", (event) => updateRules(event.target.value));

  function loadCountries() {
    if (countries) return countries;
    const select = $("signup-country");
    countries = getJSON("/api/account/countries")
      .then(({ countries: list }) => {
        let names = null;
        try {
          names = new Intl.DisplayNames([navigator.language || "en"], { type: "region" });
        } catch (error) {
          names = null;
        }
        const nameOf = (region) => {
          try {
            return (names && names.of(region)) || region;
          } catch (error) {
            return region;
          }
        };
        const items = list
          .map(({ region, code }) => ({ region, code, name: nameOf(region) }))
          .sort((a, b) => a.name.localeCompare(b.name));
        select.textContent = "";
        for (const item of items) {
          // The code first: a narrow picker cuts the end of the text off, not the start.
          const option = element("option", "", `+${item.code} ${item.name}`);
          option.value = item.region;
          select.append(option);
        }
        select.value = guessRegion(items);
      })
      .catch(() => {
        countries = null;
      });
    return countries;
  }

  /* The visitor's own country, as far as their browser's language says (en-AU is AU). */
  function guessRegion(items) {
    const known = new Set(items.map((item) => item.region));
    for (const tag of navigator.languages || [navigator.language || ""]) {
      const region = (tag.split("-").find((part, i) => i > 0 && /^[A-Za-z]{2}$/.test(part)) || "").toUpperCase();
      if (known.has(region)) return region;
    }
    return known.has("US") ? "US" : items[0].region;
  }

  $("signup-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formNode = event.currentTarget;
    clearErrors(formNode);
    const values = {
      name: $("signup-name").value.trim(),
      username: $("signup-username").value.trim(),
      email: $("signup-email").value.trim(),
      country: $("signup-country").value,
      phone: $("signup-phone").value.trim(),
      password: $("signup-password").value,
    };
    const errors = {};
    if (!values.name) errors.name = "Enter your name.";
    if (!values.username) errors.username = "Choose a username.";
    else if (!USERNAME.test(values.username)) errors.username = "Use 3 to 30 letters, numbers, dots, dashes or underscores, starting with a letter or a number.";
    if (!values.email) errors.email = "Enter your email address.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email)) errors.email = "Enter an email address such as name@example.com.";
    if (!values.phone) errors.phone = "Enter your phone number.";
    else if (!values.country && !values.phone.startsWith("+")) errors.phone = "Choose the country code of your phone number.";
    if (!values.password) errors.password = "Choose a password.";
    else if (Object.values(RULE_TESTS).some((test) => !test(values.password))) errors.password = "Your password needs everything in the list below.";
    if (showErrors("signup", errors)) return;
    setBusy(formNode, true);
    try {
      const reply = await sendJSON("POST", "/api/account/signup", values);
      if (!reply.ok) {
        const fieldErrors = reply.data.errors || {};
        if (!Object.keys(fieldErrors).length) showAlert($("signup-alert"), reply.data.detail || "Your account could not be created. Try again.");
        showErrors("signup", fieldErrors);
        return;
      }
      if (!(await sessionHeld())) {
        cookieBlocked($("signup-alert"), true);
        return;
      }
      signedIn(reply.data.account, `Welcome to Slideseek, ${reply.data.account.name.split(" ")[0]}.`);
    } catch (error) {
      showAlert($("signup-alert"), unreachable);
    } finally {
      setBusy(formNode, false);
    }
  });

  /* -- the signed-in view: sign out, or delete the account */

  $("signout-button").addEventListener("click", async () => {
    try {
      await sendJSON("POST", "/api/account/signout");
    } catch (error) {
      showAlert($("profile-alert"), unreachable);
      return;
    }
    setAccount(null);
    accountDialog.close();
    resultsSection.hidden = true;
    announce("You have signed out.");
  });

  $("delete-start").addEventListener("click", (event) => {
    const formNode = $("delete-form");
    formNode.hidden = !formNode.hidden;
    event.currentTarget.setAttribute("aria-expanded", String(!formNode.hidden));
    if (!formNode.hidden) $("delete-password").focus();
  });

  $("delete-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formNode = event.currentTarget;
    clearErrors(formNode);
    const password = $("delete-password").value;
    if (showErrors("delete", password ? {} : { password: "Enter your password to confirm." })) return;
    setBusy(formNode, true);
    try {
      const reply = await sendJSON("DELETE", "/api/account", { password });
      if (!reply.ok) {
        const fieldErrors = reply.data.errors || {};
        if (!Object.keys(fieldErrors).length) showAlert($("delete-alert"), reply.data.detail || "Your account could not be deleted. Try again.");
        showErrors("delete", fieldErrors);
        return;
      }
      setAccount(null);
      accountDialog.close();
      resultsSection.hidden = true;
      announce("Your account has been deleted.");
    } catch (error) {
      showAlert($("delete-alert"), unreachable);
    } finally {
      setBusy(formNode, false);
    }
  });

  /* -- the sheet itself */

  for (const button of accountDialog.querySelectorAll("[data-show-view]")) {
    button.addEventListener("click", () => {
      showView(button.dataset.showView);
      focusFirst(button.dataset.showView);
    });
  }

  for (const button of accountDialog.querySelectorAll("[data-reveal]")) {
    button.addEventListener("click", () => {
      const field = $(button.dataset.reveal);
      const show = field.type === "password";
      field.type = show ? "text" : "password";
      button.textContent = show ? "Hide" : "Show";
      button.setAttribute("aria-pressed", String(show));
      field.focus();
    });
  }

  accountButton.addEventListener("click", () => openAccount(account ? "profile" : "signin"));
  $("gate-signin").addEventListener("click", () => openAccount("signin"));
  $("gate-signup").addEventListener("click", () => openAccount("signup"));

  // Closed without signing in: whatever was waiting is dropped. Passwords never linger.
  accountDialog.addEventListener("close", () => {
    pending = null;
    for (const button of accountDialog.querySelectorAll("[data-reveal]")) {
      const field = $(button.dataset.reveal);
      field.value = "";
      field.type = "password";
      button.textContent = "Show";
      button.setAttribute("aria-pressed", "false");
    }
    updateRules("");
  });

  // A click on the dimmed page behind the sheet closes it; a drag out of a field does not.
  let pressedOnBackdrop = false;
  accountDialog.addEventListener("pointerdown", (event) => {
    pressedOnBackdrop = event.target === accountDialog;
  });
  accountDialog.addEventListener("click", (event) => {
    if (event.target === accountDialog && pressedOnBackdrop) accountDialog.close();
  });

  /* ------------------------------------------------------------ player */

  function openPlayer(result) {
    playerFrame.textContent = "";
    const source = result.player;
    if (source.kind === "own") {
      playOwn(result);
    } else if (source.kind === "youtube") {
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
    if (playerFrame.dataset.url) URL.revokeObjectURL(playerFrame.dataset.url);
    delete playerFrame.dataset.url;
  });
  player.addEventListener("click", (event) => {
    if (event.target === player) player.close();
  });

  /* ------------------------------------------------------------ your own lectures
   *
   * A recording is read in this browser and never uploaded: /capture/adder.js samples its
   * frames, grabs its slides and listens to it, and sends those to /api/library, where the
   * server finishes it with the indexer's own code. A result from one of your own lectures
   * plays from the file on your computer.
   */
  const BROWSER_STAGES = ["frames", "stills", "speech"];
  const STAGE_TEXT = {
    frames: "Not finished: its slides are still to be found",
    stills: "Not finished: its slides are still to be sent",
    speech: "Not finished: it is still to be listened to",
    building: "Building its search",
    ready: "Ready",
    failed: "Could not be finished",
  };
  let captureModule = null;

  function capture() {
    if (!captureModule) {
      captureModule = import("/capture/adder.js").catch((error) => {
        captureModule = null;
        throw error;
      });
    }
    return captureModule;
  }

  function isRecording(file) {
    const name = (file.name || "").toLowerCase();
    return (file.type || "").startsWith("video/") || /\.(mp4|m4v|mov|webm|mkv)$/.test(name);
  }

  /* What the public lectures are called on the scope switch: "MIT", if they all say so. */
  function publicLabel() {
    const firsts = new Set(((info && info.lectures) || []).map((l) => (l.course || "").split(/\s+/)[0]).filter(Boolean));
    return firsts.size === 1 ? [...firsts][0] : "Public";
  }

  async function loadYours() {
    if (info && info.own_lectures && account) {
      try {
        ownLectures = (await getJSON("/api/library")).lectures;
      } catch (error) {
        ownLectures = [];
      }
    } else {
      ownLectures = [];
    }
    renderYours();
  }

  function renderYours() {
    const enabled = Boolean(info && info.own_lectures);
    $("yours").hidden = !enabled;
    $("nav-yours").hidden = !enabled;
    if (!enabled) return;
    const list = $("yours-list");
    list.textContent = "";
    for (const lecture of ownLectures) list.append(yourRow(lecture));
    $("yours-note").textContent = account
      ? "Or drop a recording anywhere on this page."
      : "Sign in first: your lectures are kept for your account alone.";

    const ready = ownLectures.filter((lecture) => lecture.stage === "ready");
    scopeControl.hidden = !ready.length;
    if (!ready.length && scope !== "all") setScope("all");
    let group = lectureSelect.querySelector("optgroup");
    const current = lectureSelect.value;
    if (group) group.remove();
    if (ready.length) {
      group = element("optgroup");
      group.label = "Your lectures";
      for (const lecture of ready) {
        const option = element("option", "", lecture.title);
        option.value = lecture.lecture_id;
        group.append(option);
      }
      lectureSelect.append(group);
    }
    lectureSelect.value = [...lectureSelect.options].some((o) => o.value === current) ? current : "";
    lectureSelect.closest(".select").hidden = lectureSelect.options.length < 3;
  }

  function yourFacts(lecture) {
    const facts = [duration(lecture.duration_s)];
    if (adding && adding.id === lecture.lecture_id) facts.push(adding.summary);
    else if (lecture.stage === "ready") facts.push(plural(lecture.n_moments, "moment"), plural(lecture.n_slides, "slide"));
    else facts.push(STAGE_TEXT[lecture.stage] || lecture.stage);
    return facts.join(" · ");
  }

  function actionLink(label, onClick, className = "link") {
    const button = element("button", className, label);
    button.type = "button";
    button.addEventListener("click", onClick);
    return button;
  }

  function yourRow(lecture) {
    const row = element("li", "lecture lecture--own");
    row.dataset.lecture = lecture.lecture_id;
    const words = element("div", "lecture__words");
    words.append(element("h3", "lecture__title", lecture.title));
    words.append(element("p", "lecture__meta", yourFacts(lecture)));
    if (lecture.stage === "failed" && lecture.error) words.append(element("p", "lecture__error", lecture.error));

    const actions = element("div", "lecture__actions");
    const here = adding && adding.id === lecture.lecture_id;
    if (lecture.stage === "ready") {
      const only = actionLink("Search this lecture", () => {
        lectureSelect.value = lecture.lecture_id;
        setScope("all");
        input.focus();
        form.scrollIntoView({ behavior: "smooth", block: "center" });
        if (input.value.trim()) search(input.value, { push: true });
      });
      only.append(chevron());
      actions.append(only);
    } else if (here) {
      actions.append(actionLink("Show progress", () => adder.open || adder.showModal()));
    } else if (BROWSER_STAGES.includes(lecture.stage)) {
      actions.append(actionLink("Carry on", () => carryOn(lecture)));
    }
    actions.append(actionLink("Rename", () => startRename(row, lecture), "link link--quiet"));
    actions.append(actionLink("Delete", () => confirmDelete(row, lecture), "link link--quiet"));
    row.append(words, actions);
    return row;
  }

  function startRename(row, lecture) {
    const words = row.querySelector(".lecture__words");
    const formNode = element("form", "rename");
    const field = element("input", "rename__input");
    field.value = lecture.title;
    field.maxLength = 120;
    field.setAttribute("aria-label", "Title");
    const save = element("button", "link", "Save");
    save.type = "submit";
    formNode.append(field, save, actionLink("Cancel", renderYours, "link link--quiet"));
    words.replaceChildren(formNode);
    field.focus();
    field.select();
    field.addEventListener("keydown", (event) => {
      if (event.key === "Escape") renderYours();
    });
    formNode.addEventListener("submit", async (event) => {
      event.preventDefault();
      const title = field.value.trim();
      if (!title || title === lecture.title) {
        renderYours();
        return;
      }
      const reply = await sendJSON("PATCH", `/api/library/lectures/${encodeURIComponent(lecture.lecture_id)}`, { title }).catch(() => null);
      if (!reply || !reply.ok) {
        announce((reply && reply.data.detail) || "The lecture could not be renamed.");
        return;
      }
      if (adding && adding.id === lecture.lecture_id) adding.title = title;
      await loadYours();
      announce("Renamed.");
    });
  }

  function confirmDelete(row, lecture) {
    const actions = row.querySelector(".lecture__actions");
    const question = element("span", "lecture__confirm", `Delete “${lecture.title}”? This can’t be undone.`);
    const yes = actionLink("Delete", async () => {
      if (adding && adding.id === lecture.lecture_id) adding.controller.abort();
      try {
        const module = await capture();
        await module.library.remove(lecture.lecture_id);
      } catch (error) {
        announce(error.message || "The lecture could not be deleted.");
        return;
      }
      if (lectureSelect.value === lecture.lecture_id) lectureSelect.value = "";
      await loadYours();
      announce(`Deleted “${lecture.title}”.`);
    }, "link link--danger");
    actions.replaceChildren(question, yes, actionLink("Cancel", renderYours, "link link--quiet"));
    yes.focus();
  }

  /* -- the scope switch: all, yours, or the public ones */

  function setScope(next) {
    scope = next;
    for (const option of scopeControl.querySelectorAll("[data-scope]")) {
      const on = option.dataset.scope === next;
      option.setAttribute("aria-checked", String(on));
      option.tabIndex = on ? 0 : -1;
    }
  }

  function scopeChanged() {
    if (scope !== "all") lectureSelect.value = "";
    if (picture) searchPicture(picture.file, { scroll: false });
    else if (input.value.trim()) search(input.value, { push: true });
  }

  scopeControl.addEventListener("click", (event) => {
    const option = event.target.closest("[data-scope]");
    if (!option || option.dataset.scope === scope) return;
    setScope(option.dataset.scope);
    scopeChanged();
  });
  scopeControl.addEventListener("keydown", (event) => {
    const options = [...scopeControl.querySelectorAll("[data-scope]")];
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const next = options[(options.findIndex((o) => o.dataset.scope === scope) + step + options.length) % options.length];
    setScope(next.dataset.scope);
    next.focus();
    scopeChanged();
  });

  /* -- the sheet that adds a lecture */

  const STEPS = ["frames", "slides", "model", "speech", "building"];
  const stepNode = (name) => adder.querySelector(`[data-step="${name}"]`);
  const meter = $("adder-meter");

  function showAdderView(view) {
    for (const node of adder.querySelectorAll("[data-adder]")) node.hidden = node.dataset.adder !== view;
  }

  function showCannot(message) {
    $("adder-title").textContent = "Add a lecture";
    $("adder-lead").textContent = "";
    $("adder-cannot").textContent = message;
    showAdderView("cannot");
    if (!adder.open) adder.showModal();
  }

  function setStep(name, state, detail = "") {
    const node = stepNode(name);
    node.classList.toggle("is-active", state === "active");
    node.classList.toggle("is-done", state === "done");
    node.querySelector(".steps__detail").textContent = detail;
    if (state === "active") node.setAttribute("aria-current", "step");
    else node.removeAttribute("aria-current");
  }

  function finishSteps(upTo) {
    for (const name of STEPS.slice(0, STEPS.indexOf(upTo))) {
      if (!stepNode(name).classList.contains("is-done")) setStep(name, "done");
    }
  }

  function setMeter(fraction) {
    if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) meter.removeAttribute("value");
    else meter.value = Math.max(0, Math.min(1, fraction));
  }

  function minutesLeft(seconds) {
    const minutes = Math.max(1, Math.round(seconds / 60));
    return minutes === 1 ? "about a minute left" : `about ${minutes} minutes left`;
  }

  function summarise(text) {
    if (!adding) return;
    adding.summary = text;
    const row = $("yours-list").querySelector(`[data-lecture="${adding.id}"] .lecture__meta`);
    if (row) {
      const lecture = ownLectures.find((item) => item.lecture_id === adding.id);
      if (lecture) row.textContent = yourFacts(lecture);
    }
  }

  const downloads = new Map(); // speech model file -> { loaded, total }

  function progressed(step, details) {
    if (step === "made") {
      adding.id = details.lecture_id;
      loadYours();
      return;
    }
    if (step === "frames") {
      setStep("frames", "active", details.sending ? "Looking for the slide changes" : `${Math.round(100 * details.fraction)}%`);
      setMeter(details.sending ? null : details.fraction);
      summarise("Finding the slides");
    } else if (step === "slides") {
      finishSteps("slides");
      setStep("slides", "active", `${details.sent} of ${details.total}`);
      setMeter(details.total ? details.sent / details.total : null);
      summarise("Sending the slides");
    } else if (step === "sound") {
      finishSteps("model");
      setStep("speech", "active", "Reading the sound");
      setMeter(null);
      summarise("Listening");
    } else if (step === "model") {
      if (adding.cached) {
        // Read from this computer, not downloaded: a moment's work, part of listening.
        setStep("speech", "active", "Loading the speech model");
        setMeter(null);
        return;
      }
      downloads.set(details.file, { loaded: details.loaded || 0, total: details.total || 0 });
      let loaded = 0;
      let total = 0;
      for (const item of downloads.values()) {
        loaded += item.loaded;
        total += item.total;
      }
      stepNode("model").hidden = false;
      setStep("speech", "waiting"); // the sound is read; listening starts once the model is here
      setStep("model", "active", total ? `${megabytes(loaded)} of ${megabytes(total)}` : "");
      setMeter(total ? loaded / total : null);
      summarise("Getting the speech model");
    } else if (step === "speech") {
      finishSteps("speech");
      const fraction = details.count ? details.done / details.count : 0;
      const left = details.left === null ? "" : ` · ${minutesLeft(details.left)}`;
      setStep("speech", "active", `${Math.round(100 * fraction)}%${left}`);
      setMeter(fraction);
      summarise(`Listening, ${Math.round(100 * fraction)}%`);
    } else if (step === "building") {
      finishSteps("building");
      setStep("building", "active", "Reading the slides and making them searchable");
      setMeter(null);
      summarise("Building its search");
    } else if (step === "ready") {
      finishSteps("building");
      setStep("building", "done");
      setMeter(1);
    }
  }

  async function openAdder(found = null) {
    if (!account) {
      whenSignedIn(() => openAdder(found), "Sign in to add a lecture of your own.");
      return;
    }
    if (adding) {
      showAdderView("progress");
      if (!adder.open) adder.showModal();
      return;
    }
    let module;
    try {
      module = await capture();
    } catch (error) {
      announce("Adding a lecture could not start. Reload the page and try again.");
      return;
    }
    const can = await module.support();
    if (!can.ok) {
      showCannot(can.reason);
      return;
    }
    if (!found) found = await module.chooseRecording(recordingInput);
    if (!found) return;
    if (!isRecording(found.file)) {
      showCannot("That file is not a recording this page can read.");
      return;
    }
    // The recording of a lecture left unfinished: carry on with that one.
    const unfinished = ownLectures.find(
      (lecture) => BROWSER_STAGES.includes(lecture.stage) && module.sameFile(found.file, lecture.file)
    );
    if (unfinished) {
      runAdding({ file: found.file, handle: found.handle, title: unfinished.title, lecture: unfinished.lecture_id });
      return;
    }
    let length;
    try {
      length = await module.lengthOf(found.file);
    } catch (error) {
      showCannot(error.message || "This browser cannot play that file.");
      return;
    }
    choice = { ...found, duration: length.duration };
    $("adder-title").textContent = "Add a lecture";
    $("adder-lead").textContent = "";
    $("adder-file-name").textContent = found.file.name;
    $("adder-file-meta").textContent = `${duration(length.duration)} · ${megabytes(found.file.size)}`;
    $("adder-name").value = module.titleFrom(found.file.name);
    $("adder-alert").hidden = true;
    $("adder-name-error").hidden = true;
    $("adder-name").removeAttribute("aria-invalid");

    const cached = await module.speechModelCached();
    // Measured on an M4: 4.4 times faster than real time, plus a minute for the slides.
    const estimate = Math.round(length.duration / 4.4 / 60) + 1;
    const facts = [
      "It is read on this computer. The video is never uploaded.",
      cached
        ? `It takes about ${estimate} minutes on a recent computer. Keep this page open until it is done.`
        : `The first time, your browser downloads a ${megabytes(module.SPEECH.downloadBytes)} speech model, once. Then it takes about ${estimate} minutes on a recent computer. Keep this page open until it is done.`,
    ];
    const list = $("adder-facts");
    list.textContent = "";
    for (const fact of facts) list.append(element("li", "", fact));
    showAdderView("choose");
    if (!adder.open) adder.showModal();
    requestAnimationFrame(() => $("adder-name").focus());
  }

  async function runAdding({ file, handle = null, title, lecture = null }) {
    const module = await capture();
    adding = { id: lecture, title, controller: new AbortController(), summary: "Starting" };
    downloads.clear();
    for (const name of STEPS) setStep(name, "waiting");
    adding.cached = await module.speechModelCached();
    stepNode("model").hidden = adding.cached;
    setMeter(0);
    $("adder-title").textContent = title;
    $("adder-lead").textContent = file.name;
    showAdderView("progress");
    if (!adder.open) adder.showModal();
    if (lecture) loadYours();
    try {
      const done = await module.addLecture({
        file,
        handle,
        title,
        lecture,
        owner: account.username,
        signal: adding.controller.signal,
        report: progressed,
      });
      adding = null;
      module.rememberChoice(done.lecture_id, file, done.file, handle); // plays without asking again
      $("adder-title").textContent = `“${done.title}” is ready.`;
      $("adder-lead").textContent = `${plural(done.n_moments, "moment")} from ${plural(done.n_slides, "slide")}, searchable now.`;
      showAdderView("done");
      $("adder-search").onclick = () => {
        adder.close();
        lectureSelect.value = done.lecture_id;
        setScope("all");
        input.focus();
        form.scrollIntoView({ behavior: "smooth", block: "center" });
      };
      await loadYours();
      lectureSelect.value = done.lecture_id;
      if (!adder.open) announce(`“${done.title}” is ready to search.`);
    } catch (error) {
      const stopped = error.name === "AbortError";
      adding = null;
      await loadYours();
      if (stopped) {
        if (adder.open) adder.close();
        announce("Stopped. Carry on whenever you like, from Your lectures.");
      } else {
        $("adder-title").textContent = title;
        showCannot(error.message || "Something went wrong. Carry on from Your lectures to try again.");
      }
    }
  }

  /* Carry on with a lecture that was left unfinished: same file, from where it stopped. */
  async function carryOn(lecture) {
    if (adding) {
      openAdder();
      return;
    }
    let module;
    try {
      module = await capture();
    } catch (error) {
      announce("Adding a lecture could not start. Reload the page and try again.");
      return;
    }
    let file = await module.recordingOf(lecture.lecture_id, lecture.file);
    let handle = null;
    if (!file) {
      const can = await module.support();
      if (!can.ok) {
        showCannot(can.reason);
        return;
      }
      announce(`Choose “${lecture.file.name}”, the recording this lecture came from.`);
      const found = await module.chooseRecording(recordingInput);
      if (!found) return;
      if (!module.sameFile(found.file, lecture.file)) {
        announce(`That is not “${lecture.file.name}”. Choose the file this lecture was added from.`);
        return;
      }
      ({ file, handle } = found);
    }
    runAdding({ file, handle, title: lecture.title, lecture: lecture.lecture_id });
  }

  $("adder-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const title = $("adder-name").value.trim().replace(/\s+/g, " ");
    if (!title) {
      $("adder-name-error").textContent = "Give the lecture a title.";
      $("adder-name-error").hidden = false;
      $("adder-name").setAttribute("aria-invalid", "true");
      $("adder-name").focus();
      return;
    }
    if (!choice) return;
    const { file, handle } = choice;
    choice = null;
    runAdding({ file, handle, title });
  });
  $("adder-name").addEventListener("input", () => {
    $("adder-name-error").hidden = true;
    $("adder-name").removeAttribute("aria-invalid");
  });
  $("adder-other").addEventListener("click", () => openAdder());
  $("adder-stop").addEventListener("click", () => adding && adding.controller.abort());
  $("add-lecture").addEventListener("click", () => openAdder());
  $("yours-link").addEventListener("click", () => accountDialog.close());
  adder.addEventListener("close", () => {
    if (adding) announce(`Still adding “${adding.title}”. It carries on while this page is open.`);
  });
  adder.addEventListener("click", (event) => {
    if (event.target === adder) adder.close();
  });
  // Leaving while a lecture is being added would stop it; the browser asks first.
  window.addEventListener("beforeunload", (event) => {
    if (adding) event.preventDefault();
  });

  /* -- playing a result from one of your own lectures, from the file on this computer */

  function playFile(file, start) {
    const url = URL.createObjectURL(file);
    const video = element("video");
    video.src = url;
    video.controls = true;
    video.autoplay = true;
    video.playsInline = true;
    video.addEventListener("loadedmetadata", () => {
      video.currentTime = start;
    }, { once: true });
    playerFrame.replaceChildren(video);
    playerFrame.dataset.url = url;
  }

  async function playOwn(result) {
    const source = result.player;
    const box = element("div", "pick");
    playerFrame.replaceChildren(box);
    let module;
    try {
      module = await capture();
    } catch (error) {
      box.append(element("p", "pick__text", "This page could not open the recording. Reload it and try again."));
      return;
    }
    const file = await module.recordingOf(source.lecture_id, source.file);
    if (file) {
      playFile(file, source.start);
      return;
    }
    const text = element("p", "pick__text", `Choose “${source.file.name}” on this computer to play it from ${result.timestamp}. It plays from your computer and is not uploaded.`);
    const choose = element("button", "button", "Choose the recording");
    choose.type = "button";
    choose.addEventListener("click", async () => {
      const found = await module.chooseRecording(recordingInput);
      if (!found) return;
      if (!module.rememberChoice(source.lecture_id, found.file, source.file, found.handle)) {
        text.textContent = `That is not “${source.file.name}”. Choose the file this lecture was added from.`;
        return;
      }
      playFile(found.file, source.start);
    });
    box.append(text, choose);
  }

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
    if (lectureSelect.value) setScope("all");
    if (picture) searchPicture(picture.file, { scroll: false });
    else if (input.value.trim()) search(input.value, { push: true });
  });

  rerankToggle.addEventListener("change", () => {
    if (input.value.trim()) search(input.value);
  });

  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    const sheetOpen = player.open || accountDialog.open || adder.open;
    if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !sheetOpen) {
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
    const wanted = params.get("scope");
    setScope(!scopeControl.hidden && ["mine", "public"].includes(wanted) ? wanted : "all");
    if (!account) {
      // Nothing is searched until the visitor signs in; a shared link keeps its question.
      if (query) input.value = query;
      resultsSection.hidden = true;
      return;
    }
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
    const [library, visitor] = await Promise.allSettled([getJSON("/api/info"), getJSON("/api/account")]);
    setAccount(visitor.status === "fulfilled" ? visitor.value.account : null);
    if (library.status === "rejected") {
      status.textContent = "The library could not be loaded.";
      resultsSection.hidden = false;
      return;
    }
    info = library.value;
    rerankOption.hidden = !info.rerank;
    renderPictureBenchmark();
    renderStats();
    renderLectureFilter();
    renderExamples();
    renderLibrary();
    renderBenchmark();
    $("scope-public").textContent = publicLabel();
    await loadYours();
    restoreFromUrl({ fallback: true });
  }

  start();
})();
