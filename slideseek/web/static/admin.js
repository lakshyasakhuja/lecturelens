/* Slideseek accounts page, for admins.
 *
 * Plain JavaScript and no build step, like the search page. Everything comes from
 * /api/account (who is looking) and /api/admin/accounts (the accounts, and what changes
 * them). The page only hides what an admin may not do; the server is what refuses it.
 * Text from the server is only ever inserted as text nodes, never as HTML.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const PAGE = 100;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const RULE_TESTS = {
    length: (text) => [...text].length >= 8,
    upper: (text) => /\p{Lu}/u.test(text),
    lower: (text) => /\p{Ll}/u.test(text),
    number: (text) => /\p{Nd}/u.test(text),
    special: (text) => /[\p{P}\p{S}]/u.test(text),
  };
  const USERNAME = /^[A-Za-z0-9][A-Za-z0-9._-]{2,29}$/;
  const unreachable = "Slideseek could not be reached. Check your connection and try again.";

  const roster = $("roster");
  const editor = $("editor");
  const editorForm = $("editor-form");
  const filter = $("filter-query");
  const toast = $("toast");

  let query = "";
  let shown = []; // the accounts in the list, in order
  let total = 0;
  let counts = null; // { all, admins } as the server last said, kept up to date here
  let editing = null; // the account open in the editor; null for a new one
  let countries = null;
  let loading = null;

  /* ------------------------------------------------------------ helpers */

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function chevron() {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 12 12");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", "roster__chevron");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", "m4.5 2.5 3.5 3.5-3.5 3.5");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "1.6");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.append(path);
    return svg;
  }

  async function request(method, url, body) {
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

  function initials(name) {
    const words = name.trim().split(/\s+/).filter(Boolean);
    const first = (word) => (word ? [...word][0] : "");
    return (first(words[0]) + (words.length > 1 ? first(words[words.length - 1]) : "")).toUpperCase();
  }

  function plural(count, word) {
    return `${count.toLocaleString()} ${word}${count === 1 ? "" : "s"}`;
  }

  function day(iso) {
    const date = new Date(`${iso}T00:00:00Z`);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  }

  function signedInText(account) {
    if (account.sessions === null || account.sessions === undefined) return "";
    if (!account.sessions) return "Not signed in";
    return account.sessions === 1 ? "Signed in on 1 device" : `Signed in on ${account.sessions} devices`;
  }

  function tags(account) {
    const list = [];
    if (account.default) list.push(["Default admin", "default"]);
    else if (account.admin) list.push(["Admin", "admin"]);
    if (account.you) list.push(["You", "you"]);
    return list;
  }

  function renderTags(node, account) {
    node.textContent = "";
    for (const [label, kind] of tags(account)) node.append(element("li", `tag tag--${kind}`, label));
    node.hidden = !node.children.length;
  }

  /* ------------------------------------------------------------ who is looking */

  function showState(title, text) {
    $("console").hidden = true;
    $("summary").textContent = "";
    $("state-title").textContent = title;
    $("state-text").textContent = text;
    $("state").hidden = false;
    if (editor.open) editor.close();
  }

  function refused(status) {
    if (status === 401) {
      showState("Your session has ended.", "Sign in again on the search page, with an admin account, to come back here.");
      return true;
    }
    if (status === 403) {
      showState("Only admins can see this page.", "This account is not an admin account, or no longer is one.");
      return true;
    }
    return false;
  }

  /* ------------------------------------------------------------ the list */

  function renderSummary() {
    const parts = [plural(counts.all, "account"), plural(counts.admins, "admin")];
    $("summary").textContent = query
      ? `${total.toLocaleString()} of ${plural(counts.all, "account")} match.`
      : `${parts.join(", ")}.`;
  }

  function renderRow(account) {
    const item = element("li", "roster__item");
    item.dataset.id = String(account.id);
    const row = element("button", "roster__row");
    row.type = "button";
    const described = [account.name, `@${account.username}`, account.email, ...tags(account).map(([label]) => label), signedInText(account)];
    row.setAttribute("aria-label", described.filter(Boolean).join(", "));
    row.addEventListener("click", () => openEditor(account.id));

    const avatar = element("span", "avatar roster__avatar", initials(account.name));
    avatar.setAttribute("aria-hidden", "true");
    const text = element("span", "roster__text");
    text.append(element("span", "roster__name", account.name));
    text.append(element("span", "roster__meta", `@${account.username} · ${account.email}`));
    const side = element("span", "roster__side");
    const labels = element("ul", "tags");
    renderTags(labels, account);
    side.append(labels);
    const status = element("span", "roster__status", signedInText(account));
    if (account.sessions) status.classList.add("is-on");
    side.append(status);
    row.append(avatar, text, side, chevron());
    item.append(row);
    return item;
  }

  function renderList() {
    roster.textContent = "";
    for (const account of shown) roster.append(renderRow(account));
    $("empty").hidden = shown.length > 0;
    roster.hidden = shown.length === 0;
    $("more").hidden = shown.length >= total;
  }

  async function load({ more = false } = {}) {
    const offset = more ? shown.length : 0;
    const params = new URLSearchParams({ q: query, offset: String(offset), limit: String(PAGE) });
    const asked = query;
    roster.setAttribute("aria-busy", "true");
    const pendingLoad = request("GET", `/api/admin/accounts?${params}`);
    loading = pendingLoad;
    let reply;
    try {
      reply = await pendingLoad;
    } catch (error) {
      if (loading === pendingLoad) showState("The accounts could not be loaded.", unreachable);
      return;
    } finally {
      if (loading === pendingLoad) roster.setAttribute("aria-busy", "false");
    }
    if (loading !== pendingLoad || asked !== query) return; // a newer search has started
    if (!reply.ok) {
      if (!refused(reply.status)) showState("The accounts could not be loaded.", reply.data.detail || "Try again in a moment.");
      return;
    }
    shown = more ? shown.concat(reply.data.accounts) : reply.data.accounts;
    total = reply.data.total;
    counts = { all: reply.data.all, admins: reply.data.admins };
    renderSummary();
    renderList();
  }

  let filterTimer = 0;
  filter.addEventListener("input", () => {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(() => {
      query = filter.value.trim();
      load();
    }, 250);
  });
  $("filter").addEventListener("submit", (event) => {
    event.preventDefault();
    clearTimeout(filterTimer);
    query = filter.value.trim();
    load();
  });
  $("more").addEventListener("click", () => load({ more: true }));

  function replaceShown(account) {
    const at = shown.findIndex((item) => item.id === account.id);
    if (at < 0) return;
    // A change made here does not count sessions; keep what the list knew.
    shown[at] = { ...shown[at], ...account, sessions: account.sessions ?? shown[at].sessions, signed_in: account.signed_in ?? shown[at].signed_in };
    const old = roster.querySelector(`[data-id="${account.id}"]`);
    if (old) old.replaceWith(renderRow(shown[at]));
  }

  /* ------------------------------------------------------------ the editor */

  const fields = ["name", "username", "email", "country", "phone", "password"];

  function showAlert(node, message) {
    node.textContent = message;
    node.hidden = false;
  }

  function clearErrors() {
    for (const node of editor.querySelectorAll(".field__error, .auth__alert")) {
      node.textContent = "";
      node.hidden = true;
    }
    for (const node of editor.querySelectorAll("[aria-invalid]")) node.removeAttribute("aria-invalid");
  }

  function showErrors(errors) {
    let first = null;
    for (const [field, message] of Object.entries(errors)) {
      const note = $(`admin-${field}-error`);
      if (note) {
        note.textContent = message;
        note.hidden = false;
      }
      const inputs = field === "phone" ? [$("admin-country"), $("admin-phone")] : [$(`admin-${field}`)];
      for (const node of inputs.filter(Boolean)) node.setAttribute("aria-invalid", "true");
      if (!first) first = inputs.find(Boolean);
    }
    if (first) first.focus();
    return Object.keys(errors).length > 0;
  }

  editor.addEventListener("input", (event) => {
    const node = event.target;
    if (!node.id || !node.hasAttribute("aria-invalid")) return;
    const field = node.id === "admin-country" ? "admin-phone" : node.id;
    for (const id of [field, field === "admin-phone" ? "admin-country" : ""]) {
      if (id) $(id).removeAttribute("aria-invalid");
    }
    const note = $(`${field}-error`);
    if (note) {
      note.textContent = "";
      note.hidden = true;
    }
  });

  function updateRules(password) {
    for (const item of $("admin-rules").children) {
      const met = RULE_TESTS[item.dataset.rule](password);
      item.classList.toggle("is-met", met);
      item.querySelector("[data-rule-state]").textContent = met ? ", done" : ", not yet";
    }
  }

  $("admin-password").addEventListener("input", (event) => {
    updateRules(event.target.value);
    // For an existing account the rules matter only once a new password is typed.
    if (editing) $("admin-rules").hidden = !event.target.value;
  });

  function loadCountries() {
    if (countries) return countries;
    const select = $("admin-country");
    countries = fetch("/api/account/countries", { headers: { Accept: "application/json" } })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error("no countries"))))
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
        select.append(element("option", "", "Choose"));
        select.firstChild.value = "";
        for (const item of items) {
          const option = element("option", "", `+${item.code} ${item.name}`);
          option.value = item.region;
          select.append(option);
        }
        return items;
      })
      .catch(() => {
        countries = null;
        return [];
      });
    return countries;
  }

  function guessRegion(items) {
    const known = new Set(items.map((item) => item.region));
    for (const tag of navigator.languages || [navigator.language || ""]) {
      const region = (tag.split("-").find((part, i) => i > 0 && /^[A-Za-z]{2}$/.test(part)) || "").toUpperCase();
      if (known.has(region)) return region;
    }
    return known.has("US") ? "US" : "";
  }

  function setDisabled(node, disabled) {
    node.disabled = disabled;
    node.closest(".field, .setting")?.classList.toggle("is-disabled", disabled);
  }

  async function openEditor(id) {
    editing = id === null ? null : shown.find((account) => account.id === id) || null;
    const account = editing;
    clearErrors();
    editorForm.reset();
    updateRules("");

    const locked = Boolean(account && account.default && !account.you);
    const ownDefault = Boolean(account && account.default && account.you);
    const avatar = $("editor-avatar");
    avatar.textContent = account ? initials(account.name) : "";
    avatar.classList.toggle("avatar--blank", !account);
    $("editor-title").textContent = account ? account.name : "New account";
    const lead = account
      ? [`@${account.username}`, `Joined ${day(account.created)}`, signedInText(account)].filter(Boolean).join(" · ")
      : "It can be signed in to straight away, with the password you set here.";
    $("editor-lead").textContent = lead;
    renderTags($("editor-tags"), account || {});

    $("admin-name").value = account ? account.name : "";
    $("admin-username").value = account ? account.username : "";
    $("admin-email").value = account ? account.email : "";
    $("admin-phone").value = account ? account.national : "";
    $("admin-password-label").textContent = account ? "New password (optional)" : "Password";
    $("admin-password-hint").hidden = !account;
    $("admin-password-hint").textContent = account && account.you
      ? "Leave it empty to keep your password. A new one signs you out everywhere else."
      : "Leave it empty to keep the current password. A new one signs the account out everywhere.";
    $("admin-rules").hidden = Boolean(account);
    // Nothing can be changed here, so there is no password to type.
    $("admin-password").closest(".field").hidden = locked;
    $("admin-admin").checked = Boolean(account && account.admin);

    for (const field of fields) setDisabled($(`admin-${field}`), locked);
    setDisabled($("admin-username"), locked || ownDefault);
    const ownRights = Boolean(account && account.you && account.admin);
    setDisabled($("admin-admin"), locked || Boolean(account && account.default) || ownRights);
    $("admin-admin-detail").textContent = account && account.default
      ? "The default admin is always an admin."
      : ownRights
        ? "Another admin can remove your admin rights; you cannot."
        : "Can see and change every account, and create admins.";
    $("editor-submit").hidden = locked;
    $("editor-submit").textContent = account ? "Save changes" : "Create account";

    const note = $("editor-note");
    note.hidden = !(locked || ownDefault);
    note.textContent = locked
      ? "This is the default admin's account. Only the default admin can change it."
      : "This is the default admin account. Its username is set where the server runs, and it is always an admin.";

    const actions = $("editor-actions");
    actions.hidden = !account || locked;
    $("signout-everywhere").textContent = account && account.you ? "Sign out everywhere else" : "Sign out everywhere";
    $("signout-everywhere").disabled = Boolean(account && !account.you && !account.sessions);
    const removable = Boolean(account && !account.default && !account.you);
    $("remove-start").hidden = !removable;
    $("remove-start").setAttribute("aria-expanded", "false");
    $("remove-confirm").hidden = true;
    if (account) {
      $("remove-text").textContent =
        `This deletes @${account.username}'s account, with its name, email address and phone number, ` +
        "and signs it out everywhere. It cannot be undone.";
      $("remove-account").textContent = `Delete @${account.username}`;
    }

    if (!editor.open) editor.showModal();
    const items = await loadCountries();
    if (editing !== account) return; // another account was opened meanwhile
    $("admin-country").value = account ? account.country : guessRegion(items);
    if (!locked) requestAnimationFrame(() => $("admin-name").focus());
  }

  $("new-account").addEventListener("click", () => openEditor(null));

  function setBusy(busy) {
    editorForm.setAttribute("aria-busy", busy ? "true" : "false");
    $("editor-submit").disabled = busy;
  }

  editorForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearErrors();
    const account = editing;
    const values = {
      name: $("admin-name").value.trim(),
      username: $("admin-username").value.trim(),
      email: $("admin-email").value.trim(),
      country: $("admin-country").value,
      phone: $("admin-phone").value.trim(),
      admin: $("admin-admin").checked,
    };
    const password = $("admin-password").value;
    const errors = {};
    if (!values.name) errors.name = "Enter a name.";
    if (!values.username) errors.username = "Choose a username.";
    else if (!USERNAME.test(values.username)) errors.username = "Use 3 to 30 letters, numbers, dots, dashes or underscores, starting with a letter or a number.";
    if (!values.email) errors.email = "Enter an email address.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email)) errors.email = "Enter an email address such as name@example.com.";
    if (!values.phone) errors.phone = "Enter a phone number.";
    else if (!values.country && !values.phone.startsWith("+")) errors.phone = "Choose the country code of the phone number.";
    if (!account && !password) errors.password = "Choose a password.";
    else if (password && Object.values(RULE_TESTS).some((test) => !test(password))) {
      errors.password = "The password needs everything in the list below.";
      $("admin-rules").hidden = false;
    }
    if (showErrors(errors)) return;
    if (password) values.password = password;

    setBusy(true);
    try {
      const reply = account
        ? await request("PATCH", `/api/admin/accounts/${account.id}`, values)
        : await request("POST", "/api/admin/accounts", values);
      if (!reply.ok) {
        // A 403 here is this one change refused (the reason is in the alert), not the page.
        if (reply.status === 401) {
          refused(401);
          return;
        }
        const fieldErrors = reply.data.errors || {};
        if (!Object.keys(fieldErrors).length) showAlert($("editor-alert"), reply.data.detail || "That did not work. Try again.");
        showErrors(fieldErrors);
        return;
      }
      const saved = reply.data.account;
      editor.close();
      if (account) {
        if (saved.admin !== account.admin) {
          counts.admins += saved.admin ? 1 : -1;
          renderSummary();
        }
        replaceShown(saved);
        if (saved.you) setMe(saved);
        announce(password ? `Saved. @${saved.username} has a new password.` : `Saved @${saved.username}.`);
      } else {
        announce(`Created @${saved.username}${saved.admin ? ", an admin" : ""}.`);
        filter.value = "";
        query = "";
        load();
      }
    } catch (error) {
      showAlert($("editor-alert"), unreachable);
    } finally {
      setBusy(false);
    }
  });

  $("signout-everywhere").addEventListener("click", async (event) => {
    const account = editing;
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const reply = await request("POST", `/api/admin/accounts/${account.id}/signout`);
      if (!reply.ok) {
        if (reply.status === 401) refused(401);
        else showAlert($("actions-alert"), reply.data.detail || "That did not work. Try again.");
        return;
      }
      const ended = reply.data.ended;
      editor.close();
      replaceShown({ ...account, sessions: account.you ? Math.max(1, account.sessions - ended) : 0 });
      const where = ended === 1 ? "1 device" : `${ended} devices`;
      announce(account.you ? `Signed you out on ${where} elsewhere.` : `Signed @${account.username} out on ${where}.`);
    } catch (error) {
      showAlert($("actions-alert"), unreachable);
    } finally {
      button.disabled = false;
    }
  });

  $("remove-start").addEventListener("click", (event) => {
    const confirm = $("remove-confirm");
    confirm.hidden = !confirm.hidden;
    event.currentTarget.setAttribute("aria-expanded", String(!confirm.hidden));
    if (!confirm.hidden) $("remove-account").focus();
  });

  $("remove-account").addEventListener("click", async (event) => {
    const account = editing;
    const button = event.currentTarget;
    button.disabled = true;
    try {
      const reply = await request("DELETE", `/api/admin/accounts/${account.id}`);
      if (!reply.ok && reply.status !== 404) {
        if (reply.status === 401) refused(401);
        else showAlert($("actions-alert"), reply.data.detail || "That did not work. Try again.");
        return;
      }
      editor.close();
      announce(`Deleted @${account.username}.`);
      load();
    } catch (error) {
      showAlert($("actions-alert"), unreachable);
    } finally {
      button.disabled = false;
    }
  });

  for (const button of editor.querySelectorAll("[data-reveal]")) {
    button.addEventListener("click", () => {
      const field = $(button.dataset.reveal);
      const show = field.type === "password";
      field.type = show ? "text" : "password";
      button.textContent = show ? "Hide" : "Show";
      button.setAttribute("aria-pressed", String(show));
      field.focus();
    });
  }

  // Closed: the password typed for someone never lingers.
  editor.addEventListener("close", () => {
    const field = $("admin-password");
    field.value = "";
    field.type = "password";
    const reveal = editor.querySelector("[data-reveal]");
    reveal.textContent = "Show";
    reveal.setAttribute("aria-pressed", "false");
    updateRules("");
  });

  let pressedOnBackdrop = false;
  editor.addEventListener("pointerdown", (event) => {
    pressedOnBackdrop = event.target === editor;
  });
  editor.addEventListener("click", (event) => {
    if (event.target === editor && pressedOnBackdrop) editor.close();
  });

  /* ------------------------------------------------------------ start */

  function setMe(account) {
    const avatar = $("me-avatar");
    avatar.textContent = initials(account.name);
    avatar.title = `Signed in as @${account.username}`;
    avatar.hidden = false;
  }

  async function start() {
    let reply;
    try {
      reply = await request("GET", "/api/account");
    } catch (error) {
      showState("Slideseek could not be reached.", "Check your connection and try again.");
      return;
    }
    const account = reply.ok ? reply.data.account : null;
    if (!account) {
      showState("Sign in first.", "Sign in on the search page with an admin account, then come back here.");
      return;
    }
    setMe(account);
    if (!account.admin) {
      showState("Only admins can see this page.", "This account is not an admin account.");
      return;
    }
    $("console").hidden = false;
    loadCountries();
    load();
  }

  start();
})();
