"use strict";

const STORE_KEY = "legal-assistant-state";
const CONTEXT_EXCHANGES = 3;
const CONTEXT_ANSWER_CHARS = 1200;
const MAX_INPUT_HEIGHT = 160;
const FAILED_PREFIX = "Ответ не получен";
const STATUS_VOID = "утратил силу";
const LAW_TYPE = "статья закона";
const SUSPICIOUS_NOTE = "Не является содержанием документа: строка исключена из ответов";
const NO_FRAGMENT = "Фрагмент в документе не определён — откроется документ целиком.";

const state = { messages: [], docs: [], cache: new Map(), busy: false, opened: null, timer: 0 };

function byId(id) {
  return document.getElementById(id);
}

function make(tag, className, text) {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function formatDate(iso) {
  const parts = String(iso || "").split("-");
  return parts.length === 3 ? parts.reverse().join(".") : "";
}

function seconds(ms) {
  return (Math.max(0, Number(ms) || 0) / 1000).toFixed(1).replace(".", ",") + " с";
}

function flat(text) {
  return String(text || "").split(/\s+/).filter(Boolean).join(" ");
}

function isFailed(reply) {
  return Boolean(reply.failed) || (reply.warnings || []).some((warning) => warning.startsWith(FAILED_PREFIX));
}

function badge(status) {
  return make("span", "badge " + (status === STATUS_VOID ? "void" : "active"), status);
}

function validity(doc) {
  const parts = [];
  if (doc.valid_from) {
    parts.push("с " + formatDate(doc.valid_from));
  }
  if (doc.valid_to) {
    parts.push("по " + formatDate(doc.valid_to));
  }
  return parts.join(" ");
}

function save() {
  try {
    const data = { messages: state.messages, asOf: byId("as-of").value, opened: state.opened };
    sessionStorage.setItem(STORE_KEY, JSON.stringify(data));
  } catch (error) {
    return;
  }
}

function restore() {
  let saved = null;
  try {
    saved = JSON.parse(sessionStorage.getItem(STORE_KEY) || "null");
  } catch (error) {
    saved = null;
  }
  if (!saved || !Array.isArray(saved.messages)) {
    return;
  }
  state.messages = saved.messages;
  state.opened = saved.opened || null;
  byId("as-of").value = saved.asOf || "";
  const last = state.messages[state.messages.length - 1];
  if (last && last.role === "user") {
    state.messages.push(failure("страница была перезагружена до получения ответа"));
  }
}

function failure(reason) {
  return { role: "assistant", failed: true, answer: "Не удалось получить ответ: " + reason + ".", sources: [], warnings: [] };
}

function exchanges() {
  const pairs = [];
  for (let index = 0; index + 1 < state.messages.length; index += 1) {
    const user = state.messages[index];
    const reply = state.messages[index + 1];
    if (user.role === "user" && reply.role === "assistant" && !isFailed(reply)) {
      pairs.push([user, reply]);
    }
  }
  return pairs.slice(-CONTEXT_EXCHANGES);
}

function withContext(question, pairs) {
  if (!pairs.length) {
    return question;
  }
  const lines = [];
  for (const [user, reply] of pairs) {
    const answer = flat(String(reply.answer || "").replace(/\s*\[\d+\]/g, ""));
    lines.push("Пользователь: " + flat(user.text), "Ассистент: " + answer.slice(0, CONTEXT_ANSWER_CHARS));
  }
  lines.push("Пользователь: " + question);
  return lines.join("\n");
}

async function errorText(response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = typeof body.detail === "string" ? body.detail : "";
  } catch (error) {
    detail = "";
  }
  if (response.status === 422) {
    return detail || "вопрос или дата заданы неверно";
  }
  return detail || "сервер ответил с ошибкой " + response.status;
}

async function ask(text) {
  const question = String(text || "").trim();
  if (!question || state.busy) {
    return;
  }
  const asOf = byId("as-of").value || null;
  const pairs = exchanges();
  state.messages.push({ role: "user", text: question, asOf: asOf });
  setBusy(true);
  let reply;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: withContext(question, pairs), as_of_date: asOf }),
    });
    if (!response.ok) {
      throw new Error(await errorText(response));
    }
    const body = await response.json();
    reply = {
      role: "assistant",
      answer: String(body.answer || ""),
      sources: Array.isArray(body.sources) ? body.sources : [],
      warnings: Array.isArray(body.warnings) ? body.warnings : [],
      timings_ms: body.timings_ms || null,
    };
  } catch (error) {
    reply = failure(error instanceof TypeError ? "сервер недоступен" : error.message || "неизвестная ошибка");
  }
  state.messages.push(reply);
  setBusy(false);
  byId("question").focus();
}

function setBusy(busy) {
  state.busy = busy;
  window.clearInterval(state.timer);
  byId("send").disabled = busy;
  byId("new-chat").disabled = busy;
  for (const button of document.querySelectorAll(".example")) {
    button.disabled = busy;
  }
  renderMessages();
  save();
  if (!busy) {
    return;
  }
  const started = Date.now();
  state.timer = window.setInterval(() => {
    const clock = byId("pending-clock");
    if (clock) {
      clock.textContent = Math.round((Date.now() - started) / 1000) + " с";
    }
  }, 500);
}

function userBubble(message) {
  const bubble = make("div", "message user");
  bubble.append(make("p", "message-text", message.text));
  if (message.asOf) {
    bubble.append(make("span", "chip", "Дата события: " + formatDate(message.asOf)));
  }
  return bubble;
}

function fillAnswer(target, message, index) {
  const known = new Map((message.sources || []).map((source) => [String(source.n), source]));
  const answer = String(message.answer || "");
  let last = 0;
  for (const match of answer.matchAll(/\[(\d+)\]/g)) {
    const source = known.get(match[1]);
    if (!source) {
      continue;
    }
    target.append(answer.slice(last, match.index));
    const link = make("a", "cite", match[0]);
    link.href = "#" + sourceKey(index, source);
    link.title = source.title + (source.locator ? ", " + source.locator : "");
    link.addEventListener("click", (event) => {
      event.preventDefault();
      openSource(source, index);
    });
    target.append(link);
    last = match.index + match[0].length;
  }
  target.append(answer.slice(last));
}

function sourceKey(index, source) {
  return "source-" + index + "-" + source.n;
}

function warningBox(warnings) {
  const box = make("div", "warnings");
  box.setAttribute("role", "note");
  box.append(make("div", "warnings-title", "Обратите внимание"));
  const list = make("ul");
  for (const warning of warnings) {
    list.append(make("li", "", warning));
  }
  box.append(list);
  return box;
}

function sourceCard(source, index) {
  const card = make("button", "source");
  card.type = "button";
  card.id = sourceKey(index, source);
  const head = make("div", "source-head");
  head.append(make("span", "source-number", String(source.n)), make("span", "source-title", source.title), badge(source.status));
  card.append(head);
  if (source.locator) {
    card.append(make("div", "source-locator", source.locator));
  }
  card.append(make("div", "source-text" + (source.text ? "" : " missing"), source.text || NO_FRAGMENT));
  card.addEventListener("click", () => openSource(source, index));
  return card;
}

function timingText(timings) {
  return "Ответ за " + seconds(timings.total) + " · поиск " + seconds(timings.retrieval) + " · модель " + seconds(timings.llm);
}

function answerCard(message, index) {
  const card = make("article", "message assistant" + (message.failed ? " failed" : ""));
  const text = make("p", "message-text");
  fillAnswer(text, message, index);
  card.append(text);
  if (message.warnings && message.warnings.length) {
    card.append(warningBox(message.warnings));
  }
  if (message.sources && message.sources.length) {
    const sources = make("div", "sources");
    sources.append(make("div", "sources-title", "Источники"));
    for (const source of message.sources) {
      sources.append(sourceCard(source, index));
    }
    card.append(sources);
  }
  if (message.timings_ms) {
    card.append(make("div", "timings", timingText(message.timings_ms)));
  }
  return card;
}

function pendingCard() {
  const card = make("div", "message assistant pending");
  card.setAttribute("role", "status");
  const spinner = make("span", "spinner");
  spinner.setAttribute("aria-hidden", "true");
  const clock = make("span", "pending-clock", "0 с");
  clock.id = "pending-clock";
  card.append(spinner, make("span", "", "Ищу в документах и готовлю ответ…"), clock);
  return card;
}

function renderMessages() {
  const box = byId("messages");
  box.replaceChildren();
  byId("welcome").hidden = state.messages.length > 0 || state.busy;
  state.messages.forEach((message, index) => {
    box.append(message.role === "user" ? userBubble(message) : answerCard(message, index));
  });
  if (state.busy) {
    box.append(pendingCard());
  }
  markActive();
  const feed = byId("feed");
  feed.scrollTop = feed.scrollHeight;
}

function markActive() {
  const key = state.opened ? state.opened.key : "";
  for (const card of document.querySelectorAll(".source")) {
    card.classList.toggle("active", Boolean(key) && card.id === key);
  }
}

function docItem(doc) {
  const item = make("button", "doc-item");
  item.type = "button";
  const text = make("span", "doc-item-text");
  const dates = validity(doc);
  text.append(make("span", "doc-item-title", doc.title), make("span", "doc-item-meta", doc.type + (dates ? " · " + dates : "")));
  item.append(text, badge(doc.status));
  item.addEventListener("click", () => openDocument(doc.doc_id, null, ""));
  return item;
}

function renderDocs(failed) {
  const list = byId("docs-list");
  list.replaceChildren();
  byId("docs-count").textContent = failed ? "" : String(state.docs.length);
  if (failed) {
    list.append(make("p", "docs-empty", "Список документов недоступен. Обновите страницу."));
    return;
  }
  const groups = [
    ["Документы компании", state.docs.filter((doc) => doc.type !== LAW_TYPE)],
    ["Статьи законов", state.docs.filter((doc) => doc.type === LAW_TYPE)],
  ];
  for (const [title, docs] of groups) {
    if (!docs.length) {
      continue;
    }
    list.append(make("div", "docs-group", title));
    for (const doc of docs) {
      list.append(docItem(doc));
    }
  }
}

async function loadDocs() {
  try {
    const response = await fetch("/api/docs");
    if (!response.ok) {
      throw new Error(String(response.status));
    }
    state.docs = await response.json();
    renderDocs(false);
  } catch (error) {
    state.docs = [];
    renderDocs(true);
  }
}

function toggleDocs(open) {
  const panel = byId("docs-panel");
  const show = open === undefined ? panel.hidden : open;
  panel.hidden = !show;
  byId("docs-toggle").setAttribute("aria-expanded", String(show));
}

function openSource(source, index) {
  const range = source.start >= 0 && source.end > source.start ? { start: source.start, end: source.end } : null;
  openDocument(source.doc_id, range, sourceKey(index, source));
}

function viewerNote(text) {
  const note = byId("viewer-note");
  note.textContent = text || "";
  note.hidden = !text;
}

async function openDocument(docId, range, key) {
  state.opened = { doc_id: docId, start: range ? range.start : -1, end: range ? range.end : -1, key: key || "" };
  save();
  toggleDocs(false);
  markActive();
  document.body.classList.add("viewer-open");
  let doc = state.cache.get(docId);
  if (!doc) {
    viewerNote("Загружаю документ…");
    try {
      const response = await fetch("/api/docs/" + encodeURIComponent(docId));
      if (!response.ok) {
        throw new Error(response.status === 404 ? "документ не найден в базе" : "сервер ответил с ошибкой " + response.status);
      }
      doc = await response.json();
      state.cache.set(docId, doc);
    } catch (error) {
      if (state.opened && state.opened.doc_id === docId) {
        viewerNote("Не удалось открыть документ: " + (error instanceof TypeError ? "сервер недоступен" : error.message) + ".");
      }
      return;
    }
  }
  if (state.opened && state.opened.doc_id === docId) {
    renderDocument(doc, state.opened);
  }
}

function offsetMapper(text) {
  if (!/[\uD800-\uDBFF]/.test(text)) {
    return (offset) => offset;
  }
  const starts = [];
  for (let index = 0; index < text.length; index += text.codePointAt(index) > 0xffff ? 2 : 1) {
    starts.push(index);
  }
  starts.push(text.length);
  return (offset) => starts[Math.max(0, Math.min(offset, starts.length - 1))];
}

function lineClass(line) {
  const text = line.trim();
  if (text.length <= 160 && text === text.toUpperCase() && /[A-ZА-ЯЁ]/.test(text)) {
    return " doc-heading";
  }
  if (text.length <= 90 && /^\d+\.\s+\S/.test(text) && !/[.;:]$/.test(text)) {
    return " doc-section";
  }
  return "";
}

function renderDocument(doc, opened) {
  byId("viewer-empty").hidden = true;
  byId("viewer-head").hidden = false;
  byId("viewer-title").textContent = doc.title;
  byId("viewer-status").replaceChildren(badge(doc.status));
  const dates = validity(doc);
  byId("viewer-meta").textContent = doc.type + (dates ? " · действие: " + dates : "");
  byId("viewer").classList.toggle("is-void", doc.status === STATUS_VOID);
  const box = byId("doc-text");
  box.hidden = false;
  box.replaceChildren();
  const text = String(doc.text || "");
  const shift = offsetMapper(text);
  const hit = opened.start >= 0 && opened.end > opened.start ? [shift(opened.start), shift(opened.end)] : null;
  const excluded = (doc.suspicious || []).map((span) => [shift(span.start), shift(span.end)]);
  let position = 0;
  let target = null;
  for (const line of text.split("\n")) {
    const start = position;
    const end = position + line.length;
    position = end + 1;
    if (!line.trim()) {
      continue;
    }
    const row = make("p", "doc-line" + lineClass(line));
    if (excluded.some(([from, to]) => from < end && start < to)) {
      row.classList.add("suspicious");
      row.append(make("span", "suspicious-note", SUSPICIOUS_NOTE));
    }
    if (hit && hit[0] < end && start < hit[1]) {
      const from = Math.max(hit[0], start) - start;
      const to = Math.min(hit[1], end) - start;
      const mark = make("mark", "hit", line.slice(from, to));
      row.append(line.slice(0, from), mark, line.slice(to));
      target = target || mark;
    } else {
      row.append(line);
    }
    box.append(row);
  }
  const lost = opened.key && !target;
  viewerNote(lost ? "Фрагмент источника в тексте не определён, документ открыт целиком." : "");
  if (target) {
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ block: "center", behavior: calm ? "auto" : "smooth" });
  } else {
    box.scrollTop = 0;
  }
}

function submit() {
  const input = byId("question");
  if (!input.value.trim() || state.busy) {
    return;
  }
  const text = input.value;
  input.value = "";
  resizeInput();
  ask(text);
}

function resizeInput() {
  const input = byId("question");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, MAX_INPUT_HEIGHT) + "px";
}

function syncDate() {
  byId("as-of-clear").hidden = !byId("as-of").value;
  save();
}

function newChat() {
  if (state.busy) {
    return;
  }
  state.messages = [];
  if (state.opened) {
    state.opened.key = "";
  }
  renderMessages();
  save();
  byId("question").focus();
}

function bind() {
  byId("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    submit();
  });
  byId("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submit();
    }
  });
  byId("question").addEventListener("input", resizeInput);
  for (const button of document.querySelectorAll(".example")) {
    button.addEventListener("click", () => ask(button.dataset.question));
  }
  byId("as-of").addEventListener("change", syncDate);
  byId("as-of-clear").addEventListener("click", () => {
    byId("as-of").value = "";
    syncDate();
  });
  byId("docs-toggle").addEventListener("click", () => toggleDocs());
  byId("new-chat").addEventListener("click", newChat);
  byId("viewer-close").addEventListener("click", () => document.body.classList.remove("viewer-open"));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".docs-menu")) {
      toggleDocs(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    if (!byId("docs-panel").hidden) {
      toggleDocs(false);
      byId("docs-toggle").focus();
    } else {
      document.body.classList.remove("viewer-open");
    }
  });
}

function init() {
  restore();
  bind();
  byId("as-of-clear").hidden = !byId("as-of").value;
  renderMessages();
  loadDocs();
  if (state.opened) {
    const range = state.opened.start >= 0 ? { start: state.opened.start, end: state.opened.end } : null;
    openDocument(state.opened.doc_id, range, state.opened.key);
    document.body.classList.remove("viewer-open");
  }
}

init();
